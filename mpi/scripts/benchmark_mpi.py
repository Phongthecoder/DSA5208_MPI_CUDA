#!/usr/bin/env python3
"""Run strong, weak, resampler, and ESS experiments with matched inputs."""

import argparse
import csv
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path


def append_row(path, row):
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def invoke(args, ranks, command):
    launcher = [args.launcher, *args.launcher_flags, "-n", str(ranks), *command]
    max_attempts = 3 if args.retry_tcp_bind else 1
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        completed = subprocess.run(launcher, capture_output=True, text=True)
        launch_ms = (time.perf_counter() - started) * 1000
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        if completed.returncode == 0:
            break
        tcp_bind_error = ("mca_btl_tcp_endpoint_start_connect" in completed.stderr
                          and "Address already in use" in completed.stderr)
        if not tcp_bind_error or attempt == max_attempts:
            completed.check_returncode()
        print(f"P={ranks}: Open MPI TCP bind failed, retrying after 65 s "
              f"(attempt {attempt + 1}/{max_attempts}).", file=sys.stderr, flush=True)
        time.sleep(65)
    row, = csv.DictReader(completed.stdout.splitlines())
    if int(row["ranks"]) != ranks:
        raise RuntimeError("launcher returned a different MPI rank count")
    if args.expected_nodes is not None and int(row["nodes"]) != args.expected_nodes:
        raise RuntimeError("ranks did not span the requested number of compute nodes")
    return row, launch_ms


def compare_estimates(reference, row):
    for field in ("position_rmse_m", "velocity_rmse_mps", "log_likelihood"):
        if not math.isclose(float(reference[field]), float(row[field]), rel_tol=1e-8, abs_tol=1e-8):
            raise RuntimeError(f"fixed-seed filter output changed in {field}")
    if row["resampled_steps"] != reference["resampled_steps"]:
        raise RuntimeError("fixed-seed resampling decisions changed")


def filter_command(args, method, particles, policy, steps):
    return [str(args.binary), "--model", "range-bearing", "--resampler", method,
            "--particles", str(particles), "--steps", str(steps), "--data-seed", "42",
            "--filter-seed", "1729", "--resample", policy]


def time_filter(args, suite, ranks, method, particles, policy):
    invoke(args, ranks, filter_command(args, method, particles, policy, min(8, args.steps)))
    expected = None
    for repeat in range(1, args.repeats + 1):
        row, launch_ms = invoke(args, ranks, filter_command(args, method, particles, policy, args.steps))
        if expected is None:
            expected = row
        compare_estimates(expected, row)
        append_row(args.output_dir / "mpi_runs.csv",
                   {"suite": suite, "repeat": repeat, "launch_wall_ms": launch_ms, **row})
        print(f"{suite:6} P={ranks:3} N={particles:8} {method:11} ESS={policy:>6} "
              f"run {repeat}/{args.repeats}: {float(row['filter_ms']):9.2f} ms", flush=True)


def time_resampling(args, ranks):
    for weights in ("uniform", "skewed"):
        for method in ("centralized", "distributed"):
            command = [str(args.resampling_binary), str(args.ess_particles), method, weights, "20"]
            for repeat in range(1, args.repeats + 1):
                row, launch_ms = invoke(args, ranks, command)
                append_row(args.output_dir / "resampling_runs.csv",
                           {"repeat": repeat, "launch_wall_ms": launch_ms, **row})
                print(f"resampling P={ranks:3} {weights:7} {method:11} "
                      f"run {repeat}/{args.repeats}: {float(row['ms_per_resample']):9.2f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/mpi_pf"))
    parser.add_argument("--resampling-binary", type=Path)
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--launcher-flags", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--compare-ranks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--ess-rank", type=int, default=32)
    parser.add_argument("--expected-nodes", type=int)
    parser.add_argument("--strong-particles", type=int, default=2**22)
    parser.add_argument("--weak-per-rank", type=int, default=2**14)
    parser.add_argument("--ess-particles", type=int, default=2**20)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--retry-tcp-bind", action="store_true",
                        help="retry a transient Open MPI TCP bind error up to twice")
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.strong_particles, args.weak_per_rank,
           args.ess_particles, *args.ranks) < 1 or len(set(args.ranks)) != len(args.ranks):
        parser.error("workloads and rank counts must be positive and rank counts must be distinct")
    args.binary = args.binary.resolve()
    if args.resampling_binary:
        args.resampling_binary = args.resampling_binary.resolve()
    args.launcher_flags = shlex.split(args.launcher_flags)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ("mpi_runs.csv", "resampling_runs.csv")):
        parser.error("the output directory already contains benchmark results")
    for ranks in args.ranks:
        time_filter(args, "strong", ranks, "distributed", args.strong_particles, "0.5")
        time_filter(args, "weak", ranks, "distributed", args.weak_per_rank * ranks, "0.5")
        if ranks in args.compare_ranks:
            time_filter(args, "strong", ranks, "centralized", args.strong_particles, "0.5")
            if args.resampling_binary:
                time_resampling(args, ranks)
        if ranks == args.ess_rank:
            for policy in ("0.25", "0.5", "0.75", "always"):
                time_filter(args, "ess", ranks, "distributed", args.ess_particles, policy)


if __name__ == "__main__":
    main()
