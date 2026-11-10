#!/usr/bin/env python3
"""Validate and benchmark the MPI + CUDA particle filter inside a GPU allocation."""

import argparse
import csv
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ACCURACY = ("position_rmse_m", "velocity_rmse_mps", "log_likelihood")


def launch(args, ranks, *flags):
    command = [args.launcher, *args.launcher_flags, "-n", str(ranks),
               str(args.binary), "--expect-gpu", args.gpu_model, *flags]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    return result, (time.perf_counter() - start) * 1000


def require_success(result):
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        print(result.stdout, end="", file=sys.stderr, flush=True)
        result.check_returncode()


def probe(args, ranks):
    result, _ = launch(args, ranks, "--probe")
    path = args.output / f"gpu_bindings_p{ranks}.tsv"
    path.write_text(result.stdout)
    require_success(result)
    rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
    if (sorted(int(row["rank"]) for row in rows) != list(range(ranks))
            or len({(row["host"], row["pci_bus"]) for row in rows}) != ranks
            or len({row["host"] for row in rows}) != args.nodes):
        raise RuntimeError(f"invalid GPU placement in {path}")
    return all(row["cuda_aware_mpi"] == "1" for row in rows)


def validate(args, ranks, transport):
    result, _ = launch(args, ranks, "--validate", "--transport", transport)
    (args.output / f"validation_p{ranks}_{transport}.txt").write_text(result.stdout + result.stderr)
    require_success(result)
    if f"agreement on {ranks} GPU ranks" not in result.stdout:
        raise RuntimeError("validation did not report success")
    print(f"Validation passed for {ranks} GPU rank(s), {transport} transport", flush=True)


def measure(args, ranks, transport, particles, steps):
    result, launch_ms = launch(args, ranks, "--transport", transport, "--particles", str(particles),
                               "--steps", str(steps), "--data-seed", "42", "--filter-seed", "1729")
    require_success(result)
    row, = csv.DictReader(result.stdout.splitlines())
    if (int(row["ranks"]), int(row["nodes"]), int(row["particles"]), int(row["steps"])) != (
            ranks, args.nodes, particles, steps):
        raise RuntimeError("benchmark output did not match the requested workload and placement")
    if row["transport"] != transport or row["resampling"] != "0.5" or row["gpu"] != args.gpu_model:
        raise RuntimeError("unexpected benchmark configuration")
    for field in (*ACCURACY, "filter_ms", "workflow_ms"):
        if not math.isfinite(float(row[field])):
            raise RuntimeError(f"nonfinite benchmark field {field}")
    if float(row["filter_ms"]) <= 0:
        raise RuntimeError("filter time must be positive")
    return row, launch_ms


def compare(reference, row):
    for key in ACCURACY:
        if not math.isclose(float(reference[key]), float(row[key]), abs_tol=1e-7, rel_tol=1e-7):
            raise RuntimeError(f"fixed-seed result changed in {key}")
    if reference["resampled_steps"] != row["resampled_steps"]:
        raise RuntimeError("fixed-seed resampling count changed")


def benchmark(args, ranks, transport, reference):
    path = args.output / "gpu_runs.csv"
    for suite, particles in (("strong", args.strong_particles), ("weak", ranks * args.weak_per_gpu)):
        measure(args, ranks, transport, particles, min(8, args.steps))
        key = (ranks, suite)
        for repeat in range(1, args.repeats + 1):
            row, launch_ms = measure(args, ranks, transport, particles, args.steps)
            compare(reference.setdefault(key, row), row)
            record = {"suite": suite, "repeat": repeat, "launch_wall_ms": launch_ms, **row}
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record), lineterminator="\n")
                if stream.tell() == 0:
                    writer.writeheader()
                writer.writerow(record)
            print(f"{suite} GPUs={ranks} N={particles} {transport} run {repeat}/{args.repeats}: "
                  f"{float(row['filter_ms']) / 1000:.4f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--nodes", type=int, choices=(1,), default=1)
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--launcher-flags", default="")
    parser.add_argument("--gpu-model", default="NVIDIA A40")
    parser.add_argument("--strong-particles", type=int, default=2**22)
    parser.add_argument("--weak-per-gpu", type=int, default=2**20)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.strong_particles, args.weak_per_gpu, args.steps, args.repeats, args.timeout) <= 0:
        parser.error("workloads, repeats, and timeout must be positive")
    args.binary = args.binary.resolve()
    args.launcher_flags = shlex.split(args.launcher_flags)
    # Keep the PBS cpuset even for jobs submitted with the old HPC-X launcher flags.
    if os.environ.get("PBS_JOBID") and os.environ.get("CUDA_MPI_STACK") == "hpcx":
        if "--bind-to" in args.launcher_flags:
            index = args.launcher_flags.index("--bind-to") + 1
            args.launcher_flags[index] = "none"
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "gpu_runs.csv").exists():
        parser.error("use a new output directory to avoid mixing repeats")
    (args.output / "launcher_flags.txt").write_text(shlex.join(args.launcher_flags) + "\n")

    reference = {}
    cuda_aware = False
    for ranks in range(1, args.gpus + 1):
        cuda_aware = probe(args, ranks)
        validate(args, ranks, "staged")
        if not args.validate_only:
            benchmark(args, ranks, "staged", reference)

    if args.gpus == 1:
        status = "not_applicable\tA second GPU is needed for device-buffer exchange."
    elif not cuda_aware:
        status = "unavailable\tThe MPI build did not report CUDA-aware support."
    else:
        try:
            validate(args, args.gpus, "device")
            if not args.validate_only:
                benchmark(args, args.gpus, "device", reference)
            status = "pass\tCUDA-aware device-buffer path validated."
        except (subprocess.SubprocessError, RuntimeError) as error:
            print(f"Device-buffer comparison stopped: {error}", file=sys.stderr)
            status = f"failed\t{type(error).__name__}; see job output and validation log."
    (args.output / "device_transport.tsv").write_text("status\tdetail\n" + status + "\n")


if __name__ == "__main__":
    main()
