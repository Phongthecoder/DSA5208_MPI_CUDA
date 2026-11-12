#!/usr/bin/env python3
"""Validate and measure one A40 across particle counts with the existing filter."""
import argparse
import csv
import json
from pathlib import Path
import shlex

import benchmark as gpu_benchmark


def append(path, row):
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--launcher-flags", default="")
    parser.add_argument("--gpu-model", default="NVIDIA A40")
    parser.add_argument("--particle-powers", nargs="+", type=int, default=list(range(10, 23, 2)))
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.warmup_steps, args.timeout) < 1:
        parser.error("steps, repetitions, warm-up, and timeout must be positive")
    if (not args.particle_powers or len(set(args.particle_powers)) != len(args.particle_powers)
            or any(power < 0 or power > 26 for power in args.particle_powers)):
        parser.error("particle powers must be distinct integers from 0 to 26")
    args.binary = args.binary.resolve()
    args.launcher_flags = shlex.split(args.launcher_flags)
    args.nodes = 1
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in ("gpu_sweep_runs.csv", "warmup_runs.csv", "validation_p1_staged.txt"):
        if (args.output / filename).exists():
            parser.error(f"use a fresh output directory: {filename} already exists")
    config = {"particles": [2**p for p in sorted(args.particle_powers)],
              "steps": args.steps, "repeats": args.repeats, "warmup_steps": args.warmup_steps,
              "data_seed": 42, "filter_seed": 1729, "resampling": "ESS < 0.5*N; systematic",
              "gpu": args.gpu_model, "mpi_ranks": 1, "transport": "staged",
              "timing": "filter_ms includes allocation, initialization, and 256 filter steps; launch_wall_ms also includes process startup"}
    (args.output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "launcher_flags.txt").write_text(shlex.join(args.launcher_flags) + "\n")
    gpu_benchmark.probe(args, 1)
    gpu_benchmark.validate(args, 1, "staged")
    for power in sorted(args.particle_powers):
        particles = 2**power
        warm, launch_ms = gpu_benchmark.measure(args, 1, "staged", particles,
                                               min(args.warmup_steps, args.steps))
        append(args.output / "warmup_runs.csv",
               {"particle_power": power, "launch_wall_ms": launch_ms, **warm})
        reference = None
        for repeat in range(1, args.repeats + 1):
            row, launch_ms = gpu_benchmark.measure(args, 1, "staged", particles, args.steps)
            if (row["model"], row["data_seed"], row["filter_seed"]) != ("range-bearing", "42", "1729"):
                raise RuntimeError("benchmark returned different model or seeds")
            reference = reference or row
            gpu_benchmark.compare(reference, row)
            append(args.output / "gpu_sweep_runs.csv",
                   {"suite": "particle_sweep", "particle_power": power, "repeat": repeat,
                    "launch_wall_ms": launch_ms, **row})
            print(f"GPU sweep N=2^{power} ({particles}) run {repeat}/{args.repeats}: "
                  f"{float(row['filter_ms']):.4f} ms; resampled {row['resampled_steps']}/{args.steps}",
                  flush=True)
    print("PASS: single-GPU particle sweep completed with reproducible fixed-seed outputs.", flush=True)


if __name__ == "__main__":
    main()
