#!/usr/bin/env python3
"""Repeat the two serial scaling sweeps on fixed synthetic observations."""

import argparse
import csv
import statistics
import subprocess
from pathlib import Path


def run(binary, particles, steps):
    command = [str(binary), "--model", "range-bearing", "--particles", str(particles),
               "--steps", str(steps), "--data-seed", "42", "--filter-seed", "1729",
               "--resample", "0.5"]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    summary, = csv.DictReader(output.splitlines())
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/serial_pf"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    binary = args.binary.resolve()
    cases = [("vary_particles", n, 256) for n in (1024, 4096, 16384, 65536)]
    cases += [("vary_steps", 16384, t) for t in (64, 128, 256, 512)]
    raw = []
    medians = []
    for sweep, particles, steps in cases:
        run(binary, particles, steps)  # one warm-up run per configuration
        samples = [run(binary, particles, steps) for _ in range(args.repeats)]
        timing_columns = [key for key in samples[0] if key.endswith("_ms")]
        fixed = {key: value for key, value in samples[0].items() if key not in timing_columns}
        for repeat, sample in enumerate(samples, start=1):
            if any(sample[key] != value for key, value in fixed.items()):
                raise RuntimeError("a fixed-seed run produced different filter results")
            raw.append({"sweep": sweep, "repeat": repeat, **sample})
        median = {key: statistics.median(float(row[key]) for row in samples) for key in timing_columns}
        filter_times = [float(row["filter_ms"]) for row in samples]
        medians.append({"sweep": sweep, "repeats": args.repeats, **fixed,
                        **{key: f"{value:.6f}" for key, value in median.items()},
                        "filter_ms_min": min(filter_times), "filter_ms_max": max(filter_times)})
        print(f"{sweep:14} N={particles:6} T={steps:3} median filter time {median['filter_ms']:9.3f} ms",
              flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "serial_benchmarks.csv", medians)
    write_csv(args.output_dir / "serial_benchmark_runs.csv", raw)


if __name__ == "__main__":
    main()
