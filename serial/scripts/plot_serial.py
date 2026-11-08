#!/usr/bin/env python3
"""Plot the sample radar track and measured serial scaling sweeps."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_scaling(axis, rows, dimension, xlabel, title):
    sizes = [int(row[dimension]) for row in rows]
    times = [float(row["filter_ms"]) for row in rows]
    lower = [max(0.0, time - float(row["filter_ms_min"])) for time, row in zip(times, rows)]
    upper = [max(0.0, float(row["filter_ms_max"]) - time) for time, row in zip(times, rows)]
    axis.plot(sizes, [times[0] * n / sizes[0] for n in sizes], "--", color="0.4", label="Proportional reference")
    axis.errorbar(sizes, times, yerr=[lower, upper], fmt="o", capsize=3, label="Median and range")
    axis.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="Filter time (ms)", title=title)
    axis.set_xticks(sizes)
    axis.set_xticklabels([f"{size:,}" for size in sizes])
    axis.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    track = load_csv(args.results_dir / "range_bearing_track.csv")
    benchmarks = load_csv(args.results_dir / "serial_benchmarks.csv")

    figure, (trajectory, particles, steps) = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    ranges = [float(row["observed_range_m"]) for row in track]
    bearings = [float(row["observed_bearing_rad"]) for row in track]
    trajectory.scatter([r * math.cos(b) for r, b in zip(ranges, bearings)],
                       [r * math.sin(b) for r, b in zip(ranges, bearings)],
                       s=9, color="0.55", alpha=0.45, label="Radar readings")
    for prefix, label, color in (("true", "Simulated truth", "#25364a"),
                                 ("estimated", "Particle filter", "#d66d24")):
        trajectory.plot([float(row[f"{prefix}_x_m"]) for row in track],
                        [float(row[f"{prefix}_y_m"]) for row in track], color=color, label=label)
    trajectory.set(xlabel="x (m)", ylabel="y (m)", title="Synthetic range-bearing track")
    trajectory.set_aspect("equal", adjustable="datalim")
    trajectory.legend(fontsize=8)
    plot_scaling(particles, [row for row in benchmarks if row["sweep"] == "vary_particles"],
                 "particles", "Particles N", "Particle sweep at T = 256")
    plot_scaling(steps, [row for row in benchmarks if row["sweep"] == "vary_steps"],
                 "steps", "Time steps T", "Step sweep at N = 16,384")
    figure.savefig(args.results_dir / "serial_validation.png", dpi=180)


if __name__ == "__main__":
    main()
