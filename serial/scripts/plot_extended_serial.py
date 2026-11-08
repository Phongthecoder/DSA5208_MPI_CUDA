#!/usr/bin/env python3
"""Plot the extended Vanda serial benchmark summary."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def particle_label(size):
    if size & (size - 1):
        return f"{size:,}"
    return f"$2^{{{size.bit_length() - 1}}}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.summary.open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["particles"]))

    sizes = [int(row["particles"]) for row in rows]
    medians = [float(row["filter_median_ms"]) / 1000 for row in rows]
    lower = [median - float(row["filter_min_ms"]) / 1000 for row, median in zip(rows, medians)]
    upper = [float(row["filter_max_ms"]) / 1000 - median for row, median in zip(rows, medians)]
    peak_rss = [float(row["peak_rss_max_mib"]) for row in rows]
    buffer_memory = [float(row["particle_buffers_mib"]) for row in rows]
    gaussian_share = [float(row["gaussian_share_filter_pct"]) for row in rows]
    gaussian_seconds = float(rows[-1]["gaussian_total_ms"]) / 1000

    fig, (runtime, memory, gaussian) = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    fig.suptitle(f'Serial particle filter on one Vanda core | {rows[0]["steps"]} time steps', fontsize=14)
    for axis in (runtime, memory, gaussian):
        axis.set_xscale("log", base=2)
        axis.set_xlim(sizes[0] / 1.5, sizes[-1] * 1.5)
        axis.set_xticks(sizes)
        axis.set_xticklabels([particle_label(size) for size in sizes])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Particle count")
        axis.grid(axis="y", alpha=0.25)

    runtime.set(title="Filter runtime", ylabel="Seconds", yscale="log")
    runtime.plot(sizes, [medians[0] * size / sizes[0] for size in sizes], "--",
                 color="0.5", label="Proportional reference")
    runtime.errorbar(sizes, medians, yerr=[lower, upper], fmt="o", capsize=3,
                     color="#166f94", label="Median and range")
    runtime.legend(fontsize=8)

    memory.set(title="Peak resident memory", ylabel="MiB", yscale="log")
    memory.plot(sizes, peak_rss, "o-", color="#985842", label="Peak RSS, max across runs")
    memory.plot(sizes, buffer_memory, "--", color="0.5", label="Particle buffers")
    memory.legend(fontsize=8)

    gaussian.set(title="Gaussian sampling", ylabel="Share of profiled filter time (%)", ylim=(0, 100))
    gaussian.plot(sizes, gaussian_share, "o-", color="#307159")
    gaussian.annotate(f"{gaussian_seconds:.1f} s at the largest size",
                      (sizes[-1], gaussian_share[-1]), xytext=(-5, 12),
                      textcoords="offset points", ha="right", fontsize=8)
    fig.savefig(args.output, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
