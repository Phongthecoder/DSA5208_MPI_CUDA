#!/usr/bin/env python3
"""Summarize repeated MPI runs and optionally draw the scaling figure."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(paths):
    for path in paths:
        with path.open(newline="") as stream:
            yield from csv.DictReader(stream)


def summarize(paths, keys, metrics, time_column):
    groups = defaultdict(list)
    for row in read_rows(paths):
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for values, group in groups.items():
        row = dict(zip(keys, values))
        row["ranks"] = int(row["ranks"])
        row["particles"] = int(row["particles"])
        row["repeats"] = len(group)
        for metric in metrics:
            row[metric] = statistics.median(float(run[metric]) for run in group)
        row["min_ms"] = min(float(run[time_column]) for run in group)
        row["max_ms"] = max(float(run[time_column]) for run in group)
        summaries.append(row)
    return sorted(summaries, key=lambda row: (row.get("suite", ""), row["ranks"],
                                              row["resampler"], row.get("resampling", "")))


def workload(row):
    particles = row["particles"]
    if row["suite"] == "weak":
        particles //= row["ranks"]
    return (row["suite"], particles, row["steps"], row["data_seed"], row["filter_seed"], row["resampling"])


def add_scaling_metrics(rows):
    baselines = {workload(row): row["filter_ms"] for row in rows
                 if row["ranks"] == 1 and row["resampler"] == "distributed"}
    for row in rows:
        baseline = baselines.get(workload(row))
        row["speedup"] = ""
        row["efficiency"] = ""
        if baseline is not None and row["suite"] == "strong":
            row["speedup"] = baseline / row["filter_ms"]
            row["efficiency"] = row["speedup"] / row["ranks"]
        if baseline is not None and row["suite"] == "weak":
            row["efficiency"] = baseline / row["filter_ms"]


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_scaling(rows, title, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    strong = [row for row in rows if row["suite"] == "strong" and row["resampler"] == "distributed"]
    weak = [row for row in rows if row["suite"] == "weak"]
    if not strong or not weak or any(row["efficiency"] == "" for row in strong + weak):
        raise ValueError("scaling plots require a one-rank reference for each workload")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    fig.suptitle(title)
    ranks = [row["ranks"] for row in strong]
    axes[0].plot(ranks, ranks, "--", color="0.5", label="Ideal")
    axes[0].plot(ranks, [row["speedup"] for row in strong], "o-", label="Measured")
    axes[0].set(title=f'Strong scaling | N = {strong[0]["particles"]:,}', ylabel="Speedup")
    axes[0].legend()
    axes[1].plot(ranks, [100 * row["efficiency"] for row in strong], "o-", label="Strong")
    axes[1].plot([row["ranks"] for row in weak], [100 * row["efficiency"] for row in weak],
                 "s-", label="Weak")
    axes[1].set(title="Parallel efficiency", ylabel="Efficiency (%)")
    axes[1].legend()
    axes[2].plot(ranks, [row["max_rank_mpi_pct"] for row in strong], "o-", label="Strong")
    axes[2].plot([row["ranks"] for row in weak], [row["max_rank_mpi_pct"] for row in weak],
                 "s-", label="Weak")
    axes[2].set(title="Time in MPI calls", ylabel="Highest rank share (%)", ylim=(0, 100))
    axes[2].legend()
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks)
        axis.set_xticklabels([str(rank) for rank in ranks])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("MPI ranks")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180, facecolor="white")


def draw_resampling(rows, micro, title, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    fig.suptitle(title)
    ranks = sorted({row["ranks"] for row in rows if row["suite"] == "strong"})
    for method, marker in (("centralized", "s"), ("distributed", "o")):
        strong = [row for row in rows if row["suite"] == "strong" and row["resampler"] == method]
        if strong:
            axes[0].plot([row["ranks"] for row in strong], [row["filter_ms"] / 1000 for row in strong],
                         marker=marker, label=method.capitalize())
    axes[0].set(title="Filter time by resampler", xlabel="MPI ranks", ylabel="Seconds")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ranks)
    axes[0].set_xticklabels([str(rank) for rank in ranks])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend()

    max_ranks = max(row["ranks"] for row in micro)
    subset = {(row["weight_shape"], row["resampler"]): row
              for row in micro if row["ranks"] == max_ranks}
    for method, offset in (("centralized", -0.19), ("distributed", 0.19)):
        timings = [subset[(weights, method)]["ms_per_resample"] for weights in ("uniform", "skewed")]
        axes[1].bar([offset, 1 + offset], timings, width=0.36, label=method.capitalize())
    axes[1].set(title=f"Resampling only | {max_ranks} ranks", ylabel="Milliseconds per resample")
    axes[1].set_ylim(0, 1.32 * max(row["ms_per_resample"] for row in subset.values()))
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Uniform", "Skewed"])
    axes[1].legend()

    ess = [row for row in rows if row["suite"] == "ess"]
    if ess:
        for row in ess:
            x = row["filter_ms"] / 1000
            y = row["position_rmse_m"]
            axes[2].scatter(x, y)
            axes[2].annotate(row["resampling"], (x, y), xytext=(5, 5), textcoords="offset points")
        axes[2].set(title=f'ESS tradeoff | {ess[0]["ranks"]} ranks', xlabel="Filter seconds",
                    ylabel="Position RMSE (m)")
        axes[2].margins(0.15)
    else:
        axes[2].set_visible(False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--resampling", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-title")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = ("suite", "resampler", "ranks", "nodes", "particles", "steps", "data_seed", "filter_seed",
            "resampling")
    metrics = ("filter_ms", "workflow_ms", "launch_wall_ms", "initialize_ms", "propagate_weight_ms",
               "normalize_estimate_ms", "resample_ms", "reduction_mpi_ms", "resample_mpi_ms",
               "max_rank_mpi_pct", "resample_off_rank_bytes", "max_packed_particles", "max_pack_ratio",
               "position_rmse_m", "velocity_rmse_mps", "log_likelihood", "resampled_steps")
    rows = summarize(args.inputs, keys, metrics, "filter_ms")
    if not rows:
        parser.error("no filter benchmark rows")
    add_scaling_metrics(rows)
    save_csv(args.output_dir / "mpi_summary.csv", rows)
    micro = []
    if args.resampling:
        micro = summarize(args.resampling, ("ranks", "nodes", "particles", "resampler", "weight_shape"),
                          ("ms_per_resample", "mpi_ms_per_resample", "off_rank_bytes_per_resample",
                           "max_pack_ratio"), "ms_per_resample")
        if micro:
            save_csv(args.output_dir / "resampling_summary.csv", micro)
    if args.plot_title:
        draw_scaling(rows, args.plot_title, args.output_dir / "mpi_scaling.png")
        if micro:
            draw_resampling(rows, micro, args.plot_title, args.output_dir / "mpi_resampling.png")


if __name__ == "__main__":
    main()
