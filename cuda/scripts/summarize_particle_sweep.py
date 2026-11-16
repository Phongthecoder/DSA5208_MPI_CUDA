#!/usr/bin/env python3
"""Summarize and plot validated single-GPU particle-count experiments."""
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics


METRICS = ("filter_ms", "workflow_ms", "launch_wall_ms", "initialize_ms",
           "propagate_weight_ms", "normalize_estimate_ms", "resample_ms",
           "staging_ms", "reduction_mpi_ms", "resample_mpi_ms",
           "position_rmse_m", "velocity_rmse_mps", "log_likelihood", "resampled_steps")


def draw(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    x = [row["particles"] for row in rows]
    ticks = [rf"$2^{{{row['particle_power']}}}$" for row in rows]
    times = [row["filter_ms"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), constrained_layout=True)
    fig.suptitle("Vanda | One NVIDIA A40 | 256 steps | ESS < 0.5 N | Five repetitions")
    axes[0].errorbar(x, times,
                     yerr=[[r["filter_ms"] - r["min_filter_ms"] for r in rows],
                           [r["max_filter_ms"] - r["filter_ms"] for r in rows]],
                     fmt="o-", capsize=3, label="Median; min–max")
    axes[0].plot(x, [times[-1] * n / x[-1] for n in x], "--", color="0.5",
                 label="O(N) guide anchored at largest N")
    axes[0].set_yscale("log")
    axes[0].set(title="Filter runtime", ylabel="Milliseconds")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [r["million_particle_steps_per_s"] for r in rows], "o-", color="#237a57")
    axes[1].set(title="Throughput", ylabel="Million particle steps / second")
    phases = (("initialize_ms", "Allocation + initialization", "#8da0cb"),
              ("propagate_weight_ms", "Propagation + weighting", "#fc8d62"),
              ("normalize_estimate_ms", "Normalization + estimation", "#66c2a5"),
              ("resample_ms", "Resampling", "#e78ac3"))
    bottom = [0.0] * len(rows)
    totals = [sum(r[key] for key, _, _ in phases) for r in rows]
    for key, label, color in phases:
        share = [100 * r[key] / total for r, total in zip(rows, totals)]
        axes[2].bar(range(len(rows)), share, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, share)]
    axes[2].set(title="Recorded phase breakdown", ylabel="Share of summed phase times (%)",
                xlabel="Particles", ylim=(0, 100))
    axes[2].set_xticks(range(len(rows)), ticks)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    for axis in axes[:2]:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, ticks)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Particles")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.savefig(output / "single_gpu_particle_sweep.png", dpi=200, facecolor="white")
    fig.savefig(output / "single_gpu_particle_sweep.pdf", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--expected-repeats", type=int, default=5)
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in raw:
        if (row["suite"], row["gpu"], row["ranks"], row["nodes"], row["resampling"],
                row["steps"], row["data_seed"], row["filter_seed"], row["transport"]) != (
                "particle_sweep", "NVIDIA A40", "1", "1", "0.5", "256", "42", "1729", "staged"):
            raise RuntimeError("inconsistent single-GPU benchmark configuration")
        n = int(row["particles"])
        if n != 2**int(row["particle_power"]):
            raise RuntimeError("particle count differs from power label")
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
                raise RuntimeError(f"nonfinite metric: {metric}")
        if float(row["filter_ms"]) <= 0:
            raise RuntimeError("nonpositive filter runtime")
        groups[n].append(row)
    if not groups:
        parser.error("no measurement rows")
    rows = []
    for n, group in sorted(groups.items()):
        if sorted(int(row["repeat"]) for row in group) != list(range(1, args.expected_repeats + 1)):
            raise RuntimeError(f"N={n}: missing or duplicate repetitions")
        reference = group[0]
        for row in group:
            for key in ("position_rmse_m", "velocity_rmse_mps", "log_likelihood"):
                if not math.isclose(float(row[key]), float(reference[key]), rel_tol=1e-7, abs_tol=1e-7):
                    raise RuntimeError(f"N={n}: fixed-seed output differs in {key}")
            if row["resampled_steps"] != reference["resampled_steps"]:
                raise RuntimeError(f"N={n}: resampling decisions differ")
        row = {"particles": n, "particle_power": int(reference["particle_power"]),
               "repeats": len(group)}
        row.update({key: statistics.median(float(run[key]) for run in group) for key in METRICS})
        row["min_filter_ms"] = min(float(run["filter_ms"]) for run in group)
        row["max_filter_ms"] = max(float(run["filter_ms"]) for run in group)
        row["million_particle_steps_per_s"] = n * 256 / (row["filter_ms"] * 1000)
        row["us_per_particle_step"] = row["filter_ms"] * 1000 / (n * 256)
        row["adjacent_size_exponent"] = ""
        if rows:
            previous = rows[-1]
            row["adjacent_size_exponent"] = math.log(row["filter_ms"] / previous["filter_ms"]) / math.log(n / previous["particles"])
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "gpu_sweep_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        draw(rows, args.output)
    for row in rows:
        print(f"2^{row['particle_power']:2} N={row['particles']:7}: "
              f"{row['filter_ms']:9.4f} ms [{row['min_filter_ms']:.4f}, {row['max_filter_ms']:.4f}], "
              f"{row['million_particle_steps_per_s']:.2f} million particle steps/s")


if __name__ == "__main__":
    main()
