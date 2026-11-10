#!/usr/bin/env python3
"""Summarize a single Vanda CUDA sweep and compare a matching CPU MPI workload."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median


TIMES = ("filter_ms", "workflow_ms", "propagate_weight_ms", "normalize_estimate_ms",
         "resample_ms", "reduction_mpi_ms", "resample_mpi_ms", "staging_ms")
WORKLOAD = ("particles", "steps", "data_seed", "filter_seed", "resampling")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def summarize(raw, repeats):
    groups = defaultdict(list)
    for row in raw:
        groups[(row["suite"], int(row["ranks"]), row["transport"])].append(row)
    results = []
    for (suite, ranks, transport), rows in sorted(groups.items()):
        if sorted(int(row["repeat"]) for row in rows) != list(range(1, repeats + 1)):
            raise ValueError(f"expected {repeats} distinct repeats for {suite}/{ranks}/{transport}")
        fields = (*WORKLOAD, "nodes", "gpu")
        if any(tuple(row[k] for k in fields) != tuple(rows[0][k] for k in fields) for row in rows):
            raise ValueError(f"mixed workloads or hardware in {suite}/{ranks}/{transport}")
        times = [float(row["filter_ms"]) for row in rows]
        results.append({"suite": suite, "gpus": ranks, "transport": transport,
                        **{key: rows[0][key] for key in fields}, "repeats": len(rows),
                        **{key: median(float(row[key]) for row in rows) for key in TIMES},
                        "filter_min_ms": min(times), "filter_max_ms": max(times),
                        "position_rmse_m": median(float(row["position_rmse_m"]) for row in rows),
                        "resampled_steps": rows[0]["resampled_steps"]})

    baseline = {row["suite"]: row for row in results if row["gpus"] == 1 and row["transport"] == "staged"}
    for row in results:
        base = baseline[row["suite"]]
        if any(row[k] != base[k] for k in (*WORKLOAD[1:], "gpu", "nodes")):
            raise ValueError(f"inconsistent scaling baseline for {row['suite']}")
        scale = row["gpus"] if row["suite"] == "weak" else 1
        if int(row["particles"]) != scale * int(base["particles"]):
            raise ValueError(f"incorrect particle count for {row['suite']} scaling")
        ratio = base["filter_ms"] / row["filter_ms"]
        row["strong_speedup"] = ratio if row["suite"] == "strong" else ""
        row["efficiency"] = ratio / row["gpus"] if row["suite"] == "strong" else ratio
    return results


def compare_cpu(cpu_rows, gpu_rows):
    pairs = []
    for gpu in gpu_rows:
        if gpu["suite"] != "strong":
            continue
        for cpu in cpu_rows:
            if (cpu["suite"] == "strong" and cpu["resampler"] == "distributed"
                    and int(cpu["ranks"]) in (gpu["gpus"], 128)
                    and all(cpu[k] == gpu[k] for k in WORKLOAD)):
                pairs.append({"cpu_ranks": cpu["ranks"], "cpu_model": "Intel Xeon Gold 6438N",
                              "gpus": gpu["gpus"], "gpu_model": gpu["gpu"], "transport": gpu["transport"],
                              "particles": gpu["particles"], "steps": gpu["steps"],
                              "cpu_filter_ms": cpu["filter_ms"], "gpu_filter_ms": gpu["filter_ms"],
                              "cpu_over_gpu_time_ratio": float(cpu["filter_ms"]) / gpu["filter_ms"],
                              "cpu_position_rmse_m": cpu["position_rmse_m"],
                              "gpu_position_rmse_m": gpu["position_rmse_m"]})
    return pairs


def plot(rows, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
    fig.suptitle(f"MPI + CUDA particle filter | {rows[0]['gpu']}")
    for suite, ax in zip(("strong", "weak"), axes):
        chosen = [row for row in rows if row["suite"] == suite]
        base = next(row for row in chosen if row["gpus"] == 1)
        gpu_counts = sorted({row["gpus"] for row in chosen})
        for mode, marker in (("staged", "o"), ("device", "s")):
            data = [row for row in chosen if row["transport"] == mode]
            if not data:
                continue
            if mode == "device":
                data = [base, *data]  # A single GPU has no inter-rank transfer.
            y = [row["filter_ms"] / 1000 for row in data]
            low = [(row["filter_ms"] - row["filter_min_ms"]) / 1000 for row in data]
            high = [(row["filter_max_ms"] - row["filter_ms"]) / 1000 for row in data]
            ax.errorbar([row["gpus"] for row in data], y, yerr=[low, high], marker=marker,
                        capsize=4, label=mode.capitalize())
        title = f"{suite.capitalize()} | {int(base['particles']):,} particles"
        if suite == "weak":
            title += "/GPU"
        ax.set(title=title, xlabel="GPU count", ylabel="Median filter time (s)", xticks=gpu_counts)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.savefig(output, dpi=175, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_runs", type=Path)
    parser.add_argument("--cpu-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    raw = read_csv(args.gpu_runs)
    status_path = args.gpu_runs.with_name("device_transport.tsv")
    if status_path.exists() and not status_path.read_text().splitlines()[-1].startswith("pass\t"):
        raw = [row for row in raw if row["transport"] == "staged"]
    rows = summarize(raw, args.repeats)
    if not rows:
        parser.error("no completed timing groups")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "gpu_summary.csv", rows)
    if args.cpu_summary:
        pairs = compare_cpu(read_csv(args.cpu_summary), rows)
        if not pairs:
            parser.error("no CPU reference matches the CUDA strong-scaling workload")
        write_csv(args.output / "cpu_gpu_comparison.csv", pairs)
    if args.plot:
        plot(rows, args.output / "cuda_scaling.png")


if __name__ == "__main__":
    main()
