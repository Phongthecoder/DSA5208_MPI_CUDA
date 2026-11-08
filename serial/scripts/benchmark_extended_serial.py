#!/usr/bin/env python3
"""Measure serial filter time, peak RSS, and Gaussian sampling on one core."""

import argparse
import csv
import statistics
import subprocess
import tempfile
from pathlib import Path


PARTICLE_COUNTS = [2**exponent for exponent in (10, 12, 14, 16, 18, 20, 22)]


def run_case(binary, particles, steps, rss_path, profile=False):
    command = ["/usr/bin/time", "-f", "%M", "-o", str(rss_path), str(binary),
               "--model", "range-bearing", "--particles", str(particles),
               "--steps", str(steps), "--data-seed", "42", "--filter-seed", "1729",
               "--resample", "0.5"]
    if profile:
        command.append("--profile-gaussian")
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    row, = csv.DictReader(completed.stdout.splitlines())
    peak_rss = int(rss_path.read_text().strip())
    if peak_rss <= 0:
        raise RuntimeError("GNU time returned an invalid peak RSS")
    return row, peak_rss


def append_row(path, row):
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def verify_estimate(reference, candidate):
    for field, value in reference.items():
        if not field.endswith("_ms") and candidate[field] != value:
            raise RuntimeError(f"Fixed-seed filter output changed in {field}")


def benchmark_case(args, particles, rss_path, output_paths):
    run_case(args.binary, particles, min(8, args.steps), rss_path)
    filter_times = []
    peak_rss_values = []
    reference = None
    for repeat in range(1, args.repeats + 1):
        print(f"N={particles:,} T={args.steps} runtime repeat {repeat}/{args.repeats}", flush=True)
        row, peak_rss = run_case(args.binary, particles, args.steps, rss_path)
        if reference is None:
            reference = row
        verify_estimate(reference, row)
        append_row(output_paths["runtime"], {"repeat": repeat, "peak_rss_kib": peak_rss, **row})
        filter_times.append(float(row["filter_ms"]))
        peak_rss_values.append(peak_rss)

    print(f"N={particles:,} T={args.steps} Gaussian profile", flush=True)
    profile, profile_rss = run_case(args.binary, particles, args.steps, rss_path, profile=True)
    verify_estimate(reference, profile)
    prior_ms = float(profile["gaussian_prior_ms"])
    motion_ms = float(profile["gaussian_motion_ms"])
    if not (0 < prior_ms <= float(profile["initialize_ms"]) and
            0 < motion_ms <= float(profile["propagate_weight_ms"])):
        raise RuntimeError("Gaussian timings must be within their enclosing phases")
    gaussian_ms = prior_ms + motion_ms
    gaussian_draws = 4 * particles * (args.steps + 1)
    profile_filter_ms = float(profile["filter_ms"])
    gaussian_share = 100 * gaussian_ms / profile_filter_ms
    append_row(output_paths["profile"], {
        "peak_rss_kib": profile_rss,
        "gaussian_draws": gaussian_draws,
        "gaussian_total_ms": gaussian_ms,
        "gaussian_share_filter_pct": gaussian_share,
        **profile,
    })

    median_ms = statistics.median(filter_times)
    max_rss_mib = max(peak_rss_values) / 1024
    append_row(output_paths["summary"], {
        "particles": particles,
        "steps": args.steps,
        "runtime_repeats": args.repeats,
        "filter_median_ms": median_ms,
        "filter_min_ms": min(filter_times),
        "filter_max_ms": max(filter_times),
        "peak_rss_max_mib": max_rss_mib,
        "peak_rss_median_mib": statistics.median(peak_rss_values) / 1024,
        "particle_buffers_mib": int(reference["particle_buffer_bytes"]) / 1024**2,
        "gaussian_prior_ms": prior_ms,
        "gaussian_motion_ms": motion_ms,
        "gaussian_total_ms": gaussian_ms,
        "profile_filter_ms": profile_filter_ms,
        "gaussian_share_filter_pct": gaussian_share,
        "gaussian_draws": gaussian_draws,
        "gaussian_ns_per_draw": 1e6 * gaussian_ms / gaussian_draws,
        "position_rmse_m": reference["position_rmse_m"],
        "resampled_steps": reference["resampled_steps"],
    })
    print(f"N={particles:,} complete | filter {median_ms / 1000:.3f} s | "
          f"peak RSS {max_rss_mib:.2f} MiB | Gaussian {gaussian_ms / 1000:.3f} s "
          f"({gaussian_share:.1f}% of profiled filter)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/serial_pf"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, nargs="+", default=PARTICLE_COUNTS)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.steps, args.repeats, *args.particles) < 1:
        parser.error("steps, repeats and particle counts must be positive")
    args.binary = args.binary.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "runtime": args.output_dir / "extended_runtime_runs.csv",
        "profile": args.output_dir / "extended_gaussian_profiles.csv",
        "summary": args.output_dir / "extended_summary.csv",
    }
    if any(path.exists() for path in output_paths.values()):
        parser.error("the output directory already contains extended benchmark data")
    with tempfile.TemporaryDirectory(prefix="serial-rss-") as scratch:
        rss_path = Path(scratch) / "peak_rss.txt"
        for particles in args.particles:
            benchmark_case(args, particles, rss_path, output_paths)


if __name__ == "__main__":
    main()
