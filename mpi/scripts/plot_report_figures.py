"""Draw the CPU MPI report figures (Figures 3 and 4) from a benchmark folder.

Usage: python3 mpi/scripts/plot_report_figures.py mpi/benchmarks/vanda-platinum-5-repeats
Writes figures/fig3_mpi_scaling.png and figures/fig4_mpi_communication.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bench = Path(sys.argv[1])
out = bench / "figures"
out.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

rows = list(csv.DictReader(open(bench / "mpi_summary.csv")))
res = list(csv.DictReader(open(bench / "resampling_summary.csv")))
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def pick(suite, resampler, key):
    found = {int(r["ranks"]): float(r[key]) for r in rows
             if r["suite"] == suite and r["resampler"] == resampler and r[key]}
    xs = [x for x in RANKS if x in found]
    return xs, [found[x] for x in xs]


def log2_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.minorticks_off()
    ax.set_xlabel("MPI ranks (CPU cores)")


# Figure 3: (a) strong speedup, (b) efficiency
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
a.plot(RANKS, RANKS, ls="--", lw=1.2, color=INK2, label="Ideal")
xs, ys = pick("strong", "distributed", "speedup")
a.plot(xs, ys, "-o", lw=2, ms=4.5, color=BLUE, label="Distributed")
base = dict(zip(*pick("strong", "distributed", "filter_ms")))[1]
xc, tc = pick("strong", "centralized", "filter_ms")
a.plot(xc, [base / t for t in tc], "-s", lw=2, ms=4.5, color=ORANGE, label="Centralized")
a.set_xscale("log", base=2); a.set_yscale("log", base=2)
log2_axis(a)
a.set_yticks(RANKS); a.set_yticklabels([str(r) for r in RANKS])
a.set_ylabel("Strong-scaling speedup")
a.set_title("(a) Strong scaling, 4,194,304 particles", fontsize=9, loc="left", color=INK)
a.legend(loc="upper left")

xs, ys = pick("strong", "distributed", "efficiency")
b.plot(xs, [100 * y for y in ys], "-o", lw=2, ms=4.5, color=BLUE, label="Strong")
xw, yw = pick("weak", "distributed", "efficiency")
b.plot(xw, [100 * y for y in yw], "-D", lw=2, ms=4.5, color=AQUA, label="Weak")
b.axhline(100, ls="--", lw=1.2, color=INK2)
log2_axis(b)
b.set_ylim(0, 110)
b.set_ylabel("Parallel efficiency (%)")
b.set_title("(b) Efficiency, distributed resampling", fontsize=9, loc="left", color=INK)
b.legend(loc="lower left")
fig.savefig(out / "fig3_mpi_scaling.png", dpi=220)

# Figure 4: (a) weak-scaling phase maxima, (b) resampling cost per call
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
phases = [("propagate_weight_ms", "Propagate + weight", BLUE),
          ("normalize_estimate_ms", "Normalize + ESS + estimate", ORANGE),
          ("resample_ms", "Resample", AQUA)]
width = 0.26
for i, (key, label, color) in enumerate(phases):
    xs, ys = pick("weak", "distributed", key)
    pos = [RANKS.index(x) + (i - 1) * (width + 0.02) for x in xs]
    a.bar(pos, ys, width=width, color=color, label=label, edgecolor=SURF, linewidth=1)
a.set_xticks(range(len(RANKS))); a.set_xticklabels([str(r) for r in RANKS])
a.set_xlabel("MPI ranks (16,384 particles per rank)")
a.set_ylabel("Slowest-rank phase time (ms)")
a.grid(axis="x", visible=False)
a.set_title("(a) Weak-scaling phase times", fontsize=9, loc="left", color=INK)
a.legend(loc="upper left", fontsize=8)

for resampler, marker, color in [("distributed", "o", BLUE), ("centralized", "s", ORANGE)]:
    found = {int(r["ranks"]): float(r["ms_per_resample"]) for r in res
             if r["resampler"] == resampler and r["weight_shape"] == "skewed"}
    xs = [x for x in RANKS if x in found]
    b.plot(xs, [found[x] for x in xs], "-" + marker, lw=2, ms=4.5, color=color,
           label=resampler.capitalize())
log2_axis(b)
b.set_yscale("log")
b.set_ylabel("Time per resample (ms)")
b.set_title("(b) Resampling, 1,048,576 skewed weights", fontsize=9, loc="left", color=INK)
b.legend(loc="upper left")
fig.savefig(out / "fig4_mpi_communication.png", dpi=220)
print(out)
