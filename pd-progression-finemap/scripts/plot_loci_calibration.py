"""Generate calibration figures for all-loci Evo2 scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from loci import LOCI

ROOT = Path(__file__).resolve().parent.parent
EVO2_DIR = ROOT / "data" / "evo2"
SUMMARY_PATH = EVO2_DIR / "all_loci_summary.csv"

summary = pd.read_csv(SUMMARY_PATH)

# ── Figure A: Small multiples histograms ──────────────────────────────
print("Generating Figure A: small multiples histograms...")

# Compute global x-axis range
all_dlls = []
for locus in LOCI:
    cache = EVO2_DIR / f"{locus['name']}_{locus['trait']}_evo2.parquet"
    if cache.exists():
        ldf = pd.read_parquet(cache)
        scored = ldf[ldf["error"] == ""]["delta_log_likelihood"]
        all_dlls.append(scored)
global_min = min(s.min() for s in all_dlls)
global_max = max(s.max() for s in all_dlls)
xpad = (global_max - global_min) * 0.05
xrange = (global_min - xpad, global_max + xpad)

n_loci = len(LOCI)
ncols = 3
nrows = (n_loci + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharex=True)
axes_flat = axes.flatten()

for i, locus in enumerate(LOCI):
    ax = axes_flat[i]
    name = locus["name"]
    trait = locus["trait"]
    lead_rsid = locus["lead_rsid"]

    cache = EVO2_DIR / f"{name}_{trait}_evo2.parquet"
    if not cache.exists():
        ax.set_title(f"{name} — no data")
        continue

    ldf = pd.read_parquet(cache)
    scored = ldf[ldf["error"] == ""]
    dll = scored["delta_log_likelihood"]

    ax.hist(dll, bins=60, color="#4a90d9", edgecolor="white", linewidth=0.3, alpha=0.85,
            range=xrange)

    # Look up lead stats from summary
    row = summary[summary["locus"] == name]
    if not row.empty and row.iloc[0]["lead_in_scored"]:
        lead_dll = row.iloc[0]["paper_lead_delta_ll"]
        pctile = row.iloc[0]["percentile_signed"]
        ax.axvline(lead_dll, color="#d94a4a", linewidth=1.5, linestyle="--")
        ax.annotate(f"{lead_rsid}\n{pctile:.1f}%ile",
                    xy=(lead_dll, ax.get_ylim()[1] * 0.85),
                    fontsize=7, color="#d94a4a", ha="left",
                    xytext=(5, 0), textcoords="offset points")

    ax.set_title(f"{name} ({trait}, n={len(dll)})", fontsize=10)
    ax.set_ylabel("Count", fontsize=9)

# Hide unused subplots
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

for ax in axes_flat[ncols * (nrows - 1) : ncols * (nrows - 1) + ncols]:
    if ax.get_visible():
        ax.set_xlabel("delta_log_likelihood", fontsize=9)

fig.suptitle("Evo2 1B delta_log_likelihood — All Loci", fontsize=14, y=1.01)
plt.tight_layout()
hist_path = EVO2_DIR / "loci_histograms.png"
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {hist_path}")

# ── Figure B: Calibration dot plot ────────────────────────────────────
print("Generating Figure B: calibration summary...")

# Only include loci where lead is scored
plot_df = summary[summary["lead_in_scored"]].copy()

if plot_df.empty:
    print("  No loci with leads scored — skipping Figure B")
else:
    # Get p-values for sizing
    pvals = {}
    for locus in LOCI:
        cache_parquet = ROOT / "data" / "loci" / f"{locus['name']}_{locus['trait']}.parquet"
        if cache_parquet.exists():
            ldf = pd.read_parquet(cache_parquet)
            lead_row = ldf[ldf["pos"] == locus["lead_bp"]]
            if not lead_row.empty:
                pvals[locus["name"]] = lead_row.iloc[0]["pval"]
    plot_df["pval"] = plot_df["locus"].map(pvals)
    plot_df["neg_log10_p"] = -np.log10(plot_df["pval"].clip(lower=1e-300))

    # Colors by trait
    trait_colors = {"mortality": "#c0392b", "hy3": "#2980b9"}
    plot_df["color"] = plot_df["trait"].map(trait_colors)

    # Sort by percentile for visual clarity
    plot_df = plot_df.sort_values("percentile_signed")

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.8)))

    # Reference lines
    for pct in [5, 10, 25, 50]:
        ax.axvline(pct, color="#bdc3c7", linewidth=0.8, linestyle=":")
        ax.text(pct, len(plot_df) - 0.3, f"{pct}%", ha="center", va="bottom",
                fontsize=8, color="#7f8c8d")

    # Plot dots
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        size = max(40, row["neg_log10_p"] * 20)
        ax.scatter(row["percentile_signed"], idx, s=size, c=row["color"],
                   edgecolors="black", linewidth=0.5, zorder=5)
        label = f"{row['locus']} — {row['paper_lead_rsid']} (Δll={row['paper_lead_delta_ll']:.1f})"
        ax.text(row["percentile_signed"] + 1.5, idx, label, va="center", fontsize=9)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["locus"].tolist())
    ax.set_xlabel("Signed percentile (0% = most disruptive)", fontsize=11)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    ax.set_title("Where do paper lead variants fall in their\nlocus Evo2 score distributions?",
                 fontsize=13)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0392b',
               markersize=10, label='Mortality'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9',
               markersize=10, label='HY3+ progression'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    cal_path = EVO2_DIR / "loci_calibration.png"
    plt.savefig(cal_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {cal_path}")

print("\nDone.")
