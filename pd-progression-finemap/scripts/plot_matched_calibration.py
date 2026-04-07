"""Plot MAF-matched calibration figures.

Figure A: Null delta_ll distributions by MAF bin (8 panels)
Figure B: Locus vs. genome-wide matched percentile comparison (dot plot)
"""

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
NULL_SCORED_PATH = EVO2_DIR / "genome_null_scored.parquet"
MATCHED_PATH = EVO2_DIR / "matched_percentiles.csv"

MAF_BINS = [
    (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10),
    (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
]

null_df = pd.read_parquet(NULL_SCORED_PATH)
null_scored = null_df[null_df["error"] == ""].copy()
matched = pd.read_csv(MATCHED_PATH)

# ── Figure A: Null distributions by MAF bin ──────────────────────────
print("Generating Figure A: null distributions by MAF bin...")

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=False)
axes_flat = axes.flatten()

# Global x-range across all bins
all_dll = null_scored["delta_log_likelihood"]
xpad = (all_dll.max() - all_dll.min()) * 0.05
xrange = (all_dll.min() - xpad, all_dll.max() + xpad)

for i, (lo, hi) in enumerate(MAF_BINS):
    ax = axes_flat[i]
    bin_data = null_scored[null_scored["maf_bin"] == i]["delta_log_likelihood"]

    ax.hist(bin_data, bins=50, color="#7f8c8d", edgecolor="white",
            linewidth=0.3, alpha=0.85, range=xrange)

    ax.set_title(f"MAF [{lo:.3f}–{hi:.3f})\nn={len(bin_data)}, "
                 f"μ={bin_data.mean():.2f}, σ={bin_data.std():.2f}",
                 fontsize=9)
    ax.set_ylabel("Count", fontsize=8)

    # Mark lead variants that fall in this bin
    bin_leads = matched[matched["maf_bin"] == i]
    for _, lead in bin_leads.iterrows():
        if pd.notna(lead["lead_delta_ll"]):
            ax.axvline(lead["lead_delta_ll"], color="#e74c3c", linewidth=1.5,
                       linestyle="--", zorder=10)
            ax.text(lead["lead_delta_ll"], ax.get_ylim()[1] * 0.9,
                    f" {lead['locus']}", fontsize=7, color="#e74c3c",
                    ha="left", va="top")

for ax in axes_flat[4:]:
    ax.set_xlabel("delta_log_likelihood", fontsize=8)

fig.suptitle("Genome-Wide Null: Evo2 delta_ll by MAF Bin", fontsize=14, y=1.01)
plt.tight_layout()
null_hist_path = EVO2_DIR / "null_by_maf_histograms.png"
plt.savefig(null_hist_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {null_hist_path}")
plt.close()

# ── Figure B: Locus vs. genome-wide matched percentile ──────────────
print("Generating Figure B: locus vs matched percentile comparison...")

plot_df = matched[matched["null_n"] > 0].copy()
plot_df = plot_df.dropna(subset=["locus_percentile_signed", "null_percentile_signed"])

if plot_df.empty:
    print("  No data for Figure B — skipping")
else:
    plot_df = plot_df.sort_values("null_percentile_signed")

    trait_colors = {"mortality": "#c0392b", "hy3": "#2980b9"}
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 1.0)))

    # Reference lines
    for pct in [5, 10, 25, 50]:
        ax.axvline(pct, color="#bdc3c7", linewidth=0.8, linestyle=":")
        ax.text(pct, len(plot_df) - 0.3, f"{pct}%", ha="center", va="bottom",
                fontsize=8, color="#7f8c8d")

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        color = trait_colors.get(row["trait"], "#333333")
        extrap_mark = "*" if row.get("extrapolated", False) else ""

        # Null percentile (filled circle)
        ax.scatter(row["null_percentile_signed"], idx, s=100, c=color,
                   edgecolors="black", linewidth=0.8, zorder=5, marker="o")

        # Locus percentile (open diamond)
        ax.scatter(row["locus_percentile_signed"], idx, s=80,
                   facecolors="none", edgecolors=color, linewidth=1.5,
                   zorder=4, marker="D")

        # Connect with line
        ax.plot([row["locus_percentile_signed"], row["null_percentile_signed"]],
                [idx, idx], color=color, linewidth=1, alpha=0.5, zorder=3)

        label = (f"{row['locus']}{extrap_mark} — {row['lead_rsid']} "
                 f"(Δll={row['lead_delta_ll']:.1f}, MAF={row['lead_maf']:.3f})")
        x_label = max(row["null_percentile_signed"], row["locus_percentile_signed"])
        ax.text(x_label + 2, idx, label, va="center", fontsize=8)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["locus"].tolist())
    ax.set_xlabel("Signed percentile (0% = most disruptive)", fontsize=11)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    ax.set_title("Lead variant percentiles:\nlocus-only (◇) vs MAF-matched genome-wide (●)",
                 fontsize=13)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0392b',
               markersize=10, label='Mortality (genome)'),
        Line2D([0], [0], marker='D', color='w', markeredgecolor='#c0392b',
               markerfacecolor='none', markersize=8, label='Mortality (locus)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9',
               markersize=10, label='HY3+ (genome)'),
        Line2D([0], [0], marker='D', color='w', markeredgecolor='#2980b9',
               markerfacecolor='none', markersize=8, label='HY3+ (locus)'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    cal_path = EVO2_DIR / "matched_calibration.png"
    plt.savefig(cal_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {cal_path}")
    plt.close()

print("\nDone.")
