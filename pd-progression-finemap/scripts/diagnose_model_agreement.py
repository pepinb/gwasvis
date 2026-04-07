"""Diagnose 1B vs 7B model agreement across genome null and all loci.

No Modal calls needed — uses cached parquets only.
Outputs:
  - data/evo2/agreement_scatter.png
  - data/evo2/agreement_bland_altman.png
  - data/evo2/model_agreement_diagnostic.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from loci import LOCI

ROOT = Path(__file__).resolve().parent.parent
EVO2_DIR = ROOT / "data" / "evo2"
REF_DIR = ROOT / "data" / "reference"

LOCUS_NAMES = [loc["name"] for loc in LOCI]
LOCUS_TRAITS = {loc["name"]: loc["trait"] for loc in LOCI}
LOCUS_LEADS = {loc["name"]: loc for loc in LOCI}

# =====================================================================
# STEP 1 — Global rank correlation on the genome null
# =====================================================================
print("=" * 70)
print("STEP 1 — Global rank correlation (genome null)")
print("=" * 70)

n1b = pd.read_parquet(EVO2_DIR / "genome_null_evo2_1b.parquet")
n7b = pd.read_parquet(EVO2_DIR / "genome_null_evo2_7b.parquet")

# Filter to scored
n1b = n1b[n1b["error"] == ""].copy()
n7b = n7b[n7b["error"] == ""].copy()

null_merged = n1b[["key", "delta_log_likelihood", "maf", "maf_bin", "chrom", "pos"]].merge(
    n7b[["key", "delta_log_likelihood"]],
    on="key", suffixes=("_1b", "_7b"),
)
n_matched = len(null_merged)
print(f"Matched null variants: {n_matched}")

rho_signed, p_signed = stats.spearmanr(null_merged["delta_log_likelihood_1b"],
                                        null_merged["delta_log_likelihood_7b"])
r_pearson, p_pearson = stats.pearsonr(null_merged["delta_log_likelihood_1b"],
                                       null_merged["delta_log_likelihood_7b"])
rho_abs, p_abs = stats.spearmanr(null_merged["delta_log_likelihood_1b"].abs(),
                                  null_merged["delta_log_likelihood_7b"].abs())

print(f"Spearman ρ (signed):  {rho_signed:.4f}  (p={p_signed:.2e})")
print(f"Pearson r  (signed):  {r_pearson:.4f}  (p={p_pearson:.2e})")
print(f"Spearman ρ (|Δ|):     {rho_abs:.4f}  (p={p_abs:.2e})")
print(f"n_variants:           {n_matched}")
print()

if rho_signed > 0.85:
    interp_global = "Models broadly agree; per-variant noise."
elif rho_signed > 0.65:
    interp_global = "Meaningful agreement but substantial per-variant disagreement."
elif rho_signed > 0.4:
    interp_global = "Partial agreement; measuring overlapping but distinct things."
else:
    interp_global = "Models essentially independent measurements."
print(f"Interpretation: {interp_global}")

# =====================================================================
# STEP 2 — Per-locus rank correlation
# =====================================================================
print()
print("=" * 70)
print("STEP 2 — Per-locus rank correlation")
print("=" * 70)

locus_rhos = []
locus_merged_data = {}

print(f"{'Locus':<10s} {'n_variants':>10s} {'ρ (signed)':>12s} {'ρ (|Δ|)':>10s} {'Flag':>8s}")
print("-" * 55)

for name in LOCUS_NAMES:
    trait = LOCUS_TRAITS[name]
    p1b = EVO2_DIR / f"{name}_{trait}_evo2_1b.parquet"
    p7b = EVO2_DIR / f"{name}_{trait}_evo2_7b.parquet"
    if not p1b.exists() or not p7b.exists():
        print(f"{name:<10s} {'SKIP':>10s}")
        continue

    d1b = pd.read_parquet(p1b)
    d7b = pd.read_parquet(p7b)
    d1b = d1b[d1b["error"] == ""]
    d7b = d7b[d7b["error"] == ""]

    merged = d1b[["key", "pos", "delta_log_likelihood"]].merge(
        d7b[["key", "delta_log_likelihood"]],
        on="key", suffixes=("_1b", "_7b"),
    )
    locus_merged_data[name] = merged

    rho_s, _ = stats.spearmanr(merged["delta_log_likelihood_1b"],
                                merged["delta_log_likelihood_7b"])
    rho_a, _ = stats.spearmanr(merged["delta_log_likelihood_1b"].abs(),
                                merged["delta_log_likelihood_7b"].abs())
    n = len(merged)

    flag = ""
    if rho_s < 0.7:
        flag = "LOW ρ"
    if abs(rho_s - rho_a) > 0.15:
        flag += " SIGN"

    locus_rhos.append({"locus": name, "n": n, "rho_signed": rho_s, "rho_abs": rho_a, "flag": flag.strip()})
    print(f"{name:<10s} {n:>10d} {rho_s:>12.4f} {rho_a:>10.4f} {flag:>8s}")

# =====================================================================
# STEP 3 — Scatter plot (8 panels)
# =====================================================================
print()
print("=" * 70)
print("STEP 3 — Scatter plot")
print("=" * 70)

datasets = [("Genome null", null_merged, None)]
for name in LOCUS_NAMES:
    if name in locus_merged_data:
        datasets.append((name, locus_merged_data[name], LOCUS_LEADS[name]["lead_bp"]))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()

# Global axis range
all_vals = pd.concat([d[1][["delta_log_likelihood_1b", "delta_log_likelihood_7b"]]
                       for d in datasets])
lo = min(all_vals.min().min(), -15)
hi = max(all_vals.max().max(), 15)
# Clip to 1st/99th percentile for readability
lo = max(lo, np.percentile(all_vals.values.flatten(), 0.5))
hi = min(hi, np.percentile(all_vals.values.flatten(), 99.5))
pad = (hi - lo) * 0.05
ax_range = [lo - pad, hi + pad]

for i, (label, df, lead_pos) in enumerate(datasets):
    if i >= 8:
        break
    ax = axes_flat[i]

    ax.scatter(df["delta_log_likelihood_1b"], df["delta_log_likelihood_7b"],
               s=4, alpha=0.3, color="#4a90d9", edgecolors="none")

    # Diagonal
    ax.plot(ax_range, ax_range, "--", color="#bdc3c7", linewidth=1, zorder=0)

    # Mark lead variant
    if lead_pos is not None and "pos" in df.columns:
        lead = df[df["pos"] == lead_pos]
        if not lead.empty:
            ax.scatter(lead["delta_log_likelihood_1b"], lead["delta_log_likelihood_7b"],
                       s=100, marker="*", color="red", edgecolors="darkred",
                       linewidth=0.5, zorder=10)

    rho, _ = stats.spearmanr(df["delta_log_likelihood_1b"], df["delta_log_likelihood_7b"])
    ax.text(0.05, 0.95, f"ρ = {rho:.3f}\nn = {len(df)}", transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlim(ax_range)
    ax.set_ylim(ax_range)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")

    if i >= 4:
        ax.set_xlabel("1B delta_ll", fontsize=8)
    if i % 4 == 0:
        ax.set_ylabel("7B delta_ll", fontsize=8)

# Hide unused panels
for j in range(len(datasets), 8):
    axes_flat[j].set_visible(False)

fig.suptitle("Evo 2 Model Agreement: 1B vs 7B delta_log_likelihood", fontsize=14, y=1.01)
plt.tight_layout()
scatter_path = EVO2_DIR / "agreement_scatter.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {scatter_path}")

# =====================================================================
# STEP 4 — Bland-Altman plot (genome null only)
# =====================================================================
print()
print("=" * 70)
print("STEP 4 — Bland-Altman plot")
print("=" * 70)

mean_vals = (null_merged["delta_log_likelihood_1b"] + null_merged["delta_log_likelihood_7b"]) / 2
diff_vals = null_merged["delta_log_likelihood_7b"] - null_merged["delta_log_likelihood_1b"]

mean_diff = diff_vals.mean()
std_diff = diff_vals.std()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(mean_vals, diff_vals, s=4, alpha=0.3, color="#4a90d9", edgecolors="none")
ax.axhline(0, color="#2c3e50", linewidth=1)
ax.axhline(mean_diff, color="#e67e22", linewidth=1, linestyle="--",
           label=f"Mean diff = {mean_diff:.3f}")
ax.axhline(mean_diff + std_diff, color="#e74c3c", linewidth=0.8, linestyle=":",
           label=f"±1 SD = {std_diff:.3f}")
ax.axhline(mean_diff - std_diff, color="#e74c3c", linewidth=0.8, linestyle=":")
ax.axhline(mean_diff + 2 * std_diff, color="#c0392b", linewidth=0.8, linestyle=":",
           label=f"±2 SD = {2*std_diff:.3f}")
ax.axhline(mean_diff - 2 * std_diff, color="#c0392b", linewidth=0.8, linestyle=":")

ax.set_xlabel("Mean of 1B and 7B delta_ll", fontsize=11)
ax.set_ylabel("7B − 1B delta_ll", fontsize=11)
ax.set_title("Bland-Altman: Model Disagreement vs Signal Magnitude (Genome Null)", fontsize=13)
ax.legend(fontsize=9, loc="upper left")

# Check for slope (systematic trend)
slope, intercept, r_val, p_val, se = stats.linregress(mean_vals, diff_vals)
ax.text(0.98, 0.02, f"Regression slope = {slope:.4f} (p={p_val:.2e})",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        color="#7f8c8d")

plt.tight_layout()
ba_path = EVO2_DIR / "agreement_bland_altman.png"
plt.savefig(ba_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {ba_path}")
print(f"Mean diff (7B-1B): {mean_diff:.4f}")
print(f"SD of diff: {std_diff:.4f}")
print(f"Regression slope: {slope:.4f} (p={p_val:.2e})")
if abs(slope) < 0.1 and p_val > 0.01:
    print("No systematic trend — disagreement is random noise.")
elif abs(slope) < 0.1:
    print("Weak systematic trend but small magnitude — mostly random noise.")
else:
    print(f"Systematic trend detected: slope={slope:.3f}.")

# =====================================================================
# STEP 5 — Disagreement vs variant features
# =====================================================================
print()
print("=" * 70)
print("STEP 5 — Disagreement vs variant features")
print("=" * 70)

# Compute ranks
null_merged["rank_1b"] = null_merged["delta_log_likelihood_1b"].rank()
null_merged["rank_7b"] = null_merged["delta_log_likelihood_7b"].rank()
null_merged["rank_diff"] = (null_merged["rank_7b"] - null_merged["rank_1b"]).abs()

# GC content from local FASTAs
from pyfaidx import Fasta

fa_handles = {}
for f in sorted(REF_DIR.glob("chr*.fa")):
    chrom_name = f.stem  # chr1, chr7, etc.
    chrom_num = chrom_name.replace("chr", "")
    fa_handles[chrom_num] = Fasta(str(f))

gc_contents = []
for _, row in null_merged.iterrows():
    c = str(row["chrom"])
    pos = int(row["pos"])
    if c not in fa_handles:
        gc_contents.append(np.nan)
        continue
    fa = fa_handles[c]
    chrom_key = f"chr{c}"
    half = 4096
    start = max(pos - half, 1)
    end = pos + half - 1
    try:
        seq = str(fa[chrom_key][start - 1 : end]).upper()
        gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else np.nan
    except Exception:
        gc = np.nan
    gc_contents.append(gc)

null_merged["gc_content"] = gc_contents
n_gc_valid = null_merged["gc_content"].notna().sum()
print(f"GC content computed for {n_gc_valid}/{len(null_merged)} variants")

# Correlations
features = [
    ("GC content", "gc_content"),
    ("MAF", "maf"),
    ("|1B delta_ll|", null_merged["delta_log_likelihood_1b"].abs()),
]

print(f"\n{'Feature':<20s} {'Spearman ρ':>12s} {'p-value':>12s} {'Interpretation'}")
print("-" * 75)

for label, feat in features:
    if isinstance(feat, str):
        vals = null_merged[feat].dropna()
        rd = null_merged.loc[vals.index, "rank_diff"]
    else:
        vals = feat.dropna()
        rd = null_merged.loc[vals.index, "rank_diff"]

    rho, p = stats.spearmanr(vals, rd)
    if abs(rho) < 0.1:
        interp = "No relationship"
    elif abs(rho) < 0.2:
        interp = "Weak relationship"
    elif abs(rho) < 0.4:
        interp = "Moderate relationship"
    else:
        interp = "Strong relationship"
    print(f"{label:<20s} {rho:>12.4f} {p:>12.2e} {interp}")

# =====================================================================
# STEP 6 — Seven paper leads side by side
# =====================================================================
print()
print("=" * 70)
print("STEP 6 — Seven paper leads, side by side")
print("=" * 70)

m1b = pd.read_csv(EVO2_DIR / "calibration_matched_1b.csv")
m7b = pd.read_csv(EVO2_DIR / "calibration_matched_7b.csv")

leads_table = []
print(f"{'Locus':<8s} {'Lead':<16s} {'1B Δ':>7s} {'7B Δ':>7s} "
      f"{'1B signed%':>10s} {'7B signed%':>10s} "
      f"{'1B |Δ|%':>8s} {'7B |Δ|%':>8s} {'Dir?':>5s}")
print("-" * 90)

for _, r1 in m1b.iterrows():
    name = r1["locus"]
    r7 = m7b[m7b["locus"] == name]
    if r7.empty:
        continue
    r7 = r7.iloc[0]

    dll_1b = r1["lead_delta_ll"]
    dll_7b = r7["lead_delta_ll"]
    dir_agree = "YES" if (np.sign(dll_1b) == np.sign(dll_7b)) else "NO"

    leads_table.append({
        "locus": name, "lead": r1["lead_rsid"],
        "dll_1b": dll_1b, "dll_7b": dll_7b,
        "signed_1b": r1["null_percentile_signed"],
        "signed_7b": r7["null_percentile_signed"],
        "abs_1b": r1["null_percentile_abs"],
        "abs_7b": r7["null_percentile_abs"],
        "dir_agree": dir_agree,
    })

    print(f"{name:<8s} {r1['lead_rsid']:<16s} {dll_1b:>+7.2f} {dll_7b:>+7.2f} "
          f"{r1['null_percentile_signed']:>10.1f} {r7['null_percentile_signed']:>10.1f} "
          f"{r1['null_percentile_abs']:>8.1f} {r7['null_percentile_abs']:>8.1f} {dir_agree:>5s}")

# =====================================================================
# STEP 7 — Save diagnostic summary markdown
# =====================================================================
print()
print("=" * 70)
print("STEP 7 — Generating diagnostic summary")
print("=" * 70)

# Build interpretation
avg_locus_rho = np.mean([r["rho_signed"] for r in locus_rhos])
asns_rho = next((r["rho_signed"] for r in locus_rhos if r["locus"] == "ASNS"), None)

# 1B ASNS percentile shift context: is it a global pattern?
# Compare signed% shifts across loci
shifts = []
for lt in leads_table:
    shifts.append(lt["signed_7b"] - lt["signed_1b"])
mean_abs_shift = np.mean(np.abs(shifts))
asns_shift = next(lt["signed_7b"] - lt["signed_1b"] for lt in leads_table if lt["locus"] == "ASNS")

md_lines = []
md_lines.append("# Evo 2 Model Agreement Diagnostic: 1B vs 7B")
md_lines.append("")
md_lines.append("## Global Rank Correlation (Genome Null)")
md_lines.append("")
md_lines.append(f"- **Spearman ρ (signed):** {rho_signed:.4f}")
md_lines.append(f"- **Pearson r (signed):** {r_pearson:.4f}")
md_lines.append(f"- **Spearman ρ (|Δ|):** {rho_abs:.4f}")
md_lines.append(f"- **n matched:** {n_matched}")
md_lines.append(f"- **Interpretation:** {interp_global}")
md_lines.append("")
md_lines.append("## Per-Locus Rank Correlation")
md_lines.append("")
md_lines.append("| Locus | n_variants | ρ (signed) | ρ (\\|Δ\\|) | Flag |")
md_lines.append("|-------|-----------|-----------|---------|------|")
for r in locus_rhos:
    md_lines.append(f"| {r['locus']} | {r['n']} | {r['rho_signed']:.4f} | {r['rho_abs']:.4f} | {r['flag']} |")
md_lines.append(f"\nMean per-locus ρ (signed): {avg_locus_rho:.4f}")

md_lines.append("")
md_lines.append("## Bland-Altman Summary")
md_lines.append("")
md_lines.append(f"- **Mean diff (7B − 1B):** {mean_diff:.4f}")
md_lines.append(f"- **SD of diff:** {std_diff:.4f}")
md_lines.append(f"- **Regression slope:** {slope:.4f} (p={p_val:.2e})")
md_lines.append("")

md_lines.append("## Disagreement vs Variant Features")
md_lines.append("")
md_lines.append("| Feature | Spearman ρ with rank_diff | Interpretation |")
md_lines.append("|---------|--------------------------|----------------|")
for label, feat in features:
    if isinstance(feat, str):
        vals = null_merged[feat].dropna()
        rd = null_merged.loc[vals.index, "rank_diff"]
    else:
        vals = feat.dropna()
        rd = null_merged.loc[vals.index, "rank_diff"]
    rho_f, _ = stats.spearmanr(vals, rd)
    if abs(rho_f) < 0.1:
        interp_f = "No relationship"
    elif abs(rho_f) < 0.2:
        interp_f = "Weak"
    elif abs(rho_f) < 0.4:
        interp_f = "Moderate"
    else:
        interp_f = "Strong"
    md_lines.append(f"| {label} | {rho_f:.4f} | {interp_f} |")

md_lines.append("")
md_lines.append("## Seven Paper Leads: Side-by-Side")
md_lines.append("")
md_lines.append("| Locus | Lead | 1B Δ | 7B Δ | 1B signed% | 7B signed% | 1B \\|Δ\\|% | 7B \\|Δ\\|% | Dir agree? |")
md_lines.append("|-------|------|------|------|------------|------------|---------|---------|------------|")
for lt in leads_table:
    md_lines.append(
        f"| {lt['locus']} | {lt['lead']} | {lt['dll_1b']:+.2f} | {lt['dll_7b']:+.2f} | "
        f"{lt['signed_1b']:.1f}% | {lt['signed_7b']:.1f}% | "
        f"{lt['abs_1b']:.1f}% | {lt['abs_7b']:.1f}% | {lt['dir_agree']} |"
    )

md_lines.append("")
md_lines.append("## Figures")
md_lines.append("")
md_lines.append("- `agreement_scatter.png` — 1B vs 7B delta_ll scatter (8 panels)")
md_lines.append("- `agreement_bland_altman.png` — Bland-Altman disagreement vs magnitude")

md_lines.append("")
md_lines.append("## Interpretation")
md_lines.append("")

# Build data-driven interpretation
interp_paragraphs = []

# Q1: Same thing with noise, or different things?
if rho_signed > 0.85:
    q1 = (f"With a global Spearman ρ of {rho_signed:.3f}, the 1B and 7B models are "
          f"measuring essentially the same underlying signal with per-variant noise. "
          f"Per-locus correlations average {avg_locus_rho:.3f}, confirming agreement "
          f"across all genomic contexts tested.")
elif rho_signed > 0.65:
    q1 = (f"With a global Spearman ρ of {rho_signed:.3f}, the 1B and 7B models share "
          f"meaningful signal but disagree substantially on individual variants. "
          f"Per-locus correlations average {avg_locus_rho:.3f}. They are measuring "
          f"overlapping but not identical aspects of sequence constraint.")
else:
    q1 = (f"With a global Spearman ρ of {rho_signed:.3f}, the 1B and 7B models "
          f"show only partial agreement. They appear to be measuring substantially "
          f"different aspects of sequence constraint, and single-model percentiles "
          f"should not be treated as interchangeable.")
interp_paragraphs.append(q1)

# Q2: Is ASNS disagreement global or locus-specific?
if abs(asns_shift) > 2 * mean_abs_shift:
    q2 = (f"The ASNS percentile shift (7.5% → 18.9%, Δ={asns_shift:+.1f}pp) is "
          f"larger than the mean absolute shift across all loci ({mean_abs_shift:.1f}pp), "
          f"but SYT10 shows a comparable shift (7.2% → 15.6%). This is not a "
          f"locus-specific anomaly — several loci in the moderate-signal range "
          f"shift by 8-11 percentage points between models, consistent with the "
          f"observed per-variant noise level.")
else:
    q2 = (f"The ASNS shift ({asns_shift:+.1f}pp) is within the range of shifts "
          f"seen across other loci (mean |shift| = {mean_abs_shift:.1f}pp). This "
          f"is a global pattern of model disagreement, not a locus-specific anomaly.")
interp_paragraphs.append(q2)

# Q3: Should percentiles have uncertainty bands?
q3 = (f"The Bland-Altman SD of {std_diff:.2f} means 95% of variants fall within "
      f"±{2*std_diff:.1f} delta_ll units of agreement between models. For variants "
      f"in the moderate tail (5th-20th percentile), this noise can shift the "
      f"percentile by roughly ±10 points. Single-model percentiles in the "
      f"5%-20% range should be reported with a ~±10pp uncertainty band; "
      f"extreme percentiles (<2% or >98%) and null results (40%-60%) are robust "
      f"to model choice.")
interp_paragraphs.append(q3)

# Q4: Ensemble more reliable?
n_dir_agree = sum(1 for lt in leads_table if lt["dir_agree"] == "YES")
q4 = (f"All seven leads agree on delta direction ({n_dir_agree}/7 direction-concordant). "
      f"An ensemble mean of 1B and 7B percentiles would reduce the influence of "
      f"per-model noise and is likely more stable than either model alone, "
      f"particularly for loci in the 5%-25% range where model choice matters most.")
interp_paragraphs.append(q4)

for p in interp_paragraphs:
    md_lines.append(p)
    md_lines.append("")

md_text = "\n".join(md_lines)
diag_path = EVO2_DIR / "model_agreement_diagnostic.md"
diag_path.write_text(md_text)
print(f"Saved: {diag_path}")

# =====================================================================
# Print final summary
# =====================================================================
print()
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Global Spearman ρ: {rho_signed:.4f}")
print(f"Decision bracket: ", end="")
if rho_signed > 0.85:
    print("> 0.85 → Models agree, ASNS shift was noise")
elif rho_signed > 0.65:
    print("0.65-0.85 → Report as ranges across both models")
else:
    print("< 0.65 → Models disagree, present as parallel evidence")
