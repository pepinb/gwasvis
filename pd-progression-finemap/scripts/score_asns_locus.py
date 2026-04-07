"""Phase 2: Score all SNVs in the ASNS locus with Evo2 and produce a null distribution."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evo2.client import Evo2Client
from evo2.reference import LocalFastaProvider

ROOT = Path(__file__).resolve().parent.parent
FASTA_PATH = ROOT / "data" / "reference" / "chr7.fa"
PARQUET_PATH = ROOT / "data" / "loci" / "ASNS_hy3.parquet"
CACHE_PATH = ROOT / "data" / "evo2" / "ASNS_hy3_evo2.parquet"
HIST_PATH = ROOT / "data" / "evo2" / "asns_locus_histogram.png"
LEAD_RSID = "rs145274312"
LEAD_POS = 97470925

# ── Load locus data ──────────────────────────────────────────────────
print("=== Phase 2: Score ASNS Locus ===\n")
df = pd.read_parquet(PARQUET_PATH)
print(f"Total variants in locus: {len(df)}")

# Filter to SNVs with valid ACGT alleles
valid_bases = set("ACGT")
snv_mask = (df["a1"].str.len() == 1) & (df["a2"].str.len() == 1)
acgt_mask = df["a1"].str.upper().isin(valid_bases) & df["a2"].str.upper().isin(valid_bases)
indels = (~snv_mask).sum()
non_acgt = (snv_mask & ~acgt_mask).sum()
snv_df = df[snv_mask & acgt_mask].copy()
print(f"SNVs to score: {len(snv_df)}")
print(f"Indels skipped: {indels}")
print(f"Non-ACGT dropped: {non_acgt}")

# ── Determine ref/alt from FASTA ─────────────────────────────────────
print(f"\nLoading reference FASTA: {FASTA_PATH}")
reference = LocalFastaProvider(str(FASTA_PATH))

# Look up the reference base for each variant to assign ref/alt
from pyfaidx import Fasta
fa = Fasta(str(FASTA_PATH))

variants: list[dict] = []
for _, row in snv_df.iterrows():
    pos = int(row["pos"])
    a1 = row["a1"].upper()
    a2 = row["a2"].upper()
    ref_base = str(fa["chr7"][pos - 1 : pos]).upper()

    if ref_base == a2:
        ref, alt = a2, a1
    elif ref_base == a1:
        ref, alt = a1, a2
    else:
        # Neither allele matches reference — skip
        continue

    variants.append({
        "chrom": "7",
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "rsid": row["rsid"],
    })

print(f"Variants with ref allele resolved: {len(variants)}")

# ── Score ─────────────────────────────────────────────────────────────
print()
client = Evo2Client()
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
result_df = client.score_variants_batch(
    variants=variants,
    reference=reference,
    cache_path=CACHE_PATH,
)
elapsed = time.perf_counter() - t0
print(f"\n  Total time: {elapsed:.1f}s")

# ── Summary stats ─────────────────────────────────────────────────────
print("\n=== Results ===\n")

scored = result_df[result_df["error"] == ""].copy()
failed = result_df[result_df["error"] != ""]
print(f"Scored: {len(scored)}")
print(f"Failed: {len(failed)}")
if len(failed) > 0:
    print(f"  Failure reasons:")
    for reason, count in failed["error"].value_counts().items():
        print(f"    {reason}: {count}")

dll = scored["delta_log_likelihood"]
pcts = np.percentile(dll, [1, 5, 25, 50, 75, 95, 99])
print(f"\ndelta_log_likelihood distribution (n={len(dll)}):")
print(f"  min  = {dll.min():.4f}")
print(f"  max  = {dll.max():.4f}")
print(f"  mean = {dll.mean():.4f}")
print(f"  std  = {dll.std():.4f}")
print(f"  p01  = {pcts[0]:.4f}")
print(f"  p05  = {pcts[1]:.4f}")
print(f"  p25  = {pcts[2]:.4f}")
print(f"  p50  = {pcts[3]:.4f}")
print(f"  p75  = {pcts[4]:.4f}")
print(f"  p95  = {pcts[5]:.4f}")
print(f"  p99  = {pcts[6]:.4f}")

# ── Lead variant stats ────────────────────────────────────────────────
lead_row = scored[scored["pos"] == LEAD_POS]
if lead_row.empty:
    print(f"\nWARNING: {LEAD_RSID} not found in scored results!")
else:
    lead_dll = lead_row.iloc[0]["delta_log_likelihood"]
    print(f"\n{LEAD_RSID} (pos={LEAD_POS}):")
    print(f"  delta_log_likelihood = {lead_dll:.6f}")

    # Rank by delta_ll (most negative first)
    scored_sorted = scored.sort_values("delta_log_likelihood")
    rank_neg = (scored_sorted["delta_log_likelihood"].values <= lead_dll).sum()
    pctile_neg = rank_neg / len(scored) * 100
    print(f"  Rank by delta_ll (most negative first): {rank_neg} / {len(scored)}")
    print(f"  Percentile (most negative = 0th): {pctile_neg:.1f}%")

    # Rank by abs(delta_ll) (most disruptive in either direction)
    scored["abs_dll"] = scored["delta_log_likelihood"].abs()
    scored_abs_sorted = scored.sort_values("abs_dll", ascending=False)
    abs_lead = abs(lead_dll)
    rank_abs = (scored["abs_dll"].values >= abs_lead).sum()
    pctile_abs = (1 - rank_abs / len(scored)) * 100
    print(f"  Rank by |delta_ll| (most disruptive first): {rank_abs} / {len(scored)}")
    print(f"  Percentile by |delta_ll|: {pctile_abs:.1f}%")

# ── Histogram ─────────────────────────────────────────────────────────
print(f"\nGenerating histogram: {HIST_PATH}")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(dll, bins=80, color="#4a90d9", edgecolor="white", linewidth=0.5, alpha=0.85)

if not lead_row.empty:
    ax.axvline(lead_dll, color="#d94a4a", linewidth=2, linestyle="--",
               label=f"{LEAD_RSID} ({lead_dll:.2f})")
    ax.legend(fontsize=11)

ax.set_xlabel("delta_log_likelihood (alt − ref)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"ASNS Locus — Evo2 1B delta_log_likelihood (n={len(dll)} SNVs)", fontsize=13)

if not lead_row.empty:
    ax.text(0.98, 0.95,
            f"{LEAD_RSID}: percentile = {pctile_neg:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig(HIST_PATH, dpi=150)
print(f"Saved histogram to {HIST_PATH}")

print("\n=== Phase 2 complete ===")
