"""Compute MAF-matched genome-wide percentiles for each paper lead variant.

For each of the 7 lead variants, finds the closest MAF bin in the scored
genome-wide null and computes:
  - signed percentile (fraction with delta_ll ≤ lead's delta_ll)
  - |delta| percentile (fraction with |delta_ll| ≥ |lead's delta_ll|)

If a lead's MAF falls outside the null's MAF range, uses the nearest bin
and flags it as extrapolated.

Outputs: data/evo2/matched_percentiles.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from loci import LOCI

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="1b", help="Evo2 model version (default: 1b)")
args = parser.parse_args()
MODEL = args.model

ROOT = Path(__file__).resolve().parent.parent
EVO2_DIR = ROOT / "data" / "evo2"
SUMMARY_PATH = EVO2_DIR / f"calibration_summary_{MODEL}.csv"
NULL_SCORED_PATH = EVO2_DIR / f"genome_null_evo2_{MODEL}.parquet"
OUT_PATH = EVO2_DIR / f"calibration_matched_{MODEL}.csv"

MAF_BINS = [
    (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10),
    (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
]

print("=== Compute MAF-Matched Percentiles ===\n")

# Load data
summary = pd.read_csv(SUMMARY_PATH)
null_df = pd.read_parquet(NULL_SCORED_PATH)
null_scored = null_df[null_df["error"] == ""].copy()
print(f"Null scored: {len(null_scored)} variants")
print(f"Summary: {len(summary)} loci\n")

# Load per-locus parquets to get lead MAF from GWAS
# (p-value is available in the parquet; we need freq/maf column)
loci_dir = ROOT / "data" / "loci"


def get_lead_maf(locus: dict) -> float | None:
    """Try to get lead variant MAF from locus parquet or 1000G."""
    parquet = loci_dir / f"{locus['name']}_{locus['trait']}.parquet"
    if not parquet.exists():
        return None
    ldf = pd.read_parquet(parquet)
    lead = ldf[ldf["pos"] == locus["lead_bp"]]
    if lead.empty:
        return None
    # Check common column names for frequency
    for col in ["freq", "maf", "eaf", "a1freq", "frq", "FRQ"]:
        if col in lead.columns:
            val = lead.iloc[0][col]
            if pd.notna(val):
                return min(float(val), 1.0 - float(val))
    return None


def assign_maf_bin(maf: float) -> tuple[int, bool]:
    """Assign MAF to a bin. Returns (bin_index, is_extrapolated)."""
    for i, (lo, hi) in enumerate(MAF_BINS):
        if lo <= maf < hi:
            return i, False
    if maf == 0.50:
        return len(MAF_BINS) - 1, False
    # Extrapolation: use nearest bin
    if maf < MAF_BINS[0][0]:
        return 0, True
    if maf >= MAF_BINS[-1][1]:
        return len(MAF_BINS) - 1, True
    return 0, True


# Compute matched percentiles
rows = []
for locus in LOCI:
    name = locus["name"]
    trait = locus["trait"]

    srow = summary[summary["locus"] == name]
    if srow.empty or not srow.iloc[0]["lead_in_scored"]:
        print(f"  {name}: lead not scored — skipping")
        rows.append({
            "locus": name, "trait": trait,
            "lead_rsid": locus["lead_rsid"],
            "lead_delta_ll": float("nan"),
            "lead_maf": float("nan"),
            "maf_bin": float("nan"),
            "extrapolated": False,
            "locus_percentile_signed": float("nan"),
            "null_percentile_signed": float("nan"),
            "null_percentile_abs": float("nan"),
            "null_n": 0,
        })
        continue

    lead_dll = srow.iloc[0]["paper_lead_delta_ll"]
    locus_pctile = srow.iloc[0]["percentile_signed"]

    # Get MAF
    lead_maf = get_lead_maf(locus)
    if lead_maf is None or lead_maf == 0:
        # Fall back: use overall null (no MAF matching)
        print(f"  {name}: no MAF available — using full null")
        null_dll = null_scored["delta_log_likelihood"].values
        maf_bin_idx = -1
        extrapolated = True
    else:
        maf_bin_idx, extrapolated = assign_maf_bin(lead_maf)
        null_bin = null_scored[null_scored["maf_bin"] == maf_bin_idx]
        null_dll = null_bin["delta_log_likelihood"].values
        bin_lo, bin_hi = MAF_BINS[maf_bin_idx]
        extrap_flag = " (EXTRAPOLATED)" if extrapolated else ""
        print(f"  {name}: MAF={lead_maf:.4f} → bin {maf_bin_idx} "
              f"[{bin_lo:.3f}-{bin_hi:.3f}), n={len(null_dll)}{extrap_flag}")

    # Signed percentile: fraction with delta_ll ≤ lead's
    null_pctile_signed = (null_dll <= lead_dll).sum() / len(null_dll) * 100

    # |delta| percentile: fraction with |delta_ll| ≥ |lead's|
    abs_lead = abs(lead_dll)
    null_pctile_abs = (np.abs(null_dll) >= abs_lead).sum() / len(null_dll) * 100

    print(f"    delta_ll={lead_dll:.2f}, locus_pctile={locus_pctile:.1f}%, "
          f"null_pctile_signed={null_pctile_signed:.1f}%, "
          f"null_pctile_abs={null_pctile_abs:.1f}%")

    rows.append({
        "locus": name, "trait": trait,
        "lead_rsid": locus["lead_rsid"],
        "lead_delta_ll": lead_dll,
        "lead_maf": lead_maf if lead_maf is not None else float("nan"),
        "maf_bin": maf_bin_idx,
        "extrapolated": extrapolated,
        "locus_percentile_signed": locus_pctile,
        "null_percentile_signed": null_pctile_signed,
        "null_percentile_abs": null_pctile_abs,
        "null_n": len(null_dll),
    })

result = pd.DataFrame(rows)
result.to_csv(OUT_PATH, index=False)

# Print summary table
print(f"\n{'='*90}")
print(f"{'Locus':<8s} {'Trait':<10s} {'rsid':<16s} {'MAF':>6s} {'Bin':>4s} "
      f"{'ΔLL':>8s} {'Locus%':>7s} {'Null%':>7s} {'|Δ|%':>7s} {'n_null':>7s} {'Extrap':>7s}")
print(f"{'-'*8} {'-'*10} {'-'*16} {'-'*6} {'-'*4} "
      f"{'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for _, r in result.iterrows():
    maf_s = f"{r['lead_maf']:.4f}" if pd.notna(r["lead_maf"]) else "—"
    dll_s = f"{r['lead_delta_ll']:.2f}" if pd.notna(r["lead_delta_ll"]) else "—"
    lp = f"{r['locus_percentile_signed']:.1f}" if pd.notna(r["locus_percentile_signed"]) else "—"
    np_s = f"{r['null_percentile_signed']:.1f}" if pd.notna(r["null_percentile_signed"]) else "—"
    ap = f"{r['null_percentile_abs']:.1f}" if pd.notna(r["null_percentile_abs"]) else "—"
    nn = f"{int(r['null_n'])}" if r["null_n"] > 0 else "—"
    ext = "yes" if r["extrapolated"] else "no"
    print(f"{r['locus']:<8s} {r['trait']:<10s} {r['lead_rsid']:<16s} {maf_s:>6s} "
          f"{int(r['maf_bin']):>4d} {dll_s:>8s} {lp:>7s} {np_s:>7s} {ap:>7s} {nn:>7s} {ext:>7s}")

print(f"\nSaved to {OUT_PATH}")
print("\n=== Done ===")
