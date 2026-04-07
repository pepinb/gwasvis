"""Compute ensemble (mean of 1B and 7B) Evo2 scores.

For each variant scored by both models, computes:
  delta_ll_ensemble = (delta_ll_1b + delta_ll_7b) / 2

Produces per-locus and genome-null ensemble parquets, then
runs compute_matched_percentiles.py --model ensemble.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from loci import LOCI

ROOT = Path(__file__).resolve().parent.parent
EVO2_DIR = ROOT / "data" / "evo2"

print("=== Compute Ensemble (1B + 7B) ===\n")


def make_ensemble(path_1b: Path, path_7b: Path, out_path: Path, extra_cols: list[str] | None = None) -> pd.DataFrame | None:
    """Inner-join 1B and 7B on key, average delta_ll, save."""
    if not path_1b.exists() or not path_7b.exists():
        return None

    df_1b = pd.read_parquet(path_1b)
    df_7b = pd.read_parquet(path_7b)

    # Keep only successfully scored variants
    df_1b = df_1b[df_1b["error"] == ""].copy()
    df_7b = df_7b[df_7b["error"] == ""].copy()

    # Inner join on key
    merged = df_1b.merge(
        df_7b[["key", "delta_log_likelihood"]],
        on="key",
        how="inner",
        suffixes=("_1b", "_7b"),
    )

    # Compute ensemble
    merged["delta_log_likelihood"] = (
        merged["delta_log_likelihood_1b"] + merged["delta_log_likelihood_7b"]
    ) / 2.0

    # Build output with same schema as single-model parquets
    keep_cols = ["key", "chrom", "pos", "ref", "alt", "rsid", "delta_log_likelihood"]
    # Carry forward log_likelihood_ref/alt from 1B (just for schema compat)
    if "log_likelihood_ref" in df_1b.columns:
        merged["log_likelihood_ref"] = merged.get("log_likelihood_ref", float("nan"))
        merged["log_likelihood_alt"] = merged.get("log_likelihood_alt", float("nan"))
        keep_cols += ["log_likelihood_ref", "log_likelihood_alt"]
    merged["error"] = ""
    keep_cols.append("error")

    # Carry forward extra columns (e.g. maf, maf_bin for genome null)
    if extra_cols:
        for col in extra_cols:
            if col in df_1b.columns and col not in merged.columns:
                # Re-merge from 1B to get these columns
                merged = merged.merge(df_1b[["key", col]], on="key", how="left")
            if col in merged.columns:
                keep_cols.append(col)

    out = merged[[c for c in keep_cols if c in merged.columns]]
    out.to_parquet(out_path, index=False)
    return out


# --- Per-locus ensemble ---
print("Per-locus ensembles:")
for locus in LOCI:
    name, trait = locus["name"], locus["trait"]
    path_1b = EVO2_DIR / f"{name}_{trait}_evo2_1b.parquet"
    path_7b = EVO2_DIR / f"{name}_{trait}_evo2_7b.parquet"
    out_path = EVO2_DIR / f"{name}_{trait}_evo2_ensemble.parquet"

    result = make_ensemble(path_1b, path_7b, out_path)
    if result is not None:
        print(f"  {name}: {len(result)} variants → {out_path.name}")
    else:
        print(f"  {name}: MISSING 1B or 7B parquet — skipped")

# --- Genome null ensemble ---
print("\nGenome null ensemble:")
null_1b = EVO2_DIR / "genome_null_evo2_1b.parquet"
null_7b = EVO2_DIR / "genome_null_evo2_7b.parquet"
null_out = EVO2_DIR / "genome_null_evo2_ensemble.parquet"

null_result = make_ensemble(null_1b, null_7b, null_out, extra_cols=["maf", "maf_bin"])
if null_result is not None:
    print(f"  {len(null_result)} variants → {null_out.name}")
else:
    print("  MISSING 1B or 7B null parquet — skipped")

# --- Build calibration_summary_ensemble.csv ---
# The matched percentiles script needs a calibration_summary file.
# Build it from the ensemble locus parquets.
print("\nBuilding calibration_summary_ensemble.csv...")
summary_rows = []
for locus in LOCI:
    name, trait = locus["name"], locus["trait"]
    ens_path = EVO2_DIR / f"{name}_{trait}_evo2_ensemble.parquet"
    if not ens_path.exists():
        continue
    df = pd.read_parquet(ens_path)
    scored = df[df["error"] == ""]
    lead_pos = locus["lead_bp"]
    lead_row = scored[scored["pos"] == lead_pos]
    if not lead_row.empty:
        dll = float(lead_row.iloc[0]["delta_log_likelihood"])
        n = len(scored)
        pct_signed = (scored["delta_log_likelihood"] <= dll).sum() / n * 100
        pct_abs = (scored["delta_log_likelihood"].abs() >= abs(dll)).sum() / n * 100
        rank_signed = int((scored["delta_log_likelihood"] <= dll).sum())
        rank_abs = int((scored["delta_log_likelihood"].abs() >= abs(dll)).sum())
        summary_rows.append({
            "locus": name, "trait": trait,
            "paper_lead_rsid": locus["lead_rsid"],
            "paper_lead_chr": locus["chr"],
            "paper_lead_pos": lead_pos,
            "paper_lead_delta_ll": dll,
            "percentile_signed": pct_signed,
            "percentile_abs": pct_abs,
            "rank_signed": rank_signed,
            "rank_abs": rank_abs,
            "n_scored": n,
            "lead_in_scored": True,
        })
    else:
        summary_rows.append({
            "locus": name, "trait": trait,
            "paper_lead_rsid": locus["lead_rsid"],
            "paper_lead_chr": locus["chr"],
            "paper_lead_pos": lead_pos,
            "paper_lead_delta_ll": float("nan"),
            "percentile_signed": float("nan"),
            "percentile_abs": float("nan"),
            "rank_signed": 0,
            "rank_abs": 0,
            "n_scored": len(pd.read_parquet(ens_path).query("error == ''")),
            "lead_in_scored": False,
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = EVO2_DIR / "calibration_summary_ensemble.csv"
summary_df.to_csv(summary_path, index=False)
print(f"  Saved {summary_path.name}")

# --- Run compute_matched_percentiles.py --model ensemble ---
print("\nRunning compute_matched_percentiles.py --model ensemble...")
script = ROOT / "scripts" / "compute_matched_percentiles.py"
result = subprocess.run(
    [sys.executable, str(script), "--model", "ensemble"],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)

# --- Seven-leads comparison table ---
print("\n" + "=" * 100)
print("Seven-Leads Ensemble Comparison")
print("=" * 100)

cal_1b = pd.read_csv(EVO2_DIR / "calibration_matched_1b.csv")
cal_7b = pd.read_csv(EVO2_DIR / "calibration_matched_7b.csv")
cal_ens = pd.read_csv(EVO2_DIR / "calibration_matched_ensemble.csv")

header = (f"{'Locus':<8s} {'Lead':<16s} {'1B Δ':>8s} {'7B Δ':>8s} {'Ens Δ':>8s} "
          f"{'1B match%':>10s} {'7B match%':>10s} {'Ens match%':>11s}")
print(header)
print("-" * len(header))

for _, r1b in cal_1b.iterrows():
    name = r1b["locus"]
    r7b = cal_7b[cal_7b["locus"] == name].iloc[0]
    rens = cal_ens[cal_ens["locus"] == name].iloc[0]

    dll_1b = f"{r1b['lead_delta_ll']:.2f}" if pd.notna(r1b["lead_delta_ll"]) else "--"
    dll_7b = f"{r7b['lead_delta_ll']:.2f}" if pd.notna(r7b["lead_delta_ll"]) else "--"
    dll_ens = f"{rens['lead_delta_ll']:.2f}" if pd.notna(rens["lead_delta_ll"]) else "--"

    pct_1b = f"{r1b['null_percentile_signed']:.1f}%" if pd.notna(r1b["null_percentile_signed"]) else "--"
    pct_7b = f"{r7b['null_percentile_signed']:.1f}%" if pd.notna(r7b["null_percentile_signed"]) else "--"
    pct_ens = f"{rens['null_percentile_signed']:.1f}%" if pd.notna(rens["null_percentile_signed"]) else "--"

    print(f"{name:<8s} {r1b['lead_rsid']:<16s} {dll_1b:>8s} {dll_7b:>8s} {dll_ens:>8s} "
          f"{pct_1b:>10s} {pct_7b:>10s} {pct_ens:>11s}")

print("\n=== Done ===")
