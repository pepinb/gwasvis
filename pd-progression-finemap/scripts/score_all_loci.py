"""Phase 2.5: Score all seven loci and compare lead variant percentiles."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evo2.client import Evo2Client
from evo2.reference import LocalFastaProvider
from pyfaidx import Fasta

# Import locus definitions from the pipeline
from loci import LOCI

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="1b", help="Evo2 model version (default: 1b)")
args = parser.parse_args()
MODEL = args.model

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "data" / "reference"
LOCI_DIR = ROOT / "data" / "loci"
EVO2_DIR = ROOT / "data" / "evo2"

print("=== Phase 2.5: Score All Loci ===\n")

# ── Initialize providers ──────────────────────────────────────────────
reference = LocalFastaProvider(str(REF_DIR))
client = Evo2Client()
EVO2_DIR.mkdir(parents=True, exist_ok=True)

# Collect per-chromosome Fasta handles for ref allele lookup
chroms_needed = sorted(set(loc["chr"] for loc in LOCI))
fa_handles: dict[str, Fasta] = {}
for c in chroms_needed:
    fa_path = REF_DIR / f"chr{c}.fa"
    if fa_path.exists():
        fa_handles[c] = Fasta(str(fa_path))
    else:
        print(f"WARNING: {fa_path} not found — loci on chr{c} will fail")

# ── Score each locus ──────────────────────────────────────────────────
summary_rows: list[dict] = []
total_t0 = time.perf_counter()

for locus in LOCI:
    name = locus["name"]
    trait = locus["trait"]
    chrom = locus["chr"]
    lead_pos = locus["lead_bp"]
    lead_rsid = locus["lead_rsid"]

    parquet_path = LOCI_DIR / f"{name}_{trait}.parquet"
    cache_path = EVO2_DIR / f"{name}_{trait}_evo2_{MODEL}.parquet"

    print(f"\n{'='*60}")
    print(f"Locus: {name} ({trait}) — lead: {lead_rsid} chr{chrom}:{lead_pos}")
    print(f"{'='*60}")

    if not parquet_path.exists():
        print(f"  SKIP: {parquet_path.name} not found")
        summary_rows.append({
            "locus": name, "trait": trait,
            "paper_lead_rsid": lead_rsid,
            "paper_lead_chr": chrom, "paper_lead_pos": lead_pos,
            "paper_lead_delta_ll": float("nan"),
            "percentile_signed": float("nan"), "percentile_abs": float("nan"),
            "rank_signed": float("nan"), "rank_abs": float("nan"),
            "n_scored": 0, "lead_in_scored": False,
        })
        continue

    df = pd.read_parquet(parquet_path)
    # Filter to SNVs with valid ACGT alleles
    valid_bases = set("ACGT")
    snv_mask = (df["a1"].str.len() == 1) & (df["a2"].str.len() == 1)
    acgt_mask = (
        df["a1"].str.upper().isin(valid_bases)
        & df["a2"].str.upper().isin(valid_bases)
    )
    snv_df = df[snv_mask & acgt_mask].copy()
    n_total = len(df)
    n_snv = len(snv_df)
    n_indel = (~snv_mask).sum()
    n_bad = (snv_mask & ~acgt_mask).sum()
    print(f"  Total: {n_total}, SNVs: {n_snv}, Indels: {n_indel}, Non-ACGT: {n_bad}")

    if chrom not in fa_handles:
        print(f"  SKIP: no FASTA for chr{chrom}")
        continue

    fa = fa_handles[chrom]
    chrom_key = f"chr{chrom}"

    # Determine ref/alt from FASTA
    variants: list[dict] = []
    for _, row in snv_df.iterrows():
        pos = int(row["pos"])
        a1 = row["a1"].upper()
        a2 = row["a2"].upper()
        ref_base = str(fa[chrom_key][pos - 1 : pos]).upper()
        if ref_base == a2:
            ref, alt = a2, a1
        elif ref_base == a1:
            ref, alt = a1, a2
        else:
            continue
        variants.append({
            "chrom": chrom, "pos": pos,
            "ref": ref, "alt": alt, "rsid": row["rsid"],
        })

    print(f"  Ref allele resolved: {len(variants)}")

    t0 = time.perf_counter()
    result_df = client.score_variants_batch(
        variants=variants,
        reference=reference,
        cache_path=cache_path,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.1f}s")

    # Compute lead variant stats
    scored = result_df[result_df["error"] == ""].copy()
    n_scored = len(scored)

    lead_row = scored[scored["pos"] == lead_pos]
    lead_in_scored = not lead_row.empty

    if lead_in_scored:
        lead_dll = lead_row.iloc[0]["delta_log_likelihood"]
        rank_signed = int((scored["delta_log_likelihood"].values <= lead_dll).sum())
        pctile_signed = rank_signed / n_scored * 100
        scored["abs_dll"] = scored["delta_log_likelihood"].abs()
        abs_lead = abs(lead_dll)
        rank_abs = int((scored["abs_dll"].values >= abs_lead).sum())
        pctile_abs = (1 - rank_abs / n_scored) * 100
        print(f"  Lead {lead_rsid}: delta_ll={lead_dll:.4f}, "
              f"percentile={pctile_signed:.1f}%, "
              f"|delta| rank={rank_abs}/{n_scored}")
    else:
        lead_dll = float("nan")
        rank_signed = float("nan")
        pctile_signed = float("nan")
        rank_abs = float("nan")
        pctile_abs = float("nan")
        # Check why
        in_parquet = lead_pos in df["pos"].values
        print(f"  WARNING: Lead {lead_rsid} (pos={lead_pos}) not in scored set")
        print(f"    In parquet: {in_parquet}")

    summary_rows.append({
        "locus": name, "trait": trait,
        "paper_lead_rsid": lead_rsid,
        "paper_lead_chr": chrom, "paper_lead_pos": lead_pos,
        "paper_lead_delta_ll": lead_dll,
        "percentile_signed": pctile_signed,
        "percentile_abs": pctile_abs,
        "rank_signed": rank_signed,
        "rank_abs": rank_abs,
        "n_scored": n_scored,
        "lead_in_scored": lead_in_scored,
    })

total_elapsed = time.perf_counter() - total_t0
print(f"\n\nTotal scoring time: {total_elapsed:.1f}s")

# ── Save summary ──────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_path = EVO2_DIR / f"calibration_summary_{MODEL}.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved summary to {summary_path}")

# ── Print markdown table ──────────────────────────────────────────────
print("\n### Locus Calibration Summary\n")
print("| Locus | Trait | Lead rsid | delta_ll | Percentile | |delta| rank | n_scored | In set |")
print("|-------|-------|-----------|----------|------------|-------------|----------|--------|")
for _, r in summary_df.iterrows():
    dll = f"{r['paper_lead_delta_ll']:.2f}" if pd.notna(r['paper_lead_delta_ll']) else "—"
    pct = f"{r['percentile_signed']:.1f}%" if pd.notna(r['percentile_signed']) else "—"
    arank = f"{int(r['rank_abs'])}/{int(r['n_scored'])}" if pd.notna(r['rank_abs']) else "—"
    in_set = "yes" if r['lead_in_scored'] else "no"
    print(f"| {r['locus']:<5s} | {r['trait']:<9s} | {r['paper_lead_rsid']:<13s} | "
          f"{dll:>8s} | {pct:>10s} | {arank:>11s} | {int(r['n_scored']):>8d} | {in_set:<6s} |")

# ── Per-locus distribution stats ──────────────────────────────────────
print("\n### Per-Locus Distribution Stats\n")
print("| Locus | n | mean | std | min | max |")
print("|-------|---|------|-----|-----|-----|")
for locus in LOCI:
    name = locus["name"]
    trait = locus["trait"]
    cache = EVO2_DIR / f"{name}_{trait}_evo2.parquet"
    if not cache.exists():
        continue
    ldf = pd.read_parquet(cache)
    scored = ldf[ldf["error"] == ""]["delta_log_likelihood"]
    print(f"| {name:<5s} | {len(scored)} | {scored.mean():.2f} | {scored.std():.2f} | "
          f"{scored.min():.2f} | {scored.max():.2f} |")

print("\n=== Phase 2.5 complete ===")
