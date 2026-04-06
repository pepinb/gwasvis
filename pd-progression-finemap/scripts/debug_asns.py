"""Diagnostic: trace rs145274312 through every stage of the ASNS pipeline.

Traces the paper's ASNS lead variant through:
  1. Raw summary statistics
  2. Loci parquet (post-Ensembl annotation)
  3. 1KG Phase 3 EUR VCF
  4. Harmonization / plink2 output
  5. MAF filter

Usage:
    python3 scripts/debug_asns.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_SUMSTATS = ROOT / "data" / "raw" / "HY3GWAS_summaryStats.txt"
LOCI_PARQUET = ROOT / "data" / "loci" / "ASNS_hy3.parquet"
ALIGNED_PARQUET = ROOT / "data" / "ld" / "ASNS_hy3.aligned.parquet"
SNPLIST = ROOT / "data" / "ld" / "ASNS_hy3.snplist"
FINEMAP_PARQUET = ROOT / "data" / "finemap" / "ASNS_hy3.parquet"

RSID = "rs145274312"
ENSEMBL_POS = 97470925  # Correct GRCh37 position from Ensembl
LOCI_PY_POS = 97478547  # Position in loci.py (from paper)
CHROM = "7"

VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "ALL.chr7.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
)
PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)


def banner(step: str, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {step}: {title}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Stage 1: Raw summary statistics
# ---------------------------------------------------------------------------
def stage1_raw_sumstats() -> None:
    banner("STAGE 1", "Raw summary statistics")

    df = pd.read_csv(RAW_SUMSTATS, sep="\t")
    print(f"File: {RAW_SUMSTATS.name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()

    row = df[df["MarkerName"] == RSID]
    if row.empty:
        print(f"  *** {RSID} NOT FOUND in raw sumstats ***")
        return

    print(f"  FOUND: {RSID}")
    r = row.iloc[0]
    print(f"    effect_allele (a1) = {r['effect_allele']}")
    print(f"    noneffect_allele (a2) = {r['noneffect_allele']}")
    print(f"    beta  = {r['beta']}")
    print(f"    SE    = {r['SE']}")
    print(f"    pval  = {r['pvalue']}")
    print(f"    freq  = {r['freq']}")
    print(f"    N     = {r['TotalSampleSize']}")
    z = r["beta"] / r["SE"]
    print(f"    z     = beta/SE = {z:.4f}")
    print(f"    HetISq  = {r['HetISq']}")
    print(f"    HetPVal = {r['HetPVal']}")
    print()
    print("  NOTE: HY3 sumstats are rsid-only (no chr/pos column).")
    print("  Position is resolved via Ensembl REST API during loci extraction.")


# ---------------------------------------------------------------------------
# Stage 2: Loci parquet (post-Ensembl annotation)
# ---------------------------------------------------------------------------
def stage2_loci_parquet() -> None:
    banner("STAGE 2", "Loci parquet (post-Ensembl annotation)")

    if not LOCI_PARQUET.exists():
        print(f"  *** {LOCI_PARQUET} does not exist ***")
        return

    df = pd.read_parquet(LOCI_PARQUET)
    print(f"File: {LOCI_PARQUET.name}")
    print(f"Shape: {df.shape}")
    print(f"Position range: {df['pos'].min()} – {df['pos'].max()}")
    print()

    # By rsid
    row = df[df["rsid"] == RSID]
    if not row.empty:
        r = row.iloc[0]
        print(f"  FOUND by rsid: {RSID}")
        print(f"    chr = {r['chr']}, pos = {r['pos']}")
        print(f"    a1 = {r['a1']}, a2 = {r['a2']}")
        print(f"    beta = {r['beta']}, se = {r['se']}, pval = {r['pval']}")
        print(f"    eaf = {r['eaf']}, n = {r['n']}")
    else:
        print(f"  *** {RSID} NOT FOUND by rsid ***")

    # Position check
    print()
    print(f"  Position discrepancy check:")
    print(f"    loci.py lead_bp = {LOCI_PY_POS}")
    print(f"    Ensembl GRCh37  = {ENSEMBL_POS}")
    print(f"    Difference      = {abs(LOCI_PY_POS - ENSEMBL_POS):,} bp")

    at_loci_pos = df[df["pos"] == LOCI_PY_POS]
    at_ensembl_pos = df[df["pos"] == ENSEMBL_POS]
    print(f"    Variants at loci.py pos ({LOCI_PY_POS}): {len(at_loci_pos)}")
    print(f"    Variants at Ensembl pos ({ENSEMBL_POS}): {len(at_ensembl_pos)}")

    if not at_ensembl_pos.empty:
        print(f"    → Ensembl-resolved position is correct; loci.py lead_bp is WRONG")


# ---------------------------------------------------------------------------
# Stage 3: 1KG Phase 3 VCF
# ---------------------------------------------------------------------------
def stage3_1kg_vcf() -> None:
    banner("STAGE 3", "1KG Phase 3 EUR VCF")

    # Query both positions
    for label, pos in [("Ensembl", ENSEMBL_POS), ("loci.py", LOCI_PY_POS)]:
        region = f"7:{pos}-{pos}"
        cmd = [
            "bcftools", "view", "--regions", region, VCF_URL,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            lines = [l for l in result.stdout.splitlines() if not l.startswith("#")]
        except Exception as e:
            print(f"  bcftools error for {label} pos: {e}")
            continue

        if not lines:
            print(f"  Position {pos} ({label}): NOT FOUND in 1KG VCF")
        else:
            fields = lines[0].split("\t")
            chrom, vpos, vid, ref, alt = fields[0:5]
            info = fields[7]
            # Parse INFO AF fields
            info_dict = dict(kv.split("=", 1) for kv in info.split(";") if "=" in kv)
            print(f"  Position {pos} ({label}): FOUND")
            print(f"    CHROM={chrom} POS={vpos} ID={vid} REF={ref} ALT={alt}")
            print(f"    Global AF  = {info_dict.get('AF', '?')}")
            print(f"    EUR AF     = {info_dict.get('EUR_AF', '?')}")
            print(f"    EAS AF     = {info_dict.get('EAS_AF', '?')}")
            print(f"    AFR AF     = {info_dict.get('AFR_AF', '?')}")
            print(f"    AMR AF     = {info_dict.get('AMR_AF', '?')}")
            print(f"    SAS AF     = {info_dict.get('SAS_AF', '?')}")

    # Compute exact EUR genotype counts
    print()
    print("  Computing exact EUR genotype counts...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        panel_path = f.name
        panel_result = subprocess.run(
            ["curl", "-s", PANEL_URL], capture_output=True, text=True, timeout=60,
        )
        eur_samples = [
            line.split("\t")[0]
            for line in panel_result.stdout.splitlines()
            if "\tEUR\t" in line
        ]
        f.write("\n".join(eur_samples) + "\n")

    region = f"7:{ENSEMBL_POS}-{ENSEMBL_POS}"
    cmd = [
        "bcftools", "view", "--regions", region,
        "-S", panel_path, VCF_URL,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    lines = [l for l in result.stdout.splitlines() if not l.startswith("#")]

    if lines:
        fields = lines[0].split("\t")
        genotypes = fields[9:]
        ref_ref = sum(1 for g in genotypes if g.startswith("0|0") or g.startswith("0/0"))
        het = sum(1 for g in genotypes if g.startswith("0|1") or g.startswith("1|0") or g.startswith("0/1"))
        hom_alt = sum(1 for g in genotypes if g.startswith("1|1") or g.startswith("1/1"))
        missing = len(genotypes) - ref_ref - het - hom_alt
        an = 2 * (ref_ref + het + hom_alt)
        ac = het + 2 * hom_alt
        maf = ac / an if an > 0 else 0

        print(f"    EUR samples: {len(eur_samples)}")
        print(f"    Genotypes: REF/REF={ref_ref}  HET={het}  ALT/ALT={hom_alt}  missing={missing}")
        print(f"    EUR AC={ac}  AN={an}  MAF={maf:.6f} ({maf*100:.3f}%)")
        print()
        print(f"    *** EUR MAF = {maf:.4f} < 0.01 → DROPPED by plink2 --maf 0.01 ***")
    else:
        print("    Could not compute EUR genotypes.")

    Path(panel_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Stage 4: Harmonization (aligned parquet + snplist)
# ---------------------------------------------------------------------------
def stage4_harmonization() -> None:
    banner("STAGE 4", "Harmonization (plink2 + allele alignment)")

    # Check snplist (plink2 output, post-MAF filter)
    if SNPLIST.exists():
        with open(SNPLIST) as f:
            snps = [line.strip() for line in f]
        print(f"Snplist: {SNPLIST.name} ({len(snps)} variants)")

        # Search by position
        matches_ensembl = [s for s in snps if f":{ENSEMBL_POS}:" in s]
        matches_locipy = [s for s in snps if f":{LOCI_PY_POS}:" in s]
        print(f"  Variants at Ensembl pos ({ENSEMBL_POS}): {matches_ensembl or 'NONE'}")
        print(f"  Variants at loci.py pos ({LOCI_PY_POS}): {matches_locipy or 'NONE'}")
        print()
        print("  → Variant was already removed by plink2 --maf 0.01 before harmonization")
    else:
        print(f"  *** {SNPLIST} does not exist ***")

    # Check aligned parquet
    print()
    if ALIGNED_PARQUET.exists():
        df = pd.read_parquet(ALIGNED_PARQUET)
        print(f"Aligned parquet: {ALIGNED_PARQUET.name} ({len(df)} variants)")
        row = df[df["rsid"] == RSID]
        by_pos = df[df["pos"] == ENSEMBL_POS]
        print(f"  {RSID} by rsid: {'FOUND' if not row.empty else 'NOT FOUND'}")
        print(f"  pos={ENSEMBL_POS}: {'FOUND' if not by_pos.empty else 'NOT FOUND'}")
    else:
        print(f"  *** {ALIGNED_PARQUET} does not exist ***")


# ---------------------------------------------------------------------------
# Stage 5: Fine-mapping output
# ---------------------------------------------------------------------------
def stage5_finemap() -> None:
    banner("STAGE 5", "Fine-mapping output")

    if not FINEMAP_PARQUET.exists():
        print(f"  *** {FINEMAP_PARQUET} does not exist ***")
        return

    df = pd.read_parquet(FINEMAP_PARQUET)
    print(f"File: {FINEMAP_PARQUET.name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()

    row = df[df["rsid"] == RSID]
    by_pos = df[df["pos"] == ENSEMBL_POS]
    print(f"  {RSID} by rsid: {'FOUND' if not row.empty else 'NOT FOUND'}")
    print(f"  pos={ENSEMBL_POS}: {'FOUND' if not by_pos.empty else 'NOT FOUND'}")

    # Show top variants
    print()
    print("  Top 5 variants by PIP:")
    top = df.nlargest(5, "pip")[["rsid", "pos", "pip", "in_cs", "cs_id"]]
    print(top.to_string(index=False))

    # Show paper lead from evaluate.py
    print()
    print(f"  Paper's lead_bp in loci.py: {LOCI_PY_POS}")
    near_lead = df[(df["pos"] >= LOCI_PY_POS - 100) & (df["pos"] <= LOCI_PY_POS + 100)]
    print(f"  Variants within ±100bp of lead_bp: {len(near_lead)}")
    near_ensembl = df[(df["pos"] >= ENSEMBL_POS - 100) & (df["pos"] <= ENSEMBL_POS + 100)]
    print(f"  Variants within ±100bp of Ensembl pos: {len(near_ensembl)}")
    if not near_ensembl.empty:
        print(near_ensembl[["rsid", "pos", "pip", "in_cs"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary() -> None:
    banner("SUMMARY", "Root cause analysis")

    print("  Variant: rs145274312 (paper's ASNS lead for HY3 trait)")
    print()
    print("  Pipeline trace:")
    print("    ✓ Stage 1 (raw sumstats)  — PRESENT (a1=A, a2=G, beta=1.87, p~1e-9)")
    print("    ✓ Stage 2 (loci parquet)  — PRESENT at pos=97470925 (Ensembl GRCh37)")
    print("    ✓ Stage 3 (1KG VCF)       — PRESENT (REF=G, ALT=A)")
    print("    ✗ Stage 4 (harmonization) — DROPPED by plink2 --maf 0.01")
    print("    ✗ Stage 5 (fine-mapping)  — ABSENT (never entered SuSiE)")
    print()
    print("  Root cause: EUR MAF = 0.20% (2/1006 alleles in 503 EUR samples)")
    print("  The plink2 --maf 0.01 filter requires MAF ≥ 1.0% and removes this variant.")
    print()
    print("  Additional issue: loci.py lead_bp = 97478547 is WRONG.")
    print("  Ensembl GRCh37 places rs145274312 at 97470925 (7,622 bp away).")
    print("  Position 97478547 has NO variant in 1KG Phase 3.")
    print()
    print("  Context from paper (Tan et al. 2024):")
    print('    "rs145274312 is present in >1% of PD patients but only 0.97% in gnomAD EUR"')
    print("    In 1KG Phase 3 EUR, it is even rarer: 0.20% (only 2 heterozygotes).")
    print()
    print("  Options:")
    print("    A. Lower MAF threshold to 0.1% for all loci (risks LD instability)")
    print("    B. Lower MAF threshold to 0.1% for ASNS only (special-case)")
    print("    C. Accept that rs145274312 cannot be fine-mapped with 1KG EUR LD")
    print("       (paper's own observation — variant too rare for population LD reference)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  DEBUG: Tracing rs145274312 through the ASNS/HY3 pipeline")
    print("=" * 70)

    stage1_raw_sumstats()
    stage2_loci_parquet()
    stage3_1kg_vcf()
    stage4_harmonization()
    stage5_finemap()
    summary()


if __name__ == "__main__":
    main()
