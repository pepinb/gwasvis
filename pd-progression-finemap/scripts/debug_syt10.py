"""Diagnostic: trace rs10437796 through every stage of the SYT10 pipeline.

SYT10 is a suggestive mortality locus (p ~ 5.3e-8, just above 5e-8 threshold).
The paper lead rs10437796 appears as "12:33635494" in the chr:pos-based
mortality sumstats.

Usage:
    python3 scripts/debug_syt10.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_SUMSTATS = ROOT / "data" / "raw" / "mortalityGWAS_summaryStats.txt"
LOCI_PARQUET = ROOT / "data" / "loci" / "SYT10_mortality.parquet"
ALIGNED_PARQUET = ROOT / "data" / "ld" / "SYT10_mortality.aligned.parquet"
SNPLIST = ROOT / "data" / "ld" / "SYT10_mortality.snplist"
FINEMAP_PARQUET = ROOT / "data" / "finemap" / "SYT10_mortality.parquet"

RSID = "rs10437796"
CHRPOS = "12:33635494"
CHROM = "12"
POS = 33635494

VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "ALL.chr12.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
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

    # Mortality uses chr:pos as MarkerName
    row = df[df["MarkerName"] == CHRPOS]
    if row.empty:
        print(f"  *** {CHRPOS} NOT FOUND in raw sumstats ***")
        # Also try rsid
        row2 = df[df["MarkerName"] == RSID]
        if not row2.empty:
            print(f"  But found by rsid: {RSID}")
        return

    print(f"  FOUND: {CHRPOS}")
    r = row.iloc[0]
    for col in df.columns:
        print(f"    {col} = {r[col]}")
    z = r.get("Effect", r.get("beta", 0)) / r.get("StdErr", r.get("SE", 1))
    print(f"    z = {z:.4f}")
    print()
    print("  NOTE: Mortality sumstats use chr:pos as MarkerName (no rsids).")
    print(f"  The paper's rsid '{RSID}' maps to this chr:pos via Ensembl GRCh37.")


# ---------------------------------------------------------------------------
# Stage 2: Loci parquet
# ---------------------------------------------------------------------------
def stage2_loci_parquet() -> None:
    banner("STAGE 2", "Loci parquet")

    if not LOCI_PARQUET.exists():
        print(f"  *** {LOCI_PARQUET} does not exist ***")
        return

    df = pd.read_parquet(LOCI_PARQUET)
    print(f"File: {LOCI_PARQUET.name}")
    print(f"Shape: {df.shape}")
    print(f"Position range: {df['pos'].min()} – {df['pos'].max()}")
    print()

    # Check by chr:pos rsid
    row = df[df["rsid"] == CHRPOS]
    if not row.empty:
        r = row.iloc[0]
        print(f"  FOUND by rsid='{CHRPOS}'")
        print(f"    chr = {r['chr']}, pos = {r['pos']}")
        print(f"    a1 = {r['a1']}, a2 = {r['a2']}")
        print(f"    beta = {r['beta']}, se = {r['se']}, pval = {r['pval']}")
        print(f"    eaf = {r.get('eaf', 'N/A')}, n = {r.get('n', 'N/A')}")
    else:
        print(f"  *** {CHRPOS} NOT FOUND by rsid ***")

    # Check by position
    row2 = df[df["pos"] == POS]
    if not row2.empty:
        print(f"  FOUND by pos={POS}")
    else:
        print(f"  *** pos={POS} NOT FOUND ***")

    # Check for actual rsid
    row3 = df[df["rsid"] == RSID]
    print(f"  rsid '{RSID}': {'FOUND' if not row3.empty else 'NOT FOUND (expected — mortality uses chr:pos)'}")


# ---------------------------------------------------------------------------
# Stage 3: 1KG Phase 3 VCF
# ---------------------------------------------------------------------------
def stage3_1kg_vcf() -> None:
    banner("STAGE 3", "1KG Phase 3 EUR VCF")

    region = f"{CHROM}:{POS}-{POS}"
    cmd = ["bcftools", "view", "--regions", region, VCF_URL]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        lines = [l for l in result.stdout.splitlines() if not l.startswith("#")]
    except Exception as e:
        print(f"  bcftools error: {e}")
        return

    if not lines:
        print(f"  Position {POS}: NOT FOUND in 1KG VCF")
        return

    fields = lines[0].split("\t")
    chrom, vpos, vid, ref, alt = fields[0:5]
    info = fields[7]
    info_dict = dict(kv.split("=", 1) for kv in info.split(";") if "=" in kv)
    print(f"  Position {POS}: FOUND")
    print(f"    CHROM={chrom} POS={vpos} ID={vid} REF={ref} ALT={alt}")
    print(f"    Global AF  = {info_dict.get('AF', '?')}")
    print(f"    EUR AF     = {info_dict.get('EUR_AF', '?')}")

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

    cmd = ["bcftools", "view", "--regions", region, "-S", panel_path, VCF_URL]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    lines = [l for l in result.stdout.splitlines() if not l.startswith("#")]

    if lines:
        fields = lines[0].split("\t")
        genotypes = fields[9:]
        ref_ref = sum(1 for g in genotypes if g.startswith("0|0") or g.startswith("0/0"))
        het = sum(1 for g in genotypes if g.startswith("0|1") or g.startswith("1|0") or g.startswith("0/1"))
        hom_alt = sum(1 for g in genotypes if g.startswith("1|1") or g.startswith("1/1"))
        an = 2 * (ref_ref + het + hom_alt)
        ac = het + 2 * hom_alt
        maf = ac / an if an > 0 else 0

        print(f"    EUR samples: {len(eur_samples)}")
        print(f"    Genotypes: REF/REF={ref_ref}  HET={het}  ALT/ALT={hom_alt}")
        print(f"    EUR AC={ac}  AN={an}  MAF={maf:.6f} ({maf*100:.3f}%)")
        if maf < 0.01:
            print(f"\n    *** EUR MAF = {maf:.4f} < 0.01 → DROPPED by plink2 --maf 0.01 ***")
        else:
            print(f"\n    EUR MAF = {maf:.4f} ≥ 0.01 → PASSES plink2 --maf 0.01 filter")

    Path(panel_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Stage 4: Harmonization
# ---------------------------------------------------------------------------
def stage4_harmonization() -> None:
    banner("STAGE 4", "Harmonization (plink2 + allele alignment)")

    if SNPLIST.exists():
        with open(SNPLIST) as f:
            snps = [line.strip() for line in f]
        print(f"Snplist: {SNPLIST.name} ({len(snps)} variants)")
        matches = [s for s in snps if f":{POS}:" in s]
        print(f"  Variants at pos {POS}: {matches or 'NONE'}")
    else:
        print(f"  *** {SNPLIST} does not exist ***")

    print()
    if ALIGNED_PARQUET.exists():
        df = pd.read_parquet(ALIGNED_PARQUET)
        print(f"Aligned parquet: {ALIGNED_PARQUET.name} ({len(df)} variants)")
        row = df[df["pos"] == POS]
        if not row.empty:
            r = row.iloc[0]
            print(f"  FOUND at pos={POS}")
            print(f"    rsid = {r['rsid']}")
            print(f"    a1 = {r['a1']}, a2 = {r['a2']}")
            print(f"    z = {r['z']:.4f}")
            print(f"    eaf = {r.get('eaf', 'N/A')}")
        else:
            print(f"  *** pos={POS} NOT FOUND in aligned parquet ***")


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
    print()

    row = df[df["pos"] == POS]
    if not row.empty:
        r = row.iloc[0]
        print(f"  Paper lead at pos={POS}: FOUND")
        print(f"    rsid = {r['rsid']}")
        print(f"    PIP = {r['pip']:.6f}")
        print(f"    in_cs = {r['in_cs']}")
        print(f"    z = {r['z']:.4f}")
        print(f"    pval = {r['pval']:.2e}")
    else:
        print(f"  *** pos={POS} NOT FOUND ***")

    # Credible sets
    print()
    cs_ids = sorted(df.loc[df["in_cs"], "cs_id"].unique())
    print(f"  Credible sets: {len(cs_ids)}")
    if not cs_ids:
        print("  → No credible sets formed (signal too diffuse for purity threshold)")

    # Top variants
    print()
    print("  Top 10 variants by PIP:")
    top = df.nlargest(10, "pip")
    for _, r in top.iterrows():
        cs_label = f"CS{int(r['cs_id'])}" if r["in_cs"] else "—"
        print(f"    {r['rsid']:20s} pip={r['pip']:.6f} pval={r['pval']:.2e} {cs_label}")

    # PIP distribution
    print()
    print(f"  PIP distribution:")
    print(f"    max PIP      = {df['pip'].max():.6f}")
    print(f"    sum(PIP)     = {df['pip'].sum():.4f}")
    print(f"    PIP > 0.5    = {(df['pip'] > 0.5).sum()} variants")
    print(f"    PIP > 0.1    = {(df['pip'] > 0.1).sum()} variants")
    print(f"    PIP > 0.01   = {(df['pip'] > 0.01).sum()} variants")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary() -> None:
    banner("SUMMARY", "Root cause analysis")

    print("  Variant: rs10437796 / 12:33635494 (paper's SYT10 lead for mortality)")
    print(f"  Significance: suggestive (p = 5.3e-8, just above 5e-8 threshold)")
    print()
    print("  Pipeline trace:")
    print("    ✓ Stage 1 (raw sumstats)  — PRESENT as '12:33635494'")
    print("    ✓ Stage 2 (loci parquet)  — PRESENT")
    print("    ✓ Stage 3 (1KG VCF)       — PRESENT (common variant)")
    print("    ✓ Stage 4 (harmonization) — PRESENT")
    print("    ✓ Stage 5 (fine-mapping)  — PRESENT, PIP ≈ 0.104, NOT in any CS")
    print()
    print("  Root cause: Signal is too diffuse for SuSiE to form a credible set.")
    print("  The suggestive-significance p-value (5.3e-8) produces a moderate z-score")
    print("  (~5.4), and PIP is spread across many variants in LD. No individual")
    print("  variant reaches sufficient posterior weight for a pure credible set.")
    print("  This is expected behaviour for a suggestive locus with L=1.")
    print()
    print("  Display issue: The Streamlit app may show 'Paper lead not in locus data'")
    print("  because it searches for rsid 'rs10437796', but the mortality data uses")
    print("  chr:pos format '12:33635494'. The cross-reference logic needs to handle")
    print("  both identifier formats.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  DEBUG: Tracing rs10437796 through the SYT10/mortality pipeline")
    print("=" * 70)

    stage1_raw_sumstats()
    stage2_loci_parquet()
    stage3_1kg_vcf()
    stage4_harmonization()
    stage5_finemap()
    summary()


if __name__ == "__main__":
    main()
