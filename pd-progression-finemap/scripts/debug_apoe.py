#!/usr/bin/env python3
"""APOE mortality fine-mapping diagnostic — trace allele alignment step by step.

Run from the project root:
    python scripts/debug_apoe.py
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
LOCI_DIR = PROJECT / "data" / "loci"
LD_DIR = PROJECT / "data" / "ld"
REF_DIR = LD_DIR / "1kg_eur_hg19"

LEAD_POS = 45411941  # rs429358
CHROM = "19"
WINDOW = 500_000
START = LEAD_POS - WINDOW
END = LEAD_POS + WINDOW

# 1KG VCF base
_1KG_VCF_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
_1KG_PANEL_URL = f"{_1KG_VCF_BASE}/integrated_call_samples_v3.20130502.ALL.panel"

_AMBIGUOUS_PAIRS = frozenset({("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")})

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── STEP 2A: Inspect raw sumstats ──────────────────────────────────────────

section("STEP 2A — Raw sumstats for APOE mortality")

ss_path = LOCI_DIR / "APOE_mortality.parquet"
ss = pd.read_parquet(ss_path)

print(f"  Shape: {ss.shape}")
print(f"  Columns: {list(ss.columns)}")
print(f"\n  Head (5 rows):")
print(ss.head().to_string(index=False))

# Find lead SNP
lead_mask = ss["pos"] == LEAD_POS
if lead_mask.any():
    lead = ss.loc[lead_mask].iloc[0]
    print(f"\n  Lead SNP at 19:{LEAD_POS}:")
    for col in ss.columns:
        print(f"    {col:>12s} = {lead[col]}")
else:
    print(f"\n  !! Lead SNP at pos {LEAD_POS} NOT FOUND")

# rsid format
print(f"\n  rsid examples (first 5): {list(ss['rsid'].head())}")
print(f"  rsid contains ':': {ss['rsid'].str.contains(':').mean()*100:.1f}%")

# Allele length distribution (spot indels)
print(f"\n  a1 length distribution:")
print(ss["a1"].str.len().value_counts().to_string())
print(f"\n  a2 length distribution:")
print(ss["a2"].str.len().value_counts().to_string())

# Ambiguous strand SNPs
ambig = ss.apply(
    lambda r: (r["a1"].upper(), r["a2"].upper()) in _AMBIGUOUS_PAIRS, axis=1
)
print(f"\n  Ambiguous strand (A/T, C/G): {ambig.sum()} / {len(ss)} ({ambig.mean()*100:.1f}%)")


# ── STEP 2B: Load 1KG VCF slice ───────────────────────────────────────────

section("STEP 2B — 1KG VCF slice for APOE region")

import shutil

bcftools = shutil.which("bcftools")
plink2 = shutil.which("plink2")
print(f"  bcftools: {bcftools}")
print(f"  plink2:   {plink2}")

with tempfile.TemporaryDirectory(prefix="apoe_debug_") as tmpdir:
    tmp = Path(tmpdir)

    # Get EUR samples
    import requests
    panel_path = REF_DIR / "1kg_panel.txt"
    if not panel_path.exists():
        REF_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_1KG_PANEL_URL, timeout=60)
        resp.raise_for_status()
        panel_path.write_text(resp.text)
    panel_df = pd.read_csv(panel_path, sep="\t")
    eur_samples = panel_df.loc[panel_df["super_pop"] == "EUR", "sample"].tolist()
    keep_file = tmp / "eur_samples.txt"
    keep_file.write_text("\n".join(eur_samples) + "\n")
    print(f"  EUR samples: {len(eur_samples)}")

    # Stream region from remote VCF
    vcf_url = f"{_1KG_VCF_BASE}/ALL.chr{CHROM}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    region_vcf = tmp / "region.vcf.gz"
    print(f"  Streaming chr{CHROM}:{START}-{END} from 1KG...")

    cmd = [
        bcftools, "view",
        "--regions", f"{CHROM}:{START}-{END}",
        "--types", "snps",
        "--min-alleles", "2",
        "--max-alleles", "2",
        "-O", "z", "-o", str(region_vcf),
        vcf_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Index
    subprocess.run([bcftools, "index", str(region_vcf)], check=True, capture_output=True)

    # Filter to EUR, drop monomorphic
    eur_vcf = tmp / "region_eur.vcf.gz"
    cmd = [
        bcftools, "view",
        "--samples-file", str(keep_file),
        "--min-ac", "1",
        "-O", "z", "-o", str(eur_vcf),
        str(region_vcf),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Convert to plink2 with chr:pos:ref:alt IDs
    plink_prefix = tmp / "region_eur"
    cmd = [
        plink2,
        "--vcf", str(eur_vcf),
        "--maf", "0.01",
        "--snps-only",
        "--set-all-var-ids", "@:#:$r:$a",
        "--make-pgen",
        "--out", str(plink_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Read pvar
    pvar = pd.read_csv(
        plink_prefix.with_suffix(".pvar"), sep="\t", comment="#",
        names=["chr", "pos", "varid", "ref", "alt"],
        usecols=[0, 1, 2, 3, 4], dtype=str,
    )
    pvar["pos_int"] = pd.to_numeric(pvar["pos"], errors="coerce")
    print(f"  Variants in plink2 pvar: {len(pvar)}")
    print(f"\n  First 5 pvar rows:")
    print(pvar.head().to_string(index=False))

    # Find lead SNP in pvar
    lead_pvar = pvar[pvar["pos_int"] == LEAD_POS]
    if not lead_pvar.empty:
        lp = lead_pvar.iloc[0]
        print(f"\n  Lead SNP in VCF (pos={LEAD_POS}):")
        print(f"    varid: {lp['varid']}")
        print(f"    REF:   {lp['ref']}")
        print(f"    ALT:   {lp['alt']}")
    else:
        print(f"\n  !! Lead SNP at pos {LEAD_POS} NOT in 1KG pvar")

    # ── STEP 2C: Compare alleles ──────────────────────────────────────────

    section("STEP 2C — Allele comparison at rs429358")

    if lead_mask.any() and not lead_pvar.empty:
        ss_a1 = lead["a1"]
        ss_a2 = lead["a2"]
        vcf_ref = lp["ref"]
        vcf_alt = lp["alt"]

        print(f"  Sumstats: a1 (effect) = {ss_a1}, a2 (non-effect) = {ss_a2}")
        print(f"  VCF:      REF = {vcf_ref}, ALT = {vcf_alt}")
        print()

        if ss_a1 == vcf_ref and ss_a2 == vcf_alt:
            print(f"  Classification: 'match' (a1==REF, a2==ALT)")
            print(f"  → Effect allele is REF")
            print(f"  → plink2 LD is oriented to ALT allele dosages")
            print(f"  → z-score SHOULD be flipped to align with LD")
            print(f"  → Current code does NOT flip → BUG!")
        elif ss_a1 == vcf_alt and ss_a2 == vcf_ref:
            print(f"  Classification: 'flip' (a1==ALT, a2==REF)")
            print(f"  → Effect allele is ALT")
            print(f"  → plink2 LD is oriented to ALT allele dosages")
            print(f"  → z-score is already aligned, should NOT be flipped")
            print(f"  → Current code DOES flip → BUG!")
        elif (ss_a1, ss_a2) in _AMBIGUOUS_PAIRS:
            print(f"  Classification: AMBIGUOUS (strand-ambiguous pair)")
        else:
            print(f"  Classification: MISMATCH (alleles don't match)")

        # Show current (wrong) vs correct z-score
        beta = float(lead["beta"])
        se = float(lead["se"])
        raw_z = beta / se
        print(f"\n  beta = {beta}, se = {se}")
        print(f"  raw z = beta/se = {raw_z:.4f}")

        # What the current code produces:
        if ss_a1 == vcf_alt and ss_a2 == vcf_ref:
            current_z = -raw_z  # current code flips for "flip" case
            correct_z = raw_z   # but should NOT flip
        elif ss_a1 == vcf_ref and ss_a2 == vcf_alt:
            current_z = raw_z   # current code keeps for "match" case
            correct_z = -raw_z  # but SHOULD flip
        else:
            current_z = raw_z
            correct_z = raw_z

        print(f"  Current (buggy) z: {current_z:.4f}")
        print(f"  Correct z:         {correct_z:.4f}")

    # ── STEP 2D: Check allele alignment across ALL variants ──────────────

    section("STEP 2D — Allele alignment audit for all APOE variants")

    # Remove ambiguous-strand SNPs from sumstats (matches production code)
    ss_clean = ss[~ambig].copy().reset_index(drop=True)
    print(f"  Sumstats after removing ambiguous: {len(ss_clean)}")

    # Build position lookup from pvar
    pvar_by_pos = {}
    for _, row in pvar.iterrows():
        pk = f"{row['chr']}:{row['pos']}"
        pvar_by_pos[pk] = {"ref": row["ref"], "alt": row["alt"], "varid": row["varid"]}

    # Classify every sumstats variant
    n_match = n_flip = n_ambig = n_mismatch = n_missing = n_indel = 0
    sample_matches = []
    sample_flips = []
    sample_mismatches = []

    for _, row in ss_clean.iterrows():
        pk = f"{row['chr']}:{int(row['pos'])}"
        entry = pvar_by_pos.get(pk)
        if entry is None:
            n_missing += 1
            continue

        s1, s2 = str(row["a1"]).upper(), str(row["a2"]).upper()
        r_ref, r_alt = entry["ref"].upper(), entry["alt"].upper()

        # Skip indels
        if len(s1) > 1 or len(s2) > 1 or len(r_ref) > 1 or len(r_alt) > 1:
            n_indel += 1
            continue

        if s1 == r_ref and s2 == r_alt:
            n_match += 1
            if len(sample_matches) < 3:
                sample_matches.append((pk, s1, s2, r_ref, r_alt))
        elif s1 == r_alt and s2 == r_ref:
            n_flip += 1
            if len(sample_flips) < 3:
                sample_flips.append((pk, s1, s2, r_ref, r_alt))
        elif (s1, s2) in _AMBIGUOUS_PAIRS:
            n_ambig += 1
        else:
            n_mismatch += 1
            if len(sample_mismatches) < 3:
                sample_mismatches.append((pk, s1, s2, r_ref, r_alt))

    total_found = n_match + n_flip + n_ambig + n_mismatch + n_indel
    print(f"  Variants found in 1KG: {total_found}")
    print(f"  Not in 1KG: {n_missing}")
    print(f"\n  Alignment breakdown:")
    print(f"    MATCH   (a1==REF, a2==ALT): {n_match:5d}  ← current code: no z-flip (WRONG)")
    print(f"    FLIP    (a1==ALT, a2==REF): {n_flip:5d}  ← current code: z-flip   (WRONG)")
    print(f"    AMBIG   (strand-ambiguous): {n_ambig:5d}  ← should drop")
    print(f"    MISMATCH                  : {n_mismatch:5d}  ← dropped")
    print(f"    INDEL                     : {n_indel:5d}  ← dropped")

    if sample_matches:
        print(f"\n  Sample MATCH variants (a1==REF):")
        for pk, s1, s2, r, a in sample_matches:
            print(f"    {pk}: ss({s1}/{s2}) vcf(REF={r}/ALT={a})")
    if sample_flips:
        print(f"\n  Sample FLIP variants (a1==ALT):")
        for pk, s1, s2, r, a in sample_flips:
            print(f"    {pk}: ss({s1}/{s2}) vcf(REF={r}/ALT={a})")
    if sample_mismatches:
        print(f"\n  Sample MISMATCH variants:")
        for pk, s1, s2, r, a in sample_mismatches:
            print(f"    {pk}: ss({s1}/{s2}) vcf(REF={r}/ALT={a})")

    # ── STEP 2E: Show what correct harmonization does to z-scores ────────

    section("STEP 2E — Z-score comparison: current (buggy) vs correct")

    # Pick 5 MATCH and 5 FLIP variants near the lead
    near_lead = ss_clean[
        (ss_clean["pos"] >= LEAD_POS - 10000) &
        (ss_clean["pos"] <= LEAD_POS + 10000)
    ].head(10)

    print(f"  Variants near lead (±10kb), showing z-score impact:")
    print(f"  {'pos':>12s}  {'a1':>3s} {'a2':>3s}  {'REF':>3s} {'ALT':>3s}  "
          f"{'class':>6s}  {'raw_z':>8s}  {'cur_z':>8s}  {'fix_z':>8s}")
    print(f"  {'-'*75}")

    for _, row in near_lead.iterrows():
        pk = f"{row['chr']}:{int(row['pos'])}"
        entry = pvar_by_pos.get(pk)
        if entry is None:
            continue

        s1, s2 = str(row["a1"]).upper(), str(row["a2"]).upper()
        r_ref, r_alt = entry["ref"].upper(), entry["alt"].upper()

        raw_z = float(row["beta"]) / float(row["se"])

        if s1 == r_ref and s2 == r_alt:
            cls = "MATCH"
            cur_z = raw_z     # current: no flip
            fix_z = -raw_z    # correct: flip (effect allele is REF, LD is ALT)
        elif s1 == r_alt and s2 == r_ref:
            cls = "FLIP"
            cur_z = -raw_z    # current: flips
            fix_z = raw_z     # correct: no flip (effect allele is ALT = LD orientation)
        else:
            cls = "OTHER"
            cur_z = raw_z
            fix_z = raw_z

        print(f"  {int(row['pos']):>12d}  {s1:>3s} {s2:>3s}  {r_ref:>3s} {r_alt:>3s}  "
              f"{cls:>6s}  {raw_z:>8.3f}  {cur_z:>8.3f}  {fix_z:>8.3f}")

    # ── STEP 2F: Check sample size ────────────────────────────────────────

    section("STEP 2F — Sample size check")

    if "n" in ss.columns:
        print(f"  n column stats:")
        print(f"    median: {ss['n'].median():.0f}")
        print(f"    mean:   {ss['n'].mean():.0f}")
        print(f"    min:    {ss['n'].min():.0f}")
        print(f"    max:    {ss['n'].max():.0f}")
        print(f"    at lead: {float(lead['n']):.0f}")
        print(f"  Paper reports ~5744 for mortality GWAS")
    else:
        print(f"  !! No 'n' column in sumstats")

    # ── STEP 2G: z-score sanity check ─────────────────────────────────────

    section("STEP 2G — z-score sanity check for lead SNP")

    if lead_mask.any():
        beta = float(lead["beta"])
        se = float(lead["se"])
        pval = float(lead["pval"])
        z_raw = beta / se
        import scipy.stats as stats
        z_from_p = stats.norm.isf(pval / 2)  # two-sided

        print(f"  beta = {beta}")
        print(f"  se   = {se}")
        print(f"  pval = {pval:.2e}")
        print(f"  z from beta/se = {z_raw:.4f}")
        print(f"  |z| from pval  = {z_from_p:.4f}")
        print(f"  Match: {'YES' if abs(abs(z_raw) - z_from_p) < 0.1 else 'NO'}")

    section("DIAGNOSTIC COMPLETE")
    print("  Root cause: z-score flip logic in _alleles_match is backwards.")
    print("  When effect allele == VCF ALT → z is correct (matches LD) → do NOT flip")
    print("  When effect allele == VCF REF → z is opposite to LD → MUST flip")
    print("  Current code does the opposite, corrupting ALL z-scores.")
