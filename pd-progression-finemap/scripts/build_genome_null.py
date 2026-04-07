"""Build a MAF-matched genome-wide null for Evo2 delta_ll calibration.

Samples ~6,400 biallelic SNVs stratified by MAF from six chromosomes
(1, 2, 4, 7, 12, 19) using 1000 Genomes Phase 3 EUR allele frequencies.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from loci import LOCI

ROOT = Path(__file__).resolve().parent.parent
EVO2_DIR = ROOT / "data" / "evo2"
EVO2_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EVO2_DIR / "genome_null_candidates.parquet"

RNG = random.Random(42)
CHROMS = ["1", "2", "4", "7", "12", "19"]
CHROM_LENGTHS = {
    "1": 249250621, "2": 243199373, "4": 191154276,
    "7": 159138663, "12": 133851895, "19": 59128983,
}

# MAF bins (log-scale edges)
MAF_BINS = [
    (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10),
    (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
]
PER_BIN = 800

# Exclusion zones: ±1Mb around each paper lead
EXCLUDE_ZONES = []
for loc in LOCI:
    EXCLUDE_ZONES.append((loc["chr"], loc["lead_bp"] - 1_000_000, loc["lead_bp"] + 1_000_000))

VCF_URL_TEMPLATE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
)


def is_excluded(chrom: str, pos: int) -> bool:
    for exc_chrom, exc_start, exc_end in EXCLUDE_ZONES:
        if chrom == exc_chrom and exc_start <= pos <= exc_end:
            return True
    return False


def assign_maf_bin(maf: float) -> int | None:
    for i, (lo, hi) in enumerate(MAF_BINS):
        if lo <= maf < hi:
            return i
    if maf == 0.50:
        return len(MAF_BINS) - 1
    return None


def query_region(chrom: str, start: int, end: int) -> list[dict]:
    """Query 1000G VCF for biallelic SNVs with EUR_AF in a region."""
    url = VCF_URL_TEMPLATE.format(chrom=chrom)
    region = f"{chrom}:{start}-{end}"
    cmd = [
        "bcftools", "query", "-r", region,
        "-f", "%CHROM\t%POS\t%REF\t%ALT\t%INFO/EUR_AF\n",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return []
    except subprocess.TimeoutExpired:
        return []

    valid_bases = set("ACGT")
    variants = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        chrom_v, pos_s, ref, alt, eur_af_s = parts
        # Biallelic SNV only
        if len(ref) != 1 or len(alt) != 1:
            continue
        if ref not in valid_bases or alt not in valid_bases:
            continue
        if "," in eur_af_s:
            continue  # multiallelic
        try:
            eur_af = float(eur_af_s)
        except ValueError:
            continue
        maf = min(eur_af, 1 - eur_af)
        if maf < 0.005 or maf > 0.50:
            continue
        pos = int(pos_s)
        if is_excluded(chrom_v, pos):
            continue
        maf_bin = assign_maf_bin(maf)
        if maf_bin is None:
            continue
        variants.append({
            "chrom": chrom_v, "pos": pos, "ref": ref, "alt": alt,
            "rsid": ".", "maf": maf, "maf_bin": maf_bin,
        })
    return variants


print("=== Building Genome-Wide Null ===\n")

# Sample random 1Mb regions across chromosomes
REGION_SIZE = 1_000_000
REGIONS_PER_CHROM = 25  # ~25Mb per chromosome, 150Mb total

regions = []
for chrom in CHROMS:
    chrom_len = CHROM_LENGTHS[chrom]
    # Avoid telomeres (first/last 5Mb)
    safe_start = 5_000_000
    safe_end = chrom_len - 5_000_000 - REGION_SIZE
    if safe_end <= safe_start:
        safe_end = chrom_len - REGION_SIZE
    for _ in range(REGIONS_PER_CHROM):
        start = RNG.randint(safe_start, safe_end)
        regions.append((chrom, start, start + REGION_SIZE))

RNG.shuffle(regions)
print(f"Querying {len(regions)} random 1Mb regions across {len(CHROMS)} chromosomes...")

# Query regions, collecting variants into bins
all_variants: list[dict] = []
bin_counts = [0] * len(MAF_BINS)

for i, (chrom, start, end) in enumerate(regions):
    variants = query_region(chrom, start, end)
    all_variants.extend(variants)
    for v in variants:
        bin_counts[v["maf_bin"]] += 1

    # Progress every 10 regions
    if (i + 1) % 10 == 0:
        total = len(all_variants)
        min_bin = min(bin_counts)
        print(f"  [{i+1}/{len(regions)}] {total} variants collected, "
              f"smallest bin: {min_bin}")

    # Early termination if all bins have enough
    if all(c >= PER_BIN * 2 for c in bin_counts):
        print(f"  All bins saturated after {i+1} regions — stopping early")
        break

print(f"\nTotal candidate pool: {len(all_variants)}")
print(f"\nPer-bin counts (before sampling):")
for i, (lo, hi) in enumerate(MAF_BINS):
    print(f"  Bin {i} [{lo:.3f}-{hi:.3f}): {bin_counts[i]}")
    if bin_counts[i] < 200:
        print(f"    WARNING: fewer than 200 candidates in this bin!")
        # Don't exit — just warn

# Stratified subsample
print(f"\nStratified sampling {PER_BIN} per bin...")
pool_df = pd.DataFrame(all_variants)
sampled_frames = []
for i, (lo, hi) in enumerate(MAF_BINS):
    bin_df = pool_df[pool_df["maf_bin"] == i]
    n_sample = min(PER_BIN, len(bin_df))
    sampled = bin_df.sample(n=n_sample, random_state=42)
    sampled_frames.append(sampled)
    print(f"  Bin {i} [{lo:.3f}-{hi:.3f}): sampled {n_sample}/{len(bin_df)}")

result_df = pd.concat(sampled_frames, ignore_index=True)
result_df.to_parquet(OUT_PATH, index=False)

print(f"\nFinal sample: {len(result_df)} variants")
print(f"MAF distribution:")
print(f"  mean = {result_df['maf'].mean():.4f}")
print(f"  std  = {result_df['maf'].std():.4f}")
print(f"  min  = {result_df['maf'].min():.4f}")
print(f"  max  = {result_df['maf'].max():.4f}")
print(f"\nSaved to {OUT_PATH}")
print("\n=== Done ===")
