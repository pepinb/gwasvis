"""Score genome-wide null candidates with Evo2.

Reads genome_null_candidates.parquet (built by build_genome_null.py),
scores all ~6,400 biallelic SNVs via Modal parallel map, and saves
scored results to genome_null_evo2_{model}.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evo2.client import Evo2Client
from evo2.reference import LocalFastaProvider

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="1b", help="Evo2 model version (default: 1b)")
args = parser.parse_args()
MODEL = args.model

ROOT = Path(__file__).resolve().parent.parent
EVO2_DIR = ROOT / "data" / "evo2"
REF_DIR = ROOT / "data" / "reference"
CANDIDATES_PATH = EVO2_DIR / "genome_null_candidates.parquet"
SCORED_PATH = EVO2_DIR / f"genome_null_evo2_{MODEL}.parquet"

print("=== Score Genome-Wide Null ===\n")

if not CANDIDATES_PATH.exists():
    print(f"ERROR: {CANDIDATES_PATH} not found — run build_genome_null.py first")
    sys.exit(1)

candidates = pd.read_parquet(CANDIDATES_PATH)
print(f"Loaded {len(candidates)} null candidates")
print(f"MAF bins: {candidates['maf_bin'].value_counts().sort_index().to_dict()}")

# Build variant list
variants = []
for _, row in candidates.iterrows():
    variants.append({
        "chrom": str(row["chrom"]),
        "pos": int(row["pos"]),
        "ref": row["ref"],
        "alt": row["alt"],
        "rsid": row.get("rsid", "."),
    })

# Initialize providers
reference = LocalFastaProvider(str(REF_DIR))
client = Evo2Client()

t0 = time.perf_counter()
result_df = client.score_variants_batch(
    variants=variants,
    reference=reference,
    cache_path=SCORED_PATH,
)
elapsed = time.perf_counter() - t0

scored = result_df[result_df["error"] == ""]
failed = result_df[result_df["error"] != ""]
print(f"\nScoring complete in {elapsed:.1f}s")
print(f"  Scored: {len(scored)}")
print(f"  Failed: {len(failed)}")

if len(failed) > 0:
    print(f"\n  Failure breakdown:")
    for err, count in failed["error"].value_counts().head(10).items():
        print(f"    {count}x: {err[:80]}")

# Merge MAF info back onto scored results
candidates_key = candidates.apply(
    lambda r: f"{r['chrom']}:{int(r['pos'])}:{r['ref']}:{r['alt']}", axis=1
)
candidates["key"] = candidates_key
maf_lookup = candidates.set_index("key")[["maf", "maf_bin"]]
result_df = result_df.join(maf_lookup, on="key", how="left")
result_df.to_parquet(SCORED_PATH, index=False)

scored_with_maf = result_df[result_df["error"] == ""]
print(f"\nDelta_ll distribution (scored null):")
dll = scored_with_maf["delta_log_likelihood"]
print(f"  n     = {len(dll)}")
print(f"  mean  = {dll.mean():.4f}")
print(f"  std   = {dll.std():.4f}")
print(f"  min   = {dll.min():.4f}")
print(f"  max   = {dll.max():.4f}")

print(f"\nPer-MAF-bin stats:")
for maf_bin in sorted(scored_with_maf["maf_bin"].dropna().unique()):
    bin_dll = scored_with_maf[scored_with_maf["maf_bin"] == maf_bin]["delta_log_likelihood"]
    print(f"  Bin {int(maf_bin)}: n={len(bin_dll)}, "
          f"mean={bin_dll.mean():.4f}, std={bin_dll.std():.4f}")

print(f"\nSaved to {SCORED_PATH}")
print("\n=== Done ===")
