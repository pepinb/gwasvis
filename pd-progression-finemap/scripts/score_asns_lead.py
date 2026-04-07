"""Phase 1 proof-of-life: score ASNS lead variant rs145274312 with Evo2."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evo2.client import Evo2Client
from evo2.reference import EnsemblRESTProvider, LocalFastaProvider
from evo2.variants import build_variant_windows

# ── Hardcoded variant ────────────────────────────────────────────────
RSID = "rs145274312"
CHROM = "7"
POS = 97470925
REF = "G"
ALT = "A"
WINDOW_SIZE = 8192

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["ensembl", "local"], default="ensembl")
args = parser.parse_args()

# ── Cross-check against parquet ──────────────────────────────────────
print("=== Phase 1: Score ASNS lead variant ===\n")

parquet_path = Path(__file__).resolve().parent.parent / "data" / "loci" / "ASNS_hy3.parquet"
df = pd.read_parquet(parquet_path)
row = df[df["pos"] == POS]
if row.empty:
    print(f"ERROR: pos={POS} not found in {parquet_path.name}")
    sys.exit(1)

row = row.iloc[0]
print(f"Parquet row for {RSID}:")
for col in row.index:
    print(f"  {col:12s} = {row[col]}")

# a1 = effect allele, a2 = other allele in the GWAS
# On GRCh37, ref=G alt=A for this variant
parquet_alleles = {row["a1"].upper(), row["a2"].upper()}
hardcoded_alleles = {REF, ALT}
if parquet_alleles != hardcoded_alleles:
    print(f"\nERROR: Allele mismatch — parquet has {parquet_alleles}, hardcoded {hardcoded_alleles}")
    sys.exit(1)
print(f"\nAlleles match: {hardcoded_alleles}\n")

# ── Fetch reference window ───────────────────────────────────────────
if args.backend == "ensembl":
    print(f"Using Ensembl REST backend")
    reference = EnsemblRESTProvider()
else:
    fasta_path = Path(__file__).resolve().parent.parent / "data" / "reference" / "chr7.fa"
    print(f"Using local FASTA backend: {fasta_path}")
    reference = LocalFastaProvider(str(fasta_path))

print(f"Fetching {WINDOW_SIZE}bp window around {CHROM}:{POS}...")
ref_window, alt_window = build_variant_windows(
    chrom=CHROM, pos=POS, ref_allele=REF, alt_allele=ALT,
    reference=reference, window_size=WINDOW_SIZE,
)
print(f"  Window: {CHROM}:{POS - WINDOW_SIZE // 2}-{POS + WINDOW_SIZE // 2 - 1}")
print(f"  ref_window length: {len(ref_window)}")
print(f"  alt_window length: {len(alt_window)}")

# Visual confirmation around center
half = WINDOW_SIZE // 2
print(f"\n  ref[4090:4106]: {ref_window[4090:4106]}")
print(f"  alt[4090:4106]: {alt_window[4090:4106]}")
print(f"                        {'↑':>7}")
print(f"  ref center [{half}]: {ref_window[half]}")
print(f"  alt center [{half}]: {alt_window[half]}")

assert ref_window[half] == REF, f"Center mismatch: expected {REF}, got {ref_window[half]}"
assert alt_window[half] == ALT, f"Center mismatch: expected {ALT}, got {alt_window[half]}"
print("  Center bases confirmed.\n")

# ── Score with Evo2 via Modal ────────────────────────────────────────
print("Calling Evo2 Modal service (score_ref_alt with embeddings)...")
client = Evo2Client()
result = client.score_variant(ref_window, alt_window, return_embedding=True)

print(f"\n  delta_log_likelihood : {result['delta_log_likelihood']:.6f}")
print(f"  log_likelihood_ref   : {result['log_likelihood_ref']:.4f}")
print(f"  log_likelihood_alt   : {result['log_likelihood_alt']:.4f}")
print(f"  embedding dim        : {len(result['embedding_ref'])}")
print(f"  embedding_layer      : {result['embedding_layer']}")
print(f"  embedding_position   : {result['embedding_position']}")

# ── Save results ─────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent.parent / "data" / "evo2"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "asns_lead_phase1.json"

payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model": "evo2_1b",
    "window_size": WINDOW_SIZE,
    "backend": args.backend,
    "variant": {
        "rsid": RSID,
        "chrom": CHROM,
        "pos": POS,
        "ref": REF,
        "alt": ALT,
    },
    "scores": {
        "log_likelihood_ref": result["log_likelihood_ref"],
        "log_likelihood_alt": result["log_likelihood_alt"],
        "delta_log_likelihood": result["delta_log_likelihood"],
    },
    "embeddings": {
        "layer": result["embedding_layer"],
        "position": result["embedding_position"],
        "ref": result["embedding_ref"],
        "alt": result["embedding_alt"],
    },
}

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"\nSaved to {out_path}")
print("\n=== Phase 1 complete ===")
