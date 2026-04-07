"""BRCA1 sanity check: score a known pathogenic nonsense variant.

BRCA1 c.5503C>T (p.Arg1835Ter) — ClinVar 55601
Genomic GRCh37: chr17:41197784 G>A (minus strand gene, so cDNA C>T = genomic G>A)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evo2.client import Evo2Client
from evo2.reference import EnsemblRESTProvider
from evo2.variants import build_variant_windows

CHROM = "17"
POS = 41197784
REF = "G"
ALT = "A"
WINDOW_SIZE = 8192

print("=== BRCA1 Sanity Check: c.5503C>T (p.Arg1835Ter) ===\n")
print(f"Variant: chr{CHROM}:{POS} {REF}>{ALT}")
print(f"Expected: known pathogenic nonsense, should give clearly negative delta\n")

print(f"Fetching {WINDOW_SIZE}bp window from Ensembl GRCh37...")
reference = EnsemblRESTProvider()
ref_window, alt_window = build_variant_windows(
    chrom=CHROM, pos=POS, ref_allele=REF, alt_allele=ALT,
    reference=reference, window_size=WINDOW_SIZE,
)

half = WINDOW_SIZE // 2
print(f"  ref[{half-6}:{half+10}]: {ref_window[half-6:half+10]}")
print(f"  alt[{half-6}:{half+10}]: {alt_window[half-6:half+10]}")
print(f"  ref center [{half}]: {ref_window[half]}")
print(f"  alt center [{half}]: {alt_window[half]}")
print()

print("Calling Evo2 Modal service...")
client = Evo2Client()
result = client.score_variant(ref_window, alt_window, return_embedding=False)

print(f"\n  delta_log_likelihood : {result['delta_log_likelihood']:.6f}")
print(f"  log_likelihood_ref   : {result['log_likelihood_ref']:.4f}")
print(f"  log_likelihood_alt   : {result['log_likelihood_alt']:.4f}")

if result["delta_log_likelihood"] == 0.0:
    print("\n  FAIL: delta is exactly zero — scoring bug persists")
    sys.exit(1)
else:
    print(f"\n  PASS: non-zero delta ({result['delta_log_likelihood']:.6f})")

print("\n=== BRCA1 sanity check complete ===")
