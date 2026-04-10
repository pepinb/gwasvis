"""End-to-end BRCA1 c.5503C>T sanity check for the eQTL embeddings pipeline.

Exercises the full stack that downstream eQTL training will use:
    Evo2Client → Modal (evo2_7b, blocks.26, 128bp pool, RC augmentation)
    → parquet cache → pd.Series result

Usage:
    python3 scripts/brca1_sanity.py

Expected output (from Evo 2 paper, Extended Data Fig. 5):
    delta_log_likelihood ≈ -4.7 ± 0.5

Requires:
    - MODAL_TOKEN_ID in the environment
    - Network access to the Ensembl REST API (for the 8192 bp window)
    - The evo2-modal service deployed with score_variant_with_rc
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evo2.client import Evo2Client  # noqa: E402

WINDOW_SIZE = 8192
CENTER_INDEX = WINDOW_SIZE // 2

# BRCA1 NM_007294.4:c.5503C>T (p.Arg1835Ter), GRCh38 coordinates.
# BRCA1 is on the reverse strand, so coding C>T is a G>A on +strand.
BRCA1 = {
    "chrom": "17",
    "hg38_bp": 43057078,
    "ref": "G",
    "alt": "A",
    "gene": "BRCA1",
    "hgvs_c": "NM_007294.4:c.5503C>T",
    "hgvs_p": "p.Arg1835Ter",
}

CACHE_PATH = ROOT / "data" / "evo2" / "cache" / "embeddings_7b_blocks26_w128.parquet"
WINDOW_CACHE = ROOT / "data" / "evo2" / "cache" / "brca1_c5503ct_hg38_8192.json"

EXPECTED_DELTA = -4.7
TOLERANCE = 0.5


def fetch_brca1_window() -> dict:
    """Fetch an 8192 bp window centered on the variant from Ensembl REST."""
    if WINDOW_CACHE.exists():
        return json.loads(WINDOW_CACHE.read_text())

    chrom = BRCA1["chrom"]
    pos = BRCA1["hg38_bp"]
    start = pos - CENTER_INDEX
    end = start + WINDOW_SIZE - 1

    url = (
        f"https://rest.ensembl.org/sequence/region/human/"
        f"{chrom}:{start}..{end}:1?content-type=text/plain"
    )
    print(f"Fetching {WINDOW_SIZE} bp from Ensembl: chr{chrom}:{start}-{end}")
    req = urllib.request.Request(
        url,
        headers={"Accept": "text/plain", "User-Agent": "gwasvis-brca1-sanity"},
    )
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        seq = resp.read().decode("ascii").strip().upper()

    if len(seq) != WINDOW_SIZE:
        raise RuntimeError(
            f"Ensembl returned {len(seq)} bp, expected {WINDOW_SIZE}"
        )

    actual_ref = seq[CENTER_INDEX]
    if actual_ref != BRCA1["ref"]:
        raise RuntimeError(
            f"Reference allele mismatch at chr{chrom}:{pos}: "
            f"expected {BRCA1['ref']}, got {actual_ref}. "
            f"Update BRCA1['hg38_bp'] or verify coordinate."
        )

    alt_seq = seq[:CENTER_INDEX] + BRCA1["alt"] + seq[CENTER_INDEX + 1:]

    payload = {
        "variant": BRCA1,
        "ref_sequence": seq,
        "alt_sequence": alt_seq,
        "assembly": "hg38",
        "window_start_1based": start,
        "window_end_1based": end,
    }
    WINDOW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    WINDOW_CACHE.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    print("=== BRCA1 c.5503C>T end-to-end sanity check ===\n")

    try:
        window = fetch_brca1_window()
    except (urllib.error.URLError, RuntimeError) as exc:
        print(f"FAIL: could not prepare BRCA1 window: {exc}", file=sys.stderr)
        return 2

    client = Evo2Client(cache_path=CACHE_PATH)
    print(f"Cache: {CACHE_PATH}")
    print(f"Variant: chr{BRCA1['chrom']}:{BRCA1['hg38_bp']} "
          f"{BRCA1['ref']}>{BRCA1['alt']} ({BRCA1['hgvs_c']})")
    print("Layer: blocks.26, pool_window: 128, RC augmentation: yes\n")

    result = client.score_variant_with_rc(
        chrom=BRCA1["chrom"],
        position=BRCA1["hg38_bp"],
        ref=BRCA1["ref"],
        alt=BRCA1["alt"],
        ref_sequence=window["ref_sequence"],
        alt_sequence=window["alt_sequence"],
        assembly="hg38",
        layer_names=["blocks.26"],
        pool_window=128,
    )

    layer = result["blocks.26"]
    delta_sum = layer["delta_log_likelihood"]
    delta_fwd = layer["delta_log_likelihood_fwd"]
    delta_rc = layer["delta_log_likelihood_rc"]

    # Evo 2 paper Extended Data Fig. 5 reports delta_log_likelihood on a
    # per-kilobase basis (sum of token-level log-prob deltas across the
    # 8192 bp window, normalised to 1000 bp). This converts our raw sum
    # into the same scale as the published BRCA1 numbers.
    delta_per_kb = delta_sum * 1000.0 / WINDOW_SIZE
    delta_fwd_per_kb = delta_fwd * 1000.0 / WINDOW_SIZE
    delta_rc_per_kb = delta_rc * 1000.0 / WINDOW_SIZE

    hidden_dim = len(layer["ref_fwd_embedding"])
    concat_dim = hidden_dim * 4

    print(f"delta_log_likelihood (sum) = {delta_sum:+.4f}  over {WINDOW_SIZE} bp")
    print(f"  forward                  = {delta_fwd:+.4f}")
    print(f"  reverse-complement       = {delta_rc:+.4f}")
    print()
    print(f"delta_log_likelihood /kb   = {delta_per_kb:+.4f}  (paper scale)")
    print(f"  forward                  = {delta_fwd_per_kb:+.4f}")
    print(f"  reverse-complement       = {delta_rc_per_kb:+.4f}")
    print()
    print(f"embedding hidden_dim       = {hidden_dim}")
    print(f"concat feature length      = {concat_dim}  (4 × hidden_dim)")
    print(f"model                      = {layer['model']}")
    print()

    within_tolerance = abs(delta_per_kb - EXPECTED_DELTA) < TOLERANCE
    status = "PASS" if within_tolerance else "FAIL"
    print(
        f"[{status}] expected delta/kb ≈ {EXPECTED_DELTA:+.2f} ± {TOLERANCE}, "
        f"got {delta_per_kb:+.4f}"
    )

    if not within_tolerance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
