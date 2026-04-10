#!/usr/bin/env python3
"""Build eQTL probe training datasets for Whole_Blood and Brain_Cortex.

Usage:
    python3 scripts/build_eqtl_dataset.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eqtl.dataset import build_training_dataset, parse_variant_ids, PD_HOLDOUT_CHROMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TISSUES = ["Whole_Blood", "Brain_Cortex"]


def print_report(all_meta: list[dict]) -> None:
    """Print final summary report."""
    print("\n" + "=" * 70)
    print("FINAL REPORT: eQTL Probe Training Datasets")
    print("=" * 70)

    for meta in all_meta:
        tissue = meta["tissue"]
        parquet_path = ROOT / "data" / "eqtl" / f"{tissue}_train.parquet"
        df = pd.read_parquet(parquet_path)

        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        total = len(df)

        print(f"\n--- {tissue} ---")
        print(f"  Positives:  {n_pos:,}")
        print(f"  Negatives:  {n_neg:,}")
        print(f"  Total:      {total:,}")
        print(f"  Balance:    {n_pos}/{n_neg} = {100*n_pos/total:.1f}% / {100*n_neg/total:.1f}%")
        print(f"  Dropped (no match): {meta['n_dropped_no_match']}")
        print(f"  Dropped (no MAF, positives): {meta['n_dropped_no_maf_pos']}")
        print(f"  Dropped (no TSS, positives): {meta['n_dropped_no_tss_pos']}")

        # Chromosome distribution
        parsed = parse_variant_ids(df["variant_id"])
        chrom_counts = parsed["chr_num"].value_counts().sort_index()
        print(f"\n  Chromosome distribution:")
        for ch, cnt in chrom_counts.items():
            print(f"    chr{ch}: {cnt:,} ({100*cnt/total:.1f}%)")

        # Hard assertion
        violating = parsed["chr_num"].isin(PD_HOLDOUT_CHROMS)
        assert not violating.any(), (
            f"FATAL: {violating.sum()} variants on PD held-out chromosomes!"
        )
        print(f"\n  HARD ASSERTION PASSED: 0 variants on chr {sorted(PD_HOLDOUT_CHROMS)}")

        # Sanity check: matching quality
        pos_rows = df[df["label"] == 1]
        neg_rows = df[df["label"] == 0]
        print(f"\n  Matching sanity check:")
        print(f"    Mean MAF  — pos: {pos_rows['maf'].mean():.4f}, neg: {neg_rows['maf'].mean():.4f}")
        print(f"    Mean TSS  — pos: {pos_rows['tss_distance'].mean():,.0f} bp, "
              f"neg: {neg_rows['tss_distance'].mean():,.0f} bp")
        print(f"    Med. MAF  — pos: {pos_rows['maf'].median():.4f}, neg: {neg_rows['maf'].median():.4f}")
        print(f"    Med. TSS  — pos: {pos_rows['tss_distance'].median():,.0f} bp, "
              f"neg: {neg_rows['tss_distance'].median():,.0f} bp")

    print("\n" + "=" * 70)
    print("All datasets built successfully.")
    print("=" * 70)


def main():
    gencode_path = ROOT / "data" / "gencode" / "gencode.v26.annotation.gtf.gz"
    output_dir = ROOT / "data" / "eqtl"

    all_meta = []
    for tissue in TISSUES:
        dapg_path = ROOT / "data" / "eqtl" / "raw" / f"{tissue}.variants_pip.txt.gz"
        signif_path = ROOT / "data" / "gtex" / f"{tissue}.v8.signif_variant_gene_pairs.txt.gz"

        if not dapg_path.exists():
            log.error("DAP-G file not found: %s", dapg_path)
            sys.exit(1)
        if not signif_path.exists():
            log.error("Signif pairs file not found: %s", signif_path)
            sys.exit(1)

        meta = build_training_dataset(
            tissue=tissue,
            dapg_path=dapg_path,
            signif_path=signif_path,
            gencode_path=gencode_path,
            output_dir=output_dir,
            seed=42,
        )
        all_meta.append(meta)

    print_report(all_meta)


if __name__ == "__main__":
    main()
