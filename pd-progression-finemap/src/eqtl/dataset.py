"""Build matched positive/negative eQTL probe training datasets from DAP-G output.

Construction:
  1. Load DAP-G variants_pip file, filter to SNVs.
  2. Positives = PIP > 0.5, deduplicated by variant_id (keep highest PIP).
  3. Background pool = PIP < 0.01, deduplicated by variant_id.
  4. Drop PD held-out chromosomes (1, 2, 4, 7, 12, 19).
  5. Annotate MAF from GTEx signif_variant_gene_pairs.
  6. Annotate TSS distance from GENCODE v26.
  7. Bin MAF (8 bins) and TSS distance (5 bins).
  8. 1:1 matched negative sampling within (chr, MAF bin, TSS bin) cells.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from loci import PD_HOLDOUT_CHROMS

log = logging.getLogger(__name__)

MAF_BIN_EDGES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
TSS_BIN_EDGES = [0, 1_000, 5_000, 25_000, 100_000, 1_000_000]
TSS_BIN_LABELS = ["0-1kb", "1-5kb", "5-25kb", "25-100kb", "100kb-1Mb"]

NUCLEOTIDES = set("ACGT")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_variant_ids(variant_ids: pd.Series) -> pd.DataFrame:
    """Parse chr{N}_{pos}_{ref}_{alt}_b38 into component columns."""
    parts = variant_ids.str.split("_", expand=True)
    chrom_str = parts[0].str.replace("chr", "", regex=False)
    return pd.DataFrame({
        "chr_num": pd.to_numeric(chrom_str, errors="coerce").astype("Int64"),
        "chr": parts[0],
        "pos": parts[1].astype(int),
        "ref": parts[2],
        "alt": parts[3],
    })


def is_snv(ref: pd.Series, alt: pd.Series) -> pd.Series:
    """True when both ref and alt are single nucleotides in {A,C,G,T}."""
    return (
        (ref.str.len() == 1)
        & (alt.str.len() == 1)
        & ref.isin(NUCLEOTIDES)
        & alt.isin(NUCLEOTIDES)
    )


# ---------------------------------------------------------------------------
# Reference loaders
# ---------------------------------------------------------------------------

def load_maf_lookup(signif_path: Path) -> pd.Series:
    """Build variant_id → maf mapping from signif_variant_gene_pairs file.

    Multiple gene pairs may report the same variant; MAF is identical across
    pairs so we just drop duplicates.
    """
    log.info("Loading MAF from %s", signif_path)
    df = pd.read_csv(
        signif_path, sep="\t",
        usecols=["variant_id", "maf"],
        dtype={"variant_id": str, "maf": float},
    )
    df = df.drop_duplicates(subset="variant_id")
    return df.set_index("variant_id")["maf"]


def load_tss_map(gencode_path: Path) -> dict[str, int]:
    """Build gene_id → TSS position from GENCODE v26 GTF.

    TSS = start for + strand genes, end for - strand genes.
    Uses gene-level rows only. Gene IDs include version suffix to match DAP-G.
    """
    import polars as pl
    import gtfparse
    log.info("Loading GENCODE v26 from %s", gencode_path)
    gtf = gtfparse.read_gtf(str(gencode_path))
    # gtfparse returns a Polars DataFrame
    genes = gtf.filter(pl.col("feature") == "gene")
    genes = genes.with_columns(
        pl.when(pl.col("strand") == "+")
        .then(pl.col("start"))
        .otherwise(pl.col("end"))
        .alias("tss")
    )
    return dict(zip(
        genes["gene_id"].to_list(),
        genes["tss"].to_list(),
    ))


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_maf(maf: pd.Series) -> pd.Series:
    """Assign MAF values to 8 bins. Boundary variants snap to nearest bin."""
    bins = pd.cut(
        maf,
        bins=MAF_BIN_EDGES,
        labels=[f"{MAF_BIN_EDGES[i]}-{MAF_BIN_EDGES[i+1]}"
                for i in range(len(MAF_BIN_EDGES) - 1)],
        include_lowest=True,
        right=True,
    )
    # Snap out-of-range values to nearest bin
    below = maf < MAF_BIN_EDGES[0]
    above = maf > MAF_BIN_EDGES[-1]
    first_label = f"{MAF_BIN_EDGES[0]}-{MAF_BIN_EDGES[1]}"
    last_label = f"{MAF_BIN_EDGES[-2]}-{MAF_BIN_EDGES[-1]}"
    if below.any():
        log.info("Snapping %d variants with MAF < %s to first bin", below.sum(), MAF_BIN_EDGES[0])
        bins = bins.cat.add_categories([first_label]) if first_label not in bins.cat.categories else bins
        bins[below] = first_label
    if above.any():
        log.info("Snapping %d variants with MAF > %s to last bin", above.sum(), MAF_BIN_EDGES[-1])
        bins = bins.cat.add_categories([last_label]) if last_label not in bins.cat.categories else bins
        bins[above] = last_label
    return bins.astype(str)


def bin_tss_distance(dist: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Assign TSS distance to 5 bins. Returns (bin labels, drop mask for > 1Mb)."""
    drop_mask = dist > TSS_BIN_EDGES[-1]
    bins = pd.cut(
        dist,
        bins=TSS_BIN_EDGES,
        labels=TSS_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    return bins.astype(str), drop_mask


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_training_dataset(
    tissue: str,
    dapg_path: Path,
    signif_path: Path,
    gencode_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Build matched pos/neg training dataset for one tissue.

    Returns metadata dict.
    """
    rng = np.random.default_rng(seed)
    log.info("=" * 60)
    log.info("Building dataset for %s", tissue)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load DAP-G
    # ------------------------------------------------------------------
    log.info("Loading DAP-G file: %s", dapg_path)
    dapg = pd.read_csv(dapg_path, sep="\t", dtype={"cluster_id": str})
    log.info("  Raw rows: %d", len(dapg))

    # Parse variant_id
    parsed = parse_variant_ids(dapg["variant_id"])
    dapg = pd.concat([dapg, parsed], axis=1)

    # ------------------------------------------------------------------
    # 2. Filter to SNVs
    # ------------------------------------------------------------------
    snv_mask = is_snv(dapg["ref"], dapg["alt"])
    dapg = dapg[snv_mask].copy()
    log.info("  After SNV filter: %d", len(dapg))

    # ------------------------------------------------------------------
    # 3. Positives and background pool
    # ------------------------------------------------------------------
    pos_df = dapg[dapg["pip"] > 0.5].copy()
    pos_df = pos_df.sort_values("pip", ascending=False).drop_duplicates(
        subset="variant_id", keep="first"
    )
    log.info("  Positives (PIP>0.5, deduped): %d", len(pos_df))

    bg_df = dapg[dapg["pip"] < 0.01].copy()
    bg_df = bg_df.drop_duplicates(subset="variant_id", keep="first")
    log.info("  Background pool (PIP<0.01, deduped): %d", len(bg_df))

    # ------------------------------------------------------------------
    # 4. Drop PD held-out chromosomes
    # ------------------------------------------------------------------
    pos_before = len(pos_df)
    pos_df = pos_df[~pos_df["chr_num"].isin(PD_HOLDOUT_CHROMS)].copy()
    log.info("  Positives after chr hold-out: %d (dropped %d)",
             len(pos_df), pos_before - len(pos_df))

    bg_before = len(bg_df)
    bg_df = bg_df[~bg_df["chr_num"].isin(PD_HOLDOUT_CHROMS)].copy()
    log.info("  Background after chr hold-out: %d (dropped %d)",
             len(bg_df), bg_before - len(bg_df))

    # ------------------------------------------------------------------
    # 5. Annotate MAF
    # ------------------------------------------------------------------
    maf_lookup = load_maf_lookup(signif_path)
    log.info("  MAF lookup: %d unique variants", len(maf_lookup))

    pos_df["maf"] = pos_df["variant_id"].map(maf_lookup)
    pos_no_maf = pos_df["maf"].isna().sum()
    log.info("  Positives missing MAF: %d / %d", pos_no_maf, len(pos_df))
    pos_df = pos_df.dropna(subset=["maf"])

    bg_df["maf"] = bg_df["variant_id"].map(maf_lookup)
    bg_no_maf = bg_df["maf"].isna().sum()
    log.info("  Background missing MAF: %d / %d", bg_no_maf, len(bg_df) + bg_no_maf)
    bg_df = bg_df.dropna(subset=["maf"])

    # ------------------------------------------------------------------
    # 6. Annotate TSS distance
    # ------------------------------------------------------------------
    tss_map = load_tss_map(gencode_path)
    log.info("  GENCODE TSS map: %d genes", len(tss_map))

    def compute_tss_dist(row):
        tss = tss_map.get(row["gene"])
        if tss is None:
            return np.nan
        return abs(row["pos"] - tss)

    pos_df["tss_distance"] = pos_df.apply(compute_tss_dist, axis=1)
    pos_no_tss = pos_df["tss_distance"].isna().sum()
    log.info("  Positives missing TSS: %d / %d", pos_no_tss, len(pos_df))
    pos_df = pos_df.dropna(subset=["tss_distance"])
    pos_df["tss_distance"] = pos_df["tss_distance"].astype(int)

    bg_df["tss_distance"] = bg_df.apply(compute_tss_dist, axis=1)
    bg_no_tss = bg_df["tss_distance"].isna().sum()
    log.info("  Background missing TSS: %d / %d", bg_no_tss, len(bg_df))
    bg_df = bg_df.dropna(subset=["tss_distance"])
    bg_df["tss_distance"] = bg_df["tss_distance"].astype(int)

    # ------------------------------------------------------------------
    # 7. Bin MAF and TSS distance
    # ------------------------------------------------------------------
    pos_df["maf_bin"] = bin_maf(pos_df["maf"])
    bg_df["maf_bin"] = bin_maf(bg_df["maf"])

    pos_df["tss_bin"], pos_tss_drop = bin_tss_distance(pos_df["tss_distance"])
    bg_df["tss_bin"], bg_tss_drop = bin_tss_distance(bg_df["tss_distance"])

    if pos_tss_drop.any():
        log.info("  Dropping %d positives with TSS dist > 1Mb", pos_tss_drop.sum())
        pos_df = pos_df[~pos_tss_drop].copy()
    if bg_tss_drop.any():
        log.info("  Dropping %d background with TSS dist > 1Mb", bg_tss_drop.sum())
        bg_df = bg_df[~bg_tss_drop].copy()

    # Drop rows where binning produced 'nan' string
    pos_df = pos_df[pos_df["maf_bin"] != "nan"].copy()
    bg_df = bg_df[bg_df["maf_bin"] != "nan"].copy()
    pos_df = pos_df[pos_df["tss_bin"] != "nan"].copy()
    bg_df = bg_df[bg_df["tss_bin"] != "nan"].copy()

    log.info("  Positives ready for matching: %d", len(pos_df))
    log.info("  Background ready for matching: %d", len(bg_df))

    # ------------------------------------------------------------------
    # 8. Matched 1:1 negative sampling
    # ------------------------------------------------------------------
    # Index background by (chr, maf_bin, tss_bin)
    bg_df["match_key"] = (
        bg_df["chr"].astype(str) + "|" + bg_df["maf_bin"] + "|" + bg_df["tss_bin"]
    )
    bg_groups: dict[str, list[int]] = {}
    for idx, key in zip(bg_df.index, bg_df["match_key"]):
        bg_groups.setdefault(key, []).append(idx)

    matched_pos_indices = []
    matched_neg_indices = []
    n_dropped = 0

    for idx, row in pos_df.iterrows():
        key = f"{row['chr']}|{row['maf_bin']}|{row['tss_bin']}"
        candidates = bg_groups.get(key, [])
        if not candidates:
            n_dropped += 1
            continue
        chosen_i = rng.integers(len(candidates))
        neg_idx = candidates.pop(chosen_i)
        if not candidates:
            del bg_groups[key]
        matched_pos_indices.append(idx)
        matched_neg_indices.append(neg_idx)

    log.info("  Matched positives: %d, dropped (no match): %d", len(matched_pos_indices), n_dropped)

    # Build final dataframes
    cols_out = ["variant_id", "chr", "pos", "ref", "alt", "maf",
                "tss_distance", "gene", "pip"]

    pos_final = pos_df.loc[matched_pos_indices, cols_out].copy()
    pos_final["label"] = 1
    pos_final = pos_final.rename(columns={"gene": "gene_id"})

    neg_final = bg_df.loc[matched_neg_indices, cols_out].copy()
    neg_final["label"] = 0
    neg_final["pip"] = np.nan
    neg_final = neg_final.rename(columns={"gene": "gene_id"})

    train = pd.concat([pos_final, neg_final], ignore_index=True)

    # Final column order
    train = train[["variant_id", "chr", "pos", "ref", "alt", "maf",
                    "tss_distance", "gene_id", "label", "pip"]]

    # ------------------------------------------------------------------
    # 9. Hard assertion: no PD chromosomes
    # ------------------------------------------------------------------
    parsed_final = parse_variant_ids(train["variant_id"])
    violating = parsed_final["chr_num"].isin(PD_HOLDOUT_CHROMS)
    assert not violating.any(), (
        f"FATAL: {violating.sum()} variants on PD held-out chromosomes in final data!"
    )
    log.info("  ASSERTION PASSED: zero PD held-out chromosome variants")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{tissue}_train.parquet"
    train.to_parquet(parquet_path, index=False)
    log.info("  Saved %s (%d rows)", parquet_path, len(train))

    meta = {
        "tissue": tissue,
        "construction_date": datetime.now(timezone.utc).isoformat(),
        "n_positives": int(pos_final["label"].sum()),
        "n_negatives": int((neg_final["label"] == 0).sum()),
        "n_dropped_no_match": n_dropped,
        "n_dropped_no_maf_pos": int(pos_no_maf),
        "n_dropped_no_tss_pos": int(pos_no_tss),
        "n_dropped_no_maf_bg": int(bg_no_maf),
        "label_balance": f"{len(pos_final)}/{len(neg_final)}",
        "maf_source": "GTEx v8 signif_variant_gene_pairs",
        "maf_source_url": "https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL.tar",
        "gencode_version": "v26",
        "gencode_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.annotation.gtf.gz",
        "maf_bin_boundaries": MAF_BIN_EDGES,
        "tss_bin_boundaries_bp": TSS_BIN_EDGES,
        "held_out_chromosomes": sorted(PD_HOLDOUT_CHROMS),
        "random_seed": seed,
        "source_dapg_file": str(dapg_path),
    }

    meta_path = output_dir / f"{tissue}_train_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("  Saved %s", meta_path)

    return meta
