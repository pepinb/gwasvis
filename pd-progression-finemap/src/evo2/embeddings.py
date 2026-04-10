"""Build concatenated Evo 2 embedding features for eQTL probe training.

High-level flow:

    variants (DataFrame) → build_variant_window → Evo2Client.score_variant_with_rc
        → concat [ref_fwd, ref_rc, alt_fwd, alt_rc] per layer
        → feature matrix (n, 4 * hidden_dim)

This module intentionally stays thin: all the caching and remote-call logic
lives in Evo2Client. `load_or_extract` is the convenience entrypoint for
notebook / pipeline callers that want "give me embeddings for this variant
table, reuse what's cached, only call Modal for the rest".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evo2.client import DEFAULT_LAYER, DEFAULT_POOL_WINDOW, Evo2Client
from evo2.reference import FastaProvider
from evo2.variants import build_variant_window

# Concatenation order for the 4*hidden_dim feature vector. The caller can
# re-derive this via `concat_feature_order()` for downstream analysis.
CONCAT_ORDER = ("ref_fwd", "ref_rc", "alt_fwd", "alt_rc")


def concat_feature_order() -> tuple[str, ...]:
    """Return the stable concat order: (ref_fwd, ref_rc, alt_fwd, alt_rc)."""
    return CONCAT_ORDER


def _concat_one_layer(layer_result: dict) -> np.ndarray:
    """Flatten a per-layer result dict into a 1D float32 vector."""
    parts = [
        np.asarray(layer_result[f"{key}_embedding"], dtype=np.float32)
        for key in CONCAT_ORDER
    ]
    return np.concatenate(parts, axis=0)


def fetch_variant_embeddings_concat(
    variants: pd.DataFrame,
    *,
    client: Evo2Client,
    fasta_provider: FastaProvider,
    layer_name: str = DEFAULT_LAYER,
    pool_window: int = DEFAULT_POOL_WINDOW,
    label_col: str | None = "label",
    chrom_col: str = "chrom",
    position_col: str = "position",
    ref_col: str = "ref",
    alt_col: str = "alt",
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fetch concatenated Evo 2 embeddings for every row in `variants`.

    Returns
    -------
    features : np.ndarray, shape (n_success, 4 * hidden_dim), float32
        Concatenated [ref_fwd, ref_rc, alt_fwd, alt_rc] vectors for every
        variant that built a valid window AND was scored successfully.
    labels : np.ndarray, shape (n_success,)
        Pulled from `variants[label_col]` in the same order as `features`.
        Zeros if `label_col` is None or missing.
    failures : list[dict]
        One entry per variant that could not be processed, with fields
        {chrom, position, ref, alt, reason}. The caller can log / drop.

    Notes
    -----
    This function calls the client once per variant. If a variant is already
    fully cached for the requested (layer, pool_window), no remote call is
    made. Set `force=True` to recompute everything.
    """
    features: list[np.ndarray] = []
    labels: list = []
    failures: list[dict] = []

    assembly = fasta_provider.assembly

    for _, row in variants.iterrows():
        chrom = row[chrom_col]
        position = int(row[position_col])
        ref = str(row[ref_col]).upper()
        alt = str(row[alt_col]).upper()

        try:
            window = build_variant_window(
                chrom=chrom,
                position=position,
                ref=ref,
                alt=alt,
                provider=fasta_provider,
            )
        except (AssertionError, FileNotFoundError, ValueError) as exc:
            failures.append({
                "chrom": chrom, "position": position,
                "ref": ref, "alt": alt,
                "reason": f"window build failed: {exc}",
            })
            continue

        try:
            result = client.score_variant_with_rc(
                chrom=chrom,
                position=position,
                ref=ref,
                alt=alt,
                ref_sequence=window.ref_sequence,
                alt_sequence=window.alt_sequence,
                assembly=assembly,
                layer_names=[layer_name],
                pool_window=pool_window,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001 — surface any remote failure
            failures.append({
                "chrom": chrom, "position": position,
                "ref": ref, "alt": alt,
                "reason": f"score_variant_with_rc failed: {exc}",
            })
            continue

        if layer_name not in result:
            failures.append({
                "chrom": chrom, "position": position,
                "ref": ref, "alt": alt,
                "reason": f"layer {layer_name!r} missing from remote result",
            })
            continue

        features.append(_concat_one_layer(result[layer_name]))
        if label_col is not None and label_col in variants.columns:
            labels.append(row[label_col])
        else:
            labels.append(0)

    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,)),
            failures,
        )

    feature_matrix = np.stack(features, axis=0).astype(np.float32, copy=False)
    label_array = np.asarray(labels)
    return feature_matrix, label_array, failures


def load_or_extract(
    variants: pd.DataFrame,
    *,
    cache_path: Path,
    fasta_provider: FastaProvider,
    layer_name: str = DEFAULT_LAYER,
    pool_window: int = DEFAULT_POOL_WINDOW,
    label_col: str | None = "label",
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Convenience wrapper: construct an Evo2Client, run fetch, return matrices.

    This is the entrypoint you want from notebooks / scripts. It will:
      1. Build an Evo2Client pointed at `cache_path`.
      2. For each variant, build the 8192 bp window via `fasta_provider`.
      3. Look up the cached embedding; fall back to a Modal call for misses.
      4. Concatenate [ref_fwd, ref_rc, alt_fwd, alt_rc] per variant.
      5. Return (features, labels, failures).

    The cache is incremental: stopping and resuming the call only recomputes
    the tail that hasn't been persisted yet.
    """
    client = Evo2Client(cache_path=cache_path)
    return fetch_variant_embeddings_concat(
        variants,
        client=client,
        fasta_provider=fasta_provider,
        layer_name=layer_name,
        pool_window=pool_window,
        label_col=label_col,
        force=force,
    )
