"""Thin client for the Evo2 Modal service.

Two scoring paths coexist here:

* ``score_variant`` / ``score_variants_batch`` — the original Phase 2.5-2.8
  scalar scoring API used by the fine-mapping pipeline. Caches scalar
  ``delta_log_likelihood`` values keyed by ``chrom:pos:ref:alt`` strings.
* ``score_variant_with_rc`` — added for the eQTL embeddings phase. Calls the
  newer ``Evo2Model.score_variant_with_rc`` Modal method to fetch multilayer
  embeddings with RC augmentation. Has its own parquet-backed cache keyed by
  ``(chrom, position, ref, alt, assembly, layer_name, pool_window)`` with one
  row per (variant, layer), which is schema-incompatible with the scalar
  cache — hence a separate file path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import modal
import numpy as np
import pandas as pd

from evo2.reference import ReferenceProvider
from evo2.variants import build_variant_windows


# Defaults for the eQTL embeddings phase; exported for scripts that import
# them directly (e.g., src/evo2/embeddings.py).
DEFAULT_APP_NAME = "evo2-modal"
DEFAULT_CLASS_NAME = "Evo2Model"
DEFAULT_LAYER = "blocks.26"
DEFAULT_POOL_WINDOW = 128
DEFAULT_MODEL = "evo2_7b"


@dataclass(frozen=True)
class VariantKey:
    """Cache key for a single (variant, layer) embedding row."""

    chrom: str
    position: int
    ref: str
    alt: str
    assembly: str
    layer_name: str
    pool_window: int

    def as_tuple(self) -> tuple:
        return (
            str(self.chrom),
            int(self.position),
            self.ref.upper(),
            self.alt.upper(),
            self.assembly,
            self.layer_name,
            int(self.pool_window),
        )


class Evo2Client:
    # Columns stored in the RC parquet cache (one row per variant × layer).
    _RC_KEY_COLS = (
        "chrom", "position", "ref", "alt",
        "assembly", "layer_name", "pool_window",
    )
    _RC_VALUE_COLS = (
        "delta_log_likelihood",
        "delta_log_likelihood_fwd",
        "delta_log_likelihood_rc",
        "log_likelihood_ref_fwd",
        "log_likelihood_ref_rc",
        "log_likelihood_alt_fwd",
        "log_likelihood_alt_rc",
        "ref_fwd_embedding",   # np.ndarray, float32
        "ref_rc_embedding",    # np.ndarray, float32
        "alt_fwd_embedding",   # np.ndarray, float32
        "alt_rc_embedding",    # np.ndarray, float32
        "model",
    )

    def __init__(
        self,
        cache_path: Path | None = None,
        app_name: str = DEFAULT_APP_NAME,
        class_name: str = DEFAULT_CLASS_NAME,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        cls = modal.Cls.from_name(app_name, class_name)
        self._model = cls()
        self.app_name = app_name
        self.class_name = class_name
        self.model_name = model_name
        # RC cache (score_variant_with_rc). Optional — only required when the
        # caller actually invokes score_variant_with_rc.
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._rc_cache_df: pd.DataFrame | None = None

    def score_variant(
        self, ref_seq: str, alt_seq: str, return_embedding: bool = False
    ) -> dict:
        return self._model.score_ref_alt.remote(
            ref_sequence=ref_seq,
            alt_sequence=alt_seq,
            return_embedding=return_embedding,
        )

    def score_variants_batch(
        self,
        variants: list[dict],
        reference: ReferenceProvider,
        cache_path: Path,
        window_size: int = 8192,
    ) -> pd.DataFrame:
        """Score a batch of SNVs via Modal parallel map, with parquet caching."""
        import math

        # Load cache if it exists
        cached_keys: set[str] = set()
        cached_df = pd.DataFrame()
        if cache_path.exists():
            cached_df = pd.read_parquet(cache_path)
            cached_keys = set(cached_df["key"].tolist())
            print(f"  Cache: {len(cached_keys)} variants already scored")

        # Identify new variants
        def make_key(v: dict) -> str:
            return f"{v['chrom']}:{v['pos']}:{v['ref']}:{v['alt']}"

        new_variants = [v for v in variants if make_key(v) not in cached_keys]
        print(f"  New variants to score: {len(new_variants)}")

        if not new_variants:
            print("  Nothing new to score.")
            return cached_df

        # Build windows, collecting failures
        to_score: list[dict] = []
        failed_rows: list[dict] = []
        for v in new_variants:
            key = make_key(v)
            try:
                ref_window, alt_window = build_variant_windows(
                    chrom=v["chrom"],
                    pos=v["pos"],
                    ref_allele=v["ref"],
                    alt_allele=v["alt"],
                    reference=reference,
                    window_size=window_size,
                )
                to_score.append({**v, "key": key, "ref_window": ref_window, "alt_window": alt_window})
            except (ValueError, NotImplementedError, KeyError) as e:
                failed_rows.append({
                    "key": key, "chrom": v["chrom"], "pos": v["pos"],
                    "ref": v["ref"], "alt": v["alt"], "rsid": v.get("rsid", ""),
                    "delta_log_likelihood": float("nan"),
                    "log_likelihood_ref": float("nan"),
                    "log_likelihood_alt": float("nan"),
                    "error": str(e),
                })

        if failed_rows:
            print(f"  Windowing failures: {len(failed_rows)}")

        # Score via Modal .map()
        scored_rows: list[dict] = []
        if to_score:
            ref_seqs = [v["ref_window"] for v in to_score]
            alt_seqs = [v["alt_window"] for v in to_score]

            print(f"  Scoring {len(to_score)} variants via Modal .map()...")
            results = list(self._model.score_ref_alt.map(
                ref_seqs, alt_seqs,
                kwargs={"return_embedding": False},
                order_outputs=True,
                return_exceptions=True,
            ))

            for v, result in zip(to_score, results):
                if isinstance(result, Exception):
                    scored_rows.append({
                        "key": v["key"], "chrom": v["chrom"], "pos": v["pos"],
                        "ref": v["ref"], "alt": v["alt"], "rsid": v.get("rsid", ""),
                        "delta_log_likelihood": float("nan"),
                        "log_likelihood_ref": float("nan"),
                        "log_likelihood_alt": float("nan"),
                        "error": str(result),
                    })
                else:
                    scored_rows.append({
                        "key": v["key"], "chrom": v["chrom"], "pos": v["pos"],
                        "ref": v["ref"], "alt": v["alt"], "rsid": v.get("rsid", ""),
                        "delta_log_likelihood": result["delta_log_likelihood"],
                        "log_likelihood_ref": result["log_likelihood_ref"],
                        "log_likelihood_alt": result["log_likelihood_alt"],
                        "error": "",
                    })

        # Combine new + failed + cached
        new_df = pd.DataFrame(scored_rows + failed_rows)
        if not cached_df.empty:
            full_df = pd.concat([cached_df, new_df], ignore_index=True)
        else:
            full_df = new_df

        full_df = full_df.drop_duplicates(subset=["key"], keep="last")
        full_df.to_parquet(cache_path, index=False)
        print(f"  Saved {len(full_df)} rows to {cache_path}")

        n_ok = full_df["error"].eq("").sum() if "error" in full_df.columns else len(full_df)
        n_err = len(full_df) - n_ok
        print(f"  Scored: {n_ok}, Failed: {n_err}")

        return full_df

    # ------------------------------------------------------------------ rc cache

    def _load_rc_cache(self) -> pd.DataFrame:
        """Load the RC parquet cache from disk, or return an empty frame."""
        if self._rc_cache_df is not None:
            return self._rc_cache_df
        if self.cache_path is None:
            raise RuntimeError(
                "Evo2Client was constructed without cache_path; "
                "score_variant_with_rc requires an RC cache file."
            )
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
        else:
            df = pd.DataFrame(
                columns=list(self._RC_KEY_COLS) + list(self._RC_VALUE_COLS)
            )
            df = df.astype({
                "chrom": "object",
                "position": "int64",
                "ref": "object",
                "alt": "object",
                "assembly": "object",
                "layer_name": "object",
                "pool_window": "int64",
            }, errors="ignore")
        self._rc_cache_df = df
        return df

    def _save_rc_cache(self, df: pd.DataFrame) -> None:
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        self._rc_cache_df = df

    def _lookup_rc(self, key: VariantKey) -> pd.Series | None:
        df = self._load_rc_cache()
        if df.empty:
            return None
        mask = (
            (df["chrom"].astype(str) == str(key.chrom))
            & (df["position"] == int(key.position))
            & (df["ref"] == key.ref.upper())
            & (df["alt"] == key.alt.upper())
            & (df["assembly"] == key.assembly)
            & (df["layer_name"] == key.layer_name)
            & (df["pool_window"] == int(key.pool_window))
        )
        hits = df[mask]
        if hits.empty:
            return None
        return hits.iloc[0]

    def _append_rc_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        df = self._load_rc_cache()
        new_df = pd.DataFrame(rows)
        combined = pd.concat([df, new_df], ignore_index=True)
        self._save_rc_cache(combined)

    # ------------------------------------------------------------------ rc scoring

    def score_variant_with_rc(
        self,
        chrom: str | int,
        position: int,
        ref: str,
        alt: str,
        ref_sequence: str,
        alt_sequence: str,
        assembly: str,
        layer_names: Iterable[str] = (DEFAULT_LAYER,),
        pool_window: int = DEFAULT_POOL_WINDOW,
        force: bool = False,
    ) -> dict[str, dict]:
        """Score a variant via evo2-modal with RC augmentation, caching per layer.

        Returns a dict keyed by layer name. Each value is a dict with:
            delta_log_likelihood (mean of fwd+rc)
            delta_log_likelihood_fwd, delta_log_likelihood_rc
            log_likelihood_{ref,alt}_{fwd,rc}
            ref_fwd_embedding, ref_rc_embedding, alt_fwd_embedding, alt_rc_embedding
                — each a float32 np.ndarray of length hidden_dim
            model (str)

        ``ref_sequence`` and ``alt_sequence`` are the 8192 bp windows built by
        ``build_variant_window``; the caller is responsible for centering the
        variant at index 4096.

        If ``force=False`` (default) any layer already present in the cache is
        served from disk without a remote call. New layers are fetched and
        persisted. ``force=True`` recomputes everything and overwrites rows.
        """
        layer_names = list(layer_names)
        if not layer_names:
            raise ValueError("layer_names must be non-empty")

        # --- 1. cache lookup ---
        hits: dict[str, pd.Series] = {}
        missing: list[str] = []
        for layer in layer_names:
            key = VariantKey(
                chrom=str(chrom), position=int(position),
                ref=ref.upper(), alt=alt.upper(),
                assembly=assembly, layer_name=layer,
                pool_window=pool_window,
            )
            if not force:
                hit = self._lookup_rc(key)
                if hit is not None:
                    hits[layer] = hit
                    continue
            missing.append(layer)

        # --- 2. remote fetch for missing layers ---
        if missing:
            result = self._model.score_variant_with_rc.remote(
                ref_sequence,
                alt_sequence,
                layer_names=missing,
                pool_window=pool_window,
            )

            remote_model = result.get("model", self.model_name)
            embeddings = result["embeddings"]

            new_rows = []
            for layer in missing:
                layer_emb = embeddings[layer]
                new_rows.append({
                    "chrom": str(chrom),
                    "position": int(position),
                    "ref": ref.upper(),
                    "alt": alt.upper(),
                    "assembly": assembly,
                    "layer_name": layer,
                    "pool_window": int(pool_window),
                    "delta_log_likelihood": float(result["delta_log_likelihood"]),
                    "delta_log_likelihood_fwd": float(result["delta_log_likelihood_fwd"]),
                    "delta_log_likelihood_rc": float(result["delta_log_likelihood_rc"]),
                    "log_likelihood_ref_fwd": float(result["log_likelihood_ref_fwd"]),
                    "log_likelihood_ref_rc": float(result["log_likelihood_ref_rc"]),
                    "log_likelihood_alt_fwd": float(result["log_likelihood_alt_fwd"]),
                    "log_likelihood_alt_rc": float(result["log_likelihood_alt_rc"]),
                    "ref_fwd_embedding": np.asarray(layer_emb["ref_fwd"], dtype=np.float32),
                    "ref_rc_embedding":  np.asarray(layer_emb["ref_rc"],  dtype=np.float32),
                    "alt_fwd_embedding": np.asarray(layer_emb["alt_fwd"], dtype=np.float32),
                    "alt_rc_embedding":  np.asarray(layer_emb["alt_rc"],  dtype=np.float32),
                    "model": remote_model,
                })

            if force:
                # Drop any prior rows for (variant, layer) that we are overwriting.
                df = self._load_rc_cache()
                if not df.empty:
                    key_match = (
                        (df["chrom"].astype(str) == str(chrom))
                        & (df["position"] == int(position))
                        & (df["ref"] == ref.upper())
                        & (df["alt"] == alt.upper())
                        & (df["assembly"] == assembly)
                        & (df["layer_name"].isin(missing))
                        & (df["pool_window"] == int(pool_window))
                    )
                    df = df[~key_match].reset_index(drop=True)
                    self._save_rc_cache(df)
            self._append_rc_rows(new_rows)

            # Insert the freshly fetched rows into `hits`.
            for layer in missing:
                key = VariantKey(
                    chrom=str(chrom), position=int(position),
                    ref=ref.upper(), alt=alt.upper(),
                    assembly=assembly, layer_name=layer,
                    pool_window=pool_window,
                )
                hit = self._lookup_rc(key)
                assert hit is not None, f"Just-written row not found for {key}"
                hits[layer] = hit

        # --- 3. pack return value ---
        out: dict[str, dict] = {}
        for layer, row in hits.items():
            out[layer] = {
                "delta_log_likelihood": float(row["delta_log_likelihood"]),
                "delta_log_likelihood_fwd": float(row["delta_log_likelihood_fwd"]),
                "delta_log_likelihood_rc": float(row["delta_log_likelihood_rc"]),
                "log_likelihood_ref_fwd": float(row["log_likelihood_ref_fwd"]),
                "log_likelihood_ref_rc": float(row["log_likelihood_ref_rc"]),
                "log_likelihood_alt_fwd": float(row["log_likelihood_alt_fwd"]),
                "log_likelihood_alt_rc": float(row["log_likelihood_alt_rc"]),
                "ref_fwd_embedding": np.asarray(row["ref_fwd_embedding"], dtype=np.float32),
                "ref_rc_embedding":  np.asarray(row["ref_rc_embedding"],  dtype=np.float32),
                "alt_fwd_embedding": np.asarray(row["alt_fwd_embedding"], dtype=np.float32),
                "alt_rc_embedding":  np.asarray(row["alt_rc_embedding"],  dtype=np.float32),
                "model": str(row["model"]),
            }
        return out
