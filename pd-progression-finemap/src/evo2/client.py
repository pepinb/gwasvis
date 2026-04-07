"""Thin client for the Evo2 Modal service."""

from __future__ import annotations

from pathlib import Path

import modal
import pandas as pd

from evo2.reference import ReferenceProvider
from evo2.variants import build_variant_windows


class Evo2Client:
    def __init__(self):
        cls = modal.Cls.from_name("evo2-modal", "Evo2Model")
        self._model = cls()

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
