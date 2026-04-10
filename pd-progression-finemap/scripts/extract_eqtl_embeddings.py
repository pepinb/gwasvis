"""Full-scale Evo 2 7B embedding extraction for both eQTL training datasets.

Scales the pilot (100 variants) up to the full training pool::

    Whole_Blood_train.parquet   4,854 variants
    Brain_Cortex_train.parquet  2,214 variants
    ------------------------------------------
    total                       7,068 variants

After prompt 5 roughly 100 of the Whole_Blood variants are already in the
shared embedding cache; this script only calls Modal for the remaining
misses. Each miss triggers four H100 forward passes (ref_fwd, ref_rc,
alt_fwd, alt_rc) via ``Evo2Client.score_variant_with_rc``; results are
appended to the shared parquet cache so every call is a one-time cost.

For each tissue we:

1. Print a pre-flight budget plan (variants, cache hits, est runtime, est
   cost). Abort if the estimate exceeds the hard budget (240 min / $25).
2. Ensure the hg38 FASTAs for every touched chromosome are on disk
   (downloads anything missing from Ensembl release 110).
3. Extract embeddings in batches, streaming progress every batch or
   every ~60 s, whichever comes first. Per-variant failures are logged
   to ``data/eqtl/embeddings/{tissue}_extraction_failures.txt`` and do
   not abort the run (unless the cumulative failure rate exceeds 2 %).
4. Build the per-tissue features parquet::

       data/eqtl/embeddings/{tissue}_features.parquet

   Schema (one row per training variant that extracted successfully)::

       variant_id    str
       chr           str   (e.g. "chr17")
       pos           int64
       ref           str
       alt           str
       label         int64
       gene_id       str
       maf           float64
       tss_distance  int64
       embedding     bytes (16384 float32 values, order:
                            ref_fwd, ref_rc, alt_fwd, alt_rc)

5. Sanity-check the saved parquet by drawing 100 random rows, rebuilding
   the feature matrix, and running the same LogisticRegressionCV setup
   as the pilot. Compares to the pilot AUROC for Whole_Blood; emits a
   loud warning on >0.10 deviation (would signal schema corruption).

Usage::

    python3 scripts/extract_eqtl_embeddings.py

Requires:
    - MODAL_TOKEN_ID in the environment (or ~/.modal.toml)
    - Network to Ensembl FTP on first run for any missing chromosome
      FASTAs
    - evo2-modal deployed with ``score_variant_with_rc``
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
import ssl
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evo2.client import (  # noqa: E402
    DEFAULT_LAYER,
    DEFAULT_MODEL,
    DEFAULT_POOL_WINDOW,
    Evo2Client,
)
from evo2.embeddings import (  # noqa: E402
    CONCAT_ORDER,
    fetch_variant_embeddings_concat,
)
from evo2.reference import FastaProvider  # noqa: E402

# --------------------------------------------------------------------------- config

TISSUES = ["Whole_Blood", "Brain_Cortex"]

TRAIN_DIR = ROOT / "data" / "eqtl"
CACHE_PATH = ROOT / "data" / "evo2" / "cache" / "embeddings_7b_blocks26_w128.parquet"
FEATURES_DIR = ROOT / "data" / "eqtl" / "embeddings"
REFERENCE_DIR = ROOT / "data" / "reference_hg38"
PILOT_RESULT_PATH = ROOT / "data" / "eqtl" / "pilot_whole_blood.json"

ASSEMBLY = "hg38"
LAYER = DEFAULT_LAYER
POOL_WINDOW = DEFAULT_POOL_WINDOW
MODEL_NAME = DEFAULT_MODEL
HIDDEN_DIM = 4096
EXPECTED_CONCAT_DIM = HIDDEN_DIM * 4  # 16384
BYTES_PER_EMBEDDING = EXPECTED_CONCAT_DIM * 4  # float32 -> 65536

BATCH_SIZE = 100
PROGRESS_EVERY_VARIANTS = 500
PROGRESS_EVERY_SECONDS = 60.0

# Cost / runtime model (reported before extraction; actuals are tracked live)
SECONDS_PER_VARIANT = 0.9          # amortised H100 time per variant (4 passes)
H100_USD_PER_HOUR = 5.0            # rough Modal H100 price
COST_PER_VARIANT = SECONDS_PER_VARIANT * H100_USD_PER_HOUR / 3600.0
MAX_COST_USD = 25.0
MAX_RUNTIME_MINUTES = 240.0

FAILURE_RATE_LIMIT = 0.02          # abort if cumulative failure rate > 2%
FAILURE_CHECK_MIN = 500            # but only after processing at least this many

# Sanity-check settings (applied to the saved features parquet)
SANITY_SAMPLE = 100
SANITY_SEED = 0                    # distinct from the pilot's seed (42)
PROBE_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
PROBE_CV = 5
WB_PILOT_TOLERANCE = 0.10          # loud warning if sanity deviates > this

# Ensembl GRCh38 primary assembly release 110
ENSEMBL_FASTA_URL = (
    "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
    "Homo_sapiens.GRCh38.dna.chromosome.{chrom}.fa.gz"
)

PD_HOLDOUT = {1, 2, 4, 7, 12, 19}


# --------------------------------------------------------------------------- helpers


def _bare_chrom(chrom: str) -> str:
    """chr17 -> 17."""
    return str(chrom).replace("chr", "")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _download_fasta(chrom_bare: str) -> None:
    """Fetch + decompress one hg38 chromosome FASTA from Ensembl FTP."""
    target = REFERENCE_DIR / f"Homo_sapiens.GRCh38.dna.chromosome.{chrom_bare}.fa"
    if target.exists():
        return
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = target.with_suffix(target.suffix + ".gz")
    url = ENSEMBL_FASTA_URL.format(chrom=chrom_bare)

    print(f"  downloading chr{chrom_bare} <- {url}", flush=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url, headers={"User-Agent": "gwasdemo-extract/1.0"}
    )
    with urllib.request.urlopen(req, timeout=600, context=ctx) as resp, \
            open(gz_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"  decompressing chr{chrom_bare}", flush=True)
    with gzip.open(gz_path, "rb") as gz, open(target, "wb") as fa:
        shutil.copyfileobj(gz, fa)
    gz_path.unlink(missing_ok=True)


def _ensure_fastas(chroms_bare: list[str]) -> None:
    """Download any missing hg38 FASTAs."""
    missing = [
        c for c in chroms_bare
        if not (REFERENCE_DIR / f"Homo_sapiens.GRCh38.dna.chromosome.{c}.fa").exists()
    ]
    if not missing:
        print("All required hg38 FASTAs already present.")
        return
    print(f"Downloading {len(missing)} missing hg38 FASTA(s): {missing}")
    for c in sorted(missing, key=int):
        _download_fasta(c)


def _cached_variant_keys(
    cache_df: pd.DataFrame,
) -> set[tuple[str, int, str, str]]:
    """Extract the (chrom, pos, ref, alt) keys already in the cache."""
    if cache_df.empty:
        return set()
    keyed = cache_df[
        (cache_df["assembly"] == ASSEMBLY)
        & (cache_df["layer_name"] == LAYER)
        & (cache_df["pool_window"] == POOL_WINDOW)
    ]
    return {
        (str(r["chrom"]), int(r["position"]),
         str(r["ref"]).upper(), str(r["alt"]).upper())
        for _, r in keyed.iterrows()
    }


def _variant_key(row: pd.Series) -> tuple[str, int, str, str]:
    return (
        _bare_chrom(row["chr"]),
        int(row["pos"]),
        str(row["ref"]).upper(),
        str(row["alt"]).upper(),
    )


def _prepare_variants_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Rename / normalise columns to what fetch_variant_embeddings_concat wants."""
    v = train_df.rename(columns={"chr": "chrom", "pos": "position"}).copy()
    v["chrom"] = v["chrom"].map(_bare_chrom)
    v["ref"] = v["ref"].str.upper()
    v["alt"] = v["alt"].str.upper()
    for c in PD_HOLDOUT:
        assert not (v["chrom"] == str(c)).any(), (
            f"PD hold-out chromosome chr{c} leaked into training data"
        )
    return v[["chrom", "position", "ref", "alt", "label"]]


def _batches(df: pd.DataFrame, size: int):
    for start in range(0, len(df), size):
        yield df.iloc[start:start + size]


def _format_minutes(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60.0)
    return f"{int(m):d}m{int(s):02d}s"


# --------------------------------------------------------------------------- extraction


def extract_tissue(
    tissue: str,
    client: Evo2Client,
    provider: FastaProvider,
) -> dict:
    """Run the full extraction pipeline for one tissue, return a summary dict."""
    print("\n" + "#" * 70)
    print(f"# {tissue}")
    print("#" * 70)

    train_path = TRAIN_DIR / f"{tissue}_train.parquet"
    train_df = pd.read_parquet(train_path)
    n_total = len(train_df)
    print(f"Loaded {train_path.name}: {n_total} rows")

    # Pre-flight plan
    cache_df = client._load_cache()  # noqa: SLF001 — public enough for our use
    cached_keys = _cached_variant_keys(cache_df)
    train_keys = {_variant_key(r) for _, r in train_df.iterrows()}
    already_cached = len(train_keys & cached_keys)
    n_new = n_total - already_cached
    est_seconds = n_new * SECONDS_PER_VARIANT
    est_cost = n_new * COST_PER_VARIANT

    print(f"\n=== {tissue} extraction plan ===")
    print(f"Total variants:       {n_total}")
    print(f"Already cached:       {already_cached}")
    print(f"New to extract:       {n_new}")
    print(f"Est. runtime:         {est_seconds / 60:.1f} min  "
          f"(@ {SECONDS_PER_VARIANT} sec/variant on H100)")
    print(f"Est. Modal cost:      ${est_cost:.2f}   "
          f"(@ ~${H100_USD_PER_HOUR:.0f}/hr H100)")

    if est_cost > MAX_COST_USD or est_seconds / 60 > MAX_RUNTIME_MINUTES:
        raise RuntimeError(
            f"Aborting: estimated cost ${est_cost:.2f} or runtime "
            f"{est_seconds/60:.1f} min exceeds the hard budget "
            f"(${MAX_COST_USD} / {MAX_RUNTIME_MINUTES:.0f} min)."
        )

    # Make sure every touched chromosome has a local FASTA.
    needed_chroms = sorted({_bare_chrom(c) for c in train_df["chr"].unique()},
                           key=int)
    _ensure_fastas(needed_chroms)

    variants_df = _prepare_variants_df(train_df)

    # Failure log
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    failure_path = FEATURES_DIR / f"{tissue}_extraction_failures.txt"
    if failure_path.exists():
        failure_path.unlink()

    tissue_t0 = time.time()
    total_done = 0
    total_failures = 0
    all_failures: list[dict] = []
    new_calls = 0                    # how many variants the cache missed in this run
    cache_hits_before = already_cached
    last_progress_t = tissue_t0
    last_progress_done = 0

    for batch_idx, batch_df in enumerate(_batches(variants_df, BATCH_SIZE)):
        batch_size = len(batch_df)
        cache_df_before = client._load_cache()  # noqa: SLF001
        cache_rows_before = len(cache_df_before)

        _, _, failures = fetch_variant_embeddings_concat(
            batch_df,
            client=client,
            fasta_provider=provider,
            layer_name=LAYER,
            pool_window=POOL_WINDOW,
            label_col="label",
        )
        cache_df_after = client._load_cache()  # noqa: SLF001
        cache_rows_after = len(cache_df_after)
        new_calls += max(0, cache_rows_after - cache_rows_before)

        total_done += batch_size
        if failures:
            all_failures.extend(failures)
            total_failures += len(failures)
            with open(failure_path, "a") as fh:
                for f in failures:
                    fh.write(json.dumps(f) + "\n")

        # Progress reporting (every batch; keeps things simple)
        now = time.time()
        since_last = now - last_progress_t
        done_since_last = total_done - last_progress_done
        should_print = (
            (done_since_last >= PROGRESS_EVERY_VARIANTS)
            or (since_last >= PROGRESS_EVERY_SECONDS)
            or (total_done == n_total)
        )
        if should_print:
            elapsed = now - tissue_t0
            rate = total_done / elapsed if elapsed > 0 else 0.0
            remaining = n_total - total_done
            eta = remaining / rate if rate > 0 else 0.0
            running_cost = new_calls * COST_PER_VARIANT
            print(
                f"  [{total_done:>5}/{n_total}] "
                f"elapsed={_format_minutes(elapsed)}  "
                f"eta={_format_minutes(eta)}  "
                f"rate={rate:5.2f} var/s  "
                f"new_calls={new_calls}  "
                f"failures={total_failures}  "
                f"cost≈${running_cost:.2f}",
                flush=True,
            )
            last_progress_t = now
            last_progress_done = total_done

        # Abort on systemic failure: but only after we've seen enough samples
        # to distinguish noise from signal.
        if total_done >= FAILURE_CHECK_MIN:
            rate_fail = total_failures / total_done
            if rate_fail > FAILURE_RATE_LIMIT:
                raise RuntimeError(
                    f"{tissue}: failure rate {rate_fail:.2%} > "
                    f"{FAILURE_RATE_LIMIT:.0%} after {total_done} variants; "
                    f"stop and investigate. Log: {failure_path}"
                )

    extraction_seconds = time.time() - tissue_t0
    modal_cost = new_calls * COST_PER_VARIANT

    print(f"\n=== {tissue} extraction complete ===")
    print(f"Extraction time:     {extraction_seconds / 60:.1f} min "
          f"({_format_minutes(extraction_seconds)})")
    print(f"Modal cost:          ${modal_cost:.2f}   (estimated)")
    print(f"Failures:            {total_failures}")
    print(f"Cache hits (start):  {cache_hits_before}")
    print(f"New Modal calls:     {new_calls}")

    # ----------------------------- features parquet ----------------------------
    print(f"\nBuilding features parquet for {tissue}...")
    features_path = FEATURES_DIR / f"{tissue}_features.parquet"
    n_rows, file_size_mb = _build_features_parquet(
        tissue=tissue,
        train_df=train_df,
        client=client,
        output_path=features_path,
    )
    print(f"Final row count:     {n_rows}")
    print(f"Features file:       {features_path}")
    print(f"File size:           {file_size_mb:.1f} MB")

    return {
        "tissue": tissue,
        "n_total": n_total,
        "cache_hits_before": cache_hits_before,
        "new_modal_calls": new_calls,
        "failures": total_failures,
        "failures_path": str(failure_path) if total_failures else None,
        "extraction_seconds": extraction_seconds,
        "modal_cost_estimate": modal_cost,
        "features_path": str(features_path),
        "features_rows": n_rows,
        "features_size_mb": file_size_mb,
        "sampled_failures": all_failures[:10],
    }


def _build_features_parquet(
    tissue: str,
    train_df: pd.DataFrame,
    client: Evo2Client,
    output_path: Path,
) -> tuple[int, float]:
    """Join training metadata with cache embeddings, serialise to parquet."""
    cache_df = client._load_cache()  # noqa: SLF001
    keyed = cache_df[
        (cache_df["assembly"] == ASSEMBLY)
        & (cache_df["layer_name"] == LAYER)
        & (cache_df["pool_window"] == POOL_WINDOW)
    ].copy()
    if keyed.empty:
        raise RuntimeError(
            f"{tissue}: no cache rows match layer={LAYER} pool={POOL_WINDOW} "
            f"assembly={ASSEMBLY}"
        )
    keyed["_chrom_bare"] = keyed["chrom"].astype(str)
    keyed["_pos"] = keyed["position"].astype(int)
    keyed["_ref"] = keyed["ref"].astype(str).str.upper()
    keyed["_alt"] = keyed["alt"].astype(str).str.upper()
    keyed = keyed.drop_duplicates(
        subset=["_chrom_bare", "_pos", "_ref", "_alt"], keep="last"
    )

    train_norm = train_df.copy()
    train_norm["_chrom_bare"] = train_norm["chr"].map(_bare_chrom)
    train_norm["_pos"] = train_norm["pos"].astype(int)
    train_norm["_ref"] = train_norm["ref"].astype(str).str.upper()
    train_norm["_alt"] = train_norm["alt"].astype(str).str.upper()

    merged = train_norm.merge(
        keyed[[
            "_chrom_bare", "_pos", "_ref", "_alt",
            "ref_fwd_embedding", "ref_rc_embedding",
            "alt_fwd_embedding", "alt_rc_embedding",
        ]],
        on=["_chrom_bare", "_pos", "_ref", "_alt"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError(
            f"{tissue}: features merge produced 0 rows; cache keys may be wrong"
        )

    # Build the embedding bytes column in CONCAT_ORDER.
    order_cols = [f"{name}_embedding" for name in CONCAT_ORDER]
    embeddings_bytes: list[bytes] = []
    for _, r in merged.iterrows():
        parts = [
            np.asarray(r[col], dtype=np.float32) for col in order_cols
        ]
        concat = np.concatenate(parts, axis=0)
        if concat.shape[0] != EXPECTED_CONCAT_DIM:
            raise RuntimeError(
                f"{tissue}: unexpected embedding dim {concat.shape[0]} "
                f"for variant {r.get('variant_id')}"
            )
        embeddings_bytes.append(concat.astype(np.float32).tobytes())

    out_df = pd.DataFrame({
        "variant_id": merged["variant_id"].astype(str).values,
        "chr": merged["chr"].astype(str).values,
        "pos": merged["pos"].astype("int64").values,
        "ref": merged["_ref"].values,
        "alt": merged["_alt"].values,
        "label": merged["label"].astype("int64").values,
        "gene_id": merged["gene_id"].astype(str).values,
        "maf": merged["maf"].astype("float64").values,
        "tss_distance": merged["tss_distance"].astype("int64").values,
        "embedding": embeddings_bytes,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    return len(out_df), size_mb


# --------------------------------------------------------------------------- sanity


def _decode_embedding(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != EXPECTED_CONCAT_DIM:
        raise ValueError(
            f"embedding blob has {arr.shape[0]} floats, expected {EXPECTED_CONCAT_DIM}"
        )
    return arr


def _probe(features: np.ndarray, labels: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=PROBE_CV, shuffle=True, random_state=42)
    clf = LogisticRegressionCV(
        Cs=PROBE_CS,
        cv=skf,
        scoring="roc_auc",
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
        refit=True,
    )
    clf.fit(features, labels)
    scores_matrix = next(iter(clf.scores_.values()))
    best_C = float(clf.C_[0])
    best_col = int(np.argmin(np.abs(np.array(PROBE_CS) - best_C)))
    per_fold = [float(x) for x in scores_matrix[:, best_col]]
    return {
        "mean_cv_auroc": float(np.mean(per_fold)),
        "std_cv_auroc": float(np.std(per_fold)),
        "per_fold_auroc": per_fold,
        "best_C": best_C,
    }


def sanity_check(features_path: Path, tissue: str) -> dict:
    """Re-load the saved features parquet and train a probe on a 100-row sample."""
    print(f"\n=== {tissue} sanity check ===")
    df = pd.read_parquet(features_path)
    if len(df) < SANITY_SAMPLE:
        raise RuntimeError(
            f"{tissue}: only {len(df)} rows in features parquet; need "
            f"{SANITY_SAMPLE} for sanity check"
        )

    rng = np.random.default_rng(SANITY_SEED)
    pos_idx = rng.choice(
        df.index[df["label"] == 1].values,
        size=SANITY_SAMPLE // 2, replace=False,
    )
    neg_idx = rng.choice(
        df.index[df["label"] == 0].values,
        size=SANITY_SAMPLE // 2, replace=False,
    )
    sample = df.loc[np.concatenate([pos_idx, neg_idx])].reset_index(drop=True)

    X = np.stack(
        [_decode_embedding(b) for b in sample["embedding"].values], axis=0
    ).astype(np.float32)
    y = sample["label"].astype(int).values
    assert X.shape == (SANITY_SAMPLE, EXPECTED_CONCAT_DIM)

    # Split into the four sub-vectors — same concat order as the pilot.
    ref_fwd = X[:, 0 * HIDDEN_DIM:1 * HIDDEN_DIM]
    ref_rc = X[:, 1 * HIDDEN_DIM:2 * HIDDEN_DIM]
    alt_fwd = X[:, 2 * HIDDEN_DIM:3 * HIDDEN_DIM]
    alt_rc = X[:, 3 * HIDDEN_DIM:4 * HIDDEN_DIM]
    X_delta = np.concatenate(
        [alt_fwd - ref_fwd, alt_rc - ref_rc], axis=1
    ).astype(np.float32)

    concat = _probe(X, y)
    delta = _probe(X_delta, y)

    print(f"  Sanity sample: n={len(sample)}, label={np.bincount(y).tolist()}")
    print(f"  concat (16384): mean AUROC={concat['mean_cv_auroc']:.4f}  "
          f"std={concat['std_cv_auroc']:.4f}")
    print(f"  delta  (8192):  mean AUROC={delta['mean_cv_auroc']:.4f}  "
          f"std={delta['std_cv_auroc']:.4f}")

    return {
        "concat": concat,
        "delta": delta,
        "n_sampled": int(len(sample)),
    }


def compare_to_pilot(sanity: dict) -> None:
    """Loud warning if Whole_Blood sanity drifts too far from the pilot AUROC."""
    if not PILOT_RESULT_PATH.exists():
        print("  (no pilot result to compare against)")
        return
    pilot = json.loads(PILOT_RESULT_PATH.read_text())
    pilot_concat = pilot.get("concat", {}).get("mean_cv_auroc")
    pilot_delta = pilot.get("delta", {}).get("mean_cv_auroc")
    san_concat = sanity["concat"]["mean_cv_auroc"]
    san_delta = sanity["delta"]["mean_cv_auroc"]

    warnings = []
    if pilot_concat is not None:
        diff = abs(san_concat - pilot_concat)
        print(f"  concat: sanity {san_concat:.4f} vs pilot "
              f"{pilot_concat:.4f} (|Δ|={diff:.4f})")
        if diff > WB_PILOT_TOLERANCE:
            warnings.append(
                f"concat sanity deviates {diff:.4f} > {WB_PILOT_TOLERANCE}"
            )
    if pilot_delta is not None:
        diff = abs(san_delta - pilot_delta)
        print(f"  delta:  sanity {san_delta:.4f} vs pilot "
              f"{pilot_delta:.4f} (|Δ|={diff:.4f})")
        if diff > WB_PILOT_TOLERANCE:
            warnings.append(
                f"delta sanity deviates {diff:.4f} > {WB_PILOT_TOLERANCE}"
            )

    if warnings:
        print("\n" + "!" * 70)
        print("!!! LOUD WARNING — Whole_Blood sanity check mismatch")
        for w in warnings:
            print(f"!!!   {w}")
        print("!!! Likely cause: embedding/label misalignment, wrong byte order,")
        print("!!! or concat sub-vector swap in _build_features_parquet.")
        print("!" * 70)


# --------------------------------------------------------------------------- main


def main() -> int:
    print("=== Full eQTL embedding extraction: Whole_Blood + Brain_Cortex ===")
    print(f"Model: {MODEL_NAME}, layer={LAYER}, pool_window={POOL_WINDOW}, "
          f"assembly={ASSEMBLY}")
    print(f"Cache: {CACHE_PATH}")
    print(f"Features output dir: {FEATURES_DIR}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    provider = FastaProvider(assembly=ASSEMBLY)
    client = Evo2Client(cache_path=CACHE_PATH)

    overall_t0 = time.time()
    per_tissue: dict[str, dict] = {}
    sanity_results: dict[str, dict] = {}

    for tissue in TISSUES:
        summary = extract_tissue(tissue, client=client, provider=provider)
        per_tissue[tissue] = summary
        sanity = sanity_check(Path(summary["features_path"]), tissue)
        sanity_results[tissue] = sanity
        if tissue == "Whole_Blood":
            compare_to_pilot(sanity)
        else:
            sb_delta = sanity["delta"]["mean_cv_auroc"]
            sb_concat = sanity["concat"]["mean_cv_auroc"]
            if sb_delta <= 0.55 and sb_concat <= 0.55:
                print("\n" + "!" * 70)
                print("!!! LOUD WARNING — Brain_Cortex sanity both constructions")
                print(f"!!!   concat={sb_concat:.4f}, delta={sb_delta:.4f}")
                print("!!! Expected at least one > 0.55. Investigate.")
                print("!" * 70)

    overall_runtime = time.time() - overall_t0

    total_variants = sum(s["n_total"] for s in per_tissue.values())
    total_new = sum(s["new_modal_calls"] for s in per_tissue.values())
    total_cost = sum(s["modal_cost_estimate"] for s in per_tissue.values())
    total_failures = sum(s["failures"] for s in per_tissue.values())

    cache_size_mb = (
        CACHE_PATH.stat().st_size / (1024 * 1024) if CACHE_PATH.exists() else 0.0
    )

    print("\n" + "=" * 70)
    print("=== Full extraction summary ===")
    print(f"Tissues:                {', '.join(TISSUES)}")
    print(f"Total variants:         {total_variants}")
    print(f"Total new extractions:  {total_new}")
    print(f"Cache size after:       {cache_size_mb:.1f} MB "
          f"({CACHE_PATH})")
    print(f"Total runtime:          {overall_runtime / 60:.1f} min "
          f"({_format_minutes(overall_runtime)})")
    print(f"Total Modal cost:       ${total_cost:.2f}")
    print(f"Total failures:         {total_failures}")

    wb_sanity = sanity_results.get("Whole_Blood", {})
    bc_sanity = sanity_results.get("Brain_Cortex", {})
    if wb_sanity:
        pilot = (
            json.loads(PILOT_RESULT_PATH.read_text())
            if PILOT_RESULT_PATH.exists() else {}
        )
        pilot_delta = pilot.get("delta", {}).get("mean_cv_auroc", float("nan"))
        pilot_concat = pilot.get("concat", {}).get("mean_cv_auroc", float("nan"))
        print(
            f"Whole_Blood sanity:     "
            f"delta AUROC={wb_sanity['delta']['mean_cv_auroc']:.4f} "
            f"(pilot {pilot_delta:.4f}), "
            f"concat AUROC={wb_sanity['concat']['mean_cv_auroc']:.4f} "
            f"(pilot {pilot_concat:.4f})"
        )
    if bc_sanity:
        print(
            f"Brain_Cortex sanity:    "
            f"delta AUROC={bc_sanity['delta']['mean_cv_auroc']:.4f}, "
            f"concat AUROC={bc_sanity['concat']['mean_cv_auroc']:.4f}"
        )
    print("=" * 70)

    # Persist the summary alongside the features.
    summary_path = FEATURES_DIR / "extraction_summary.json"
    summary_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model": MODEL_NAME,
        "layer": LAYER,
        "pool_window": POOL_WINDOW,
        "assembly": ASSEMBLY,
        "per_tissue": per_tissue,
        "sanity": sanity_results,
        "total_variants": total_variants,
        "total_new_extractions": total_new,
        "total_cost_estimate": total_cost,
        "total_runtime_seconds": overall_runtime,
        "cache_size_mb": cache_size_mb,
    }, indent=2, default=str))
    print(f"Wrote {summary_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(3)
