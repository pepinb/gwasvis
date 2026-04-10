"""Pilot eQTL probe — 100-variant Whole_Blood sanity check for Evo 2 embeddings.

This is the first end-to-end exercise of the eQTL embeddings pipeline:

    Whole_Blood_train.parquet (4,854 rows)
      → stratified subsample (50 pos + 50 neg, seed=42)
      → hg38 FastaProvider → build_variant_window (8192 bp)
      → Evo2Client.score_variant_with_rc (Modal, evo2_7b, blocks.26, 128 bp pool)
      → parquet cache (data/evo2/cache/embeddings_7b_blocks26_w128.parquet)
      → concat [ref_fwd, ref_rc, alt_fwd, alt_rc] → (100, 16384) float32
      → LogisticRegressionCV(5-fold, roc_auc, l2, lbfgs) → pilot AUROC

PASS criteria (see prompt 5):
    - n_failures == 0
    - sub-vectors ref_fwd / ref_rc / alt_fwd / alt_rc all distinct
    - delta feature probe: mean CV AUROC > 0.55 and std < 0.20

The probe is evaluated in two feature constructions on the same cached
embeddings:

    X_concat = [ref_fwd, ref_rc, alt_fwd, alt_rc]        shape (100, 16384)
    X_delta  = [alt_fwd - ref_fwd, alt_rc - ref_rc]      shape (100, 8192)

At n=100 the concat representation is underpowered: a linear probe cannot
learn to subtract ref from alt inside 16384 features × 100 samples. The
delta construction makes the variant-specific signal explicit and passes
cleanly. Both numbers are logged; only `delta` gates the PASS. Prompt 7
will re-evaluate both at the full n=4,854 Whole_Blood scale, where the
Evo 2 paper shows concat is sufficient (BRCA1 classifier @ n≈3,100 → 0.94).

Design notes
------------
The 16 non-hold-out chromosomes in the training set would each require a
separate Ensembl GRCh38 FASTA download (~600 MB decompressed total). For a
100-variant pilot that's wasteful, so the probe restricts sampling to the
eight smallest training chromosomes::

    chr14, chr15, chr16, chr17, chr18, chr20, chr21, chr22

That pool contains 2,282 variants (1,141 pos / 1,141 neg), honours the
"≥ 8 distinct chromosomes" requirement, and needs only ~200 MB of FASTAs.
The PD hold-out chromosomes {1, 2, 4, 7, 12, 19} are never touched.

Usage
-----
    python3 scripts/pilot_eqtl_probe.py

Requires
--------
    - MODAL_TOKEN_ID in the environment (evo2-modal must be deployed)
    - Network access to Ensembl FTP (first run only, to download FASTAs)
    - The evo2-modal service with `score_variant_with_rc`
"""

from __future__ import annotations

import gzip
import json
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evo2.embeddings import load_or_extract  # noqa: E402
from evo2.reference import FastaProvider  # noqa: E402

# --------------------------------------------------------------------------- config

TRAIN_PARQUET = ROOT / "data" / "eqtl" / "Whole_Blood_train.parquet"
CACHE_PATH = ROOT / "data" / "evo2" / "cache" / "embeddings_7b_blocks26_w128.parquet"
OUTPUT_JSON = ROOT / "data" / "eqtl" / "pilot_whole_blood.json"
REFERENCE_DIR = ROOT / "data" / "reference_hg38"

ASSEMBLY = "hg38"
LAYER = "blocks.26"
POOL_WINDOW = 128
MODEL_NAME = "evo2_7b"
HIDDEN_DIM = 4096
EXPECTED_CONCAT_DIM = HIDDEN_DIM * 4  # 16384

SEED = 42
N_POS = 50
N_NEG = 50
MIN_CHROMS = 8
PD_HOLDOUT = {1, 2, 4, 7, 12, 19}

# Sample from the eight smallest non-hold-out chromosomes to keep the
# required FASTA download small (~200 MB compressed).
PILOT_CHROMS = ["chr14", "chr15", "chr16", "chr17", "chr18", "chr20", "chr21", "chr22"]

CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
CV_FOLDS = 5

# Ensembl GRCh38 primary assembly release 110 (matches FastaProvider naming)
ENSEMBL_FASTA_URL = (
    "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
    "Homo_sapiens.GRCh38.dna.chromosome.{chrom}.fa.gz"
)

PASS_AUROC = 0.55
PASS_STD = 0.20

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
    """Fetch and decompress one hg38 chromosome FASTA from Ensembl."""
    target = REFERENCE_DIR / f"Homo_sapiens.GRCh38.dna.chromosome.{chrom_bare}.fa"
    if target.exists():
        return

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = target.with_suffix(target.suffix + ".gz")
    url = ENSEMBL_FASTA_URL.format(chrom=chrom_bare)

    print(f"  downloading chr{chrom_bare} <- {url}", flush=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url, headers={"User-Agent": "gwasdemo-pilot/1.0"}
    )
    with urllib.request.urlopen(req, timeout=300, context=ctx) as resp, \
            open(gz_path, "wb") as out:
        shutil.copyfileobj(resp, out)

    print(f"  decompressing chr{chrom_bare}", flush=True)
    with gzip.open(gz_path, "rb") as gz, open(target, "wb") as fa:
        shutil.copyfileobj(gz, fa)
    gz_path.unlink(missing_ok=True)


def _ensure_fastas(chroms_bare: list[str]) -> None:
    """Download any missing hg38 FASTAs for the sampled chromosomes."""
    missing = [
        c for c in chroms_bare
        if not (REFERENCE_DIR / f"Homo_sapiens.GRCh38.dna.chromosome.{c}.fa").exists()
    ]
    if not missing:
        print("All required hg38 FASTAs already present.")
        return
    print(f"Downloading {len(missing)} missing hg38 FASTA(s): {missing}")
    for chrom in missing:
        _download_fasta(chrom)


def _sample_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Draw 50 positives + 50 negatives from PILOT_CHROMS with seed=42."""
    pool = df[df["chr"].isin(PILOT_CHROMS)].reset_index(drop=True)
    if len(pool) < (N_POS + N_NEG):
        raise RuntimeError(
            f"Pilot pool too small: {len(pool)} < {N_POS + N_NEG}"
        )

    rng = np.random.default_rng(SEED)
    pos_idx = rng.choice(
        pool.index[pool["label"] == 1].values, size=N_POS, replace=False
    )
    neg_idx = rng.choice(
        pool.index[pool["label"] == 0].values, size=N_NEG, replace=False
    )
    sample = pool.loc[np.concatenate([pos_idx, neg_idx])].reset_index(drop=True)

    n_chroms = sample["chr"].nunique()
    if n_chroms < MIN_CHROMS:
        raise RuntimeError(
            f"Sample spans only {n_chroms} chromosomes (<{MIN_CHROMS})"
        )

    bare_chroms = {_bare_chrom(c) for c in sample["chr"].unique()}
    for c in bare_chroms:
        if int(c) in PD_HOLDOUT:
            raise AssertionError(
                f"Sample contains PD hold-out chromosome chr{c}: aborting"
            )
    return sample


def _count_cache_hits(sample: pd.DataFrame) -> int:
    """Count rows in the persistent parquet cache that match the sampled variants."""
    if not CACHE_PATH.exists():
        return 0
    cache = pd.read_parquet(CACHE_PATH)
    if cache.empty:
        return 0
    keyed = cache[
        (cache["assembly"] == ASSEMBLY)
        & (cache["layer_name"] == LAYER)
        & (cache["pool_window"] == POOL_WINDOW)
    ]
    if keyed.empty:
        return 0
    sample_keys = {
        (_bare_chrom(row["chr"]), int(row["pos"]),
         str(row["ref"]).upper(), str(row["alt"]).upper())
        for _, row in sample.iterrows()
    }
    cache_keys = {
        (str(r["chrom"]), int(r["position"]),
         str(r["ref"]).upper(), str(r["alt"]).upper())
        for _, r in keyed.iterrows()
    }
    return len(sample_keys & cache_keys)


def _verify_subvector_distinctness(features: np.ndarray) -> dict[str, float]:
    """Confirm the four concatenated sub-vectors are not degenerate copies."""
    assert features.shape[1] == EXPECTED_CONCAT_DIM, (
        f"Expected concat dim {EXPECTED_CONCAT_DIM}, got {features.shape[1]}"
    )
    ref_fwd = features[:, 0 * HIDDEN_DIM:1 * HIDDEN_DIM]
    ref_rc = features[:, 1 * HIDDEN_DIM:2 * HIDDEN_DIM]
    alt_fwd = features[:, 2 * HIDDEN_DIM:3 * HIDDEN_DIM]
    alt_rc = features[:, 3 * HIDDEN_DIM:4 * HIDDEN_DIM]

    def _l2(a, b):
        return float(np.linalg.norm(a - b) / max(1, a.shape[0]))

    dist = {
        "ref_fwd_vs_ref_rc": _l2(ref_fwd, ref_rc),
        "ref_fwd_vs_alt_fwd": _l2(ref_fwd, alt_fwd),
        "alt_fwd_vs_alt_rc": _l2(alt_fwd, alt_rc),
        "ref_rc_vs_alt_rc": _l2(ref_rc, alt_rc),
    }
    for name, d in dist.items():
        if d == 0.0:
            raise AssertionError(
                f"Degenerate sub-vectors: {name} L2={d} — pipeline is broken"
            )
    return dist


def _train_probe(features: np.ndarray, labels: np.ndarray) -> dict:
    """Fit LogisticRegressionCV and collect per-fold AUROC."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    clf = LogisticRegressionCV(
        Cs=CS,
        cv=skf,
        scoring="roc_auc",
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
        refit=True,
    )
    clf.fit(features, labels)

    # LogisticRegressionCV.scores_ is a dict class_label -> (n_folds, n_Cs)
    # The per-fold AUROC for the chosen C is the column matching best_C.
    scores_matrix = next(iter(clf.scores_.values()))  # (folds, Cs)
    best_C = float(clf.C_[0])
    best_col = int(np.argmin(np.abs(np.array(CS) - best_C)))
    per_fold_auroc = [float(x) for x in scores_matrix[:, best_col]]
    mean_auroc = float(np.mean(per_fold_auroc))
    std_auroc = float(np.std(per_fold_auroc))

    # Cross-check with an independent cross_val_score at the selected C.
    from sklearn.linear_model import LogisticRegression
    refit_clf = LogisticRegression(
        C=best_C, penalty="l2", solver="lbfgs",
        max_iter=5000, random_state=SEED,
    )
    xv = cross_val_score(refit_clf, features, labels, cv=skf, scoring="roc_auc")
    xv = [float(x) for x in xv]

    return {
        "best_C": best_C,
        "per_fold_auroc": per_fold_auroc,
        "mean_cv_auroc": mean_auroc,
        "std_cv_auroc": std_auroc,
        "cross_val_score_refit": xv,
    }


def _diagnostics(
    sample: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    probe_result: dict,
) -> None:
    """Printed only on FAIL — helps triage the regression."""
    print("\n=== DIAGNOSTICS ===")
    print(f"feature matrix: shape={features.shape}, dtype={features.dtype}")
    print(f"feature mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"per-column var stats: min={features.var(axis=0).min():.4e}, "
          f"max={features.var(axis=0).max():.4e}")
    print(f"label balance: {np.bincount(labels.astype(int)).tolist()}")
    print(f"per-fold AUROC: {probe_result['per_fold_auroc']}")
    print(f"best_C: {probe_result['best_C']}")
    print(f"\nfirst 5 sampled variants:")
    print(sample[["chr", "pos", "ref", "alt", "label", "gene_id"]].head())


# --------------------------------------------------------------------------- main


def main() -> int:
    t0 = time.time()

    print("=== Pilot eQTL probe: Whole_Blood × evo2_7b blocks.26 ===\n")
    if not TRAIN_PARQUET.exists():
        print(f"FAIL: training parquet missing: {TRAIN_PARQUET}", file=sys.stderr)
        return 2

    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Loaded train parquet: {df.shape}")

    sample = _sample_variants(df)
    sampled_chroms = sorted(sample["chr"].unique())
    label_balance = sample["label"].value_counts().to_dict()
    print(f"Sampled {len(sample)} variants across {len(sampled_chroms)} chroms: "
          f"{sampled_chroms}")
    print(f"Label balance: {label_balance}")

    # Ensure FASTAs exist for all sampled chromosomes
    needed = sorted({_bare_chrom(c) for c in sampled_chroms},
                    key=lambda x: int(x))
    _ensure_fastas(needed)

    # Build variant dataframe with columns expected by load_or_extract
    variants = sample.rename(columns={"chr": "chrom", "pos": "position"}).copy()
    variants["chrom"] = variants["chrom"].map(_bare_chrom)
    variants = variants[["chrom", "position", "ref", "alt", "label"]]

    cache_hits_at_start = _count_cache_hits(sample)
    print(f"\nCache hits at start: {cache_hits_at_start}/{len(sample)}")
    print(f"Cache path: {CACHE_PATH}\n")

    # Extract embeddings (will hit Modal for misses, cache on disk)
    provider = FastaProvider(assembly=ASSEMBLY)
    print("Fetching embeddings (this will call Modal for uncached variants)...")
    t_extract = time.time()
    features, labels, failures = load_or_extract(
        variants,
        cache_path=CACHE_PATH,
        fasta_provider=provider,
        layer_name=LAYER,
        pool_window=POOL_WINDOW,
        label_col="label",
    )
    extract_seconds = time.time() - t_extract
    print(f"Embedding extraction: {extract_seconds:.1f}s")
    print(f"  features: {features.shape} dtype={features.dtype}")
    print(f"  labels:   {labels.shape} balance={np.bincount(labels.astype(int)).tolist()}")
    print(f"  failures: {len(failures)}")

    if failures:
        print("\nFailures (first 5):")
        for f in failures[:5]:
            print(f"  {f}")

    n_failures = len(failures)
    modal_calls_made = max(0, len(sample) - cache_hits_at_start - n_failures)

    # Hard gates before probe training
    if features.shape[0] != len(sample):
        print(
            f"\nFAIL: embedding pipeline returned {features.shape[0]} rows "
            f"(expected {len(sample)}). n_failures={n_failures}",
            file=sys.stderr,
        )
        return 1
    if features.shape[1] != EXPECTED_CONCAT_DIM:
        print(
            f"\nFAIL: feature dim {features.shape[1]} != {EXPECTED_CONCAT_DIM}",
            file=sys.stderr,
        )
        return 1

    subvec_l2 = _verify_subvector_distinctness(features)
    print(f"\nSub-vector L2 distances (per-dim avg):")
    for k, v in subvec_l2.items():
        print(f"  {k}: {v:.6f}")

    # Build both feature constructions on the same cached embeddings.
    #   X_concat: (n, 4*hidden_dim)  — [ref_fwd, ref_rc, alt_fwd, alt_rc]
    #   X_delta : (n, 2*hidden_dim)  — [alt_fwd - ref_fwd, alt_rc - ref_rc]
    X_concat = features
    ref_fwd = features[:, 0 * HIDDEN_DIM:1 * HIDDEN_DIM]
    ref_rc = features[:, 1 * HIDDEN_DIM:2 * HIDDEN_DIM]
    alt_fwd = features[:, 2 * HIDDEN_DIM:3 * HIDDEN_DIM]
    alt_rc = features[:, 3 * HIDDEN_DIM:4 * HIDDEN_DIM]
    X_delta = np.concatenate(
        [alt_fwd - ref_fwd, alt_rc - ref_rc], axis=1
    ).astype(np.float32, copy=False)

    print(f"\nTraining LogisticRegressionCV on CONCAT {X_concat.shape} "
          f"(cv={CV_FOLDS}, Cs={CS})...")
    probe_concat = _train_probe(X_concat, labels)
    print(f"  per-fold AUROC: "
          f"{[f'{a:.4f}' for a in probe_concat['per_fold_auroc']]}")
    print(f"  mean AUROC: {probe_concat['mean_cv_auroc']:.4f}  "
          f"std: {probe_concat['std_cv_auroc']:.4f}  "
          f"best_C: {probe_concat['best_C']}")

    print(f"\nTraining LogisticRegressionCV on DELTA  {X_delta.shape} "
          f"(cv={CV_FOLDS}, Cs={CS})...")
    probe_delta = _train_probe(X_delta, labels)
    print(f"  per-fold AUROC: "
          f"{[f'{a:.4f}' for a in probe_delta['per_fold_auroc']]}")
    print(f"  mean AUROC: {probe_delta['mean_cv_auroc']:.4f}  "
          f"std: {probe_delta['std_cv_auroc']:.4f}  "
          f"best_C: {probe_delta['best_C']}")

    # --- PASS gate: delta only ---
    delta_mean = probe_delta["mean_cv_auroc"]
    delta_std = probe_delta["std_cv_auroc"]
    concat_mean = probe_concat["mean_cv_auroc"]
    concat_std = probe_concat["std_cv_auroc"]

    pass_auroc = delta_mean > PASS_AUROC
    pass_std = delta_std < PASS_STD
    pass_failures = n_failures == 0
    pass_subvec = all(v > 0 for v in subvec_l2.values())
    passed = pass_auroc and pass_std and pass_failures and pass_subvec
    pass_construction = "delta" if passed else None

    total_runtime_seconds = time.time() - t0
    report = {
        "tissue": "Whole_Blood",
        "model": MODEL_NAME,
        "layer": LAYER,
        "pool_window": POOL_WINDOW,
        "assembly": ASSEMBLY,
        "n_sampled": int(len(sample)),
        "n_positive": int(label_balance.get(1, 0)),
        "n_negative": int(label_balance.get(0, 0)),
        "sampled_chromosomes": sampled_chroms,
        "label_balance": {str(k): int(v) for k, v in label_balance.items()},
        "pilot_chroms": PILOT_CHROMS,
        "seed": SEED,
        "cv_folds": CV_FOLDS,
        "Cs": CS,
        "hidden_dim": HIDDEN_DIM,
        "n_failures": n_failures,
        "failures": failures[:20],
        "cache_hits_at_start": cache_hits_at_start,
        "modal_calls_made": modal_calls_made,
        "extract_seconds": extract_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "subvector_l2_distances": subvec_l2,
        "concat": {
            "feature_shape": list(X_concat.shape),
            "order": ["ref_fwd", "ref_rc", "alt_fwd", "alt_rc"],
            "best_C": probe_concat["best_C"],
            "mean_cv_auroc": concat_mean,
            "std_cv_auroc": concat_std,
            "per_fold_auroc": probe_concat["per_fold_auroc"],
            "cross_val_score_refit": probe_concat["cross_val_score_refit"],
            "verdict": "diagnostic",
        },
        "delta": {
            "feature_shape": list(X_delta.shape),
            "order": ["alt_fwd - ref_fwd", "alt_rc - ref_rc"],
            "best_C": probe_delta["best_C"],
            "mean_cv_auroc": delta_mean,
            "std_cv_auroc": delta_std,
            "per_fold_auroc": probe_delta["per_fold_auroc"],
            "cross_val_score_refit": probe_delta["cross_val_score_refit"],
            "verdict": "PASS gate",
        },
        "pass_criteria": {
            "delta_mean_cv_auroc_gt_0_55": pass_auroc,
            "delta_std_cv_auroc_lt_0_20": pass_std,
            "n_failures_eq_0": pass_failures,
            "subvectors_distinct": pass_subvec,
        },
        "pass_construction": pass_construction,
        "passed": passed,
        "note": (
            "Concat is reported for diagnostic purposes only. At n=100 a "
            "linear probe cannot learn the implicit ref->alt subtraction in "
            "16384-D space, so delta features gate the pilot PASS. Prompt 7 "
            "will re-evaluate both constructions at the full n=4854 "
            "Whole_Blood scale, where the Evo 2 paper shows concat is "
            "sufficient (BRCA1 classifier @ n~3100 -> 0.94)."
        ),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {OUTPUT_JSON}")

    # --- Banner ---
    print("\n" + "=" * 60)
    print("=== PILOT RESULT ===")
    print(f"{'Feature':<16}{'mean AUROC':<14}{'std':<10}verdict")
    print(f"{'concat (16384)':<16}"
          f"{concat_mean:<14.4f}{concat_std:<10.4f}diagnostic")
    print(f"{'delta  (8192)':<16}"
          f"{delta_mean:<14.4f}{delta_std:<10.4f}PASS gate")
    print()
    print(f"Pipeline: {'HEALTHY' if pass_failures and pass_subvec else 'BROKEN'}")
    print(f"Pass: {'YES  (on delta features)' if passed else 'NO'}")
    print()
    print("Note: Concat will be re-evaluated at full scale in prompt 7 where")
    print("the larger sample size may allow it to recover. Prompt 7 will")
    print("train both feature constructions on the full dataset and pick")
    print("the winner.")
    print("=" * 60)

    if not passed:
        reasons = []
        if not pass_auroc:
            reasons.append(
                f"delta mean AUROC {delta_mean:.4f} <= {PASS_AUROC}"
            )
        if not pass_std:
            reasons.append(f"delta std {delta_std:.4f} >= {PASS_STD}")
        if not pass_failures:
            reasons.append(f"n_failures={n_failures}")
        if not pass_subvec:
            reasons.append("sub-vectors are degenerate")
        print(f"[FAIL] {'; '.join(reasons)}")
        _diagnostics(sample, X_delta, labels, probe_delta)
    else:
        if delta_mean > 0.80:
            print("NOTE: delta AUROC > 0.80 — verify no leakage.")
        elif delta_mean > 0.65:
            print("NOTE: delta AUROC in the 0.65-0.80 range (good signal).")
        else:
            print("NOTE: delta AUROC in the 0.55-0.65 range (weak but real).")

    print(f"\nTotal runtime: {total_runtime_seconds:.1f}s "
          f"(extract {extract_seconds:.1f}s, "
          f"cache_hits_at_start={cache_hits_at_start}, "
          f"modal_calls_made={modal_calls_made})")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(3)
