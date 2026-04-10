"""eQTL logistic regression probes over Evo 2 embeddings.

Feature constructions are derived from cached (n, 16384) float32 embeddings
stored in concat order [ref_fwd, ref_rc, alt_fwd, alt_rc] (4 x 4096).
"""

from __future__ import annotations

import datetime as _dt
import subprocess
import warnings
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Required to pass `groups` through LogisticRegressionCV.fit -> GroupKFold.split
sklearn.set_config(enable_metadata_routing=True)

_MAX_ITER = 500

FEATURE_CONSTRUCTIONS = ['concat', 'delta', 'hybrid']
DEFAULT_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

_SUB_DIM = 4096
_CONCAT_DIM = 4 * _SUB_DIM  # 16384


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


def _chrom_sort_key(c: str) -> tuple:
    s = str(c)
    if s.startswith('chr'):
        s = s[3:]
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def build_features(embeddings: np.ndarray, construction: str) -> np.ndarray:
    """Build feature matrix from cached (n, 16384) embeddings.

    Cache layout: [ref_fwd, ref_rc, alt_fwd, alt_rc], each 4096-D.
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != _CONCAT_DIM:
        raise ValueError(
            f'Expected (n, {_CONCAT_DIM}) embeddings, got shape {embeddings.shape}'
        )

    ref_fwd = embeddings[:, 0 * _SUB_DIM : 1 * _SUB_DIM]
    ref_rc = embeddings[:, 1 * _SUB_DIM : 2 * _SUB_DIM]
    alt_fwd = embeddings[:, 2 * _SUB_DIM : 3 * _SUB_DIM]
    alt_rc = embeddings[:, 3 * _SUB_DIM : 4 * _SUB_DIM]

    if construction == 'concat':
        return embeddings
    if construction == 'delta':
        return np.concatenate([alt_fwd - ref_fwd, alt_rc - ref_rc], axis=1)
    if construction == 'hybrid':
        return np.concatenate(
            [ref_fwd, alt_fwd - ref_fwd, ref_rc, alt_rc - ref_rc], axis=1
        )
    raise ValueError(
        f'Unknown construction {construction!r}; expected one of {FEATURE_CONSTRUCTIONS}'
    )


class EQTLProbe:
    """StandardScaler + L2 logistic regression probe over eQTL embeddings.

    Bundles a sklearn Pipeline so inference at prompt 8 applies the same
    scaling that was used at training time.
    """

    def __init__(
        self,
        C: float = 1.0,
        random_state: int = 42,
        construction: str = 'concat',
    ):
        self.C = float(C)
        self.random_state = int(random_state)
        self.construction = construction
        self.model = Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'lr',
                    LogisticRegression(
                        C=self.C,
                        solver='lbfgs',
                        max_iter=_MAX_ITER,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.metadata: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EQTLProbe':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(X.astype(np.float64, copy=False), y)
        self.metadata = {
            'model_name': 'evo2_7b',
            'layer': 'blocks.26',
            'pool_window': 128,
            'construction': self.construction,
            'feature_dim': int(X.shape[1]),
            'C': self.C,
            'training_timestamp': _dt.datetime.now(_dt.timezone.utc).isoformat(),
            'git_commit': _git_commit(),
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X.astype(np.float64, copy=False))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X.astype(np.float64, copy=False))
        lr = self.model.named_steps['lr']
        pos_idx = list(lr.classes_).index(1)
        return proba[:, pos_idx]

    def save(self, path) -> None:
        joblib.dump(
            {
                'model': self.model,
                'metadata': self.metadata,
                'C': self.C,
                'construction': self.construction,
                'random_state': self.random_state,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> 'EQTLProbe':
        obj = joblib.load(path)
        probe = cls(
            C=obj['C'],
            random_state=obj['random_state'],
            construction=obj['construction'],
        )
        probe.model = obj['model']
        probe.metadata = obj['metadata']
        return probe


def train_with_leave_chr_out(
    embeddings: np.ndarray,
    labels: np.ndarray,
    chromosomes: np.ndarray,
    construction: str,
    C_grid: list[float] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Leave-one-chromosome-out CV with inner GroupKFold for C selection."""
    if C_grid is None:
        C_grid = list(DEFAULT_C_GRID)

    X = build_features(embeddings, construction).astype(np.float64, copy=False)
    y = np.asarray(labels).astype(int)
    chroms = np.asarray(chromosomes)
    unique_chroms = sorted(np.unique(chroms).tolist(), key=_chrom_sort_key)

    per_auroc: dict[str, float] = {}
    per_n: dict[str, int] = {}
    all_C_choices: list[tuple[str, float]] = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for held_chr in unique_chroms:
            test_mask = chroms == held_chr
            train_mask = ~test_mask

            X_tr_raw, y_tr = X[train_mask], y[train_mask]
            X_te_raw, y_te = X[test_mask], y[test_mask]
            train_groups = chroms[train_mask]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            n_train_chroms = len(np.unique(train_groups))
            inner_k = int(min(5, n_train_chroms))

            cv = GroupKFold(n_splits=inner_k)
            # n_jobs=1: sklearn's native BLAS threading inside LBFGS is
            # more efficient than joblib process parallelism at this size
            # (avoids re-pickling the ~600 MB training matrix per task).
            lrcv = LogisticRegressionCV(
                Cs=C_grid,
                cv=cv,
                scoring='roc_auc',
                solver='lbfgs',
                max_iter=_MAX_ITER,
                refit=True,
                random_state=42,
                n_jobs=1,
            )
            lrcv.fit(X_tr, y_tr, groups=train_groups)

            best_C = float(lrcv.C_[0])
            all_C_choices.append((str(held_chr), best_C))

            pos_idx = list(lrcv.classes_).index(1)
            probas = lrcv.predict_proba(X_te)[:, pos_idx]
            if len(np.unique(y_te)) < 2:
                auroc = float('nan')
            else:
                auroc = float(roc_auc_score(y_te, probas))
            per_auroc[str(held_chr)] = auroc
            per_n[str(held_chr)] = int(test_mask.sum())

            if verbose:
                print(
                    f'    [{construction}] {held_chr}: AUROC={auroc:.3f} '
                    f'n={int(test_mask.sum())} best_C={best_C:g}',
                    flush=True,
                )

    valid = [v for v in per_auroc.values() if not np.isnan(v)]
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    mn = float(np.min(valid))
    mx = float(np.max(valid))

    choices = [c for _, c in all_C_choices]
    uniq, counts = np.unique(choices, return_counts=True)
    modal_best_C = float(uniq[int(np.argmax(counts))])

    return {
        'construction': construction,
        'feature_dim': int(X.shape[1]),
        'per_chromosome_auroc': per_auroc,
        'per_chromosome_n': per_n,
        'mean_auroc': mean,
        'std_auroc': std,
        'min_auroc': mn,
        'max_auroc': mx,
        'modal_best_C': modal_best_C,
        'all_C_choices': all_C_choices,
    }
