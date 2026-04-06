"""Run statistical fine-mapping to produce credible sets for each locus.

Implements SuSiE-RSS (Sum of Single Effects — Regression with Summary
Statistics) in pure Python / NumPy, following Wang et al. (2020) JRSS-B
and Zou et al. (2022) PLoS Genetics.

**Why a pure-Python implementation?**
- ``susiepy`` does not exist on PyPI.
- R (susieR) is not available in this environment and rpy2 adds a heavy
  runtime dependency.
- The SuSiE-RSS algorithm is compact (~150 lines of linear algebra) and
  benefits from being transparent and dependency-free.

The implementation mirrors susieR's ``susie_rss`` → ``susie_suff_stat``
pipeline, using the same PVE-adjusted z-score conversion, Wakefield
approximate Bayes factors for the single-effect regression (SER), and
ELBO-based convergence.

Usage:
    python -m src.finemap --locus APOE --trait mortality
    python -m src.finemap --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.ld import build_locus_ld, LD_DIR
from src.loci import LOCI, LOCI_DIR

logger = logging.getLogger(__name__)

FINEMAP_DIR = Path(__file__).resolve().parent.parent / "data" / "finemap"


# ---------------------------------------------------------------------------
# SuSiE-RSS  (pure NumPy)
# ---------------------------------------------------------------------------

def _single_effect_regression(
    Xty: np.ndarray,
    d: np.ndarray,
    residual_var: float,
    prior_var: float,
) -> dict:
    """Fit a single-effect regression (SER) — one causal variant.

    Parameters
    ----------
    Xty : (p,) residual association vector (X'r for the current residual r).
    d : (p,) diagonal of X'X (i.e. ``diag(XtX)``).
    residual_var : current estimate of sigma^2 (residual variance).
    prior_var : prior variance on the non-zero effect (V).

    Returns
    -------
    dict with keys: alpha, mu, mu2, lbf, lbf_model
    """
    p = len(Xty)

    # Posterior quantities for each variant j
    shat2 = residual_var / d                         # (p,) variance of betahat
    betahat = Xty / d                                # (p,) least-squares estimate

    # Log Bayes factor (Wakefield ABF)
    lbf = np.full(p, -np.inf)
    valid = d > 0
    tau = prior_var + shat2[valid]
    lbf[valid] = (
        -0.5 * np.log(1.0 + prior_var / shat2[valid])
        + 0.5 * betahat[valid] ** 2 * prior_var / (shat2[valid] * tau)
    )

    # Posterior inclusion probabilities (alpha)
    log_alpha = lbf - logsumexp(lbf)
    alpha = np.exp(log_alpha)

    # Posterior mean / second moment of effect (conditional on being causal)
    post_var = np.zeros(p)
    post_mean = np.zeros(p)
    post_var[valid] = prior_var * shat2[valid] / tau
    post_mean[valid] = post_var[valid] / shat2[valid] * betahat[valid]

    mu = alpha * post_mean                           # E[b_l]
    mu2 = alpha * (post_var + post_mean ** 2)        # E[b_l^2]

    # KL divergence for this effect
    lbf_model = logsumexp(lbf)   # log marginal likelihood of the SER

    return {"alpha": alpha, "mu": mu, "mu2": mu2, "lbf": lbf, "lbf_model": lbf_model}


def _compute_elbo(
    XtX: np.ndarray,
    Xty: np.ndarray,
    yty: float,
    n: int,
    residual_var: float,
    effects: list[dict],
) -> float:
    """Compute the evidence lower bound (ELBO) for the current fit."""
    p = XtX.shape[0]

    # Expected fitted values: Xb = XtX @ sum(mu_l)
    b = np.sum([e["mu"] for e in effects], axis=0)
    Xb = XtX @ b

    # Expected RSS
    E_rss = yty - 2.0 * Xty @ b
    for e in effects:
        E_rss += np.dot(e["mu2"], np.diag(XtX))
        E_rss += b @ XtX @ b  # will subtract self-interaction below
    # Correct: only cross-terms
    E_rss = yty - 2.0 * Xty @ b
    for i, ei in enumerate(effects):
        E_rss += np.dot(ei["mu2"], np.diag(XtX))
        for j, ej in enumerate(effects):
            if i != j:
                E_rss += ei["mu"] @ XtX @ ej["mu"]

    # Expected log-likelihood
    E_loglik = -0.5 * n * np.log(2.0 * np.pi * residual_var) - 0.5 / residual_var * E_rss

    # KL divergence: sum of -lbf_model for each effect (relative to uniform prior)
    kl = -sum(e["lbf_model"] for e in effects)

    return float(E_loglik - kl)


def run_susie(
    sumstats: pd.DataFrame,
    ld: np.ndarray,
    n: int,
    L: int = 10,
    coverage: float = 0.95,
    min_abs_corr: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-3,
    prior_var: float = 0.2,
) -> dict:
    """Run SuSiE-RSS fine-mapping.

    Parameters
    ----------
    sumstats : DataFrame with columns 'beta', 'se' (and optionally 'z').
    ld : (p, p) LD correlation matrix (r, *not* r²), aligned to sumstats.
    n : effective sample size.
    L : maximum number of causal effects to model.
    coverage : credible set coverage (default 0.95).
    min_abs_corr : purity threshold — minimum absolute correlation within a CS.
    max_iter : maximum IBSS iterations.
    tol : convergence tolerance on ELBO change.
    prior_var : prior variance on standardised effect sizes (default 0.2).

    Returns
    -------
    dict with keys:
        pip       — (p,) posterior inclusion probabilities
        alpha     — (L, p) per-effect inclusion probabilities
        credible_sets — list of lists of variant indices
        cs_coverage   — coverage achieved per CS
        cs_purity     — min |r| within each CS
        lead_idx  — index of top-PIP variant
        converged — bool
        elbo      — final ELBO value
        n_iter    — iterations run
    """
    p = ld.shape[0]
    assert ld.shape == (p, p), f"LD must be square, got {ld.shape}"
    assert len(sumstats) == p, f"sumstats ({len(sumstats)}) != LD ({p})"

    # -- z-scores -------------------------------------------------------------
    z = sumstats["z"].values if "z" in sumstats.columns else (
        sumstats["beta"].values / sumstats["se"].values
    )

    # -- Convert to sufficient statistics (susieR convention) -----------------
    # PVE adjustment: adj_j = (n-1) / (z_j^2 + n - 2)
    adj = (n - 1.0) / (z ** 2 + n - 2.0)
    z_adj = np.sqrt(adj) * z

    R = ld.copy()
    np.fill_diagonal(R, 1.0)                # ensure exact 1s on diagonal
    XtX = (n - 1.0) * R
    Xty = np.sqrt(n - 1.0) * z_adj
    yty = float(n - 1)
    d = np.diag(XtX)                        # (p,) diagonal

    # -- Regularise XtX slightly for numerical stability ----------------------
    XtX += 1e-6 * np.eye(p)
    d = np.diag(XtX)

    # -- Initialise effects ---------------------------------------------------
    residual_var = 1.0
    effects: list[dict] = []
    for _ in range(L):
        effects.append({
            "alpha": np.full(p, 1.0 / p),
            "mu": np.zeros(p),
            "mu2": np.zeros(p),
            "lbf": np.zeros(p),
            "lbf_model": 0.0,
        })

    # -- IBSS iterations ------------------------------------------------------
    elbo_history: list[float] = []
    converged = False

    for it in range(max_iter):
        for l_idx in range(L):
            # Current residual association
            Xtr = Xty.copy()
            for k, ek in enumerate(effects):
                if k != l_idx:
                    Xtr -= XtX @ ek["mu"]

            effects[l_idx] = _single_effect_regression(
                Xtr, d, residual_var, prior_var * (n - 1),
            )

        elbo = _compute_elbo(XtX, Xty, yty, n, residual_var, effects)
        elbo_history.append(elbo)

        if len(elbo_history) >= 2:
            delta = abs(elbo_history[-1] - elbo_history[-2])
            if delta < tol:
                converged = True
                logger.info("Converged at iteration %d (ΔELBO=%.2e)", it + 1, delta)
                break

    n_iter = len(elbo_history)
    if not converged:
        logger.warning("Did not converge after %d iterations", max_iter)

    # -- PIP: 1 - prod(1 - alpha_l) ------------------------------------------
    alpha_mat = np.array([e["alpha"] for e in effects])  # (L, p)
    pip = 1.0 - np.prod(1.0 - alpha_mat, axis=0)

    # -- Credible sets --------------------------------------------------------
    credible_sets: list[list[int]] = []
    cs_coverages: list[float] = []
    cs_purities: list[float] = []

    for l_idx in range(L):
        a = alpha_mat[l_idx]
        order = np.argsort(a)[::-1]
        cumprob = np.cumsum(a[order])
        n_in = int(np.searchsorted(cumprob, coverage) + 1)
        cs_indices = list(order[:n_in])
        achieved_cov = float(cumprob[n_in - 1])

        # Purity: minimum absolute correlation among CS members
        if len(cs_indices) <= 1:
            purity = 1.0
        else:
            ld_sub = ld[np.ix_(cs_indices, cs_indices)]
            purity = float(np.min(np.abs(ld_sub)))

        # Only keep CS with sufficient purity and non-trivial signal
        if purity >= min_abs_corr and a.max() > 1.0 / p:
            credible_sets.append(cs_indices)
            cs_coverages.append(achieved_cov)
            cs_purities.append(purity)

    lead_idx = int(np.argmax(pip))

    return {
        "pip": pip,
        "alpha": alpha_mat,
        "credible_sets": credible_sets,
        "cs_coverage": cs_coverages,
        "cs_purity": cs_purities,
        "lead_idx": lead_idx,
        "converged": converged,
        "elbo": elbo_history[-1] if elbo_history else float("nan"),
        "n_iter": n_iter,
    }


# ---------------------------------------------------------------------------
# Locus-level wrapper
# ---------------------------------------------------------------------------

def finemap_locus(locus_name: str, trait: str, L: int = 10) -> pd.DataFrame:
    """Fine-map a single locus end-to-end.

    1. Build / load the aligned LD matrix via ``build_locus_ld``.
    2. Run SuSiE-RSS.
    3. Annotate the sumstats with PIP, in_cs, cs_id.
    4. Save to ``data/finemap/{locus_name}_{trait}.parquet``.

    Returns
    -------
    pd.DataFrame  — aligned sumstats with PIP / CS annotations.
    """
    # Load LD (may build from plink2 if not cached)
    aligned_path = LD_DIR / f"{locus_name}_{trait}.aligned.parquet"
    ld_path = LD_DIR / f"{locus_name}_{trait}.ld"
    snplist_path = LD_DIR / f"{locus_name}_{trait}.snplist"

    if aligned_path.exists() and ld_path.exists() and snplist_path.exists():
        logger.info("Loading cached LD for %s_%s", locus_name, trait)
        sumstats = pd.read_parquet(aligned_path)
        ld_matrix = np.loadtxt(ld_path, dtype=np.float64)
        snp_order = snplist_path.read_text().strip().split("\n")
    else:
        sumstats, ld_matrix, snp_order = build_locus_ld(locus_name, trait)

    # Effective sample size from the sumstats
    if "n" in sumstats.columns:
        n = int(sumstats["n"].median())
    else:
        logger.warning("No 'n' column — using default n=5000")
        n = 5000

    logger.info(
        "Fine-mapping %s_%s: %d variants, n=%d, L=%d",
        locus_name, trait, len(sumstats), n, L,
    )
    print(f"  SuSiE-RSS: {len(sumstats)} variants, n={n}, L={L}")

    # z-scores should already be computed by build_locus_ld
    if "z" not in sumstats.columns:
        sumstats["z"] = sumstats["beta"] / sumstats["se"]

    result = run_susie(sumstats, ld_matrix, n=n, L=L)

    # -- Annotate sumstats with fine-mapping results --------------------------
    out = sumstats.copy()
    out["pip"] = result["pip"]
    out["in_cs"] = False
    out["cs_id"] = -1

    for cs_idx, cs in enumerate(result["credible_sets"]):
        for var_idx in cs:
            out.loc[out.index[var_idx], "in_cs"] = True
            out.loc[out.index[var_idx], "cs_id"] = cs_idx

    out["converged"] = result["converged"]

    # -- Save -----------------------------------------------------------------
    FINEMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINEMAP_DIR / f"{locus_name}_{trait}.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved fine-mapping results to %s", out_path)

    # -- Report ---------------------------------------------------------------
    print(f"  Converged: {result['converged']} ({result['n_iter']} iterations)")
    print(f"  ELBO: {result['elbo']:.2f}")
    print(f"  Credible sets: {len(result['credible_sets'])}")

    for cs_idx, (cs, cov, pur) in enumerate(
        zip(result["credible_sets"], result["cs_coverage"], result["cs_purity"])
    ):
        cs_rsids = [snp_order[i] if i < len(snp_order) else f"idx_{i}" for i in cs]
        print(f"    CS{cs_idx}: {len(cs)} variants, coverage={cov:.3f}, purity={pur:.3f}")
        if len(cs) <= 10:
            for i in cs:
                row = out.iloc[i]
                print(
                    f"      {row.get('rsid', '?'):20s}  "
                    f"PIP={result['pip'][i]:.4f}  "
                    f"p={row.get('pval', float('nan')):.2e}"
                )

    print(f"\n  Top 5 SNPs by PIP:")
    top5 = out.nlargest(5, "pip")
    for _, row in top5.iterrows():
        cs_label = f"CS{int(row['cs_id'])}" if row["in_cs"] else "—"
        print(
            f"    {row.get('rsid', '?'):20s}  "
            f"PIP={row['pip']:.4f}  "
            f"p={row.get('pval', float('nan')):.2e}  "
            f"{cs_label}"
        )

    return out


def finemap_all(L: int = 10) -> list[Path]:
    """Fine-map every locus defined in :data:`LOCI`.

    Returns a list of output parquet paths (one per successfully processed locus).
    """
    FINEMAP_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for locus in LOCI:
        name, trait = locus["name"], locus["trait"]
        parquet = LOCI_DIR / f"{name}_{trait}.parquet"
        if not parquet.exists():
            print(f"  ! {name}_{trait}: locus parquet not found, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"  {name} ({trait})")
        print(f"{'='*60}")
        try:
            finemap_locus(name, trait, L=L)
            written.append(FINEMAP_DIR / f"{name}_{trait}.parquet")
        except Exception as exc:
            logger.error("Failed for %s_%s: %s", name, trait, exc)
            print(f"  ERROR: {exc}")

    print(f"\nDone — {len(written)} locus parquets in {FINEMAP_DIR}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fine-map PD progression GWAS loci with SuSiE-RSS.",
    )
    parser.add_argument("--locus", type=str, help="Locus name (e.g. APOE).")
    parser.add_argument("--trait", type=str, help="Trait (mortality or hy3).")
    parser.add_argument("--L", type=int, default=10, help="Max causal effects (default: 10).")
    parser.add_argument("--all", action="store_true", help="Fine-map all loci.")
    args = parser.parse_args()

    if args.all:
        finemap_all(L=args.L)
        sys.exit(0)

    if args.locus and args.trait:
        finemap_locus(args.locus, args.trait, L=args.L)
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
