"""Tests for src.finemap — SuSiE-RSS pure-Python implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.finemap import run_susie, _single_effect_regression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_causal():
    """Synthetic locus: 50 variants, one strong causal at index 10."""
    np.random.seed(42)
    p = 50
    n = 5000

    # Block-diagonal LD with one block around the causal
    R = np.eye(p)
    causal = 10
    for i in range(max(0, causal - 5), min(p, causal + 6)):
        for j in range(max(0, causal - 5), min(p, causal + 6)):
            if i != j:
                R[i, j] = 0.3 * np.exp(-0.5 * abs(i - j))
    # Make symmetric PSD
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)

    # True effect only at causal
    beta_true = np.zeros(p)
    beta_true[causal] = 0.15

    # Generate z-scores: z = sqrt(n) * R @ beta + noise
    z = np.sqrt(n) * R @ beta_true + np.random.randn(p) * 0.5
    se = np.ones(p) / np.sqrt(n)
    beta = z * se

    df = pd.DataFrame({
        "rsid": [f"rs{i}" for i in range(p)],
        "chr": ["1"] * p,
        "pos": list(range(100000, 100000 + p)),
        "a1": ["A"] * p,
        "a2": ["G"] * p,
        "beta": beta,
        "se": se,
        "pval": 0.05 * np.ones(p),  # placeholder
        "z": z,
        "n": n,
    })
    return df, R, n, causal


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSER:
    """Test the single-effect regression subroutine."""

    def test_strong_signal_concentrates_alpha(self):
        """A very strong signal at one variant should give alpha ≈ 1 there."""
        p = 20
        # Simulate: n=5000, betahat[5] = 0.15, shat2 = 1/5000
        d = np.ones(p) * 5000.0
        Xty = np.zeros(p)
        Xty[5] = 5000.0 * 0.15  # betahat = Xty/d = 0.15, strong signal
        # prior_var scaled to n: V = 0.2 * (n-1) ≈ 1000
        result = _single_effect_regression(Xty, d, residual_var=1.0, prior_var=1000.0)
        assert result["alpha"][5] > 0.99
        assert np.isclose(result["alpha"].sum(), 1.0, atol=1e-6)

    def test_no_signal_uniform_alpha(self):
        """With no signal, alpha should be roughly uniform."""
        p = 20
        d = np.ones(p) * 5000.0
        Xty = np.zeros(p)
        result = _single_effect_regression(Xty, d, residual_var=1.0, prior_var=100.0)
        assert np.allclose(result["alpha"], 1.0 / p, atol=0.01)


class TestRunSusie:
    """Test the full SuSiE-RSS pipeline."""

    def test_converges(self, single_causal):
        df, R, n, causal = single_causal
        result = run_susie(df, R, n=n, L=5, max_iter=200)
        assert result["converged"], "SuSiE should converge on clean synthetic data"

    def test_causal_has_high_pip(self, single_causal):
        df, R, n, causal = single_causal
        result = run_susie(df, R, n=n, L=5, max_iter=200)
        # Causal variant should have highest PIP
        assert result["lead_idx"] == causal, (
            f"Lead should be {causal}, got {result['lead_idx']}"
        )
        assert result["pip"][causal] > 0.5, (
            f"Causal PIP={result['pip'][causal]:.3f}, expected > 0.5"
        )

    def test_credible_set_contains_causal(self, single_causal):
        df, R, n, causal = single_causal
        result = run_susie(df, R, n=n, L=5, max_iter=200)
        # At least one CS should contain the causal variant
        found = any(causal in cs for cs in result["credible_sets"])
        assert found, (
            f"Causal index {causal} not in any CS: {result['credible_sets']}"
        )

    def test_pip_shape_and_range(self, single_causal):
        df, R, n, causal = single_causal
        result = run_susie(df, R, n=n, L=5)
        assert result["pip"].shape == (len(df),)
        assert np.all(result["pip"] >= 0)
        assert np.all(result["pip"] <= 1)

    def test_alpha_shape(self, single_causal):
        df, R, n, causal = single_causal
        L = 5
        result = run_susie(df, R, n=n, L=L)
        assert result["alpha"].shape == (L, len(df))
        # Each row sums to 1
        for l in range(L):
            assert np.isclose(result["alpha"][l].sum(), 1.0, atol=1e-6)

    def test_elbo_is_finite(self, single_causal):
        df, R, n, causal = single_causal
        result = run_susie(df, R, n=n, L=5)
        assert np.isfinite(result["elbo"])
