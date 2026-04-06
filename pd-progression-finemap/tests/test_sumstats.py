"""Validate lead SNPs from the PD progression GWAS paper.

Reference: Tan et al., "Genome-wide determinants of mortality and motor
progression in Parkinson's disease", npj Parkinson's Disease (2024).
Zenodo record 8017385.

The test loads the mortality and HY3+ summary statistics and checks that
each paper-reported lead SNP is present with a p-value within one order
of magnitude of the published value.

Expects files in data/raw/ matching:
  - *MORTALITY*META*.tbl  (MarkerName = chr:bp format)
  - *HY3*META*.tbl        (MarkerName = rsID format)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.download import load_sumstats

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Lead SNP definitions from the paper
# ---------------------------------------------------------------------------

@dataclass
class LeadSNP:
    rsid: str
    chr: str
    pos: int       # hg19 / GRCh37
    gene: str
    pval: float    # published p-value
    chrpos: str | None = None  # chr:pos string for files without rsids


MORTALITY_LEADS = [
    LeadSNP("rs429358",   "19", 45411941,  "APOE",   1.4e-10, "19:45411941"),
    LeadSNP("rs4726467",  "7",  139637422, "TBXAS1", 7.7e-10, "7:139637422"),
    LeadSNP("rs10437796", "12", 33635494,  "SYT10",  5.3e-08, "12:33635494"),
]

HY3_LEADS = [
    LeadSNP("rs115217673",  "1",  2308517,   "MORN1", 3.1e-09),
    LeadSNP("rs145274312",  "7",  97478547,  "ASNS",  3.5e-09),
    LeadSNP("rs113120976",  "4",  120416730, "PDE5A", 7.0e-09),
    LeadSNP("rs141421624",  "2",  61709726,  "XPO1",  3.1e-08),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file(pattern: str) -> Path | None:
    """Glob for a single file in DATA_DIR matching *pattern*."""
    hits = sorted(DATA_DIR.glob(pattern))
    if hits:
        return hits[0]
    return None


def _parse_chrpos_marker(df: pd.DataFrame) -> pd.DataFrame:
    """If 'rsid' looks like chr:pos, split into chr and pos columns."""
    if df.empty or "rsid" not in df.columns:
        return df
    sample = df["rsid"].dropna().iloc[:20]
    if sample.str.contains(":").all():
        parts = df["rsid"].str.split(":", n=1, expand=True)
        df["chr"] = parts[0].astype(str)
        df["pos"] = pd.to_numeric(parts[1], errors="coerce")
    return df


def _find_snp(df: pd.DataFrame, snp: LeadSNP) -> pd.Series | None:
    """Look up a lead SNP by rsid first, then by chr:pos string, then by chr + pos columns."""
    if "rsid" in df.columns:
        # Strategy 1: exact rsid match
        mask = df["rsid"] == snp.rsid
        if mask.any():
            return df.loc[mask].iloc[0]
        # Strategy 2: explicit chrpos string (for mortality files with chr:pos markers)
        if snp.chrpos:
            mask = df["rsid"] == snp.chrpos
            if mask.any():
                return df.loc[mask].iloc[0]
        # Strategy 3: construct chr:pos from snp fields
        chrpos = f"{snp.chr}:{snp.pos}"
        mask = df["rsid"] == chrpos
        if mask.any():
            return df.loc[mask].iloc[0]

    # Strategy 4: match by chr + pos columns
    if "chr" in df.columns and "pos" in df.columns:
        mask = (df["chr"].astype(str) == str(snp.chr)) & (df["pos"] == snp.pos)
        if mask.any():
            return df.loc[mask].iloc[0]

    return None


def _nearby_variants(df: pd.DataFrame, snp: LeadSNP, window: int = 500_000) -> pd.DataFrame:
    """Return variants within ±window bp of the lead SNP for diagnostics."""
    if "chr" not in df.columns or "pos" not in df.columns:
        return pd.DataFrame()
    mask = (
        (df["chr"].astype(str) == str(snp.chr))
        & (df["pos"] >= snp.pos - window)
        & (df["pos"] <= snp.pos + window)
    )
    nearby = df.loc[mask].copy()
    if nearby.empty:
        return nearby
    nearby["dist_to_lead"] = (nearby["pos"] - snp.pos).abs()
    cols = [c for c in ["rsid", "chr", "pos", "pval", "dist_to_lead"] if c in nearby.columns]
    return nearby[cols].sort_values("pval").head(10)


def _assert_pval_within_order_of_magnitude(observed: float, expected: float, snp_label: str) -> None:
    """Assert that log10(observed) is within 1.0 of log10(expected)."""
    log_obs = np.log10(observed)
    log_exp = np.log10(expected)
    diff = abs(log_obs - log_exp)
    assert diff <= 1.0, (
        f"{snp_label}: p-value {observed:.2e} is {diff:.2f} orders of magnitude "
        f"from expected {expected:.2e} (tolerance: 1.0)"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mortality_df() -> pd.DataFrame:
    path = (
        _find_file("*mortality*summaryStats*")
        or _find_file("*mortality*sumstats*")
        or _find_file("*MORTALITY*META*.tbl")
        or _find_file("*mortality*meta*.tbl")
        or _find_file("*mortality*")
    )
    if path is None:
        pytest.skip(
            f"Mortality summary stats not found in {DATA_DIR}. "
            "Run: python -m src.download"
        )
    df = load_sumstats(path)
    df = _parse_chrpos_marker(df)
    return df


@pytest.fixture(scope="module")
def hy3_df() -> pd.DataFrame:
    path = (
        _find_file("*HY3*summaryStats*")
        or _find_file("*HY3*sumstats*")
        or _find_file("*HY3*META*.tbl")
        or _find_file("*hy3*meta*.tbl")
        or _find_file("*HY3*")
        or _find_file("*hy3*")
    )
    if path is None:
        pytest.skip(
            f"HY3+ summary stats not found in {DATA_DIR}. "
            "Run: python -m src.download"
        )
    df = load_sumstats(path)
    df = _parse_chrpos_marker(df)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMortalityLeadSNPs:
    """Verify that mortality lead SNPs are present with expected p-values."""

    @pytest.mark.parametrize(
        "snp",
        MORTALITY_LEADS,
        ids=[s.rsid for s in MORTALITY_LEADS],
    )
    def test_lead_snp(self, mortality_df: pd.DataFrame, snp: LeadSNP) -> None:
        row = _find_snp(mortality_df, snp)

        if row is None:
            nearby = _nearby_variants(mortality_df, snp)
            nearby_str = nearby.to_string() if not nearby.empty else "(no variants in ±500kb window)"
            pytest.fail(
                f"Lead SNP {snp.rsid} ({snp.gene}, chr{snp.chr}:{snp.pos}) "
                f"not found in mortality summary stats.\n"
                f"Closest variants by position (±500kb):\n{nearby_str}"
            )

        assert "pval" in row.index, f"No pval column for {snp.rsid}"
        observed_p = float(row["pval"])
        assert np.isfinite(observed_p) and observed_p > 0, (
            f"{snp.rsid}: invalid p-value {observed_p}"
        )
        _assert_pval_within_order_of_magnitude(observed_p, snp.pval, f"{snp.rsid} ({snp.gene})")
        print(
            f"  ✓ {snp.rsid} ({snp.gene}): observed p={observed_p:.2e}, "
            f"expected p≈{snp.pval:.2e}"
        )


class TestHY3LeadSNPs:
    """Verify that HY3+ motor-progression lead SNPs are present with expected p-values."""

    @pytest.mark.parametrize(
        "snp",
        HY3_LEADS,
        ids=[s.rsid for s in HY3_LEADS],
    )
    def test_lead_snp(self, hy3_df: pd.DataFrame, snp: LeadSNP) -> None:
        row = _find_snp(hy3_df, snp)

        if row is None:
            nearby = _nearby_variants(hy3_df, snp)
            nearby_str = nearby.to_string() if not nearby.empty else "(no variants in ±500kb window)"
            pytest.fail(
                f"Lead SNP {snp.rsid} ({snp.gene}, chr{snp.chr}:{snp.pos}) "
                f"not found in HY3+ summary stats.\n"
                f"Closest variants by position (±500kb):\n{nearby_str}"
            )

        assert "pval" in row.index, f"No pval column for {snp.rsid}"
        observed_p = float(row["pval"])
        assert np.isfinite(observed_p) and observed_p > 0, (
            f"{snp.rsid}: invalid p-value {observed_p}"
        )
        _assert_pval_within_order_of_magnitude(observed_p, snp.pval, f"{snp.rsid} ({snp.gene})")
        print(
            f"  ✓ {snp.rsid} ({snp.gene}): observed p={observed_p:.2e}, "
            f"expected p≈{snp.pval:.2e}"
        )
