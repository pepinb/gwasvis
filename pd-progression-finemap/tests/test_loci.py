"""Tests for src.loci — locus definitions and extraction logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.loci import LOCI, extract_locus


# ---------------------------------------------------------------------------
# Locus constant validation
# ---------------------------------------------------------------------------

def test_loci_count():
    """There should be 7 loci: 3 mortality + 4 HY3+."""
    assert len(LOCI) == 7


def test_loci_required_keys():
    """Every locus dict must have the required keys."""
    required = {"name", "chr", "lead_bp", "lead_rsid", "trait", "build"}
    for loc in LOCI:
        missing = required - set(loc.keys())
        assert not missing, f"Locus {loc.get('name')}: missing keys {missing}"


def test_loci_traits():
    """Traits should be exactly mortality and hy3."""
    traits = {loc["trait"] for loc in LOCI}
    assert traits == {"mortality", "hy3"}


def test_loci_build():
    """All loci should be hg19."""
    for loc in LOCI:
        assert loc["build"] == "hg19", f"{loc['name']} has build {loc['build']}"


def test_loci_unique_names():
    """Locus names should be unique."""
    names = [loc["name"] for loc in LOCI]
    assert len(names) == len(set(names))


def test_mortality_loci_present():
    """APOE, TBXAS1, SYT10 should be in the mortality loci."""
    mort_names = {loc["name"] for loc in LOCI if loc["trait"] == "mortality"}
    assert mort_names == {"APOE", "TBXAS1", "SYT10"}


def test_hy3_loci_present():
    """MORN1, ASNS, PDE5A, XPO1 should be in the HY3+ loci."""
    hy3_names = {loc["name"] for loc in LOCI if loc["trait"] == "hy3"}
    assert hy3_names == {"MORN1", "ASNS", "PDE5A", "XPO1"}


def test_syt10_is_suggestive():
    """SYT10 should be marked as suggestive."""
    syt10 = [loc for loc in LOCI if loc["name"] == "SYT10"][0]
    assert syt10.get("significance") == "suggestive"


# ---------------------------------------------------------------------------
# extract_locus logic
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sumstats() -> pd.DataFrame:
    """Synthetic sumstats for chr19 around the APOE lead (45411941)."""
    positions = list(range(44_900_000, 45_920_001, 1_000))
    return pd.DataFrame({
        "chr": ["19"] * len(positions),
        "pos": positions,
        "rsid": [f"rs_fake_{i}" for i in range(len(positions))],
        "pval": [0.5] * len(positions),
    })


def test_extract_locus_default_window(mock_sumstats):
    """Default 500 kb window around APOE lead should capture the right range."""
    apoe = [loc for loc in LOCI if loc["name"] == "APOE"][0]
    result = extract_locus(mock_sumstats, apoe)
    assert len(result) > 0
    assert result["pos"].min() >= apoe["lead_bp"] - 500_000
    assert result["pos"].max() <= apoe["lead_bp"] + 500_000
    # Should be sorted by position
    assert (result["pos"].diff().dropna() >= 0).all()


def test_extract_locus_custom_window(mock_sumstats):
    """A 100 kb window should return fewer variants than 500 kb."""
    apoe = [loc for loc in LOCI if loc["name"] == "APOE"][0]
    narrow = extract_locus(mock_sumstats, apoe, window=100_000)
    wide = extract_locus(mock_sumstats, apoe, window=500_000)
    assert len(narrow) < len(wide)
    assert narrow["pos"].min() >= apoe["lead_bp"] - 100_000
    assert narrow["pos"].max() <= apoe["lead_bp"] + 100_000


def test_extract_locus_wrong_chr(mock_sumstats):
    """Extracting a locus from the wrong chromosome should return empty."""
    morn1 = [loc for loc in LOCI if loc["name"] == "MORN1"][0]
    result = extract_locus(mock_sumstats, morn1)
    assert len(result) == 0
