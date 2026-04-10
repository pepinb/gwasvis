"""Streamlit-friendly loaders for Evo2 scored data.

All functions return cached DataFrames via @st.cache_data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

EVO2_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "evo2"
EQTL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "eqtl"


@st.cache_data
def list_available_models() -> list[str]:
    """Scan data/evo2/ for model versions that have a calibration_summary file."""
    models = []
    for p in sorted(EVO2_DIR.glob("calibration_summary_*.csv")):
        # calibration_summary_1b.csv -> "1b"
        suffix = p.stem.removeprefix("calibration_summary_")
        if suffix:
            models.append(suffix)
    return models


@st.cache_data
def load_evo2_scores(locus: str, trait: str, model: str = "1b") -> pd.DataFrame | None:
    """Load per-locus Evo2 scored parquet."""
    path = EVO2_DIR / f"{locus}_{trait}_evo2_{model}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_calibration_summary(model: str = "1b") -> pd.DataFrame | None:
    """Load calibration_summary_{model}.csv (per-locus scoring stats)."""
    path = EVO2_DIR / f"calibration_summary_{model}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_calibration_matched(model: str = "1b") -> pd.DataFrame | None:
    """Load calibration_matched_{model}.csv (locus + genome-wide percentiles)."""
    path = EVO2_DIR / f"calibration_matched_{model}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_genome_null(model: str = "1b") -> pd.DataFrame | None:
    """Load the scored genome-wide null for the given model."""
    path = EVO2_DIR / f"genome_null_evo2_{model}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# eQTL probe loaders (prompt 8B/8C)
# ---------------------------------------------------------------------------


@st.cache_data
def load_eqtl_probe_scorecard() -> pd.DataFrame | None:
    """Load the per-locus eQTL probe scorecard written by
    scripts/score_pd_loci_with_probes.py."""
    path = EQTL_DIR / "probe_scorecard.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_eqtl_probe_locus(locus: str, trait: str) -> pd.DataFrame | None:
    """Load per-variant eQTL probe scores for one PD locus."""
    path = EQTL_DIR / "locus_scores" / f"{locus}_{trait}_eqtl_probes.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
