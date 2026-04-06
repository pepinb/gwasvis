"""Tests for src.download – column mapping and loading logic."""

import textwrap
from pathlib import Path

import pandas as pd

from src.download import load_sumstats, CANONICAL_COLS


def _write_tsv(tmp_path: Path, header: str, row: str) -> Path:
    p = tmp_path / "test_sumstats.tsv"
    p.write_text(textwrap.dedent(f"{header}\n{row}\n"))
    return p


def test_load_metal_format(tmp_path):
    """METAL-style columns should map to canonical names."""
    path = _write_tsv(
        tmp_path,
        "MarkerName\tAllele1\tAllele2\tEffect\tStdErr\tP-value\tFreq1\tWeight",
        "rs123\ta\tc\t0.05\t0.01\t1e-8\t0.3\t5000",
    )
    df = load_sumstats(path)
    assert "rsid" in df.columns
    assert "a1" in df.columns
    assert "beta" in df.columns
    assert "pval" in df.columns
    assert df.iloc[0]["rsid"] == "rs123"
    assert df.iloc[0]["a1"] == "A"  # uppercased
    assert df.iloc[0]["beta"] == 0.05
    assert df.iloc[0]["pval"] == 1e-8
    assert df.iloc[0]["eaf"] == 0.3
    assert df.iloc[0]["n"] == 5000


def test_load_gwas_catalog_format(tmp_path):
    """GWAS-catalog-style columns should also map correctly."""
    path = _write_tsv(
        tmp_path,
        "chromosome\tbase_pair_location\trsid\teffect_allele\tother_allele\tbeta\tstandard_error\tp_value\teffect_allele_frequency\tn_samples",
        "1\t12345\trs456\tG\tA\t-0.02\t0.005\t0.03\t0.45\t10000",
    )
    df = load_sumstats(path)
    assert df.iloc[0]["chr"] == "1"
    assert df.iloc[0]["pos"] == 12345
    assert df.iloc[0]["a2"] == "A"
    assert df.iloc[0]["se"] == 0.005


def test_canonical_column_order(tmp_path):
    """Canonical columns should appear first in the output."""
    path = _write_tsv(
        tmp_path,
        "MarkerName\tAllele1\tAllele2\tEffect\tStdErr\tP-value\tDirection",
        "rs1\ta\tg\t0.1\t0.02\t0.001\t++",
    )
    df = load_sumstats(path)
    present = [c for c in CANONICAL_COLS if c in df.columns]
    assert list(df.columns[: len(present)]) == present
