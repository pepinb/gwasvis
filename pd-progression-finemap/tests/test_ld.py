"""Tests for src.ld — allele alignment and LD helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ld import _classify_alleles, _is_ambiguous, _AMBIGUOUS_PAIRS, harmonize_alleles


# ---------------------------------------------------------------------------
# Ambiguous strand detection
# ---------------------------------------------------------------------------

class TestAmbiguousStrand:
    def test_at_is_ambiguous(self):
        assert _is_ambiguous("A", "T")
        assert _is_ambiguous("T", "A")

    def test_cg_is_ambiguous(self):
        assert _is_ambiguous("C", "G")
        assert _is_ambiguous("G", "C")

    def test_ac_not_ambiguous(self):
        assert not _is_ambiguous("A", "C")

    def test_ag_not_ambiguous(self):
        assert not _is_ambiguous("A", "G")

    def test_case_insensitive(self):
        assert _is_ambiguous("a", "t")
        assert _is_ambiguous("c", "g")


# ---------------------------------------------------------------------------
# Allele classification (plink2 LD is oriented to ALT allele)
# ---------------------------------------------------------------------------

class TestClassifyAlleles:
    def test_effect_is_alt_keep(self):
        # sumstats a1=G (effect) == VCF ALT=G → keep, no flip needed
        assert _classify_alleles("G", "A", "A", "G") == "keep"

    def test_effect_is_ref_flip(self):
        # sumstats a1=A (effect) == VCF REF=A → flip z to align with ALT
        assert _classify_alleles("A", "G", "A", "G") == "flip"

    def test_mismatch_drop(self):
        assert _classify_alleles("A", "G", "C", "T") == "drop"

    def test_case_insensitive(self):
        assert _classify_alleles("a", "g", "G", "A") == "keep"
        assert _classify_alleles("a", "g", "A", "G") == "flip"

    def test_partial_overlap_drops(self):
        assert _classify_alleles("A", "G", "A", "C") == "drop"


# ---------------------------------------------------------------------------
# Harmonize alleles — integration test with small synthetic data
# ---------------------------------------------------------------------------

class TestHarmonizeAlleles:
    def test_flip_negates_z_and_eaf(self):
        """When effect allele == REF, z must be negated and eaf inverted."""
        # 3 variants: keep, flip, keep
        sumstats = pd.DataFrame({
            "rsid": ["rs1", "rs2", "rs3"],
            "chr": ["1", "1", "1"],
            "pos": [100, 200, 300],
            "a1": ["G", "A", "T"],  # effect alleles
            "a2": ["A", "G", "C"],
            "beta": [0.5, 0.3, -0.2],
            "se": [0.1, 0.1, 0.1],
            "eaf": [0.3, 0.4, 0.6],
        })
        ref_snps = ["rs1", "rs2", "rs3"]
        # VCF REF/ALT: rs1 REF=A ALT=G, rs2 REF=A ALT=G, rs3 REF=C ALT=T
        ref_alleles = {
            "rs1": ("A", "G"),  # a1=G==ALT → keep
            "rs2": ("A", "G"),  # a1=A==REF → flip
            "rs3": ("C", "T"),  # a1=T==ALT → keep
        }
        ld_raw = np.eye(3)

        aligned, ld_sub, snp_order, report = harmonize_alleles(
            sumstats, ref_snps, ref_alleles, ld_raw, match_by="rsid",
        )

        assert report["n_keep"] == 2
        assert report["n_flip"] == 1
        assert len(aligned) == 3

        # rs1: keep → z = 0.5/0.1 = 5.0
        assert aligned.iloc[0]["z"] == pytest.approx(5.0)
        assert aligned.iloc[0]["eaf"] == pytest.approx(0.3)

        # rs2: flip → z = -(0.3/0.1) = -3.0, eaf = 1-0.4 = 0.6
        assert aligned.iloc[1]["z"] == pytest.approx(-3.0)
        assert aligned.iloc[1]["eaf"] == pytest.approx(0.6)

        # rs3: keep → z = -0.2/0.1 = -2.0
        assert aligned.iloc[2]["z"] == pytest.approx(-2.0)
        assert aligned.iloc[2]["eaf"] == pytest.approx(0.6)

    def test_position_based_matching(self):
        """Verify harmonize works with match_by='pos'."""
        sumstats = pd.DataFrame({
            "rsid": ["19:100", "19:200"],
            "chr": ["19", "19"],
            "pos": [100, 200],
            "a1": ["C", "T"],
            "a2": ["T", "C"],
            "beta": [0.5, 0.3],
            "se": [0.1, 0.1],
            "eaf": [0.3, 0.7],
        })
        ref_snps = ["19:100:T:C", "19:200:C:T"]
        ref_alleles = {
            "19:100": ("T", "C"),  # a1=C==ALT → keep
            "19:200": ("C", "T"),  # a1=T==ALT → keep
        }
        ld_raw = np.eye(2)

        aligned, ld_sub, snp_order, report = harmonize_alleles(
            sumstats, ref_snps, ref_alleles, ld_raw, match_by="pos",
        )

        assert report["n_keep"] == 2
        assert report["n_flip"] == 0
        assert len(aligned) == 2

    def test_ambiguous_dropped(self):
        """A/T and C/G SNPs should be dropped."""
        sumstats = pd.DataFrame({
            "rsid": ["rs1"],
            "chr": ["1"],
            "pos": [100],
            "a1": ["A"],
            "a2": ["T"],
            "beta": [0.5],
            "se": [0.1],
        })
        ref_snps = ["rs1"]
        ref_alleles = {"rs1": ("A", "T")}
        ld_raw = np.array([[1.0]])

        aligned, ld_sub, snp_order, report = harmonize_alleles(
            sumstats, ref_snps, ref_alleles, ld_raw, match_by="rsid",
        )

        assert report["n_ambiguous_dropped"] == 1
        assert report["n_final"] == 0
