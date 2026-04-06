"""Tests for src.ld — allele alignment and LD helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ld import _alleles_match, _is_ambiguous, _AMBIGUOUS_PAIRS


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
# Allele matching
# ---------------------------------------------------------------------------

class TestAllelesMatch:
    def test_exact_match(self):
        assert _alleles_match("A", "G", "A", "G") == "match"

    def test_flip(self):
        assert _alleles_match("A", "G", "G", "A") == "flip"

    def test_drop_mismatch(self):
        assert _alleles_match("A", "G", "C", "T") == "drop"

    def test_case_insensitive(self):
        assert _alleles_match("a", "g", "A", "G") == "match"
        assert _alleles_match("a", "g", "G", "A") == "flip"

    def test_same_alleles_both_sides_match(self):
        # e.g. A/C vs A/C
        assert _alleles_match("A", "C", "A", "C") == "match"

    def test_partial_overlap_drops(self):
        # one allele matches but not both
        assert _alleles_match("A", "G", "A", "C") == "drop"
