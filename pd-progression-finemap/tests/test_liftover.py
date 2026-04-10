"""Regression tests for hg19/hg38 liftover of PD lead variants.

(a) test_reference_alleles_match: verify ref allele in both assemblies
(b) test_evo2_score_consistency: score on Evo 2 1B via Modal, compare hg19 vs hg38
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# gwasvis's dict-based `LOCI` lives alongside a dataclass tuple `LOCI_OBJECTS`
# added during the eQTL embeddings reconciliation. This test file needs the
# attribute-based Locus objects (hg19_bp/hg38_bp/ref/alt/rsid/chr as int).
from loci import LOCI_OBJECTS as LOCI
from evo2.reference import FastaProvider
from evo2.variants import build_variant_window

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# (a) Reference allele check in both assemblies
# ---------------------------------------------------------------------------

class TestReferenceAllelesMatch:
    """For each PD lead variant, the ref allele must match the FASTA base
    at the correct position in BOTH hg19 and hg38."""

    @pytest.fixture(scope="class")
    def hg19_provider(self):
        return FastaProvider(assembly="hg19", data_root=DATA_ROOT)

    @pytest.fixture(scope="class")
    def hg38_provider(self):
        return FastaProvider(assembly="hg38", data_root=DATA_ROOT)

    @pytest.mark.parametrize("locus", LOCI, ids=[l.rsid for l in LOCI])
    def test_hg19_ref_allele(self, locus, hg19_provider):
        actual = hg19_provider.get_base(locus.chr, locus.hg19_bp)
        assert actual == locus.ref, (
            f"{locus.name} ({locus.rsid}): hg19 chr{locus.chr}:{locus.hg19_bp} "
            f"expected ref={locus.ref!r}, got {actual!r}"
        )

    @pytest.mark.parametrize("locus", LOCI, ids=[l.rsid for l in LOCI])
    def test_hg38_ref_allele(self, locus, hg38_provider):
        actual = hg38_provider.get_base(locus.chr, locus.hg38_bp)
        assert actual == locus.ref, (
            f"{locus.name} ({locus.rsid}): hg38 chr{locus.chr}:{locus.hg38_bp} "
            f"expected ref={locus.ref!r}, got {actual!r}"
        )

    @pytest.mark.parametrize("locus", LOCI, ids=[l.rsid for l in LOCI])
    def test_variant_window_hg19(self, locus, hg19_provider):
        """build_variant_window should not raise (ref allele assertion inside)."""
        window = build_variant_window(
            chrom=locus.chr,
            position=locus.hg19_bp,
            ref=locus.ref,
            alt=locus.alt,
            provider=hg19_provider,
        )
        assert len(window.ref_sequence) == 8192
        assert window.ref_sequence[4096] == locus.ref
        assert window.alt_sequence[4096] == locus.alt

    @pytest.mark.parametrize("locus", LOCI, ids=[l.rsid for l in LOCI])
    def test_variant_window_hg38(self, locus, hg38_provider):
        """build_variant_window should not raise (ref allele assertion inside)."""
        window = build_variant_window(
            chrom=locus.chr,
            position=locus.hg38_bp,
            ref=locus.ref,
            alt=locus.alt,
            provider=hg38_provider,
        )
        assert len(window.ref_sequence) == 8192
        assert window.ref_sequence[4096] == locus.ref
        assert window.alt_sequence[4096] == locus.alt


# ---------------------------------------------------------------------------
# (b) Evo 2 scoring consistency across assemblies
# ---------------------------------------------------------------------------

class TestEvo2ScoreConsistency:
    """Score each lead variant on Evo 2 1B using both hg19 and hg38 windows.
    Requires Modal service. Skip if MODAL_TOKEN_ID not set."""

    @pytest.fixture(scope="class", autouse=True)
    def check_modal(self):
        if not os.environ.get("MODAL_TOKEN_ID"):
            pytest.skip("MODAL_TOKEN_ID not set — skipping Evo 2 scoring tests")

    @pytest.fixture(scope="class")
    def hg19_provider(self):
        return FastaProvider(assembly="hg19", data_root=DATA_ROOT)

    @pytest.fixture(scope="class")
    def hg38_provider(self):
        return FastaProvider(assembly="hg38", data_root=DATA_ROOT)

    @pytest.fixture(scope="class")
    def evo2_model(self):
        import modal
        Evo2Model = modal.Cls.from_name("evo2-modal", "Evo2Model")
        return Evo2Model()

    def test_score_consistency(self, hg19_provider, hg38_provider, evo2_model):
        results = []

        for locus in LOCI:
            # Build windows in both assemblies
            w19 = build_variant_window(
                locus.chr, locus.hg19_bp, locus.ref, locus.alt, hg19_provider
            )
            w38 = build_variant_window(
                locus.chr, locus.hg38_bp, locus.ref, locus.alt, hg38_provider
            )

            # Score via Modal
            r19 = evo2_model.score_ref_alt.remote(w19.ref_sequence, w19.alt_sequence)
            r38 = evo2_model.score_ref_alt.remote(w38.ref_sequence, w38.alt_sequence)

            d19 = r19["delta_log_likelihood"]
            d38 = r38["delta_log_likelihood"]
            diff = abs(d19 - d38)

            if diff < 0.2:
                status = "PASS"
            elif diff < 1.0:
                status = "WARN"
            else:
                status = "FAIL"

            results.append({
                "locus": locus.name,
                "lead_rsid": locus.rsid,
                "hg19_delta_ll": d19,
                "hg38_delta_ll": d38,
                "abs_diff": diff,
                "status": status,
            })

        # Save results
        out_dir = DATA_ROOT / "evo2"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hg19_vs_hg38_sanity.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Report
        for r in results:
            print(
                f"  {r['locus']:8s} {r['lead_rsid']:15s} "
                f"hg19={r['hg19_delta_ll']:+.4f}  hg38={r['hg38_delta_ll']:+.4f}  "
                f"diff={r['abs_diff']:.4f}  {r['status']}"
            )

        # Assert no FAILs
        fails = [r for r in results if r["status"] == "FAIL"]
        assert not fails, (
            f"{len(fails)} loci FAILED (abs_diff > 1.0): "
            + ", ".join(f"{r['locus']} ({r['abs_diff']:.4f})" for r in fails)
        )
