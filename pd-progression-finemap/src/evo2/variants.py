"""Build ref/alt sequence windows around a variant for Evo2 scoring.

Two flavours:

* ``build_variant_windows`` (plural): original API used by the Phase 2.5-2.8
  fine-mapping pipeline. Returns a ``(ref_window, alt_window)`` tuple and
  takes a ``ReferenceProvider``.
* ``build_variant_window`` (singular): added for the eQTL embeddings phase.
  Returns a structured ``VariantWindow`` dataclass and takes a
  ``FastaProvider`` so the resulting window knows which assembly it came from.
"""

from __future__ import annotations

from dataclasses import dataclass

from evo2.reference import FastaProvider, ReferenceProvider


def build_variant_windows(
    chrom: str,
    pos: int,
    ref_allele: str,
    alt_allele: str,
    reference: ReferenceProvider,
    window_size: int = 8192,
) -> tuple[str, str]:
    """Return (ref_window, alt_window) centered on the variant.

    Coordinates are 1-based. Only SNVs are supported in Phase 1.
    """
    if len(ref_allele) != 1 or len(alt_allele) != 1:
        raise NotImplementedError("Indels not supported in Phase 1")

    half = window_size // 2
    start = pos - half          # 1-based
    end = pos + half - 1        # 1-based inclusive, total length = window_size

    ref_window = reference.fetch(chrom, start, end)

    if len(ref_window) != window_size:
        raise ValueError(
            f"Expected {window_size}bp window, got {len(ref_window)}bp "
            f"for {chrom}:{start}-{end}"
        )

    center_base = ref_window[half]
    if center_base != ref_allele.upper():
        raise ValueError(
            f"Ref allele mismatch at {chrom}:{pos} — "
            f"expected {ref_allele.upper()!r}, "
            f"observed {center_base!r} at window position {half}"
        )

    alt_window = ref_window[:half] + alt_allele.upper() + ref_window[half + 1 :]
    return ref_window, alt_window


# ---------------------------------------------------------------------------
# Singular build_variant_window (eQTL embeddings phase)
# ---------------------------------------------------------------------------

WINDOW_SIZE = 8192
CENTER_INDEX = WINDOW_SIZE // 2  # 4096


@dataclass(frozen=True)
class VariantWindow:
    """An 8192 bp window centered on a variant, ready for Evo 2 scoring."""

    ref_sequence: str
    alt_sequence: str
    chrom: str
    position: int
    ref_allele: str
    alt_allele: str
    assembly: str


def build_variant_window(
    chrom: int | str,
    position: int,
    ref: str,
    alt: str,
    provider: FastaProvider,
) -> VariantWindow:
    """Build an 8192 bp window centered on a SNV for Evo 2 scoring.

    Parameters
    ----------
    chrom : int or str
        Chromosome (e.g., 19, 'chr19', '19').
    position : int
        1-based genomic coordinate of the variant.
    ref : str
        Expected reference allele (single nucleotide for SNVs).
    alt : str
        Alternate allele (single nucleotide for SNVs).
    provider : FastaProvider
        Reference FASTA provider (carries assembly info).

    Returns
    -------
    VariantWindow
        Window with ref and alt sequences.

    Raises
    ------
    AssertionError
        If the reference base at the center position does not match ``ref``.
    """
    window_start = position - CENTER_INDEX  # 1-based
    window_end = window_start + WINDOW_SIZE - 1  # 1-based inclusive

    seq = provider.get_sequence(chrom, window_start, window_end)
    assert len(seq) == WINDOW_SIZE, (
        f"Expected {WINDOW_SIZE} bp window, got {len(seq)} bp. "
        f"Position {position} may be too close to chromosome boundary."
    )

    actual_ref = seq[CENTER_INDEX]
    assert actual_ref == ref.upper(), (
        f"Reference allele mismatch at {chrom}:{position} ({provider.assembly}): "
        f"expected {ref!r}, got {actual_ref!r} from FASTA. "
        f"Check coordinate system or assembly."
    )

    alt_seq = seq[:CENTER_INDEX] + alt.upper() + seq[CENTER_INDEX + 1 :]

    return VariantWindow(
        ref_sequence=seq,
        alt_sequence=alt_seq,
        chrom=str(chrom),
        position=position,
        ref_allele=ref.upper(),
        alt_allele=alt.upper(),
        assembly=provider.assembly,
    )
