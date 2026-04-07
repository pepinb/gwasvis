"""Build ref/alt sequence windows around a variant for Evo2 scoring."""

from __future__ import annotations

from evo2.reference import ReferenceProvider


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
