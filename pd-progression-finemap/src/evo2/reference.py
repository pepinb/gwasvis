"""Reference genome sequence providers.

Two families of providers live here:

* The original Phase 2.5-2.8 fine-mapping pipeline uses the ``ReferenceProvider``
  Protocol with ``EnsemblRESTProvider`` and ``LocalFastaProvider``. These are
  general-purpose and hg19-only.
* The eQTL embeddings phase (hg38) added ``FastaProvider`` — an Ensembl
  per-chromosome FASTA loader that accepts an ``assembly`` parameter. It also
  satisfies the ``ReferenceProvider`` protocol via a ``fetch()`` shim, so new
  code can plug it into the existing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import requests


@runtime_checkable
class ReferenceProvider(Protocol):
    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch sequence for chrom:start-end (1-based inclusive). Returns uppercase ACGTN."""
        ...


class EnsemblRESTProvider:
    """Fetch reference sequence from the Ensembl GRCh37 REST API."""

    BASE_URL = "https://grch37.rest.ensembl.org/sequence/region/human"

    def fetch(self, chrom: str, start: int, end: int) -> str:
        chrom = chrom.removeprefix("chr")
        url = f"{self.BASE_URL}/{chrom}:{start}..{end}"
        resp = requests.get(
            url,
            headers={"Content-Type": "text/plain"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ensembl REST error {resp.status_code} for {chrom}:{start}-{end}: "
                f"{resp.text[:200]}"
            )
        return resp.text.strip().upper()


class LocalFastaProvider:
    """Fetch reference sequence from local FASTA file(s) via pyfaidx.

    Accepts either a single FASTA path or a directory containing
    per-chromosome files named chr{N}.fa.
    """

    def __init__(self, fasta_path: str):
        import os

        from pyfaidx import Fasta

        path = os.path.expanduser(fasta_path)
        if os.path.isdir(path):
            # Load all chr*.fa files from the directory
            self._fastas: dict[str, Fasta] = {}
            for f in sorted(os.listdir(path)):
                if f.endswith(".fa") and f.startswith("chr"):
                    fa = Fasta(os.path.join(path, f))
                    for key in fa.keys():
                        self._fastas[key] = fa
        else:
            fa = Fasta(path)
            self._fastas = {key: fa for key in fa.keys()}

        self._chroms = set(self._fastas.keys())

    def _normalize_chrom(self, chrom: str) -> str:
        if chrom in self._chroms:
            return chrom
        alt = "chr" + chrom if not chrom.startswith("chr") else chrom.removeprefix("chr")
        if alt in self._chroms:
            return alt
        raise KeyError(f"Chromosome {chrom!r} (tried also {alt!r}) not in FASTA")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        chrom = self._normalize_chrom(chrom)
        # pyfaidx uses 0-based slicing; our API is 1-based inclusive
        return str(self._fastas[chrom][chrom][start - 1 : end]).upper()


# ---------------------------------------------------------------------------
# FastaProvider — hg19/hg38 aware, Ensembl per-chromosome layout
# ---------------------------------------------------------------------------
# Used by the eQTL embeddings phase (src/evo2/embeddings.py,
# src/evo2/variants.build_variant_window, tests/test_liftover.py).

_ASSEMBLY_DIRS = {
    "hg19": "reference",
    "hg38": "reference_hg38",
}

# Ensembl FASTA file naming patterns per assembly
_FASTA_PATTERNS = {
    "hg19": "Homo_sapiens.GRCh37.dna.chromosome.{chrom}.fa",
    "hg38": "Homo_sapiens.GRCh38.dna.chromosome.{chrom}.fa",
}


def _normalise_chrom(chrom: str | int) -> str:
    """Strip 'chr' prefix and return bare chromosome name for Ensembl FASTAs."""
    return str(chrom).replace("chr", "")


class FastaProvider:
    """Lazy-loading, caching FASTA provider for one assembly.

    Loads Ensembl per-chromosome FASTA files from ``data/reference/`` (hg19)
    or ``data/reference_hg38/`` (hg38). Files must be decompressed (.fa) so
    pyfaidx can index them.

    Ensembl FASTAs use bare chromosome names (e.g., "1", "19"), not "chr1".
    This provider normalises chromosome input to match.

    Parameters
    ----------
    assembly : {'hg19', 'hg38'}
        Genome assembly to load. Default is 'hg19' for backward compatibility
        with hg19-only call sites.
    data_root : Path, optional
        Root data directory. Default is ``<project>/data/``.
    """

    def __init__(
        self,
        assembly: str = "hg19",
        data_root: Path | None = None,
    ) -> None:
        if assembly not in _ASSEMBLY_DIRS:
            raise ValueError(
                f"Unknown assembly {assembly!r}. Use 'hg19' or 'hg38'."
            )
        self.assembly = assembly
        if data_root is None:
            data_root = Path(__file__).resolve().parent.parent.parent / "data"
        self._dir = Path(data_root) / _ASSEMBLY_DIRS[assembly]
        # Lazy Fasta cache (import here so the module stays importable even
        # when pyfaidx is not installed).
        self._cache: dict[str, object] = {}

    def _get_fasta(self, chrom: str):
        from pyfaidx import Fasta

        if chrom not in self._cache:
            fname = _FASTA_PATTERNS[self.assembly].format(chrom=chrom)
            path = self._dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"FASTA not found: {path}\n"
                    f"Download {self.assembly} chr{chrom} into {self._dir}/"
                )
            self._cache[chrom] = Fasta(str(path))
        return self._cache[chrom]

    def get_base(self, chrom: str | int, pos: int) -> str:
        """Return the reference base at a 1-based position."""
        chrom_key = _normalise_chrom(chrom)
        fasta = self._get_fasta(chrom_key)
        # pyfaidx uses 0-based slicing; single base at pos means [pos-1:pos]
        return str(fasta[chrom_key][pos - 1 : pos]).upper()

    def get_sequence(self, chrom: str | int, start: int, end: int) -> str:
        """Return the reference sequence at 1-based ``[start, end]`` inclusive."""
        chrom_key = _normalise_chrom(chrom)
        fasta = self._get_fasta(chrom_key)
        # pyfaidx 0-based slicing: [start-1 : end]
        return str(fasta[chrom_key][start - 1 : end]).upper()

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """ReferenceProvider-compatible shim, delegates to ``get_sequence``."""
        return self.get_sequence(chrom, start, end)
