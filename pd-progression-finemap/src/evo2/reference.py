"""Reference genome sequence providers."""

from __future__ import annotations

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
