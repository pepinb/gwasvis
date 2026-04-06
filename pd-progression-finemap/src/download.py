"""Download GWAS summary statistics and reference data for PD progression loci.

Data source: Zenodo record 8017385
    "Genome-wide determinants of mortality and motor progression in
     Parkinson's disease" — summary statistics from METAL meta-analysis.

Usage:
    python -m src.download            # download all files
    python -m src.download --list     # list remote files only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ZENODO_RECORD_ID = "8017385"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Column mapping: METAL / common GWAS variants → canonical names
# ---------------------------------------------------------------------------
# Left side: lowercased column names seen in the wild.
# Right side: our canonical column names.
_COLUMN_MAP: dict[str, str] = {
    # Marker / variant ID
    "markername": "rsid",
    "snp": "rsid",
    "rsid": "rsid",
    "marker": "rsid",
    "variant_id": "rsid",
    # Chromosome
    "chr": "chr",
    "chromosome": "chr",
    "chrom": "chr",
    "#chr": "chr",
    "hm_chrom": "chr",
    # Position
    "pos": "pos",
    "position": "pos",
    "bp": "pos",
    "base_pair_location": "pos",
    "hm_pos": "pos",
    # Alleles
    "allele1": "a1",
    "a1": "a1",
    "effect_allele": "a1",
    "alt": "a1",
    "allele2": "a2",
    "a2": "a2",
    "other_allele": "a2",
    "non_effect_allele": "a2",
    "noneffect_allele": "a2",
    "ref": "a2",
    # Effect size
    "effect": "beta",
    "beta": "beta",
    "coeff": "beta",
    "log_or": "beta",
    # Standard error
    "stderr": "se",
    "se": "se",
    "standard_error": "se",
    # P-value
    "p-value": "pval",
    "pvalue": "pval",
    "pval": "pval",
    "p_value": "pval",
    "p": "pval",
    # Allele frequency
    "freq1": "eaf",
    "freq": "eaf",
    "eaf": "eaf",
    "maf": "eaf",
    "effect_allele_frequency": "eaf",
    # Sample size
    "n": "n",
    "weight": "n",
    "totalsamplesize": "n",
    "n_samples": "n",
}

CANONICAL_COLS = ["chr", "pos", "rsid", "a1", "a2", "beta", "se", "pval", "eaf", "n"]


# ---------------------------------------------------------------------------
# Zenodo helpers
# ---------------------------------------------------------------------------

def fetch_record_metadata() -> dict:
    """Fetch the Zenodo record JSON and return it."""
    resp = requests.get(ZENODO_API_URL, timeout=60)
    resp.raise_for_status()
    return resp.json()


def list_remote_files() -> list[dict]:
    """Return a list of dicts with keys: filename, size_bytes, download_url."""
    meta = fetch_record_metadata()
    files = []
    for f in meta.get("files", []):
        files.append(
            {
                "filename": f["key"],
                "size_bytes": f["size"],
                "download_url": f["links"]["self"],
            }
        )
    return files


def print_remote_files() -> None:
    """Fetch and pretty-print the files in the Zenodo record."""
    files = list_remote_files()
    print(f"\nZenodo record {ZENODO_RECORD_ID}  —  {len(files)} file(s)\n")
    print(f"  {'Filename':<60s}  {'Size':>10s}  URL")
    print(f"  {'-'*60}  {'-'*10}  {'-'*40}")
    for f in files:
        size_mb = f["size_bytes"] / (1024 * 1024)
        print(f"  {f['filename']:<60s}  {size_mb:>8.1f} MB  {f['download_url']}")
    print()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download *url* to *dest* with a progress indicator."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                bar = "#" * int(pct // 2)
                print(f"\r  [{bar:<50s}] {pct:5.1f}%  ({downloaded / 1e6:.1f} MB)", end="", flush=True)
    print()  # newline after progress bar


def download_all(dest_dir: Path | None = None) -> list[Path]:
    """Download every file in the Zenodo record to *dest_dir*.

    Files that already exist (same name) are skipped.

    Returns a list of local file paths.
    """
    dest_dir = Path(dest_dir) if dest_dir else RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = list_remote_files()
    paths: list[Path] = []

    for f in files:
        dest = dest_dir / f["filename"]
        if dest.exists():
            logger.info("Skipping %s (already exists)", dest.name)
            print(f"  ✓ {dest.name} — already exists, skipping")
            paths.append(dest)
            continue
        logger.info("Downloading %s (%.1f MB)", dest.name, f["size_bytes"] / 1e6)
        print(f"  ↓ {dest.name} ({f['size_bytes'] / 1e6:.1f} MB)")
        _download_file(f["download_url"], dest)
        paths.append(dest)

    return paths


# ---------------------------------------------------------------------------
# Load & harmonise summary statistics
# ---------------------------------------------------------------------------

def load_sumstats(path: str | Path) -> pd.DataFrame:
    """Read a GWAS summary-statistics file and return a DataFrame with canonical columns.

    The function auto-detects separators (tab or whitespace) and maps
    whichever column names are present (METAL-style, GWAS Catalog-style,
    etc.) to a canonical set:

        chr, pos, rsid, a1, a2, beta, se, pval, eaf, n

    Columns that cannot be mapped are kept under their original names.
    A log message records the original columns and the mapping applied.

    Parameters
    ----------
    path : str or Path
        Path to a summary statistics file (plain text, .gz, .tsv, .csv).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    logger.info("Loading summary statistics from %s", path)

    # Detect separator — try tab first, fall back to whitespace
    df = pd.read_csv(path, sep="\t", nrows=5, comment="#")
    if len(df.columns) <= 2:
        df = pd.read_csv(path, sep=r"\s+", nrows=5, comment="#")
        sep = r"\s+"
    else:
        sep = "\t"

    df = pd.read_csv(path, sep=sep, comment="#")
    original_cols = list(df.columns)
    logger.info("Original columns (%d): %s", len(original_cols), original_cols)

    # Build rename map for columns present in this file
    rename: dict[str, str] = {}
    for col in df.columns:
        canonical = _COLUMN_MAP.get(col.lower().strip())
        if canonical and canonical not in rename.values():
            rename[col] = canonical

    logger.info("Column mapping applied: %s", rename)
    df = df.rename(columns=rename)

    # Coerce key numeric columns
    for col in ("pos", "beta", "se", "pval", "eaf", "n"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "chr" in df.columns:
        df["chr"] = df["chr"].astype(str).str.replace("chr", "", case=False)

    # Uppercase alleles
    for col in ("a1", "a2"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()

    # Reorder: canonical columns first, then any extras
    front = [c for c in CANONICAL_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    logger.info(
        "Loaded %d variants, canonical columns present: %s",
        len(df),
        [c for c in CANONICAL_COLS if c in df.columns],
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download PD progression GWAS summary statistics from Zenodo.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List remote files and exit (no download).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Destination directory (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Load and preview a local summary-statistics file, then exit.",
    )
    args = parser.parse_args()

    if args.load:
        df = load_sumstats(args.load)
        print(df.head(10).to_string())
        print(f"\n{len(df)} variants, columns: {list(df.columns)}")
        sys.exit(0)

    if args.list:
        print_remote_files()
        sys.exit(0)

    print(f"Downloading files from Zenodo record {ZENODO_RECORD_ID} …\n")
    paths = download_all(dest_dir=args.dest)
    print(f"\nDone — {len(paths)} file(s) in {paths[0].parent if paths else RAW_DIR}")


if __name__ == "__main__":
    main()
