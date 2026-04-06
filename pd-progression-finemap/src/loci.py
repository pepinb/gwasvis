"""Define and extract genomic loci around lead SNPs from GWAS summary statistics.

Locus definitions come from Tables 2 & 3 of Tan et al. (2024),
"Genome-wide determinants of mortality and motor progression in
Parkinson's disease", npj Parkinson's Disease.  Coordinates are hg19.

Usage:
    python -m src.loci                  # extract all loci to data/loci/
    python -m src.loci --window 1000000 # use a 1 Mb window
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.download import load_sumstats

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LOCI_DIR = Path(__file__).resolve().parent.parent / "data" / "loci"

# ---------------------------------------------------------------------------
# Locus definitions — Tables 2 (mortality) & 3 (HY3+) of Tan et al. 2024
# ---------------------------------------------------------------------------
# Significance:
#   genome-wide significant  p < 5e-8
#   suggestive               5e-8 <= p < 1e-6 (included for completeness)

LOCI: list[dict] = [
    # ── Mortality (Table 2) ──────────────────────────────────────────────
    {
        "name": "APOE",
        "chr": "19",
        "lead_bp": 45411941,
        "lead_rsid": "rs429358",
        "trait": "mortality",
        "build": "hg19",
    },
    {
        "name": "TBXAS1",
        "chr": "7",
        "lead_bp": 139563129,
        "lead_rsid": "rs4726467",
        "trait": "mortality",
        "build": "hg19",
    },
    {
        "name": "SYT10",
        "chr": "12",
        "lead_bp": 33586576,
        "lead_rsid": "rs10437796",
        "trait": "mortality",
        "build": "hg19",
        "significance": "suggestive",
    },
    # ── HY3+ motor progression (Table 3) ─────────────────────────────────
    {
        "name": "MORN1",
        "chr": "1",
        "lead_bp": 2308517,
        "lead_rsid": "rs115217673",
        "trait": "hy3",
        "build": "hg19",
    },
    {
        "name": "ASNS",
        "chr": "7",
        "lead_bp": 97478547,
        "lead_rsid": "rs145274312",
        "trait": "hy3",
        "build": "hg19",
    },
    {
        "name": "PDE5A",
        "chr": "4",
        "lead_bp": 120416730,
        "lead_rsid": "rs113120976",
        "trait": "hy3",
        "build": "hg19",
    },
    {
        "name": "XPO1",
        "chr": "2",
        "lead_bp": 61709726,
        "lead_rsid": "rs141421624",
        "trait": "hy3",
        "build": "hg19",
    },
]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _parse_chrpos_markers(df: pd.DataFrame) -> pd.DataFrame:
    """If the rsid column contains chr:pos strings, split into chr and pos."""
    if "rsid" not in df.columns or df.empty:
        return df
    sample = df["rsid"].dropna().iloc[:50]
    if not sample.str.contains(":").all():
        return df
    parts = df["rsid"].str.split(":", n=1, expand=True)
    df["chr"] = parts[0].astype(str)
    df["pos"] = pd.to_numeric(parts[1], errors="coerce")
    return df


def extract_locus(
    sumstats_df: pd.DataFrame,
    locus: dict,
    window: int = 500_000,
) -> pd.DataFrame:
    """Return the sumstats slice for a given locus ± window, sorted by position.

    Parameters
    ----------
    sumstats_df : pd.DataFrame
        Full summary statistics with canonical columns (chr, pos, …).
    locus : dict
        One element of :data:`LOCI`.
    window : int
        Half-window in base pairs around ``lead_bp`` (default 500 kb).

    Returns
    -------
    pd.DataFrame
        Subset of *sumstats_df* for the locus region, sorted by ``pos``.
    """
    chrom = str(locus["chr"])
    start = locus["lead_bp"] - window
    end = locus["lead_bp"] + window

    mask = (
        (sumstats_df["chr"].astype(str) == chrom)
        & (sumstats_df["pos"] >= start)
        & (sumstats_df["pos"] <= end)
    )
    region = sumstats_df.loc[mask].copy()
    region = region.sort_values("pos").reset_index(drop=True)
    return region


def _find_sumstats_file(trait: str) -> Path | None:
    """Find a summary-stats file in RAW_DIR for the given trait."""
    if trait == "mortality":
        patterns = ["*MORTALITY*META*.tbl", "*mortality*meta*.tbl"]
    elif trait == "hy3":
        patterns = ["*HY3*META*.tbl", "*hy3*meta*.tbl"]
    else:
        patterns = [f"*{trait}*"]
    for pat in patterns:
        hits = sorted(RAW_DIR.glob(pat))
        if hits:
            return hits[0]
    return None


def extract_all(window: int = 500_000) -> list[Path]:
    """Load both trait sumstats, extract every locus, save to parquet.

    Saves each locus to ``data/loci/{name}_{trait}.parquet`` and prints
    the variant count per locus.

    Returns
    -------
    list[Path]
        Paths to the written parquet files.
    """
    LOCI_DIR.mkdir(parents=True, exist_ok=True)

    # Group loci by trait so we only load each file once
    traits = sorted({loc["trait"] for loc in LOCI})
    trait_dfs: dict[str, pd.DataFrame] = {}

    for trait in traits:
        path = _find_sumstats_file(trait)
        if path is None:
            logger.warning(
                "No summary-stats file found for trait '%s' in %s — skipping",
                trait,
                RAW_DIR,
            )
            print(f"  ! No file found for trait '{trait}' in {RAW_DIR}")
            continue
        logger.info("Loading %s sumstats from %s", trait, path.name)
        print(f"  Loading {trait}: {path.name}")
        df = load_sumstats(path)
        df = _parse_chrpos_markers(df)
        if "chr" not in df.columns or "pos" not in df.columns:
            logger.error(
                "File %s has no chr/pos columns after parsing — cannot extract loci",
                path.name,
            )
            print(f"  ! {path.name}: missing chr/pos columns, skipping")
            continue
        trait_dfs[trait] = df

    # Extract each locus
    written: list[Path] = []
    print(f"\n  {'Locus':<10s}  {'Trait':<12s}  {'Region':>30s}  {'Variants':>10s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*30}  {'-'*10}")

    for locus in LOCI:
        trait = locus["trait"]
        if trait not in trait_dfs:
            print(f"  {locus['name']:<10s}  {trait:<12s}  {'— no data —':>30s}  {'—':>10s}")
            continue

        region = extract_locus(trait_dfs[trait], locus, window=window)
        start = locus["lead_bp"] - window
        end = locus["lead_bp"] + window
        region_str = f"chr{locus['chr']}:{start:,}-{end:,}"
        n_variants = len(region)

        out_path = LOCI_DIR / f"{locus['name']}_{trait}.parquet"
        region.to_parquet(out_path, index=False)
        written.append(out_path)

        print(f"  {locus['name']:<10s}  {trait:<12s}  {region_str:>30s}  {n_variants:>10,d}")
        logger.info(
            "Locus %s (%s): %d variants, saved to %s",
            locus["name"],
            trait,
            n_variants,
            out_path.name,
        )

    return written


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
        description="Extract genomic loci from PD progression GWAS summary statistics.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500_000,
        help="Half-window in bp around each lead SNP (default: 500000).",
    )
    args = parser.parse_args()

    print(f"Extracting {len(LOCI)} loci (±{args.window / 1e6:.1f} Mb) …\n")
    paths = extract_all(window=args.window)
    print(f"\nDone — {len(paths)} parquet file(s) written to {LOCI_DIR}")


if __name__ == "__main__":
    main()
