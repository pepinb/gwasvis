"""Evaluate and summarise fine-mapping results, including annotation overlap.

Loads every ``data/finemap/*.parquet``, cross-references with the paper's
lead SNPs, and produces a one-row-per-locus summary table.

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.finemap import LOCUS_L
from src.loci import LOCI

logger = logging.getLogger(__name__)

FINEMAP_DIR = Path(__file__).resolve().parent.parent / "data" / "finemap"

# Quick lookup: (locus_name, trait) → paper lead rsid
_PAPER_LEADS: dict[tuple[str, str], str] = {
    (loc["name"], loc["trait"]): loc["lead_rsid"] for loc in LOCI
}

# Also look up by chr:pos for mortality loci
_PAPER_LEAD_CHRPOS: dict[tuple[str, str], str] = {
    (loc["name"], loc["trait"]): loc.get("lead_chrpos", f"{loc['chr']}:{loc['lead_bp']}")
    for loc in LOCI
}


def _find_paper_lead(df: pd.DataFrame, name: str, trait: str) -> pd.Series | None:
    """Look up the paper's lead SNP by rsid first, then by chr:pos."""
    key = (name, trait)
    # Try rsid
    rsid = _PAPER_LEADS.get(key)
    if rsid and "rsid" in df.columns:
        rows = df[df["rsid"] == rsid]
        if not rows.empty:
            return rows.iloc[0]
    # Try chr:pos
    chrpos = _PAPER_LEAD_CHRPOS.get(key)
    if chrpos and "rsid" in df.columns:
        rows = df[df["rsid"] == chrpos]
        if not rows.empty:
            return rows.iloc[0]
    # Try by position
    locus_def = next((l for l in LOCI if l["name"] == name and l["trait"] == trait), None)
    if locus_def and "pos" in df.columns:
        rows = df[df["pos"] == locus_def["lead_bp"]]
        if not rows.empty:
            return rows.iloc[0]
    return None


def summarize_finemap() -> pd.DataFrame:
    """Load all fine-mapping parquets and build a summary table.

    Columns:
        locus, trait, n_vars, L, converged, top_pip, top_variant,
        cs_size, paper_lead_in_cs, paper_lead_pip

    The table is saved to ``data/finemap/baseline_summary.csv`` and
    returned as a DataFrame.
    """
    rows: list[dict] = []

    for locus in LOCI:
        name = locus["name"]
        trait = locus["trait"]
        L = LOCUS_L.get((name, trait), 1)
        path = FINEMAP_DIR / f"{name}_{trait}.parquet"

        if not path.exists():
            logger.warning("No finemap result for %s_%s", name, trait)
            rows.append({
                "locus": name, "trait": trait, "n_vars": None, "L": L,
                "converged": None, "top_pip": None, "top_variant": None,
                "cs_size": None, "paper_lead_in_cs": None, "paper_lead_pip": None,
            })
            continue

        df = pd.read_parquet(path)
        n_variants = len(df)

        # Credible sets
        cs_ids = sorted(df.loc[df["in_cs"], "cs_id"].unique())
        n_cs = len(cs_ids)
        cs_size = int((df["cs_id"] == cs_ids[0]).sum()) if n_cs > 0 else 0

        # Top PIP variant
        top_idx = df["pip"].idxmax()
        top_pip = float(df.loc[top_idx, "pip"])
        top_variant = str(df.loc[top_idx, "rsid"]) if "rsid" in df.columns else "?"

        # Paper lead SNP
        lead_row = _find_paper_lead(df, name, trait)
        if lead_row is not None:
            paper_in_cs = bool(lead_row["in_cs"])
            paper_lead_pip = round(float(lead_row["pip"]), 4)
        else:
            paper_in_cs = False
            paper_lead_pip = None

        converged = bool(df["converged"].iloc[0]) if "converged" in df.columns else None

        # Locus note (e.g. paper lead dropped by MAF filter)
        locus_note = None
        if "locus_note" in df.columns:
            note_vals = df["locus_note"].dropna().unique()
            if len(note_vals) > 0:
                locus_note = str(note_vals[0])

        rows.append({
            "locus": name,
            "trait": trait,
            "n_vars": n_variants,
            "L": L,
            "converged": converged,
            "top_pip": round(top_pip, 4),
            "top_variant": top_variant,
            "cs_size": cs_size,
            "paper_lead_in_cs": paper_in_cs,
            "paper_lead_pip": paper_lead_pip,
            "locus_note": locus_note,
        })

    summary = pd.DataFrame(rows)

    # Save
    FINEMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINEMAP_DIR / "baseline_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Saved summary to %s", out_path)

    # Print as markdown table
    print("\n## Fine-mapping summary\n")
    print(summary.to_markdown(index=False))
    print(f"\nSaved to {out_path}")

    return summary


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
        description="Summarise fine-mapping results across all loci.",
    )
    parser.parse_args()

    summarize_finemap()


if __name__ == "__main__":
    main()
