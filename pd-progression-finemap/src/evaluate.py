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

from src.loci import LOCI

logger = logging.getLogger(__name__)

FINEMAP_DIR = Path(__file__).resolve().parent.parent / "data" / "finemap"

# Quick lookup: (locus_name, trait) → paper lead rsid
_PAPER_LEADS: dict[tuple[str, str], str] = {
    (loc["name"], loc["trait"]): loc["lead_rsid"] for loc in LOCI
}


def summarize_finemap() -> pd.DataFrame:
    """Load all fine-mapping parquets and build a summary table.

    Columns:
        locus, trait, n_variants, n_credible_sets, cs_size,
        top_pip, top_rsid, paper_lead_rsid, paper_lead_in_cs, converged

    The table is saved to ``data/finemap/baseline_summary.csv`` and
    returned as a DataFrame.
    """
    rows: list[dict] = []

    for locus in LOCI:
        name = locus["name"]
        trait = locus["trait"]
        path = FINEMAP_DIR / f"{name}_{trait}.parquet"

        if not path.exists():
            logger.warning("No finemap result for %s_%s", name, trait)
            rows.append({
                "locus": name,
                "trait": trait,
                "n_variants": None,
                "n_credible_sets": None,
                "cs_size": None,
                "top_pip": None,
                "top_rsid": None,
                "paper_lead_rsid": locus["lead_rsid"],
                "paper_lead_in_cs": None,
                "converged": None,
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
        top_rsid = str(df.loc[top_idx, "rsid"]) if "rsid" in df.columns else "?"

        # Paper lead SNP
        paper_rsid = _PAPER_LEADS.get((name, trait), "?")
        paper_in_cs = False
        if "rsid" in df.columns:
            lead_rows = df[df["rsid"] == paper_rsid]
            if not lead_rows.empty:
                paper_in_cs = bool(lead_rows.iloc[0]["in_cs"])

        # Convergence flag (stored per-row, but uniform within a locus)
        converged = bool(df["converged"].iloc[0]) if "converged" in df.columns else None

        rows.append({
            "locus": name,
            "trait": trait,
            "n_variants": n_variants,
            "n_credible_sets": n_cs,
            "cs_size": cs_size,
            "top_pip": round(top_pip, 4),
            "top_rsid": top_rsid,
            "paper_lead_rsid": paper_rsid,
            "paper_lead_in_cs": paper_in_cs,
            "converged": converged,
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
