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

from src.finemap import LOCUS_L, _get_1kg_eur_af
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

# Paper-reported p-values from Tables 2 (mortality) & 3 (HY3) of Tan et al. 2024.
# Hardcoded because dropped variants can't be recovered from the pipeline.
_PAPER_PVALS: dict[tuple[str, str], float] = {
    ("APOE", "mortality"): 1.35e-10,
    ("TBXAS1", "mortality"): 7.71e-10,
    ("SYT10", "mortality"): 5.31e-8,
    ("MORN1", "hy3"): 3.09e-9,
    ("ASNS", "hy3"): 3.49e-9,
    ("PDE5A", "hy3"): 6.95e-9,
    ("XPO1", "hy3"): 3.08e-8,
}

# ---------------------------------------------------------------------------
# Locus status taxonomy
# ---------------------------------------------------------------------------

REPLICATED = "REPLICATED"
CAUTION = "CAUTION"
NOT_TESTABLE = "NOT_TESTABLE"
FAILED = "FAILED"


def classify_locus(
    *,
    converged: bool | None,
    paper_lead_in_cs: bool,
    paper_lead_pip: float | None,
    locus_note: str | None,
) -> tuple[str, str]:
    """Classify a locus into one of four status categories.

    Returns (status, reason) tuple.
    """
    # FAILED: SuSiE did not converge or no result
    if converged is None or converged is False:
        return FAILED, "SuSiE did not converge or no result available."

    # NOT_TESTABLE: paper lead dropped before fine-mapping
    if locus_note:
        return NOT_TESTABLE, locus_note

    # REPLICATED: paper lead in credible set with PIP >= 0.1
    if paper_lead_in_cs and paper_lead_pip is not None and paper_lead_pip >= 0.1:
        return REPLICATED, ""

    # CAUTION: everything else (lead present but weak/absent from CS)
    parts: list[str] = []
    if not paper_lead_in_cs:
        parts.append("paper lead not in any credible set")
    if paper_lead_pip is not None and paper_lead_pip < 0.1:
        parts.append(f"paper lead PIP = {paper_lead_pip:.4f} (< 0.1)")
    return CAUTION, "SuSiE converged but " + "; ".join(parts) + "."


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
        locus, trait, paper_lead, paper_pval, n_vars, L, converged,
        top_pip, top_variant, cs_size, paper_lead_in_cs, paper_lead_pip,
        reference_maf, locus_status, reason, locus_note

    The table is saved to ``data/finemap/baseline_summary.csv`` and
    returned as a DataFrame.
    """
    rows: list[dict] = []

    for locus in LOCI:
        name = locus["name"]
        trait = locus["trait"]
        chrom = str(locus["chr"])
        lead_pos = locus["lead_bp"]
        lead_rsid = locus["lead_rsid"]
        L = LOCUS_L.get((name, trait), 1)
        paper_pval = _PAPER_PVALS.get((name, trait))
        path = FINEMAP_DIR / f"{name}_{trait}.parquet"

        if not path.exists():
            logger.warning("No finemap result for %s_%s", name, trait)
            rows.append({
                "locus": name, "trait": trait, "paper_lead": lead_rsid,
                "paper_pval": paper_pval,
                "n_vars": None, "L": L,
                "converged": None, "top_pip": None, "top_variant": None,
                "cs_size": None, "paper_lead_in_cs": None, "paper_lead_pip": None,
                "reference_maf": None, "locus_status": FAILED, "reason": "No finemap result.",
                "locus_note": None,
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

        # Reference MAF for paper lead
        logger.info("Querying 1KG EUR AF for %s %s (chr%s:%d)…", name, lead_rsid, chrom, lead_pos)
        ref_maf = _get_1kg_eur_af(chrom, lead_pos)

        # Classify status
        status, reason = classify_locus(
            converged=converged,
            paper_lead_in_cs=paper_in_cs,
            paper_lead_pip=paper_lead_pip,
            locus_note=locus_note,
        )

        rows.append({
            "locus": name,
            "trait": trait,
            "paper_lead": lead_rsid,
            "paper_pval": paper_pval,
            "n_vars": n_variants,
            "L": L,
            "converged": converged,
            "top_pip": round(top_pip, 4),
            "top_variant": top_variant,
            "cs_size": cs_size,
            "paper_lead_in_cs": paper_in_cs,
            "paper_lead_pip": paper_lead_pip,
            "reference_maf": ref_maf,
            "locus_status": status,
            "reason": reason if reason else None,
            "locus_note": locus_note,
        })

    summary = pd.DataFrame(rows)

    # Save
    FINEMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINEMAP_DIR / "baseline_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Saved summary to %s", out_path)

    # Print as markdown table (compact columns for terminal)
    display_cols = [
        "locus", "trait", "paper_lead", "locus_status", "top_pip",
        "top_variant", "cs_size", "paper_lead_in_cs", "paper_lead_pip",
        "reference_maf",
    ]
    print("\n## Fine-mapping summary\n")
    print(summary[display_cols].to_markdown(index=False))
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
