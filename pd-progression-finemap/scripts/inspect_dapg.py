#!/usr/bin/env python3
"""Inspect GTEx v8 DAP-G fine-mapping files for Whole_Blood and Brain_Cortex.

Produces a markdown inspection report at docs/dapg_inspection.md and prints
the same report to stdout.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "eqtl" / "raw"
OUT_MD = ROOT / "docs" / "dapg_inspection.md"

TISSUES = ["Whole_Blood", "Brain_Cortex"]
PD_CHROMS = {1, 2, 4, 7, 12, 19}  # held-out chromosomes for PD loci

PIP_THRESHOLDS = [
    ("> 0.9", lambda p: p > 0.9),
    ("> 0.5", lambda p: p > 0.5),
    ("> 0.1", lambda p: p > 0.1),
    ("> 0.01", lambda p: p > 0.01),
    ("≤ 0.01", lambda p: p <= 0.01),
]


def parse_variant_id(vid: str):
    """Parse chr{N}_{pos}_{ref}_{alt}_b38 → (chrom_int, pos, ref, alt)."""
    parts = vid.split("_")
    chrom_str = parts[0].replace("chr", "").replace("X", "23").replace("Y", "24")
    try:
        chrom = int(chrom_str)
    except ValueError:
        chrom = -1
    return chrom, parts[1], parts[2], parts[3]


def is_snv(ref: str, alt: str) -> bool:
    return len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT"


def inspect_tissue(tissue: str) -> str:
    path = RAW_DIR / f"{tissue}.variants_pip.txt.gz"
    if not path.exists():
        return f"### {tissue}\n\n**FILE NOT FOUND**: {path}\n"

    df = pd.read_csv(path, sep="\t", dtype={"cluster_id": str})

    lines = []
    lines.append(f"### {tissue.replace('_', ' ')}")
    lines.append("")
    lines.append(f"**File**: `{path.relative_to(ROOT)}`  ")
    lines.append(f"**Rows**: {len(df):,}  ")
    lines.append(f"**Unique variant_ids**: {df['variant_id'].nunique():,}  ")
    lines.append(f"**Unique genes**: {df['gene'].nunique():,}  ")
    lines.append("")

    # Schema
    lines.append("#### Column schema")
    lines.append("")
    lines.append("| Column | dtype |")
    lines.append("|--------|-------|")
    for col in df.columns:
        lines.append(f"| {col} | {df[col].dtype} |")
    lines.append("")

    # Variant ID examples
    examples = df["variant_id"].drop_duplicates().head(3).tolist()
    lines.append("#### Variant ID examples")
    lines.append("")
    for ex in examples:
        lines.append(f"- `{ex}`")
    lines.append("")

    # PIP distribution
    lines.append("#### PIP distribution")
    lines.append("")
    lines.append("| Threshold | Variants | % of total |")
    lines.append("|-----------|----------|------------|")
    for label, filt in PIP_THRESHOLDS:
        n = filt(df["pip"]).sum()
        pct = 100 * n / len(df)
        lines.append(f"| PIP {label} | {n:,} | {pct:.1f}% |")
    lines.append("")

    # Chromosome distribution
    parsed = df["variant_id"].apply(parse_variant_id)
    df["chrom"] = parsed.apply(lambda x: x[0])
    df["ref"] = parsed.apply(lambda x: x[2])
    df["alt"] = parsed.apply(lambda x: x[3])

    lines.append("#### Chromosome distribution (all variants)")
    lines.append("")
    chrom_counts = df["chrom"].value_counts().sort_index()
    lines.append("| Chr | Count | % |")
    lines.append("|-----|-------|---|")
    for ch, cnt in chrom_counts.items():
        pct = 100 * cnt / len(df)
        marker = " *" if ch in PD_CHROMS else ""
        lines.append(f"| {ch}{marker} | {cnt:,} | {pct:.1f}% |")
    lines.append("")
    lines.append("\\* = PD held-out chromosome")
    lines.append("")

    # PD chrom breakdown for pip > 0.5
    high_pip = df[df["pip"] > 0.5]
    pd_high = high_pip[high_pip["chrom"].isin(PD_CHROMS)]
    other_high = high_pip[~high_pip["chrom"].isin(PD_CHROMS)]

    lines.append("#### PIP > 0.5 by PD-held-out vs rest")
    lines.append("")
    lines.append("| Group | Count | % of PIP>0.5 |")
    lines.append("|-------|-------|--------------|")
    n_high = len(high_pip)
    if n_high > 0:
        lines.append(
            f"| PD chroms ({', '.join(str(c) for c in sorted(PD_CHROMS))}) "
            f"| {len(pd_high):,} | {100*len(pd_high)/n_high:.1f}% |"
        )
        lines.append(
            f"| Other chroms | {len(other_high):,} | {100*len(other_high)/n_high:.1f}% |"
        )
        lines.append(f"| **Total** | **{n_high:,}** | **100%** |")
    else:
        lines.append("| (none) | 0 | — |")
    lines.append("")

    # SNV vs indel
    df["is_snv"] = df.apply(lambda r: is_snv(r["ref"], r["alt"]), axis=1)
    n_snv = df["is_snv"].sum()
    n_indel = len(df) - n_snv
    lines.append("#### SNV vs indel")
    lines.append("")
    lines.append("| Type | Count | % |")
    lines.append("|------|-------|---|")
    lines.append(f"| SNV | {n_snv:,} | {100*n_snv/len(df):.1f}% |")
    lines.append(f"| Indel | {n_indel:,} | {100*n_indel/len(df):.1f}% |")
    lines.append("")

    return "\n".join(lines)


def main():
    sections = []
    sections.append("# DAP-G Fine-Mapping Inspection Report")
    sections.append("")
    sections.append(
        "GTEx v8 DAP-G eQTL credible sets (Zenodo 3517189). "
        "Two tissues extracted for PD progression fine-mapping."
    )
    sections.append("")

    for tissue in TISSUES:
        print(f"Processing {tissue}...", file=sys.stderr)
        sections.append(inspect_tissue(tissue))

    report = "\n".join(sections)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(report)
    print(report)
    print(f"\n--- Report saved to {OUT_MD} ---", file=sys.stderr)


if __name__ == "__main__":
    main()
