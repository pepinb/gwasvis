"""Compute or retrieve linkage disequilibrium matrices for fine-mapping regions.

Uses 1000 Genomes Phase 3 EUR samples (hg19) as the LD reference panel.
Requires plink2 on $PATH.

Workflow:
    1. download_1kg_eur()   — fetch pgen/pvar/psam, filter to EUR
    2. build_locus_ld()     — extract locus variants, compute LD, align to sumstats

Usage:
    python -m src.ld --download             # download 1KG EUR reference
    python -m src.ld --locus APOE --trait mortality   # build LD for one locus
    python -m src.ld --all                  # build LD for every locus
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.loci import LOCI, LOCI_DIR

logger = logging.getLogger(__name__)

LD_DIR = Path(__file__).resolve().parent.parent / "data" / "ld"
REF_DIR = LD_DIR / "1kg_eur_hg19"

# ---------------------------------------------------------------------------
# 1000 Genomes Phase 3 URLs (plink2 format, hg19)
# https://www.cog-genomics.org/plink/2.0/resources#1kg_phase3
# ---------------------------------------------------------------------------
_1KG_FILES = {
    "all_phase3.pgen.zst": "https://www.dropbox.com/s/afvvf1e15gqzsqo/all_phase3.pgen.zst?dl=1",
    "all_phase3.pvar.zst": "https://www.dropbox.com/s/op9osq6luy3pjg8/all_phase3.pvar.zst?dl=1",
    "phase3_corrected.psam": "https://www.dropbox.com/s/yozrzsdrwqej63q/phase3_corrected.psam?dl=1",
}

# Ambiguous-strand allele pairs (complement is the same pair)
_AMBIGUOUS_PAIRS = frozenset({("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")})


# ---------------------------------------------------------------------------
# plink2 helpers
# ---------------------------------------------------------------------------

def _plink2_path() -> str:
    """Return the plink2 binary path, or exit with install instructions."""
    path = shutil.which("plink2")
    if path is None:
        print(
            "\n  plink2 is not installed or not on $PATH.\n"
            "\n"
            "  Install instructions:\n"
            "    macOS (Homebrew):  brew install plink2\n"
            "    Ubuntu / Debian:   sudo apt-get install plink2\n"
            "    Conda:             conda install -c bioconda plink2\n"
            "    Manual:            https://www.cog-genomics.org/plink/2.0/\n",
        )
        sys.exit(1)
    return path


def _run_plink2(args: list[str], description: str = "") -> subprocess.CompletedProcess:
    """Run plink2 with *args*, logging the command and checking for errors."""
    cmd = [_plink2_path()] + args
    logger.info("Running: %s", " ".join(cmd))
    if description:
        print(f"  plink2: {description}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("plink2 stderr:\n%s", result.stderr)
        raise RuntimeError(f"plink2 failed (exit {result.returncode}):\n{result.stderr}")
    return result


# ---------------------------------------------------------------------------
# Step 1: Download 1000 Genomes EUR reference
# ---------------------------------------------------------------------------

def _stream_download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress indicator (requests-based)."""
    import requests

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(
                    f"\r    [{('#' * int(pct // 2)):<50s}] {pct:5.1f}%",
                    end="",
                    flush=True,
                )
    print()


def download_1kg_eur() -> Path:
    """Download 1KG Phase 3 plink2 files and filter to EUR samples.

    Files are saved to ``data/ld/1kg_eur_hg19/``.  If the final EUR-only
    pgen already exists the download is skipped.

    Returns
    -------
    Path
        Directory containing EUR-only plink2 fileset.
    """
    _plink2_path()  # fail fast if plink2 missing
    REF_DIR.mkdir(parents=True, exist_ok=True)

    eur_pgen = REF_DIR / "1kg_eur_hg19.pgen"
    if eur_pgen.exists():
        print(f"  EUR reference already exists at {REF_DIR}")
        return REF_DIR

    # -- Download raw files ---------------------------------------------------
    for fname, url in _1KG_FILES.items():
        dest = REF_DIR / fname
        if dest.exists():
            print(f"  ✓ {fname} — already downloaded")
            continue
        print(f"  ↓ {fname}")
        _stream_download(url, dest)

    # -- Decompress pgen.zst --------------------------------------------------
    raw_pgen = REF_DIR / "all_phase3.pgen"
    if not raw_pgen.exists():
        print("  Decompressing all_phase3.pgen.zst …")
        _run_plink2(
            ["--zst-decompress", str(REF_DIR / "all_phase3.pgen.zst"),
             str(raw_pgen)],
            description="decompress pgen",
        )

    # -- Create EUR keep-list from psam ---------------------------------------
    psam = REF_DIR / "phase3_corrected.psam"
    eur_keep = REF_DIR / "eur_samples.txt"
    if not eur_keep.exists():
        psam_df = pd.read_csv(psam, sep="\t")
        # psam has columns: #IID, SEX, SuperPop, Population
        # for --keep we need FID IID (plink2 with --no-fid just needs IID)
        eur = psam_df.loc[psam_df["SuperPop"] == "EUR", "#IID"]
        eur.to_csv(eur_keep, index=False, header=False)
        print(f"  Wrote {len(eur)} EUR sample IDs to {eur_keep.name}")

    # -- Symlink psam so plink2 can find the fileset as "all_phase3" ----------
    expected_psam = REF_DIR / "all_phase3.psam"
    if not expected_psam.exists():
        expected_psam.symlink_to(psam.name)

    # -- Filter to EUR, autosomes, biallelic SNPs, MAF > 0.01 ----------------
    _run_plink2(
        [
            "--pfile", str(REF_DIR / "all_phase3"), "vzs",
            "--keep", str(eur_keep),
            "--autosome",
            "--max-alleles", "2",
            "--min-alleles", "2",
            "--maf", "0.01",
            "--snps-only",
            "--make-pgen",
            "--out", str(REF_DIR / "1kg_eur_hg19"),
        ],
        description="filter to EUR, autosomes, biallelic SNPs, MAF > 1%",
    )

    print(f"  EUR reference panel ready at {REF_DIR}")
    return REF_DIR


# ---------------------------------------------------------------------------
# Step 2: Build LD matrix for a locus
# ---------------------------------------------------------------------------

def _load_locus_sumstats(locus_name: str, trait: str) -> pd.DataFrame:
    """Load the parquet for a locus and ensure required columns exist."""
    path = LOCI_DIR / f"{locus_name}_{trait}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Locus file not found: {path}\n"
            "Run: python -m src.loci"
        )
    df = pd.read_parquet(path)
    for col in ("rsid", "chr", "pos", "a1", "a2", "beta", "se", "pval"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df


def _is_ambiguous(a1: str, a2: str) -> bool:
    return (a1.upper(), a2.upper()) in _AMBIGUOUS_PAIRS


def _alleles_match(sum_a1: str, sum_a2: str, ref_a1: str, ref_a2: str) -> str:
    """Compare sumstats alleles to reference. Returns 'match', 'flip', or 'drop'.

    'match' — same orientation (a1/a2 agree).
    'flip'  — alleles are swapped: sumstats a1 == ref a2 and vice versa.
    'drop'  — alleles incompatible (tri-allelic, indel mismatch, etc.).
    """
    s1, s2 = sum_a1.upper(), sum_a2.upper()
    r1, r2 = ref_a1.upper(), ref_a2.upper()
    if s1 == r1 and s2 == r2:
        return "match"
    if s1 == r2 and s2 == r1:
        return "flip"
    return "drop"


def build_locus_ld(
    locus_name: str,
    trait: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Compute an LD matrix for *locus_name* / *trait* and align to sumstats.

    Steps:
        1. Load ``data/loci/{locus_name}_{trait}.parquet``
        2. Write variant list, run plink2 ``--r-unphased square``
        3. Read the LD matrix and variant IDs
        4. Align rows/cols to the sumstats, handling allele flips and
           dropping ambiguous-strand SNPs (A/T, C/G)

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, list[str]]
        (aligned_sumstats, ld_matrix, snp_order)
        All three are aligned: ``len(df) == ld.shape[0] == ld.shape[1]``.
    """
    plink2 = _plink2_path()
    ref_prefix = REF_DIR / "1kg_eur_hg19"
    if not ref_prefix.with_suffix(".pgen").exists():
        raise FileNotFoundError(
            f"EUR reference not found at {ref_prefix}.pgen\n"
            "Run: python -m src.ld --download"
        )

    sumstats = _load_locus_sumstats(locus_name, trait)
    logger.info(
        "Locus %s_%s: %d variants in sumstats", locus_name, trait, len(sumstats)
    )

    # -- Drop ambiguous-strand SNPs from sumstats before extraction -----------
    ambig_mask = sumstats.apply(
        lambda r: _is_ambiguous(str(r["a1"]), str(r["a2"])), axis=1
    )
    n_ambig = ambig_mask.sum()
    if n_ambig:
        logger.info("Dropping %d ambiguous-strand SNPs (A/T or C/G)", n_ambig)
        sumstats = sumstats.loc[~ambig_mask].reset_index(drop=True)

    with tempfile.TemporaryDirectory(prefix="ld_") as tmpdir:
        tmp = Path(tmpdir)

        # -- Write SNP list for --extract ------------------------------------
        snp_list = tmp / "snps.txt"
        snp_list.write_text("\n".join(sumstats["rsid"].dropna().unique()))

        # -- Run plink2 to compute LD ----------------------------------------
        out_prefix = tmp / "ld_out"
        _run_plink2(
            [
                "--pfile", str(ref_prefix),
                "--extract", str(snp_list),
                "--r-unphased", "square",
                "--out", str(out_prefix),
            ],
            description=f"LD matrix for {locus_name}_{trait}",
        )

        # -- Locate output files ---------------------------------------------
        # plink2 writes: {prefix}.unphased.vcor1 (matrix)
        #                {prefix}.unphased.vcor1.vars (variant IDs)
        vcor_file = out_prefix.with_suffix(".unphased.vcor1")
        vars_file = Path(str(vcor_file) + ".vars")

        if not vcor_file.exists() or not vars_file.exists():
            # Fallback: scan tmpdir for any .vcor* file
            vcor_candidates = list(tmp.glob("*.vcor*"))
            vars_candidates = list(tmp.glob("*.vars"))
            logger.warning(
                "Expected output not found. vcor candidates: %s, vars: %s",
                vcor_candidates, vars_candidates,
            )
            if not vcor_candidates:
                raise FileNotFoundError(
                    f"plink2 LD output not found in {tmpdir}. "
                    "Check plink2 version (need v2.0+)."
                )
            vcor_file = vcor_candidates[0]
            vars_file = vars_candidates[0] if vars_candidates else None

        # -- Read LD matrix ---------------------------------------------------
        ld_raw = np.loadtxt(vcor_file, dtype=np.float64)
        if ld_raw.ndim == 1:
            ld_raw = ld_raw.reshape(1, 1)

        # -- Read variant order -----------------------------------------------
        if vars_file and vars_file.exists():
            ref_snps = vars_file.read_text().strip().split("\n")
        else:
            raise FileNotFoundError("No .vars file found — cannot determine SNP order")

        assert ld_raw.shape == (len(ref_snps), len(ref_snps)), (
            f"LD shape {ld_raw.shape} doesn't match {len(ref_snps)} variants"
        )

    # -- Read the pvar to get reference alleles for the extracted SNPs --------
    pvar_path = ref_prefix.with_suffix(".pvar")
    pvar = pd.read_csv(pvar_path, sep="\t", comment="#",
                       names=["chr", "pos", "rsid", "ref", "alt"],
                       usecols=[0, 1, 2, 3, 4], dtype=str)
    pvar_lookup = pvar.set_index("rsid")[["ref", "alt"]].to_dict("index")

    # -- Align: intersect sumstats ↔ ref, handle allele flips ----------------
    ref_set = set(ref_snps)
    ref_idx = {snp: i for i, snp in enumerate(ref_snps)}

    keep_rows: list[int] = []        # indices into sumstats
    keep_ref_idx: list[int] = []     # indices into ld_raw / ref_snps
    flip_positions: list[int] = []   # positions in the *output* arrays to flip
    n_match = n_flip = n_drop = 0

    for i, row in sumstats.iterrows():
        rsid = row["rsid"]
        if rsid not in ref_set:
            continue
        # Look up reference alleles
        ref_alleles = pvar_lookup.get(rsid)
        if ref_alleles is None:
            continue
        action = _alleles_match(
            str(row["a1"]), str(row["a2"]),
            ref_alleles["ref"], ref_alleles["alt"],
        )
        if action == "drop":
            n_drop += 1
            continue
        keep_rows.append(i)
        keep_ref_idx.append(ref_idx[rsid])
        if action == "flip":
            flip_positions.append(len(keep_rows) - 1)
            n_flip += 1
        else:
            n_match += 1

    logger.info(
        "Allele alignment: %d match, %d flipped, %d dropped, %d not in ref",
        n_match, n_flip, n_drop, len(sumstats) - n_match - n_flip - n_drop,
    )
    print(
        f"    Alleles: {n_match} match, {n_flip} flipped, {n_drop} dropped, "
        f"{len(sumstats) - n_match - n_flip - n_drop} not in ref"
    )

    if not keep_rows:
        raise ValueError(
            f"No overlapping SNPs between sumstats and reference for "
            f"{locus_name}_{trait}"
        )

    # -- Subset and align -----------------------------------------------------
    aligned_ss = sumstats.iloc[keep_rows].reset_index(drop=True)
    ld_sub = ld_raw[np.ix_(keep_ref_idx, keep_ref_idx)]
    snp_order = [ref_snps[j] for j in keep_ref_idx]

    # -- Flip z-scores for swapped alleles ------------------------------------
    # z = beta / se; flipping the effect allele negates z
    if flip_positions:
        z = aligned_ss["beta"].values / aligned_ss["se"].values
        for pos in flip_positions:
            z[pos] = -z[pos]
        aligned_ss = aligned_ss.copy()
        aligned_ss["z"] = z
        logger.info("Flipped z-scores at %d positions", len(flip_positions))
    else:
        aligned_ss = aligned_ss.copy()
        aligned_ss["z"] = aligned_ss["beta"].values / aligned_ss["se"].values

    # -- Save outputs ---------------------------------------------------------
    LD_DIR.mkdir(parents=True, exist_ok=True)
    ld_out = LD_DIR / f"{locus_name}_{trait}.ld"
    np.savetxt(ld_out, ld_sub, fmt="%.6f", delimiter="\t")

    snp_out = LD_DIR / f"{locus_name}_{trait}.snplist"
    snp_out.write_text("\n".join(snp_order) + "\n")

    ss_out = LD_DIR / f"{locus_name}_{trait}.aligned.parquet"
    aligned_ss.to_parquet(ss_out, index=False)

    logger.info(
        "Saved LD (%d x %d) to %s", ld_sub.shape[0], ld_sub.shape[1], ld_out
    )

    # -- Final consistency check ----------------------------------------------
    assert len(aligned_ss) == ld_sub.shape[0] == ld_sub.shape[1], (
        f"Dimension mismatch: df={len(aligned_ss)}, "
        f"ld={ld_sub.shape[0]}x{ld_sub.shape[1]}"
    )

    print(
        f"    LD matrix: {ld_sub.shape[0]} x {ld_sub.shape[1]} "
        f"→ {ld_out.name}"
    )

    return aligned_ss, ld_sub, snp_order


def build_all_ld() -> None:
    """Build LD matrices for every locus defined in LOCI."""
    for locus in LOCI:
        name, trait = locus["name"], locus["trait"]
        parquet = LOCI_DIR / f"{name}_{trait}.parquet"
        if not parquet.exists():
            print(f"  ! {name}_{trait}: locus parquet not found, skipping")
            continue
        print(f"  {name} ({trait})")
        try:
            build_locus_ld(name, trait)
        except Exception as exc:
            logger.error("Failed for %s_%s: %s", name, trait, exc)
            print(f"    ERROR: {exc}")


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
        description="Build LD matrices from 1000 Genomes EUR for fine-mapping.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and prepare the 1KG EUR reference panel.",
    )
    parser.add_argument("--locus", type=str, help="Locus name (e.g. APOE).")
    parser.add_argument("--trait", type=str, help="Trait name (mortality or hy3).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build LD for all loci.",
    )
    args = parser.parse_args()

    if args.download:
        print("Downloading 1000 Genomes Phase 3 EUR reference …\n")
        download_1kg_eur()
        sys.exit(0)

    if args.all:
        print("Building LD matrices for all loci …\n")
        build_all_ld()
        sys.exit(0)

    if args.locus and args.trait:
        print(f"Building LD for {args.locus} ({args.trait}) …\n")
        df, ld, snps = build_locus_ld(args.locus, args.trait)
        print(f"\n  Result: {len(df)} variants, LD {ld.shape[0]}×{ld.shape[1]}")
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
