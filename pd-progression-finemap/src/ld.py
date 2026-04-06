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

# 1KG Phase 3 VCFs on EBI FTP (tabix-indexed, can stream regions via bcftools)
_1KG_VCF_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
_1KG_PANEL_URL = f"{_1KG_VCF_BASE}/integrated_call_samples_v3.20130502.ALL.panel"

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


def _bcftools_path() -> str:
    """Return the bcftools binary path, or exit with install instructions."""
    path = shutil.which("bcftools")
    if path is None:
        print(
            "\n  bcftools is not installed or not on $PATH.\n"
            "\n"
            "  Install instructions:\n"
            "    macOS (Homebrew):  brew install bcftools\n"
            "    Ubuntu / Debian:   sudo apt-get install bcftools\n"
            "    Conda:             conda install -c bioconda bcftools\n",
        )
        sys.exit(1)
    return path


def _get_eur_samples() -> list[str]:
    """Download the 1KG panel file and return EUR sample IDs."""
    import requests
    panel_path = REF_DIR / "1kg_panel.txt"
    if not panel_path.exists():
        REF_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_1KG_PANEL_URL, timeout=60)
        resp.raise_for_status()
        panel_path.write_text(resp.text)
    df = pd.read_csv(panel_path, sep="\t")
    return df.loc[df["super_pop"] == "EUR", "sample"].tolist()


def _extract_region_vcf(chrom: str, start: int, end: int, out_vcf: Path) -> None:
    """Use bcftools to stream a region from the remote 1KG VCF."""
    bcftools = _bcftools_path()
    vcf_url = f"{_1KG_VCF_BASE}/ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    region = f"{chrom}:{start}-{end}"

    logger.info("Extracting region %s from remote 1KG VCF", region)
    cmd = [
        bcftools, "view",
        "--regions", region,
        "--types", "snps",
        "--min-alleles", "2",
        "--max-alleles", "2",
        "-O", "z",
        "-o", str(out_vcf),
        vcf_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"bcftools failed:\n{result.stderr}")
    # Index the output
    subprocess.run(["bcftools", "index", str(out_vcf)], check=True, capture_output=True)


def build_locus_ld_from_vcf(
    locus_name: str,
    trait: str,
    window: int = 500_000,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build LD matrix for a locus using remote 1KG VCFs via bcftools.

    This is an alternative to build_locus_ld() that doesn't require
    downloading the full 1KG pgen reference. It streams just the
    region we need from EBI FTP.
    """
    from src.loci import LOCI, LOCI_DIR

    plink2 = _plink2_path()
    bcftools = _bcftools_path()

    # Load locus sumstats
    sumstats = _load_locus_sumstats(locus_name, trait)
    logger.info("Locus %s_%s: %d variants in sumstats", locus_name, trait, len(sumstats))

    # Find locus definition for the region bounds
    locus = None
    for loc in LOCI:
        if loc["name"] == locus_name and loc["trait"] == trait:
            locus = loc
            break
    if locus is None:
        raise ValueError(f"Unknown locus: {locus_name}_{trait}")

    chrom = str(locus["chr"])
    start = max(1, locus["lead_bp"] - window)
    end = locus["lead_bp"] + window

    # Note: ambiguous-strand SNPs are handled inside harmonize_alleles()

    with tempfile.TemporaryDirectory(prefix="ld_vcf_") as tmpdir:
        tmp = Path(tmpdir)

        # Step 1: Get EUR sample list
        eur_samples = _get_eur_samples()
        keep_file = tmp / "eur_samples.txt"
        keep_file.write_text("\n".join(eur_samples) + "\n")
        print(f"    EUR samples: {len(eur_samples)}")

        # Step 2: Extract region from remote VCF, filter to EUR, biallelic SNPs
        region_vcf = tmp / "region.vcf.gz"
        print(f"    Streaming chr{chrom}:{start:,}-{end:,} from 1KG EBI FTP…")
        _extract_region_vcf(chrom, start, end, region_vcf)

        # Step 3: Filter to EUR samples only
        eur_vcf = tmp / "region_eur.vcf.gz"
        cmd = [
            bcftools, "view",
            "--samples-file", str(keep_file),
            "--min-ac", "1",  # drop monomorphic in EUR
            "-O", "z",
            "-o", str(eur_vcf),
            str(region_vcf),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Step 4: Convert to plink2 format
        # Use chr:pos:ref:alt as variant ID since 1KG VCFs lack rsids
        plink_prefix = tmp / "region_eur"
        _run_plink2(
            [
                "--vcf", str(eur_vcf),
                "--maf", "0.01",
                "--snps-only",
                "--set-all-var-ids", "@:#:$r:$a",
                "--make-pgen",
                "--out", str(plink_prefix),
            ],
            description=f"convert VCF to plink2 for {locus_name}",
        )

        # Step 5: Read pvar to build position-based lookups
        # Variant IDs in pvar are now chr:pos:ref:alt (set by --set-all-var-ids)
        pvar_path = plink_prefix.with_suffix(".pvar")
        pvar = pd.read_csv(pvar_path, sep="\t", comment="#",
                           names=["chr", "pos", "varid", "ref", "alt"],
                           usecols=[0, 1, 2, 3, 4], dtype=str)
        pvar["chr"] = pvar["chr"].astype(str)

        # Build position→(ref, alt) lookup for harmonize_alleles
        pvar_alleles: dict[str, tuple[str, str]] = {}
        pvar_varid_by_pos: dict[str, str] = {}
        for _, prow in pvar.iterrows():
            pk = f"{prow['chr']}:{prow['pos']}"
            pvar_alleles[pk] = (prow["ref"], prow["alt"])
            pvar_varid_by_pos[pk] = prow["varid"]

        # Find overlapping variants by position for --extract
        sumstats_pos_keys = (
            sumstats["chr"].astype(str) + ":" + sumstats["pos"].astype(int).astype(str)
        ).values
        extract_varids = []
        for pk in sumstats_pos_keys:
            varid = pvar_varid_by_pos.get(pk)
            if varid:
                extract_varids.append(varid)
        extract_varids_set = set(extract_varids)
        logger.info("Position-based matching: %d sumstats → %d in 1KG ref",
                     len(sumstats), len(extract_varids_set))

        if not extract_varids_set:
            raise ValueError(f"No overlapping variants for {locus_name}_{trait}")

        snp_list = tmp / "snps.txt"
        snp_list.write_text("\n".join(sorted(extract_varids_set)))
        print(f"    Variants overlapping with 1KG: {len(extract_varids_set)}")

        # Step 6: Run plink2 to compute LD
        ld_prefix = tmp / "ld_out"
        _run_plink2(
            [
                "--pfile", str(plink_prefix),
                "--extract", str(snp_list),
                "--r-unphased", "square",
                "--out", str(ld_prefix),
            ],
            description=f"LD matrix for {locus_name}_{trait}",
        )

        # Step 7: Read LD output
        vcor_file = ld_prefix.with_suffix(".unphased.vcor1")
        vars_file = Path(str(vcor_file) + ".vars")

        if not vcor_file.exists():
            vcor_candidates = list(tmp.glob("ld_out*.vcor*"))
            vars_candidates = list(tmp.glob("ld_out*.vars"))
            if not vcor_candidates:
                raise FileNotFoundError(f"plink2 LD output not found in {tmpdir}")
            vcor_file = vcor_candidates[0]
            vars_file = vars_candidates[0] if vars_candidates else None

        ld_raw = np.loadtxt(vcor_file, dtype=np.float64)
        if ld_raw.ndim == 1:
            ld_raw = ld_raw.reshape(1, 1)

        if vars_file and vars_file.exists():
            ref_snps = vars_file.read_text().strip().split("\n")
        else:
            raise FileNotFoundError("No .vars file — cannot determine SNP order")

    # Step 8: Harmonize alleles using the shared function
    # ref_snps has IDs like "19:45411941:T:C" — parse to build allele lookup
    ref_alleles_by_pos: dict[str, tuple[str, str]] = {}
    for varid in ref_snps:
        parts = varid.split(":")
        if len(parts) >= 4:
            pk = f"{parts[0]}:{parts[1]}"
            ref_alleles_by_pos[pk] = (parts[2], parts[3])

    aligned_ss, ld_sub, snp_order, report = harmonize_alleles(
        sumstats, ref_snps, ref_alleles_by_pos, ld_raw, match_by="pos",
    )

    logger.info(
        "Allele harmonization: %d keep, %d flipped, %d ambig, %d mismatch, %d not in ref",
        report["n_keep"], report["n_flip"], report["n_ambiguous_dropped"],
        report["n_mismatch_dropped"], report["n_not_in_ref"],
    )
    print(
        f"    Alleles: {report['n_keep']} keep, {report['n_flip']} flipped, "
        f"{report['n_ambiguous_dropped']} ambig, {report['n_mismatch_dropped']} mismatch, "
        f"{report['n_not_in_ref']} not in ref"
    )

    if report["n_final"] == 0:
        raise ValueError(f"No overlapping SNPs for {locus_name}_{trait}")

    # Save outputs
    LD_DIR.mkdir(parents=True, exist_ok=True)
    ld_out = LD_DIR / f"{locus_name}_{trait}.ld"
    np.savetxt(ld_out, ld_sub, fmt="%.6f", delimiter="\t")

    snp_out = LD_DIR / f"{locus_name}_{trait}.snplist"
    snp_out.write_text("\n".join(snp_order) + "\n")

    ss_out = LD_DIR / f"{locus_name}_{trait}.aligned.parquet"
    aligned_ss.to_parquet(ss_out, index=False)

    assert len(aligned_ss) == ld_sub.shape[0] == ld_sub.shape[1]

    print(
        f"    LD matrix: {ld_sub.shape[0]} x {ld_sub.shape[1]} → {ld_out.name}"
    )

    return aligned_ss, ld_sub, snp_order


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


def _classify_alleles(
    sum_a1: str, sum_a2: str, vcf_ref: str, vcf_alt: str,
) -> str:
    """Classify the allele relationship between sumstats and VCF reference.

    plink2 LD is computed on ALT allele dosages, so z-scores must be
    oriented to the ALT allele for SuSiE-RSS.

    Returns
    -------
    'keep'  — sumstats effect allele (a1) == VCF ALT → z already aligned.
    'flip'  — sumstats effect allele (a1) == VCF REF → must negate z
              (and set eaf = 1 - eaf) to align with LD.
    'drop'  — alleles incompatible (tri-allelic, indel mismatch, etc.).
    """
    s1, s2 = sum_a1.upper(), sum_a2.upper()
    ref, alt = vcf_ref.upper(), vcf_alt.upper()
    if s1 == alt and s2 == ref:
        # Effect allele == ALT → z already matches LD orientation
        return "keep"
    if s1 == ref and s2 == alt:
        # Effect allele == REF → must flip z to orient to ALT
        return "flip"
    return "drop"


def harmonize_alleles(
    sumstats_df: pd.DataFrame,
    ref_snps: list[str],
    ref_alleles: dict[str, tuple[str, str]],
    ld_raw: np.ndarray,
    *,
    match_by: str = "rsid",
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """Align sumstats z-scores to the LD reference panel allele coding.

    plink2 computes LD correlations based on ALT allele dosages. This
    function ensures that z-scores are oriented so that the *effect*
    direction corresponds to the ALT allele in the reference.

    Parameters
    ----------
    sumstats_df : DataFrame with columns rsid, chr, pos, a1, a2, beta, se, eaf.
    ref_snps : ordered list of variant IDs from plink2 .vars file.
    ref_alleles : mapping of variant key → (VCF REF, VCF ALT).
        When *match_by* is ``"rsid"``, keys are rsids.
        When *match_by* is ``"pos"``, keys are ``"chr:pos"`` strings.
    ld_raw : (n_ref, n_ref) raw LD correlation matrix aligned to *ref_snps*.
    match_by : ``"rsid"`` or ``"pos"`` — how to join sumstats to reference.

    Returns
    -------
    (aligned_df, ld_sub, snp_order, report)
        aligned_df : subset of sumstats with correct z-scores and eaf.
        ld_sub : LD submatrix aligned to aligned_df.
        snp_order : ref variant IDs in the same order.
        report : dict with counts {n_keep, n_flip, n_ambiguous_dropped,
                 n_mismatch_dropped, n_indel_dropped, n_not_in_ref, n_final}.
    """
    # Build ref index: key → (index_in_ld, ref_allele, alt_allele, varid)
    ref_idx_map: dict[str, tuple[int, str, str, str]] = {}
    for idx, varid in enumerate(ref_snps):
        if match_by == "pos":
            parts = varid.split(":")
            if len(parts) >= 2:
                key = f"{parts[0]}:{parts[1]}"
            else:
                continue
        else:
            key = varid
        alleles = ref_alleles.get(key)
        if alleles:
            ref_idx_map[key] = (idx, alleles[0], alleles[1], varid)

    keep_rows: list[int] = []
    keep_ref_idx: list[int] = []
    flip_positions: list[int] = []
    n_keep = n_flip = n_ambig = n_mismatch = n_indel = n_not_in_ref = 0

    for i, row in sumstats_df.iterrows():
        if match_by == "pos":
            key = f"{row['chr']}:{int(row['pos'])}"
        else:
            key = row["rsid"]

        entry = ref_idx_map.get(key)
        if entry is None:
            n_not_in_ref += 1
            continue

        ref_i, vcf_ref, vcf_alt, ref_varid = entry
        s1, s2 = str(row["a1"]).upper(), str(row["a2"]).upper()

        # Skip indels and multi-allelic
        if len(s1) > 1 or len(s2) > 1 or len(vcf_ref) > 1 or len(vcf_alt) > 1:
            n_indel += 1
            continue

        # Skip ambiguous strand
        if _is_ambiguous(s1, s2):
            n_ambig += 1
            continue

        action = _classify_alleles(s1, s2, vcf_ref, vcf_alt)

        if action == "drop":
            n_mismatch += 1
            continue

        keep_rows.append(i)
        keep_ref_idx.append(ref_i)
        if action == "flip":
            flip_positions.append(len(keep_rows) - 1)
            n_flip += 1
        else:
            n_keep += 1

    report = {
        "n_keep": n_keep,
        "n_flip": n_flip,
        "n_ambiguous_dropped": n_ambig,
        "n_mismatch_dropped": n_mismatch,
        "n_indel_dropped": n_indel,
        "n_not_in_ref": n_not_in_ref,
        "n_final": len(keep_rows),
    }

    if not keep_rows:
        return pd.DataFrame(), np.array([]), [], report

    aligned_ss = sumstats_df.iloc[keep_rows].reset_index(drop=True).copy()
    ld_sub = ld_raw[np.ix_(keep_ref_idx, keep_ref_idx)]
    snp_order = [ref_snps[j] for j in keep_ref_idx]

    # Compute z-scores
    z = aligned_ss["beta"].values / aligned_ss["se"].values

    # Flip z-scores AND eaf for variants where effect allele == VCF REF
    if flip_positions:
        for pos in flip_positions:
            z[pos] = -z[pos]
        if "eaf" in aligned_ss.columns:
            eaf = aligned_ss["eaf"].values.copy()
            for pos in flip_positions:
                eaf[pos] = 1.0 - eaf[pos]
            aligned_ss["eaf"] = eaf

    aligned_ss["z"] = z

    return aligned_ss, ld_sub, snp_order, report


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

    # Note: ambiguous-strand SNPs are handled inside harmonize_alleles()

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
    pvar_lookup: dict[str, tuple[str, str]] = {
        row["rsid"]: (row["ref"], row["alt"])
        for _, row in pvar.iterrows()
        if pd.notna(row["rsid"])
    }

    # -- Harmonize alleles via the shared function ----------------------------
    aligned_ss, ld_sub, snp_order, report = harmonize_alleles(
        sumstats, ref_snps, pvar_lookup, ld_raw, match_by="rsid",
    )

    logger.info(
        "Allele harmonization: %d keep, %d flipped, %d ambig, %d mismatch, %d not in ref",
        report["n_keep"], report["n_flip"], report["n_ambiguous_dropped"],
        report["n_mismatch_dropped"], report["n_not_in_ref"],
    )
    print(
        f"    Alleles: {report['n_keep']} keep, {report['n_flip']} flipped, "
        f"{report['n_ambiguous_dropped']} ambig, {report['n_mismatch_dropped']} mismatch, "
        f"{report['n_not_in_ref']} not in ref"
    )

    if report["n_final"] == 0:
        raise ValueError(
            f"No overlapping SNPs between sumstats and reference for "
            f"{locus_name}_{trait}"
        )

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

    assert len(aligned_ss) == ld_sub.shape[0] == ld_sub.shape[1]

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
