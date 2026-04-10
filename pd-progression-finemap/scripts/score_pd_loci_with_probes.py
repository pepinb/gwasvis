"""Score PD progression locus variants with the trained eQTL embedding probes.

This is prompt 8B of the PD GWAS × Evo 2 embeddings project. It takes the
Phase 2.x locus variant lists (already SNV-filtered, with paper leads present),
lifts them hg19 → hg38, verifies ref alleles against the hg38 FASTA, extracts
Evo 2 7B blocks.26 concatenated embeddings via Modal (reusing the existing
cache), and applies both the Whole_Blood and Brain_Cortex probes trained in
prompt 7.

Headline output is ``data/eqtl/probe_scorecard.csv``, with one row per PD lead
and columns for each tissue's per-locus percentile rank of the paper lead
variant, plus a STRONG/MODERATE/WEAK/NONE interpretation tier.

Run:

    python scripts/score_pd_loci_with_probes.py            # full pipeline
    python scripts/score_pd_loci_with_probes.py --preflight # pre-flight only

Outputs
-------
data/eqtl/locus_scores/<name>_<trait>_eqtl_probes.parquet
data/eqtl/probe_scorecard.csv
data/eqtl/locus_scores/extraction_failures.txt (if any)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eqtl.probes import EQTLProbe
from evo2.client import Evo2Client
from evo2.reference import FastaProvider
from evo2.variants import build_variant_window
from loci import LOCI, LOCI_OBJECTS

# ---------------------------------------------------------------------- config

CACHE_PATH = ROOT / "data" / "evo2" / "cache" / "embeddings_7b_blocks26_w128.parquet"
PROBES_DIR = ROOT / "data" / "eqtl" / "probes"
LOCUS_SCORES_DIR = ROOT / "data" / "eqtl" / "locus_scores"
SCORECARD_PATH = ROOT / "data" / "eqtl" / "probe_scorecard.csv"
LIKELIHOOD_PATH = ROOT / "data" / "evo2" / "calibration_matched_ensemble.csv"

LAYER = "blocks.26"
POOL_WINDOW = 128
ASSEMBLY = "hg38"
FAILURE_RATE_STOP = 0.02  # 2%
COST_PER_VARIANT_USD = 0.001
COST_STOP_USD = 25.0
BATCH_SIZE = 100       # Modal .map() batch
FLUSH_EVERY = 500      # cache flush interval (rows)

# Map Locus.name → filename stem in data/evo2/ (Phase 2.x)
LOCUS_FILE_STEMS = {
    "APOE":   "APOE_mortality",
    "TBXAS1": "TBXAS1_mortality",
    "SYT10":  "SYT10_mortality",
    "MORN1":  "MORN1_hy3",
    "ASNS":   "ASNS_hy3",
    "PDE5A":  "PDE5A_hy3",
    "XPO1":   "XPO1_hy3",
}

# Hardcoded paper eQTL status strings for the scorecard (from Tan et al. 2024
# npj Parkinson's Disease supplementary tables).
PAPER_EQTL_STATUS = {
    "APOE":   "Known APOE locus; many eQTLs in LD",
    "TBXAS1": "Suggestive blood eQTL (TBXAS1)",
    "SYT10":  "Brain eQTL reported (SYT10, cortex)",
    "MORN1":  "No strong eQTL (negative control)",
    "ASNS":   "Rare variant; no bulk eQTL expected",
    "PDE5A":  "Blood eQTL suggestive (PDE5A)",
    "XPO1":   "Dual-tissue eQTL (XPO1, blood + brain)",
}


# ---------------------------------------------------------------- variant list


def _load_phase2_variants(name: str) -> pd.DataFrame:
    """Load the Phase 2.x per-locus Evo 2 7B scalar parquet as the variant source."""
    stem = LOCUS_FILE_STEMS[name]
    df = pd.read_parquet(ROOT / "data" / "evo2" / f"{stem}_evo2_7b.parquet")
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""].copy()
    # already SNV in the Phase 2.x extraction, but double-check
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    return df.reset_index(drop=True)


def build_variant_list() -> tuple[pd.DataFrame, dict]:
    """Build the full variant DataFrame (all 7 loci) with hg38 coordinates.

    Returns
    -------
    variants : pd.DataFrame
        Columns: locus, trait, chrom, position (hg38), ref, alt, rsid,
        hg19_pos, distance_to_lead_hg38, is_paper_lead.
    stats : dict
        Per-locus counts (raw, lifted, verified, lead_found).
    """
    from pyliftover import LiftOver
    lo = LiftOver("hg19", "hg38")
    fasta_hg38 = FastaProvider(assembly="hg38")

    out_rows: list[dict] = []
    stats: dict[str, dict] = {}

    for locus in LOCI_OBJECTS:
        name = locus.name
        trait = locus.trait
        df = _load_phase2_variants(name)
        n_raw = len(df)
        lead_hg38 = locus.hg38_bp

        lifted_rows = []
        n_lift_fail = 0
        n_ref_mismatch = 0
        for _, row in df.iterrows():
            chrom = str(row["chrom"]).replace("chr", "")
            hg19_pos = int(row["pos"])
            ref = str(row["ref"]).upper()
            alt = str(row["alt"]).upper()

            # Liftover (pyliftover uses 0-based coordinates)
            conv = lo.convert_coordinate(f"chr{chrom}", hg19_pos - 1)
            if not conv:
                n_lift_fail += 1
                continue
            hg38_chrom = conv[0][0].replace("chr", "")
            hg38_pos = conv[0][1] + 1  # back to 1-based
            strand = conv[0][2]

            if hg38_chrom != chrom:
                n_lift_fail += 1
                continue
            if strand != "+":
                # would require allele flip; skip (rare for PD loci)
                n_lift_fail += 1
                continue

            # Verify ref allele against hg38 FASTA
            try:
                fasta_ref = fasta_hg38.get_base(chrom, hg38_pos)
            except (FileNotFoundError, KeyError):
                n_lift_fail += 1
                continue
            if fasta_ref != ref:
                n_ref_mismatch += 1
                continue

            lifted_rows.append({
                "locus": name,
                "trait": trait,
                "chrom": chrom,
                "position": hg38_pos,
                "ref": ref,
                "alt": alt,
                "rsid": row["rsid"],
                "hg19_pos": hg19_pos,
                "distance_to_lead_hg38": abs(hg38_pos - lead_hg38),
                "delta_log_likelihood": float(row.get("delta_log_likelihood", np.nan)),
            })

        locus_df = pd.DataFrame(lifted_rows)
        # mark paper lead
        locus_df["is_paper_lead"] = (
            (locus_df["position"] == lead_hg38)
            & (locus_df["ref"] == locus.ref)
            & (locus_df["alt"] == locus.alt)
        )
        lead_found = int(locus_df["is_paper_lead"].sum())
        stats[name] = {
            "raw": n_raw,
            "lifted": len(locus_df),
            "lift_fail": n_lift_fail,
            "ref_mismatch": n_ref_mismatch,
            "lead_found": lead_found,
        }
        out_rows.append(locus_df)

    variants = pd.concat(out_rows, ignore_index=True)
    return variants, stats


# ---------------------------------------------------------- batched extraction


def extract_embeddings_batched(
    variants: pd.DataFrame,
    *,
    cache_path: Path,
    fasta_provider: FastaProvider,
    layer: str = LAYER,
    pool_window: int = POOL_WINDOW,
    batch_size: int = BATCH_SIZE,
    flush_every: int = FLUSH_EVERY,
) -> tuple[np.ndarray, list[int], list[dict]]:
    """Extract concat-order [ref_fwd, ref_rc, alt_fwd, alt_rc] embeddings for
    each variant, using Modal .map() for parallel remote calls and periodic
    cache flushes to avoid O(n) disk rewrites per variant.

    Returns
    -------
    features : np.ndarray, shape (n_kept, 4*hidden_dim)
    kept_indices : list[int]  (indices into variants DataFrame, in order)
    failures : list[dict]
    """
    import time

    client = Evo2Client(cache_path=cache_path)
    assembly = fasta_provider.assembly

    # --- 1. load cache once, build in-memory lookup ---
    cache_df = client._load_rc_cache()  # noqa: SLF001
    def _build_lookup(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        mask = (
            (df["assembly"] == assembly)
            & (df["layer_name"] == layer)
            & (df["pool_window"] == pool_window)
        )
        sub = df[mask]
        keys = list(zip(
            sub["chrom"].astype(str),
            sub["position"].astype(int),
            sub["ref"].astype(str).str.upper(),
            sub["alt"].astype(str).str.upper(),
        ))
        return dict(zip(keys, sub.index))

    cache_lookup = _build_lookup(cache_df)

    # --- 2. classify variants: cached / to-extract / window-fail ---
    to_extract: list[tuple[int, str, int, str, str, str, str]] = []
    failures: list[dict] = []
    classification: list[str] = []  # per variant: 'cached', 'new', 'failed'

    for orig_idx, row in enumerate(variants.itertuples(index=False)):
        chrom = str(row.chrom)
        pos = int(row.position)
        ref = row.ref.upper()
        alt = row.alt.upper()
        if (chrom, pos, ref, alt) in cache_lookup:
            classification.append("cached")
            continue
        try:
            w = build_variant_window(
                chrom=chrom, position=pos, ref=ref, alt=alt,
                provider=fasta_provider,
            )
        except (AssertionError, FileNotFoundError, ValueError) as exc:
            failures.append({
                "orig_idx": orig_idx, "chrom": chrom, "position": pos,
                "ref": ref, "alt": alt,
                "reason": f"window build failed: {exc}",
            })
            classification.append("failed")
            continue
        to_extract.append((orig_idx, chrom, pos, ref, alt, w.ref_sequence, w.alt_sequence))
        classification.append("new")

    n_cached = classification.count("cached")
    n_new = len(to_extract)
    n_window_fail = classification.count("failed")
    print(f"  classification: cached={n_cached}  new={n_new}  window_fail={n_window_fail}")

    # --- 3. batched .map() calls ---
    new_rows: list[dict] = []
    pending_flush = 0
    t0 = time.time()

    for batch_start in range(0, len(to_extract), batch_size):
        batch = to_extract[batch_start:batch_start + batch_size]
        ref_seqs = [v[5] for v in batch]
        alt_seqs = [v[6] for v in batch]

        try:
            results = list(client._model.score_variant_with_rc.map(
                ref_seqs, alt_seqs,
                kwargs={"layer_names": [layer], "pool_window": pool_window},
                order_outputs=True,
                return_exceptions=True,
            ))
        except Exception as exc:  # noqa: BLE001
            for vinfo in batch:
                failures.append({
                    "orig_idx": vinfo[0], "chrom": vinfo[1], "position": vinfo[2],
                    "ref": vinfo[3], "alt": vinfo[4],
                    "reason": f"batch map error: {exc}",
                })
            continue

        for vinfo, result in zip(batch, results):
            orig_idx, chrom, pos, ref, alt, _, _ = vinfo
            if isinstance(result, Exception):
                failures.append({
                    "orig_idx": orig_idx, "chrom": chrom, "position": pos,
                    "ref": ref, "alt": alt,
                    "reason": f"score_variant_with_rc failed: {result}",
                })
                continue
            emb = result["embeddings"][layer]
            row_dict = {
                "chrom": chrom, "position": pos, "ref": ref, "alt": alt,
                "assembly": assembly, "layer_name": layer,
                "pool_window": int(pool_window),
                "delta_log_likelihood": float(result["delta_log_likelihood"]),
                "delta_log_likelihood_fwd": float(result["delta_log_likelihood_fwd"]),
                "delta_log_likelihood_rc": float(result["delta_log_likelihood_rc"]),
                "log_likelihood_ref_fwd": float(result["log_likelihood_ref_fwd"]),
                "log_likelihood_ref_rc": float(result["log_likelihood_ref_rc"]),
                "log_likelihood_alt_fwd": float(result["log_likelihood_alt_fwd"]),
                "log_likelihood_alt_rc": float(result["log_likelihood_alt_rc"]),
                "ref_fwd_embedding": np.asarray(emb["ref_fwd"], dtype=np.float32),
                "ref_rc_embedding":  np.asarray(emb["ref_rc"],  dtype=np.float32),
                "alt_fwd_embedding": np.asarray(emb["alt_fwd"], dtype=np.float32),
                "alt_rc_embedding":  np.asarray(emb["alt_rc"],  dtype=np.float32),
                "model": result.get("model", "evo2_7b"),
            }
            new_rows.append(row_dict)
            pending_flush += 1

        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (len(to_extract) - done) / rate if rate > 0 else 0.0
        print(f"  [{done:5d}/{len(to_extract)}] new_rows={len(new_rows)} "
              f"failures={len(failures)} rate={rate:.1f} var/s eta={eta/60:.1f}min",
              flush=True)

        if pending_flush >= flush_every:
            print(f"    flushing {pending_flush} rows to cache...", flush=True)
            client._append_rc_rows(new_rows[-pending_flush:])
            pending_flush = 0

    if pending_flush > 0:
        print(f"  final flush: {pending_flush} rows", flush=True)
        client._append_rc_rows(new_rows[-pending_flush:])

    # --- 4. rebuild feature matrix from cache in input order ---
    cache_df_final = client._load_rc_cache()  # noqa: SLF001
    cache_lookup_final = _build_lookup(cache_df_final)

    failed_idx_set = {f["orig_idx"] for f in failures}
    features: list[np.ndarray] = []
    kept_indices: list[int] = []
    for orig_idx, row in enumerate(variants.itertuples(index=False)):
        if orig_idx in failed_idx_set:
            continue
        key = (str(row.chrom), int(row.position), row.ref.upper(), row.alt.upper())
        if key not in cache_lookup_final:
            failures.append({
                "orig_idx": orig_idx, "chrom": key[0], "position": key[1],
                "ref": key[2], "alt": key[3],
                "reason": "missing from cache after extraction",
            })
            continue
        cache_row = cache_df_final.loc[cache_lookup_final[key]]
        parts = [
            np.asarray(cache_row[f"{k}_embedding"], dtype=np.float32)
            for k in ("ref_fwd", "ref_rc", "alt_fwd", "alt_rc")
        ]
        features.append(np.concatenate(parts, axis=0))
        kept_indices.append(orig_idx)

    if features:
        feature_matrix = np.stack(features, axis=0).astype(np.float32, copy=False)
    else:
        feature_matrix = np.empty((0, 0), dtype=np.float32)

    return feature_matrix, kept_indices, failures


# --------------------------------------------------------------------- probes


def apply_probes(features: np.ndarray) -> dict[str, np.ndarray]:
    """Load both probes and return dict tissue → positive-class probability.

    EQTLProbe.predict_proba already returns a 1-D positive-class array.
    """
    wb = EQTLProbe.load(PROBES_DIR / "Whole_Blood_evo2_7b_blocks26.joblib")
    bc = EQTLProbe.load(PROBES_DIR / "Brain_Cortex_evo2_7b_blocks26.joblib")
    for tissue, probe in (("Whole_Blood", wb), ("Brain_Cortex", bc)):
        if probe.construction != "concat":
            raise RuntimeError(
                f"{tissue} probe construction is {probe.construction!r}; "
                "expected 'concat'."
            )
    return {
        "Whole_Blood": wb.predict_proba(features),
        "Brain_Cortex": bc.predict_proba(features),
    }


# ------------------------------------------------------------------ scorecard


def _tier(percentile: float) -> str:
    if percentile >= 95:
        return "STRONG"
    if percentile >= 80:
        return "MODERATE"
    if percentile >= 50:
        return "WEAK"
    return "NONE"


def build_scorecard(variants: pd.DataFrame) -> pd.DataFrame:
    """Build per-locus scorecard DataFrame with headline numbers."""
    rows = []
    for locus in LOCI_OBJECTS:
        sub = variants[variants["locus"] == locus.name].copy()
        lead = sub[sub["is_paper_lead"]]
        if lead.empty:
            rows.append({
                "locus": locus.name,
                "trait": locus.trait,
                "chrom": locus.chr,
                "hg38_bp": locus.hg38_bp,
                "rsid": locus.rsid,
                "n_variants": len(sub),
                "wb_lead_prob": np.nan,
                "wb_lead_percentile": np.nan,
                "wb_tier": "MISSING",
                "bc_lead_prob": np.nan,
                "bc_lead_percentile": np.nan,
                "bc_tier": "MISSING",
                "paper_eqtl_status": PAPER_EQTL_STATUS[locus.name],
            })
            continue
        wb_prob = float(lead["wb_prob"].iloc[0])
        bc_prob = float(lead["bc_prob"].iloc[0])
        wb_pct = float((sub["wb_prob"] <= wb_prob).mean() * 100.0)
        bc_pct = float((sub["bc_prob"] <= bc_prob).mean() * 100.0)
        rows.append({
            "locus": locus.name,
            "trait": locus.trait,
            "chrom": locus.chr,
            "hg38_bp": locus.hg38_bp,
            "rsid": locus.rsid,
            "n_variants": len(sub),
            "wb_lead_prob": wb_prob,
            "wb_lead_percentile": wb_pct,
            "wb_tier": _tier(wb_pct),
            "bc_lead_prob": bc_prob,
            "bc_lead_percentile": bc_pct,
            "bc_tier": _tier(bc_pct),
            "paper_eqtl_status": PAPER_EQTL_STATUS[locus.name],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- output


def print_preflight(variants: pd.DataFrame, stats: dict) -> float:
    """Print pre-flight summary and return cost estimate in USD."""
    cache = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame()
    if not cache.empty:
        cache_keys = set(
            cache[["chrom", "position", "ref", "alt", "assembly", "layer_name", "pool_window"]]
            .astype({"chrom": "string"})
            .apply(tuple, axis=1)
        )
    else:
        cache_keys = set()

    variant_keys = variants[["chrom", "position", "ref", "alt"]].astype({"chrom": "string"}).apply(
        lambda r: (r["chrom"], int(r["position"]), r["ref"], r["alt"], ASSEMBLY, LAYER, POOL_WINDOW),
        axis=1,
    )
    already_cached = sum(1 for k in variant_keys if k in cache_keys)
    to_extract = len(variants) - already_cached
    cost_est = to_extract * COST_PER_VARIANT_USD

    print("=" * 72)
    print("PRE-FLIGHT SUMMARY — score_pd_loci_with_probes")
    print("=" * 72)
    print(f"{'locus':8s} {'trait':12s} {'raw':>6s} {'lifted':>6s} {'lift_fail':>9s} "
          f"{'refmm':>6s} {'lead':>5s}")
    for locus in LOCI_OBJECTS:
        s = stats[locus.name]
        print(f"{locus.name:8s} {locus.trait:12s} {s['raw']:6d} {s['lifted']:6d} "
              f"{s['lift_fail']:9d} {s['ref_mismatch']:6d} {s['lead_found']:5d}")
    print("-" * 72)
    print(f"Total variants (post-liftover, verified):  {len(variants)}")
    print(f"Already in Evo 2 cache:                    {already_cached}")
    print(f"To extract via Modal:                      {to_extract}")
    print(f"Estimated cost (@${COST_PER_VARIANT_USD}/var):              ${cost_est:.2f}")
    print(f"Cost stop threshold:                       ${COST_STOP_USD:.2f}")
    print("=" * 72)
    return cost_est


def print_headline(scorecard: pd.DataFrame, variants: pd.DataFrame) -> None:
    """Print the Q1-Q7 headline answers."""
    sc = scorecard.set_index("locus")

    print()
    print("=" * 72)
    print("HEADLINE ANSWERS — do the embedding probes see the paper eQTLs?")
    print("=" * 72)

    def _fmt(locus: str, tissue: str) -> str:
        pct = sc.loc[locus, f"{tissue}_lead_percentile"]
        tier = sc.loc[locus, f"{tissue}_tier"]
        prob = sc.loc[locus, f"{tissue}_lead_prob"]
        if np.isnan(pct):
            return f"{locus} {tissue}: lead MISSING from variant set"
        return f"{locus} {tissue}: p={prob:.3f}  percentile={pct:5.1f}  tier={tier}"

    print("Q1. TBXAS1 — do we see the blood eQTL signal?")
    print("   →", _fmt("TBXAS1", "wb"))

    print("Q2. XPO1 — dual-tissue (blood + brain) signal?")
    print("   →", _fmt("XPO1", "wb"))
    print("   →", _fmt("XPO1", "bc"))

    print("Q3. MORN1 — negative control (should be WEAK/NONE)")
    print("   →", _fmt("MORN1", "wb"))
    print("   →", _fmt("MORN1", "bc"))

    print("Q4. PDE5A — blood eQTL signal?")
    print("   →", _fmt("PDE5A", "wb"))

    print("Q5. SYT10 — brain (cortex) eQTL signal?")
    print("   →", _fmt("SYT10", "bc"))

    print("Q6. ASNS — rare variant singleton (scientific curiosity)")
    print("   →", _fmt("ASNS", "wb"))
    print("   →", _fmt("ASNS", "bc"))

    print("Q7. APOE — large LD block completeness")
    print("   →", _fmt("APOE", "wb"))
    print("   →", _fmt("APOE", "bc"))

    # Summary counts
    strong_wb = (sc["wb_tier"] == "STRONG").sum()
    mod_wb = (sc["wb_tier"] == "MODERATE").sum()
    strong_bc = (sc["bc_tier"] == "STRONG").sum()
    mod_bc = (sc["bc_tier"] == "MODERATE").sum()
    print()
    print(f"Whole_Blood tiers: STRONG={strong_wb}, MODERATE={mod_wb}")
    print(f"Brain_Cortex tiers: STRONG={strong_bc}, MODERATE={mod_bc}")

    # Optional likelihood comparison
    if LIKELIHOOD_PATH.exists():
        print()
        print("Likelihood vs probe comparison:")
        try:
            lk = pd.read_csv(LIKELIHOOD_PATH)
            print("   (loaded", LIKELIHOOD_PATH.name, "shape=", lk.shape, ")")
            # conservative — just note existence; full comparison left to analysis
        except Exception as e:
            print("   (could not read likelihood file:", e, ")")
    else:
        print()
        print(f"(no likelihood file at {LIKELIHOOD_PATH} — skipping comparison)")


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true",
                        help="Run pre-flight only, no Modal calls")
    parser.add_argument("--skip-preflight-stop", action="store_true",
                        help="Skip the cost/lead preflight gate (advanced)")
    args = parser.parse_args()

    LOCUS_SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Part 1: build variant list ----------
    print("[1/6] Building variant list (hg19 → hg38 liftover + ref verification)")
    variants, stats = build_variant_list()

    # Pre-flight gate
    cost_est = print_preflight(variants, stats)
    missing_leads = [name for name, s in stats.items() if s["lead_found"] == 0]
    if missing_leads and not args.skip_preflight_stop:
        print(f"\nERROR: paper lead missing for {missing_leads}. STOP.")
        sys.exit(1)
    if cost_est > COST_STOP_USD and not args.skip_preflight_stop:
        print(f"\nERROR: cost estimate ${cost_est:.2f} > ${COST_STOP_USD}. STOP.")
        sys.exit(1)
    if args.preflight:
        print("\n(preflight only — exiting)")
        return

    # -------- Part 2: extract embeddings ----------
    print()
    print(f"[2/6] Extracting Evo 2 embeddings (layer={LAYER}, pool={POOL_WINDOW}) "
          f"via Modal .map() batch={BATCH_SIZE} flush_every={FLUSH_EVERY}")
    fasta_hg38 = FastaProvider(assembly=ASSEMBLY)

    features, kept_indices, failures = extract_embeddings_batched(
        variants,
        cache_path=CACHE_PATH,
        fasta_provider=fasta_hg38,
        layer=LAYER,
        pool_window=POOL_WINDOW,
    )
    n_ok = features.shape[0]
    n_fail = len(failures)
    fail_rate = n_fail / max(len(variants), 1)
    print(f"  extracted: {n_ok}, failures: {n_fail} ({fail_rate*100:.2f}%)")

    if failures:
        fail_path = LOCUS_SCORES_DIR / "extraction_failures.txt"
        with fail_path.open("w") as f:
            for fr in failures:
                f.write(f"{fr}\n")
        print(f"  failure log: {fail_path}")

    if fail_rate > FAILURE_RATE_STOP:
        print(f"ERROR: failure rate {fail_rate*100:.2f}% > {FAILURE_RATE_STOP*100:.0f}%. STOP.")
        sys.exit(1)

    # Subset variants to those that extracted successfully.
    variants = variants.iloc[kept_indices].reset_index(drop=True)
    assert len(variants) == n_ok, (
        f"variants/features length mismatch: {len(variants)} vs {n_ok}"
    )

    # -------- Part 3: apply probes ----------
    print()
    print("[3/6] Applying Whole_Blood and Brain_Cortex probes")
    probas = apply_probes(features)
    variants["wb_prob"] = probas["Whole_Blood"]
    variants["bc_prob"] = probas["Brain_Cortex"]

    # Save per-locus parquets
    for locus in LOCI_OBJECTS:
        sub = variants[variants["locus"] == locus.name].copy()
        if sub.empty:
            continue
        # percentile within locus
        sub["wb_percentile"] = sub["wb_prob"].rank(pct=True) * 100.0
        sub["bc_percentile"] = sub["bc_prob"].rank(pct=True) * 100.0
        trait_slug = locus.trait.replace(" ", "_").lower()
        path = LOCUS_SCORES_DIR / f"{locus.name}_{trait_slug}_eqtl_probes.parquet"
        sub.to_parquet(path, index=False)
        print(f"  {path.name}: {len(sub)} variants")

    # -------- Part 4: scorecard ----------
    print()
    print("[4/6] Building probe_scorecard.csv")
    scorecard = build_scorecard(variants)
    scorecard.to_csv(SCORECARD_PATH, index=False)
    print(f"  wrote {SCORECARD_PATH}")
    print()
    print(scorecard.to_string(index=False))

    # -------- Part 5: headline answers ----------
    print()
    print("[5/6] Headline answers")
    print_headline(scorecard, variants)

    # -------- Part 6: commit guidance ----------
    print()
    print("[6/6] Next: review outputs then commit with")
    print("  git add scripts/score_pd_loci_with_probes.py "
          "data/eqtl/probe_scorecard.csv data/eqtl/locus_scores/")
    print("  git commit -m 'Score PD loci with eQTL probes'")


if __name__ == "__main__":
    main()
