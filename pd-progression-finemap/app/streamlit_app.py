"""Streamlit dashboard for interactive exploration of PD fine-mapping results.

Run with:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evo2.load import (
    list_available_models,
    load_calibration_matched,
    load_calibration_summary,
    load_evo2_scores,
    load_genome_null,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

FINEMAP_DIR = Path(__file__).resolve().parent.parent / "data" / "finemap"
LOCI_DIR = Path(__file__).resolve().parent.parent / "data" / "loci"
SUMMARY_CSV = FINEMAP_DIR / "baseline_summary.csv"

# Paper lead SNPs (locus_name -> info dict).
# Mortality sumstats use chr:pos as rsid; HY3 uses actual rsids.
_PAPER_LEADS: dict[str, dict] = {
    "APOE": {"rsid": "rs429358", "chrpos": "19:45411941", "pos": 45411941, "trait": "mortality"},
    "TBXAS1": {"rsid": "rs4726467", "chrpos": "7:139637422", "pos": 139637422, "trait": "mortality"},
    "SYT10": {"rsid": "rs10437796", "chrpos": "12:33635494", "pos": 33635494, "trait": "mortality"},
    "MORN1": {"rsid": "rs115217673", "pos": 2315032, "trait": "hy3"},
    "ASNS": {"rsid": "rs145274312", "pos": 97470925, "trait": "hy3"},
    "PDE5A": {"rsid": "rs113120976", "pos": 120566153, "trait": "hy3"},
    "XPO1": {"rsid": "rs141421624", "pos": 61742356, "trait": "hy3"},
}

_STATUS_BADGES: dict[str, str] = {
    "REPLICATED": "\U0001f7e2 REPLICATED",
    "CAUTION": "\U0001f7e1 CAUTION",
    "NOT_TESTABLE": "\U0001f534 NOT TESTABLE",
    "FAILED": "\u26aa FAILED",
}


def _find_paper_lead_row(df: pd.DataFrame, paper_info: dict) -> pd.Series | None:
    """Find the paper's lead variant in finemap data, trying rsid, chr:pos, and position."""
    rsid = paper_info.get("rsid", "")
    if rsid and "rsid" in df.columns:
        rows = df[df["rsid"] == rsid]
        if not rows.empty:
            return rows.iloc[0]
    chrpos = paper_info.get("chrpos", "")
    if chrpos and "rsid" in df.columns:
        rows = df[df["rsid"] == chrpos]
        if not rows.empty:
            return rows.iloc[0]
    pos = paper_info.get("pos")
    if pos is not None and "pos" in df.columns:
        rows = df[df["pos"] == pos]
        if not rows.empty:
            return rows.iloc[0]
    return None


# ---------------------------------------------------------------------------
# Data loading (finemap)
# ---------------------------------------------------------------------------


@st.cache_data
def discover_loci() -> list[dict]:
    """Scan data/finemap/ for parquet files and return available (name, trait) pairs."""
    loci = []
    for p in sorted(FINEMAP_DIR.glob("*.parquet")):
        stem = p.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            loci.append({"name": parts[0], "trait": parts[1], "path": str(p)})
    return loci


@st.cache_data
def load_locus(path: str) -> pd.DataFrame:
    """Load a finemap parquet and add -log10(p) column."""
    df = pd.read_parquet(path)
    pval = df["pval"].clip(lower=1e-300)
    df["neglog10p"] = -np.log10(pval)
    return df


@st.cache_data
def load_loci_parquet(locus: str, trait: str) -> pd.DataFrame | None:
    """Load raw GWAS locus parquet from data/loci/."""
    path = LOCI_DIR / f"{locus}_{trait}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_summary() -> pd.DataFrame | None:
    """Load baseline_summary.csv if it exists."""
    if SUMMARY_CSV.exists():
        return pd.read_csv(SUMMARY_CSV)
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_locuszoom(df: pd.DataFrame, paper_lead_rsid: str) -> go.Figure:
    """LocusZoom-style scatter: x=pos, y=-log10(p), color=PIP."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["pos"],
            y=df["neglog10p"],
            mode="markers",
            marker=dict(
                size=6,
                color=df["pip"],
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title="PIP", thickness=12, len=0.6),
                line=dict(width=0.3, color="white"),
            ),
            text=df.apply(
                lambda r: (
                    f"{r['rsid']}<br>"
                    f"p = {r['pval']:.2e}<br>"
                    f"PIP = {r['pip']:.3f}"
                ),
                axis=1,
            ),
            hoverinfo="text",
            name="Variants",
        )
    )

    lead = df[df["rsid"] == paper_lead_rsid]
    if not lead.empty:
        fig.add_trace(
            go.Scatter(
                x=lead["pos"],
                y=lead["neglog10p"],
                mode="markers",
                marker=dict(
                    size=14,
                    symbol="star",
                    color="red",
                    line=dict(width=1, color="darkred"),
                ),
                text=lead.apply(
                    lambda r: (
                        f"<b>{r['rsid']}</b> (paper lead)<br>"
                        f"p = {r['pval']:.2e}<br>"
                        f"PIP = {r['pip']:.3f}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                name="Paper lead",
            )
        )

    fig.add_hline(
        y=-np.log10(5e-8),
        line_dash="dash",
        line_color="grey",
        annotation_text="p = 5e-8",
        annotation_position="top left",
    )

    fig.update_layout(
        title="LocusZoom",
        xaxis_title="Position (bp)",
        yaxis_title="-log10(p)",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False,
    )
    return fig


def make_pip_track(df: pd.DataFrame) -> go.Figure:
    """PIP bar chart: x=pos, y=pip, credible set bars highlighted."""
    in_cs = df["in_cs"].astype(bool)

    fig = go.Figure()

    bg = df[~in_cs]
    if not bg.empty:
        fig.add_trace(
            go.Bar(
                x=bg["pos"],
                y=bg["pip"],
                marker_color="lightgrey",
                name="Not in CS",
                hovertext=bg["rsid"],
                hoverinfo="text+y",
                width=800,
            )
        )

    cs = df[in_cs]
    if not cs.empty:
        fig.add_trace(
            go.Bar(
                x=cs["pos"],
                y=cs["pip"],
                marker_color="steelblue",
                name="In credible set",
                hovertext=cs.apply(
                    lambda r: f"{r['rsid']}<br>CS{int(r['cs_id'])}<br>PIP={r['pip']:.3f}",
                    axis=1,
                ),
                hoverinfo="text",
                width=800,
            )
        )

    fig.update_layout(
        title="PIP Track",
        xaxis_title="Position (bp)",
        yaxis_title="PIP",
        yaxis_range=[0, 1.05],
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
    )
    return fig


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def view_locus_detail(available: list[dict]) -> None:
    """Render the single-locus detail view (LocusZoom + PIP + CS)."""
    selected_name = st.session_state.get("selected_locus")
    selected_trait = st.session_state.get("selected_trait")

    if not selected_name or not selected_trait:
        st.info("Select a locus from the sidebar.")
        return

    match = [
        loc for loc in available
        if loc["name"] == selected_name and loc["trait"] == selected_trait
    ]
    if not match:
        st.warning("No data for this locus/trait combination.")
        return

    try:
        df = load_locus(match[0]["path"])
    except Exception as exc:
        st.error(f"Failed to load {match[0]['path']}: {exc}")
        return

    if df.empty:
        st.warning("Parquet loaded but contains no variants.")
        return

    paper_info = _PAPER_LEADS.get(selected_name, {})
    paper_lead_rsid = paper_info.get("rsid", "")
    lead_row = _find_paper_lead_row(df, paper_info)

    plot_lead_rsid = str(lead_row["rsid"]) if lead_row is not None else paper_lead_rsid

    # -- Locus note -----------------------------------------------------------
    locus_note = None
    if "locus_note" in df.columns:
        note_vals = df["locus_note"].dropna().unique()
        if len(note_vals) > 0:
            locus_note = str(note_vals[0])

    if locus_note:
        st.info(locus_note)

    # -- Layout ---------------------------------------------------------------
    col_left, col_right = st.columns([3, 1])

    with col_left:
        fig_lz = make_locuszoom(df, plot_lead_rsid)
        st.plotly_chart(fig_lz, use_container_width=True)

        fig_pip = make_pip_track(df)
        st.plotly_chart(fig_pip, use_container_width=True)

    with col_right:
        st.subheader("Summary")

        cs_variants = df[df["in_cs"]]
        n_cs = cs_variants["cs_id"].nunique() if not cs_variants.empty else 0
        cs0_size = int((df["cs_id"] == 0).sum()) if n_cs > 0 else 0

        top_row = df.loc[df["pip"].idxmax()]
        top_pip = float(top_row["pip"])
        top_rsid = str(top_row["rsid"])
        converged = bool(df["converged"].iloc[0]) if "converged" in df.columns else None

        c1, c2 = st.columns(2)
        c1.metric("Credible sets", n_cs)
        c2.metric("CS0 size", cs0_size)

        c3, c4 = st.columns(2)
        c3.metric("Top PIP", f"{top_pip:.4f}")
        c4.metric("Converged", "Yes" if converged else "No")

        st.metric("Top SNP", top_rsid)

        # Paper lead status
        if lead_row is not None:
            lead_in_cs = bool(lead_row["in_cs"])
            display_rsid = paper_lead_rsid
            data_rsid = str(lead_row["rsid"])
            if data_rsid != paper_lead_rsid:
                display_rsid = f"{paper_lead_rsid} ({data_rsid})"
            st.info(
                f"Paper lead: **{display_rsid}**\n\n"
                f"PIP = {lead_row['pip']:.4f} | "
                f"In CS: {'Yes' if lead_in_cs else 'No'}"
            )
        elif locus_note:
            pass
        else:
            st.warning(f"Paper lead {paper_lead_rsid} not in locus data")

        # Credible set table
        st.subheader("Credible set variants")
        if not cs_variants.empty:
            display_cols = ["rsid", "pos", "pip", "pval", "beta", "cs_id"]
            display_cols = [c for c in display_cols if c in cs_variants.columns]
            cs_display = cs_variants[display_cols].sort_values("pip", ascending=False)
            st.dataframe(cs_display, use_container_width=True, hide_index=True)
        else:
            st.caption("No credible set variants (purity filter may have removed all CS).")

        with st.expander("Full locus table"):
            show_cols = ["rsid", "pos", "pval", "beta", "se", "pip", "in_cs", "cs_id"]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(
                df[show_cols].sort_values("pos"),
                use_container_width=True,
                hide_index=True,
                height=400,
            )


# ---------------------------------------------------------------------------
# Evo 2 per-locus view
# ---------------------------------------------------------------------------


def view_evo2(model: str) -> None:
    """Render the Evo 2 tab for the selected locus."""
    selected_name = st.session_state.get("selected_locus")
    selected_trait = st.session_state.get("selected_trait")

    if not selected_name or not selected_trait:
        st.info("Select a locus from the sidebar.")
        return

    paper_info = _PAPER_LEADS.get(selected_name, {})
    paper_lead_rsid = paper_info.get("rsid", "")

    evo2_df = load_evo2_scores(selected_name, selected_trait, model)
    if evo2_df is None:
        st.info(
            f"No Evo 2 {model.upper()} scores available for this locus. "
            "Run `scripts/score_all_loci.py` to generate them."
        )
        return

    scored = evo2_df[evo2_df["error"] == ""].copy()
    if scored.empty:
        st.warning("All variants failed scoring.")
        return

    # Look up lead stats from calibration data
    cal_summary = load_calibration_summary(model)
    cal_matched = load_calibration_matched(model)

    lead_dll = None
    locus_pctile = None
    null_pctile_signed = None
    null_pctile_abs = None
    lead_maf = None

    if cal_summary is not None:
        srow = cal_summary[cal_summary["locus"] == selected_name]
        if not srow.empty and srow.iloc[0]["lead_in_scored"]:
            lead_dll = srow.iloc[0]["paper_lead_delta_ll"]
            locus_pctile = srow.iloc[0]["percentile_signed"]

    if cal_matched is not None:
        mrow = cal_matched[cal_matched["locus"] == selected_name]
        if not mrow.empty:
            null_pctile_signed = mrow.iloc[0]["null_percentile_signed"]
            null_pctile_abs = mrow.iloc[0]["null_percentile_abs"]
            lead_maf_val = mrow.iloc[0].get("lead_maf")
            if pd.notna(lead_maf_val):
                lead_maf = float(lead_maf_val)

    # ── PANEL A: Headline metrics ──────────────────────────────────────
    if lead_dll is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Lead delta-ll",
            f"{lead_dll:.2f}",
            help="Delta log-likelihood: negative = alt allele disrupts sequence plausibility",
        )
        c2.metric(
            "Locus percentile",
            f"{locus_pctile:.1f}%" if locus_pctile is not None else "--",
            help="Lower = more disruptive than nearby variants in this locus",
        )
        c3.metric(
            "Genome-wide matched %ile",
            f"{null_pctile_signed:.1f}%" if null_pctile_signed is not None else "--",
            help="Ranked against MAF-matched variants from across the genome",
        )
        c4.metric(
            "|delta| matched %ile",
            f"{null_pctile_abs:.1f}%" if null_pctile_abs is not None else "--",
            help="Magnitude of disruption regardless of direction",
        )

        # Interpretation box
        pct = null_pctile_signed if null_pctile_signed is not None else locus_pctile
        if selected_name == "APOE" and null_pctile_abs is not None:
            st.info(
                f"APOE e4 (C) is the ancestral allele; Evo 2 prefers it over the "
                f"human-specific reference (T). The signed percentile is misleading "
                f"here -- magnitude is the right metric, and {paper_lead_rsid} ranks "
                f"in the top {null_pctile_abs:.1f}% by |delta|."
            )
        elif pct is not None:
            if pct < 5:
                st.success(
                    f"Strong functional support: ranks in the most disruptive "
                    f"{pct:.1f}% of MAF-matched genomic variants."
                )
            elif pct < 15:
                st.info(
                    f"Moderate functional support: ranks in the more disruptive "
                    f"tail of MAF-matched variants ({pct:.1f}th percentile)."
                )
            elif pct < 35:
                st.info(
                    f"Weak signal: somewhat more disruptive than typical, but "
                    f"not extreme ({pct:.1f}th percentile)."
                )
            elif pct < 65:
                st.warning(
                    f"No clear functional signal at this position from Evo 2 "
                    f"{model.upper()} ({pct:.1f}th percentile)."
                )
            else:
                st.warning(
                    f"Alt allele scored as more plausible than reference "
                    f"({pct:.1f}th percentile); could indicate low constraint, "
                    f"ancestral allele, or model blind spot."
                )
    else:
        st.warning(f"Paper lead {paper_lead_rsid} not found in Evo 2 scored set.")

    # ── PANEL B: GWAS vs Evo 2 scatter ─────────────────────────────────
    loci_df = load_loci_parquet(selected_name, selected_trait)
    if loci_df is not None:
        # Join evo2 scores to GWAS data on position
        merged = loci_df.merge(
            scored[["pos", "delta_log_likelihood", "rsid"]].rename(
                columns={"rsid": "evo2_rsid"}
            ),
            on="pos",
            how="inner",
        )

        if not merged.empty:
            merged["neglog10p"] = -np.log10(merged["pval"].clip(lower=1e-300))

            # Try to get PIP from finemap
            finemap_path = FINEMAP_DIR / f"{selected_name}_{selected_trait}.parquet"
            if finemap_path.exists():
                fdf = pd.read_parquet(finemap_path)
                merged = merged.merge(
                    fdf[["pos", "pip"]].drop_duplicates(subset=["pos"]),
                    on="pos",
                    how="left",
                )
                merged["pip"] = merged["pip"].fillna(0)
                color_col = merged["pip"]
                colorbar_title = "PIP"
            else:
                color_col = "#4a90d9"
                colorbar_title = None

            # Identify paper lead
            lead_pos = paper_info.get("pos")
            is_lead = merged["pos"] == lead_pos

            fig = go.Figure()

            # Background variants
            bg = merged[~is_lead]
            marker_kwargs = dict(
                size=6,
                line=dict(width=0.3, color="white"),
            )
            if colorbar_title:
                marker_kwargs["color"] = bg["pip"]
                marker_kwargs["colorscale"] = "Viridis"
                marker_kwargs["cmin"] = 0
                marker_kwargs["cmax"] = 1
                marker_kwargs["colorbar"] = dict(title=colorbar_title, thickness=12, len=0.6)
            else:
                marker_kwargs["color"] = color_col

            fig.add_trace(go.Scatter(
                x=bg["neglog10p"],
                y=bg["delta_log_likelihood"],
                mode="markers",
                marker=marker_kwargs,
                text=bg.apply(
                    lambda r: (
                        f"{r['rsid']}<br>"
                        f"p = {r['pval']:.2e}<br>"
                        f"delta_ll = {r['delta_log_likelihood']:.2f}"
                        + (f"<br>PIP = {r['pip']:.3f}" if "pip" in r.index else "")
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                name="Variants",
            ))

            # Paper lead star
            lead_data = merged[is_lead]
            if not lead_data.empty:
                fig.add_trace(go.Scatter(
                    x=lead_data["neglog10p"],
                    y=lead_data["delta_log_likelihood"],
                    mode="markers",
                    marker=dict(
                        size=14,
                        symbol="star",
                        color="red",
                        line=dict(width=1, color="darkred"),
                    ),
                    text=lead_data.apply(
                        lambda r: (
                            f"<b>{paper_lead_rsid}</b> (paper lead)<br>"
                            f"p = {r['pval']:.2e}<br>"
                            f"delta_ll = {r['delta_log_likelihood']:.2f}"
                        ),
                        axis=1,
                    ),
                    hoverinfo="text",
                    name="Paper lead",
                ))

            fig.update_layout(
                title=f"GWAS significance vs Evo 2 disruption score ({model.upper()})",
                xaxis_title="-log10(p)",
                yaxis_title="delta_log_likelihood",
                height=400,
                margin=dict(l=50, r=20, t=40, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Top-left quadrant = high GWAS significance AND predicted disruption "
                "(negative delta_ll). Variants there have independent statistical and "
                "evolutionary support."
            )

    # ── PANEL C: Locus null distribution ───────────────────────────────
    dll = scored["delta_log_likelihood"]
    fig_hist = go.Figure()

    # Null overlay
    null_df = load_genome_null(model)
    if null_df is not None and cal_matched is not None:
        mrow = cal_matched[cal_matched["locus"] == selected_name]
        if not mrow.empty and pd.notna(mrow.iloc[0].get("maf_bin")):
            maf_bin = int(mrow.iloc[0]["maf_bin"])
            null_scored_bin = null_df[
                (null_df["error"] == "") & (null_df["maf_bin"] == maf_bin)
            ]["delta_log_likelihood"]
            if not null_scored_bin.empty:
                # Normalize null to same area as locus
                scale = len(dll) / len(null_scored_bin)
                fig_hist.add_trace(go.Histogram(
                    x=null_scored_bin,
                    nbinsx=60,
                    name=f"Genome null (MAF bin {maf_bin})",
                    marker_color="rgba(150, 150, 150, 0.35)",
                    histnorm="",
                ))

    fig_hist.add_trace(go.Histogram(
        x=dll,
        nbinsx=60,
        name=f"{selected_name} locus",
        marker_color="rgba(74, 144, 217, 0.7)",
    ))

    if lead_dll is not None:
        fig_hist.add_vline(
            x=lead_dll,
            line_dash="dash",
            line_color="#d94a4a",
            annotation_text=f"{paper_lead_rsid} ({locus_pctile:.1f}%ile)" if locus_pctile else paper_lead_rsid,
            annotation_position="top right",
        )

    fig_hist.update_layout(
        title=f"Locus null distribution ({selected_name}, n={len(dll)})",
        xaxis_title="delta_log_likelihood",
        yaxis_title="Count",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── PANEL D: Methodology note ──────────────────────────────────────
    with st.expander("Methodology"):
        st.markdown(f"""
**Delta log-likelihood** measures how much a single-nucleotide variant
changes the DNA sequence model's predicted probability. A negative value
means the alt allele makes the sequence less plausible (predicted disruptive);
positive means more plausible.

- **Window size**: 8,192 bp centered on the variant
- **Model**: Evo 2 {model.upper()} (StripedHyena 2 architecture)
- **Locus null**: all scored SNVs in the locus window (typically 2,000-3,000)
- **Genome-wide null**: ~6,400 random biallelic SNVs from 1000G EUR,
  stratified by MAF (800 per bin), excluding locus regions
- **Reference**: [Evo 2 (Nguyen et al., 2025)](https://arcinstitute.org/tools/evo/evo-2)

**Caveat**: The 1B model has limited noncoding resolution. Regulatory and
intronic variants may not show strong signal until the 7B or 40B models
are scored.
""")


# ---------------------------------------------------------------------------
# Calibration cross-locus view
# ---------------------------------------------------------------------------


def view_calibration(model: str) -> None:
    """Render the cross-locus calibration tab."""
    cal_matched = load_calibration_matched(model)
    if cal_matched is None:
        st.info(
            f"No calibration data for Evo 2 {model.upper()}. "
            "Run the scoring pipeline first."
        )
        return

    st.markdown(
        f"Evo 2 {model.upper()} scoring of all seven paper lead variants. "
        "APOE serves as a positive control (coding missense, top 0.1% by |delta|). "
        "The other six loci are noncoding regulatory variants where 1B "
        "resolution is known to be limited."
    )

    # ── Calibration table ──────────────────────────────────────────────
    display_rows = []
    for _, r in cal_matched.iterrows():
        pct = r["null_percentile_signed"]
        if pd.notna(pct):
            if pct < 15:
                badge = "\U0001f7e2"
            elif pct < 35:
                badge = "\U0001f7e1"
            elif pct < 65:
                badge = "\u26aa"
            else:
                badge = "\U0001f534"
        else:
            badge = "--"

        display_rows.append({
            "Locus": r["locus"],
            "Trait": r["trait"],
            "Lead rsid": r["lead_rsid"],
            "MAF": f"{r['lead_maf']:.4f}" if pd.notna(r.get("lead_maf")) else "--",
            "delta-ll": f"{r['lead_delta_ll']:.4f}" if pd.notna(r["lead_delta_ll"]) else "--",
            "Locus %ile": f"{r['locus_percentile_signed']:.1f}%" if pd.notna(r.get("locus_percentile_signed")) else "--",
            "Matched %ile": f"{pct:.1f}%" if pd.notna(pct) else "--",
            "|delta| %ile": f"{r['null_percentile_abs']:.1f}%" if pd.notna(r.get("null_percentile_abs")) else "--",
            "Signal": badge,
        })

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Signal": st.column_config.TextColumn(width="small"),
            "Matched %ile": st.column_config.TextColumn(width="small"),
            "|delta| %ile": st.column_config.TextColumn(width="small"),
            "Locus %ile": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        "**Signal key:** "
        "\U0001f7e2 matched %ile < 15% (moderate+) "
        "\U0001f7e1 15-35% (weak) "
        "\u26aa 35-65% (no signal) "
        "\U0001f534 > 65% (wrong direction)"
    )

    # ── Comparison dot plot ────────────────────────────────────────────
    plot_df = cal_matched.dropna(
        subset=["locus_percentile_signed", "null_percentile_signed"]
    ).copy()

    if not plot_df.empty:
        plot_df = plot_df.sort_values("null_percentile_signed", ascending=False)
        trait_colors = {"mortality": "#c0392b", "hy3": "#2980b9"}

        fig = go.Figure()

        # Reference lines
        for pct in [5, 15, 50]:
            fig.add_vline(x=pct, line_dash="dot", line_color="#bdc3c7")
            fig.add_annotation(
                x=pct, y=len(plot_df) - 0.3,
                text=f"{pct}%", showarrow=False,
                font=dict(size=10, color="#7f8c8d"),
            )

        loci_names = plot_df["locus"].tolist()
        for idx, (_, r) in enumerate(plot_df.iterrows()):
            color = trait_colors.get(r["trait"], "#333")

            # Connecting line
            fig.add_trace(go.Scatter(
                x=[r["locus_percentile_signed"], r["null_percentile_signed"]],
                y=[idx, idx],
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Locus percentile (open diamond)
            fig.add_trace(go.Scatter(
                x=[r["locus_percentile_signed"]],
                y=[idx],
                mode="markers",
                marker=dict(
                    size=10, symbol="diamond-open",
                    color=color, line=dict(width=2, color=color),
                ),
                text=f"{r['locus']} locus: {r['locus_percentile_signed']:.1f}%",
                hoverinfo="text",
                showlegend=False,
            ))

            # Genome-wide (filled circle)
            fig.add_trace(go.Scatter(
                x=[r["null_percentile_signed"]],
                y=[idx],
                mode="markers",
                marker=dict(size=10, color=color),
                text=f"{r['locus']} genome: {r['null_percentile_signed']:.1f}%",
                hoverinfo="text",
                showlegend=False,
            ))

        # Legend traces (invisible data, visible legend)
        for trait, color in trait_colors.items():
            label = "Mortality" if trait == "mortality" else "HY3+"
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color),
                name=f"{label} (genome)",
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, symbol="diamond-open", color=color,
                            line=dict(width=2, color=color)),
                name=f"{label} (locus)",
            ))

        fig.update_layout(
            title="Where do the seven paper lead variants fall in their Evo 2 score distributions?",
            xaxis_title="Signed percentile (0% = most disruptive)",
            xaxis_range=[-5, 105],
            yaxis=dict(
                tickvals=list(range(len(loci_names))),
                ticktext=loci_names,
            ),
            height=max(300, len(plot_df) * 55),
            margin=dict(l=80, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Headline observations ──────────────────────────────────────────
    st.markdown("### Headline observations")
    st.markdown("""
- **APOE** is the strongest signal (top 0.1% by |delta|), confirming method
  calibration on coding variants
- **ASNS** (the variant LD-based fine-mapping cannot resolve) ranks in the
  top 7.5% against a MAF-matched null, providing independent functional support
- **TBXAS1** shows no signal in 1B despite being the second-strongest paper
  hit, consistent with its eQTL/regulatory mechanism -- candidate for 7B re-scoring
- Overall, signal magnitude correlates with variant class (coding >
  intronic-near-coding > regulatory) more than with GWAS p-value
- 7B and 40B are expected to improve noncoding resolution; results for those
  models will appear in this view when available
""")


# ---------------------------------------------------------------------------
# Benchmark view (updated with Evo 2 columns)
# ---------------------------------------------------------------------------


def view_benchmark(model: str) -> None:
    """Render the benchmark comparison table."""
    summary = load_summary()
    if summary is None:
        st.error(
            "No baseline_summary.csv found. "
            "Run: `python -m src.evaluate`"
        )
        return

    st.subheader("Pipeline vs Paper -- Benchmark Table")
    st.caption(
        "Comparing our SuSiE-RSS fine-mapping results (1000G EUR LD reference) "
        "against the lead variants reported in Tan et al. (2024)."
    )

    # Load Evo 2 calibration for extra columns
    cal_matched = load_calibration_matched(model)
    evo2_lookup: dict[str, dict] = {}
    if cal_matched is not None:
        for _, r in cal_matched.iterrows():
            evo2_lookup[r["locus"]] = {
                "delta_ll": r["lead_delta_ll"],
                "matched_pct": r["null_percentile_signed"],
            }

    # Build display table
    rows: list[dict] = []
    for _, r in summary.iterrows():
        status_raw = str(r.get("locus_status", ""))
        status_badge = _STATUS_BADGES.get(status_raw, status_raw)

        paper_pval = r.get("paper_pval")
        pval_str = f"{paper_pval:.2e}" if pd.notna(paper_pval) else "N/A"

        top_variant = str(r.get("top_variant", ""))

        top_pip = r.get("top_pip")
        pip_str = f"{top_pip:.4f}" if pd.notna(top_pip) else "N/A"

        if status_raw == "NOT_TESTABLE":
            lead_cs = "N/A"
        elif r.get("paper_lead_in_cs"):
            lead_cs = "\u2713"
        else:
            lead_cs = "\u2717"

        cs_size = r.get("cs_size")
        if status_raw == "NOT_TESTABLE":
            cs_str = "N/A"
        elif pd.notna(cs_size):
            cs_str = str(int(cs_size))
        else:
            cs_str = "N/A"

        ref_maf = r.get("reference_maf")
        if pd.notna(ref_maf):
            maf_str = f"{float(ref_maf) * 100:.2f}%"
        else:
            maf_str = "absent"

        lead_pip = r.get("paper_lead_pip")
        lead_pip_str = f"{lead_pip:.4f}" if pd.notna(lead_pip) else "N/A"

        # Evo 2 columns
        locus_name = r["locus"]
        evo2_info = evo2_lookup.get(locus_name, {})
        dll = evo2_info.get("delta_ll")
        mpct = evo2_info.get("matched_pct")

        row_dict = {
            "Locus": locus_name,
            "Trait": r["trait"],
            "Paper Lead": str(r.get("paper_lead", "")),
            "Paper p-value": pval_str,
            "Top SNP": top_variant,
            "Top PIP": pip_str,
            "Lead in CS": lead_cs,
            "Lead PIP": lead_pip_str,
            "CS Size": cs_str,
            "Ref MAF (1KG EUR)": maf_str,
            f"Evo 2 {model.upper()} delta": f"{dll:.2f}" if pd.notna(dll) else "--",
            f"Evo 2 {model.upper()} %ile": f"{mpct:.1f}%" if pd.notna(mpct) else "--",
            "Status": status_badge,
        }
        rows.append(row_dict)

    bench_df = pd.DataFrame(rows)

    st.dataframe(
        bench_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="medium"),
            "Paper p-value": st.column_config.TextColumn(width="small"),
            "Top PIP": st.column_config.TextColumn(width="small"),
            "Lead PIP": st.column_config.TextColumn(width="small"),
            "Lead in CS": st.column_config.TextColumn(width="small"),
            "CS Size": st.column_config.TextColumn(width="small"),
            "Ref MAF (1KG EUR)": st.column_config.TextColumn(width="small"),
            f"Evo 2 {model.upper()} delta": st.column_config.TextColumn(width="small"),
            f"Evo 2 {model.upper()} %ile": st.column_config.TextColumn(width="small"),
        },
    )

    # Status legend
    st.markdown(
        "**Status key:** "
        "\U0001f7e2 Paper lead replicated in credible set (PIP \u2265 0.1) \u00b7 "
        "\U0001f7e1 Signal present but not fine-resolved \u00b7 "
        "\U0001f534 Paper lead absent from reference (MAF filter) \u00b7 "
        "\u26aa SuSiE did not converge"
    )

    # Expandable reason column
    reasons = summary[summary["reason"].notna() & (summary["reason"] != "")]
    if not reasons.empty:
        with st.expander("Status details"):
            for _, r in reasons.iterrows():
                status_badge = _STATUS_BADGES.get(str(r.get("locus_status", "")), "")
                st.markdown(f"**{r['locus']}** ({r['trait']}) -- {status_badge}")
                st.caption(str(r["reason"]))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="PD Fine-Mapping Viewer",
        layout="wide",
    )

    st.title("PD Progression Fine-Mapping")

    available = discover_loci()
    if not available:
        st.error(
            "No finemap parquets found in `data/finemap/`. "
            "Run: `python -m src.finemap --all`"
        )
        return

    # -- Sidebar ---------------------------------------------------------------
    with st.sidebar:
        st.header("Navigation")

        # Model selector
        available_models = list_available_models()
        if not available_models:
            available_models = ["1b"]
        selected_model = st.selectbox(
            "Evo 2 model",
            options=available_models,
            index=0,
            help=(
                "Larger models give better noncoding resolution. "
                "7B and 40B results will appear here when available."
            ),
        )

        st.divider()

        locus_names = sorted({loc["name"] for loc in available})
        selected_name = st.selectbox("Locus", locus_names)
        st.session_state["selected_locus"] = selected_name

        traits_for_locus = sorted(
            {loc["trait"] for loc in available if loc["name"] == selected_name}
        )
        selected_trait = st.radio("Trait", traits_for_locus)
        st.session_state["selected_trait"] = selected_trait

        st.divider()
        st.caption(
            "Tan et al. (2024) -- Genome-wide determinants of mortality "
            "and motor progression in Parkinson's disease"
        )

    # -- Tabs ------------------------------------------------------------------
    tab_detail, tab_evo2, tab_calibration, tab_benchmark = st.tabs(
        ["Locus Detail", "Evo 2", "Calibration", "Benchmark"]
    )

    with tab_detail:
        view_locus_detail(available)

    with tab_evo2:
        view_evo2(selected_model)

    with tab_calibration:
        view_calibration(selected_model)

    with tab_benchmark:
        view_benchmark(selected_model)


if __name__ == "__main__":
    main()
