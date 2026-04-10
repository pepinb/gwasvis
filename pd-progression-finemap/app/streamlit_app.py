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
    load_eqtl_probe_locus,
    load_eqtl_probe_scorecard,
    load_evo2_scores,
    load_genome_null,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

FINEMAP_DIR = Path(__file__).resolve().parent.parent / "data" / "finemap"
LOCI_DIR = Path(__file__).resolve().parent.parent / "data" / "loci"
SUMMARY_CSV = FINEMAP_DIR / "baseline_summary.csv"

# The three models we always show side by side
_MODELS = ["1b", "7b", "ensemble"]
_MODEL_LABELS = {"1b": "1B", "7b": "7B", "ensemble": "Ensemble"}
_MODEL_COLORS = {"1b": "steelblue", "7b": "#e67e22", "ensemble": "#2ecc71"}

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


def _fmt_pct(val, na="--") -> str:
    """Format a percentile value."""
    if pd.notna(val):
        return f"{val:.1f}%"
    return na


def _fmt_dll(val, na="--") -> str:
    """Format a delta log-likelihood value."""
    if pd.notna(val):
        return f"{val:.2f}"
    return na


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_all_matched() -> dict[str, pd.DataFrame | None]:
    """Load calibration_matched for all three models."""
    return {m: load_calibration_matched(m) for m in _MODELS}


def _load_all_summaries() -> dict[str, pd.DataFrame | None]:
    """Load calibration_summary for all three models."""
    return {m: load_calibration_summary(m) for m in _MODELS}


def _get_lead_stats(locus: str, model: str, summaries: dict, matched: dict) -> dict:
    """Extract lead variant stats for a locus/model combo."""
    stats = {
        "lead_dll": None, "locus_pctile": None,
        "null_pctile_signed": None, "null_pctile_abs": None,
        "lead_maf": None,
    }
    cal_s = summaries.get(model)
    if cal_s is not None:
        srow = cal_s[cal_s["locus"] == locus]
        if not srow.empty and srow.iloc[0]["lead_in_scored"]:
            stats["lead_dll"] = srow.iloc[0]["paper_lead_delta_ll"]
            stats["locus_pctile"] = srow.iloc[0]["percentile_signed"]

    cal_m = matched.get(model)
    if cal_m is not None:
        mrow = cal_m[cal_m["locus"] == locus]
        if not mrow.empty:
            stats["null_pctile_signed"] = mrow.iloc[0]["null_percentile_signed"]
            stats["null_pctile_abs"] = mrow.iloc[0]["null_percentile_abs"]
            maf_val = mrow.iloc[0].get("lead_maf")
            if pd.notna(maf_val):
                stats["lead_maf"] = float(maf_val)
    return stats


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
# Evo 2 per-locus view (three-model comparison)
# ---------------------------------------------------------------------------


def view_evo2() -> None:
    """Render the Evo 2 tab for the selected locus, showing 1B, 7B, and ensemble."""
    selected_name = st.session_state.get("selected_locus")
    selected_trait = st.session_state.get("selected_trait")

    if not selected_name or not selected_trait:
        st.info("Select a locus from the sidebar.")
        return

    paper_info = _PAPER_LEADS.get(selected_name, {})
    paper_lead_rsid = paper_info.get("rsid", "")

    # Load all three models' scores for this locus
    evo2_dfs: dict[str, pd.DataFrame] = {}
    scored_dfs: dict[str, pd.DataFrame] = {}
    for m in _MODELS:
        df = load_evo2_scores(selected_name, selected_trait, m)
        if df is not None:
            evo2_dfs[m] = df
            s = df[df["error"] == ""].copy()
            if not s.empty:
                scored_dfs[m] = s

    if not scored_dfs:
        st.info(
            "No Evo 2 scores available for this locus. "
            "Run `scripts/score_all_loci.py` to generate them."
        )
        return

    # Load calibration data for all models
    summaries = _load_all_summaries()
    matched = _load_all_matched()

    # Get per-model lead stats
    model_stats: dict[str, dict] = {}
    for m in _MODELS:
        model_stats[m] = _get_lead_stats(selected_name, m, summaries, matched)

    # ── PANEL A: Three-column headline metrics ────────────────────────
    has_any_lead = any(s["lead_dll"] is not None for s in model_stats.values())
    if has_any_lead:
        cols = st.columns(3)
        for i, m in enumerate(_MODELS):
            s = model_stats[m]
            label = _MODEL_LABELS[m]
            with cols[i]:
                if m == "ensemble":
                    st.markdown(f"**:green[{label}]**")
                else:
                    st.markdown(f"**{label}**")
                st.metric(
                    "delta-ll",
                    _fmt_dll(s["lead_dll"]),
                    help="Delta log-likelihood: negative = alt allele disrupts sequence plausibility",
                )
                st.metric(
                    "Matched %ile",
                    _fmt_pct(s["null_pctile_signed"]),
                    help="Ranked against MAF-matched variants from across the genome",
                )
                st.metric(
                    "|delta| %ile",
                    _fmt_pct(s["null_pctile_abs"]),
                    help="Magnitude of disruption regardless of direction",
                )

        # Interpretation box (keyed off ensemble)
        ens = model_stats.get("ensemble", {})
        pct = ens.get("null_pctile_signed")
        abs_pct = ens.get("null_pctile_abs")

        # Show range info
        pcts_1b_7b = [model_stats[m].get("null_pctile_signed") for m in ["1b", "7b"]]
        pcts_valid = [p for p in pcts_1b_7b if p is not None]

        if selected_name == "APOE" and abs_pct is not None:
            st.info(
                f"APOE e4 (C) is the ancestral allele; Evo 2 prefers it over the "
                f"human-specific reference (T). The signed percentile is misleading "
                f"here -- magnitude is the right metric, and {paper_lead_rsid} ranks "
                f"in the top {abs_pct:.1f}% by |delta|."
            )
        elif pct is not None:
            range_str = ""
            if len(pcts_valid) == 2:
                lo, hi = min(pcts_valid), max(pcts_valid)
                range_str = f" (1B/7B range: {lo:.0f}%-{hi:.0f}%)"
            if pct < 5:
                st.success(
                    f"Strong functional support: ensemble ranks in the most disruptive "
                    f"{pct:.1f}% of MAF-matched genomic variants.{range_str}"
                )
            elif pct < 15:
                st.info(
                    f"Moderate functional support: ensemble ranks in the more disruptive "
                    f"tail of MAF-matched variants ({pct:.1f}th percentile).{range_str}"
                )
            elif pct < 35:
                st.info(
                    f"Weak signal: ensemble somewhat more disruptive than typical, but "
                    f"not extreme ({pct:.1f}th percentile).{range_str}"
                )
            elif pct < 65:
                st.warning(
                    f"No clear functional signal at this position from Evo 2 ensemble "
                    f"({pct:.1f}th percentile).{range_str}"
                )
            else:
                st.warning(
                    f"Alt allele scored as more plausible than reference by ensemble "
                    f"({pct:.1f}th percentile); could indicate low constraint, "
                    f"ancestral allele, or model blind spot.{range_str}"
                )
    else:
        st.warning(f"Paper lead {paper_lead_rsid} not found in Evo 2 scored set.")

    # ── PANEL B: GWAS vs Evo 2 scatter ─────────────────────────────────
    # Model selector for scatter y-axis
    scatter_model = st.selectbox(
        "Scatter plot model",
        options=[m for m in _MODELS if m in scored_dfs],
        index=min(2, len([m for m in _MODELS if m in scored_dfs]) - 1),  # default to ensemble if available
        format_func=lambda m: _MODEL_LABELS[m],
        key="scatter_model_select",
    )
    scored = scored_dfs.get(scatter_model)

    if scored is not None:
        try:
            finemap_path = FINEMAP_DIR / f"{selected_name}_{selected_trait}.parquet"
            gwas_df = None
            has_pip = False
            if finemap_path.exists():
                gwas_df = pd.read_parquet(finemap_path)
                has_pip = "pip" in gwas_df.columns
            else:
                gwas_df = load_loci_parquet(selected_name, selected_trait)

            if gwas_df is not None and "pval" in gwas_df.columns:
                merged = gwas_df.merge(
                    scored[["pos", "delta_log_likelihood"]],
                    on="pos",
                    how="inner",
                )

                if not merged.empty:
                    merged["neglog10p"] = -np.log10(merged["pval"].clip(lower=1e-300))
                    if has_pip:
                        merged["pip"] = merged["pip"].fillna(0)

                    lead_pos = paper_info.get("pos")
                    is_lead = merged["pos"] == lead_pos

                    fig = go.Figure()

                    bg = merged[~is_lead].reset_index(drop=True)
                    marker_kwargs = dict(
                        size=6,
                        line=dict(width=0.3, color="white"),
                    )
                    if has_pip:
                        marker_kwargs["color"] = bg["pip"].tolist()
                        marker_kwargs["colorscale"] = "Viridis"
                        marker_kwargs["cmin"] = 0
                        marker_kwargs["cmax"] = 1
                        marker_kwargs["colorbar"] = dict(title="PIP", thickness=12, len=0.6)
                    else:
                        marker_kwargs["color"] = "#4a90d9"

                    fig.add_trace(go.Scatter(
                        x=bg["neglog10p"],
                        y=bg["delta_log_likelihood"],
                        mode="markers",
                        marker=marker_kwargs,
                        text=bg.apply(
                            lambda r: (
                                f"{r['rsid']}<br>"
                                f"p = {r['pval']:.2e}<br>"
                                f"delta_ll = {r['delta_log_likelihood']:.3f}"
                                + (f"<br>PIP = {r['pip']:.3f}" if has_pip else "")
                            ),
                            axis=1,
                        ),
                        hoverinfo="text",
                        name="Variants",
                    ))

                    lead_data = merged[is_lead]
                    if not lead_data.empty:
                        fig.add_trace(go.Scatter(
                            x=lead_data["neglog10p"],
                            y=lead_data["delta_log_likelihood"],
                            mode="markers",
                            marker=dict(
                                size=20,
                                symbol="star",
                                color="red",
                                line=dict(width=2, color="red"),
                            ),
                            text=lead_data.apply(
                                lambda r: (
                                    f"<b>{paper_lead_rsid}</b> (paper lead)<br>"
                                    f"p = {r['pval']:.2e}<br>"
                                    f"delta_ll = {r['delta_log_likelihood']:.3f}"
                                    + (f"<br>PIP = {r['pip']:.3f}" if has_pip else "")
                                ),
                                axis=1,
                            ),
                            hoverinfo="text",
                            name="Paper lead",
                        ))

                    fig.add_hline(y=0, line_dash="dot", line_color="#bdc3c7", line_width=1)
                    fig.add_vline(
                        x=-np.log10(5e-8), line_dash="dot", line_color="#bdc3c7",
                        line_width=1, annotation_text="p=5e-8",
                        annotation_position="top right",
                        annotation_font_size=9, annotation_font_color="#7f8c8d",
                    )

                    # Ensure paper lead is always visible on x-axis
                    lead_nlp = float(lead_data["neglog10p"].iloc[0]) if not lead_data.empty else 0
                    max_x = max(float(merged["neglog10p"].max()), lead_nlp) + 0.5
                    fig.update_layout(
                        title=f"GWAS significance vs Evo 2 {_MODEL_LABELS[scatter_model]} disruption score",
                        xaxis_title="-log10(p)",
                        yaxis_title="delta_log_likelihood",
                        height=400,
                        margin=dict(l=50, r=20, t=40, b=40),
                        showlegend=False,
                    )
                    fig.update_xaxes(range=[0, max_x])
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Bottom-right quadrant = high GWAS significance and predicted "
                        "disruption (negative delta = alt allele less plausible). Star = paper lead."
                    )
                else:
                    st.warning("No variants overlap between GWAS and Evo 2 scored data.")
            else:
                st.warning("No GWAS data available for scatter plot.")
        except Exception as exc:
            st.error(f"Error rendering GWAS vs Evo 2 scatter: {exc}")

    # ── PANEL C: Dual histogram (1B vs 7B) ─────────────────────────────
    scored_1b = scored_dfs.get("1b")
    scored_7b = scored_dfs.get("7b")

    if scored_1b is not None and scored_7b is not None:
        dll_1b = scored_1b["delta_log_likelihood"]
        dll_7b = scored_7b["delta_log_likelihood"]
        n_variants = max(len(dll_1b), len(dll_7b))

        # X-axis range: 1st-99th percentile of combined data with 10% padding
        combined = pd.concat([dll_1b, dll_7b])
        locus_lo, locus_hi = float(np.percentile(combined, 1)), float(np.percentile(combined, 99))
        pad = (locus_hi - locus_lo) * 0.10
        x_range = [locus_lo - pad, locus_hi + pad]

        fig_hist = go.Figure()

        # Null overlay (outline only, from ensemble null)
        null_maf_bin = None
        cal_m_ens = matched.get("ensemble")
        null_df = load_genome_null("ensemble")
        if null_df is not None and cal_m_ens is not None:
            mrow = cal_m_ens[cal_m_ens["locus"] == selected_name]
            if not mrow.empty and pd.notna(mrow.iloc[0].get("maf_bin")):
                null_maf_bin = int(mrow.iloc[0]["maf_bin"])
                null_scored_bin = null_df[
                    (null_df["error"] == "") & (null_df["maf_bin"] == null_maf_bin)
                ]["delta_log_likelihood"]
                if not null_scored_bin.empty:
                    fig_hist.add_trace(go.Histogram(
                        x=null_scored_bin,
                        nbinsx=60,
                        name=f"Genome null (MAF bin {null_maf_bin})",
                        marker=dict(
                            color="rgba(0, 0, 0, 0)",
                            line=dict(color="rgba(120, 120, 120, 0.5)", width=1.5),
                        ),
                        histnorm="probability density",
                    ))

        # 1B histogram
        fig_hist.add_trace(go.Histogram(
            x=dll_1b,
            nbinsx=60,
            name="1B locus distribution",
            marker_color="rgba(70, 130, 200, 0.6)",
            histnorm="probability density",
        ))

        # 7B histogram
        fig_hist.add_trace(go.Histogram(
            x=dll_7b,
            nbinsx=60,
            name="7B locus distribution",
            marker_color="rgba(230, 126, 34, 0.6)",
            histnorm="probability density",
        ))

        # Lead variant lines with staggered labels to avoid collision
        stats_1b = model_stats.get("1b", {})
        stats_7b = model_stats.get("7b", {})
        stats_ens = model_stats.get("ensemble", {})

        # Vertical lines (no built-in annotations — we add explicit annotations below)
        line_specs = []
        if stats_1b.get("lead_dll") is not None:
            pct_1b = stats_1b.get("null_pctile_signed")
            label = f"1B ({pct_1b:.0f}%)" if pct_1b is not None else "1B"
            line_specs.append((stats_1b["lead_dll"], "solid", "steelblue", label, 0.95))
        if stats_7b.get("lead_dll") is not None:
            pct_7b = stats_7b.get("null_pctile_signed")
            label = f"7B ({pct_7b:.0f}%)" if pct_7b is not None else "7B"
            line_specs.append((stats_7b["lead_dll"], "solid", "#e67e22", label, 0.85))
        if stats_ens.get("lead_dll") is not None:
            pct_ens = stats_ens.get("null_pctile_signed")
            label = f"Ens ({pct_ens:.0f}%)" if pct_ens is not None else "Ens"
            line_specs.append((stats_ens["lead_dll"], "dash", "#2ecc71", label, 0.75))

        for dll_val, dash, color, label, y_pos in line_specs:
            fig_hist.add_vline(
                x=dll_val, line_dash=dash, line_color=color, line_width=2,
            )
            # Place label at right of line if in left half, else left of line
            x_mid = (x_range[0] + x_range[1]) / 2
            fig_hist.add_annotation(
                x=dll_val, y=y_pos, xref="x", yref="paper",
                text=label, showarrow=False,
                font=dict(size=11, color=color, weight="bold"),
                xanchor="left" if dll_val < x_mid else "right",
                xshift=4 if dll_val < x_mid else -4,
                yanchor="top",
            )

        fig_hist.update_layout(
            title=f"Locus null distribution ({selected_name}, n={n_variants}) -- 1B vs 7B",
            xaxis_title="delta_log_likelihood",
            yaxis_title="Density",
            height=400,
            margin=dict(l=50, r=20, t=40, b=40),
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_hist.update_xaxes(range=x_range)

        if null_maf_bin is not None:
            fig_hist.add_annotation(
                x=x_range[1], y=1, xref="x", yref="paper",
                text="Genome null clipped to locus range",
                showarrow=False, font=dict(size=9, color="#999"),
                xanchor="right", yanchor="top",
            )

        st.plotly_chart(fig_hist, use_container_width=True)
    elif scored_dfs:
        # Fallback: show single-model histogram if only one model available
        m = list(scored_dfs.keys())[0]
        dll = scored_dfs[m]["delta_log_likelihood"]
        locus_lo, locus_hi = float(np.percentile(dll, 1)), float(np.percentile(dll, 99))
        pad = (locus_hi - locus_lo) * 0.10
        x_range = [locus_lo - pad, locus_hi + pad]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=dll, nbinsx=60, name=f"{_MODEL_LABELS[m]} variants",
            marker_color="rgba(70, 130, 200, 0.7)", histnorm="probability density",
        ))
        fig_hist.update_layout(
            title=f"Locus null distribution ({selected_name}, n={len(dll)})",
            xaxis_title="delta_log_likelihood", yaxis_title="Density",
            height=350, margin=dict(l=50, r=20, t=40, b=40),
        )
        fig_hist.update_xaxes(range=x_range)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── PANEL D: Methodology note ──────────────────────────────────────
    with st.expander("Methodology"):
        st.markdown("""
**Delta log-likelihood** measures how much a single-nucleotide variant
changes the DNA sequence model's predicted probability. A negative value
means the alt allele makes the sequence less plausible (predicted disruptive);
positive means more plausible.

- **Window size**: 8,192 bp centered on the variant
- **Models**: Evo 2 1B and 7B (StripedHyena 2 architecture); ensemble = mean of both
- **Locus null**: all scored SNVs in the locus window (typically 2,000-3,000)
- **Genome-wide null**: ~6,400 random biallelic SNVs from 1000G EUR,
  stratified by MAF (800 per bin), excluding locus regions
- **Model agreement**: Global Spearman rho = 0.695 between 1B and 7B on the
  genome-wide null. All seven leads are direction-concordant. Ensemble
  reduces per-model noise, especially in the 5%-25% range.
- **Reference**: [Evo 2 (Nguyen et al., 2025)](https://arcinstitute.org/tools/evo/evo-2)
""")


# ---------------------------------------------------------------------------
# Calibration cross-locus view (three-model comparison)
# ---------------------------------------------------------------------------


def view_calibration() -> None:
    """Render the cross-locus calibration tab with 1B, 7B, and ensemble."""
    all_matched = _load_all_matched()

    # Check we have at least one model
    available = {m: df for m, df in all_matched.items() if df is not None}
    if not available:
        st.info("No calibration data available. Run the scoring pipeline first.")
        return

    st.markdown(
        "Evo 2 scoring of all seven paper lead variants across 1B, 7B, and ensemble models. "
        "APOE serves as a positive control (coding missense). "
        "The Range column shows the spread between single-model percentiles."
    )

    # ── Three-model calibration table ─────────────────────────────────
    display_rows = []
    for locus_name, paper_info in _PAPER_LEADS.items():
        row = {
            "Locus": locus_name,
            "Trait": paper_info["trait"],
            "Lead rsid": paper_info["rsid"],
        }

        # MAF (same across models)
        maf_shown = False
        for m in _MODELS:
            df = all_matched.get(m)
            if df is not None:
                mrow = df[df["locus"] == locus_name]
                if not mrow.empty and pd.notna(mrow.iloc[0].get("lead_maf")):
                    row["MAF"] = f"{mrow.iloc[0]['lead_maf']:.4f}"
                    maf_shown = True
                    break
        if not maf_shown:
            row["MAF"] = "--"

        pcts_for_range = []
        for m in _MODELS:
            label = _MODEL_LABELS[m]
            df = all_matched.get(m)
            if df is not None:
                mrow = df[df["locus"] == locus_name]
                if not mrow.empty:
                    dll = mrow.iloc[0]["lead_delta_ll"]
                    pct = mrow.iloc[0]["null_percentile_signed"]
                    row[f"{label} delta"] = _fmt_dll(dll)
                    row[f"{label} match%"] = _fmt_pct(pct)
                    if m in ("1b", "7b") and pd.notna(pct):
                        pcts_for_range.append(pct)
                else:
                    row[f"{label} delta"] = "--"
                    row[f"{label} match%"] = "--"
            else:
                row[f"{label} delta"] = "--"
                row[f"{label} match%"] = "--"

        if len(pcts_for_range) == 2:
            lo, hi = min(pcts_for_range), max(pcts_for_range)
            row["Range"] = f"{lo:.0f}% - {hi:.0f}%"
        else:
            row["Range"] = "--"

        display_rows.append(row)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAF": st.column_config.TextColumn(width="small"),
            "1B delta": st.column_config.TextColumn(width="small"),
            "1B match%": st.column_config.TextColumn(width="small"),
            "7B delta": st.column_config.TextColumn(width="small"),
            "7B match%": st.column_config.TextColumn(width="small"),
            "Ensemble delta": st.column_config.TextColumn(width="small"),
            "Ensemble match%": st.column_config.TextColumn(width="small"),
            "Range": st.column_config.TextColumn(width="small"),
        },
    )

    # ── Three-marker comparison dot plot ──────────────────────────────
    # Build plot data from all three models
    cal_ens = all_matched.get("ensemble")
    cal_1b = all_matched.get("1b")
    cal_7b = all_matched.get("7b")

    if cal_ens is not None:
        # Sort by ensemble percentile
        plot_base = cal_ens.dropna(subset=["null_percentile_signed"]).copy()
        plot_base = plot_base.sort_values("null_percentile_signed", ascending=False)

        if not plot_base.empty:
            trait_colors = {"mortality": "#c0392b", "hy3": "#2980b9"}

            fig = go.Figure()

            # Reference lines
            for pct in [5, 15, 50]:
                fig.add_vline(x=pct, line_dash="dot", line_color="#bdc3c7")
                fig.add_annotation(
                    x=pct, y=len(plot_base) - 0.3,
                    text=f"{pct}%", showarrow=False,
                    font=dict(size=10, color="#7f8c8d"),
                )

            loci_names = plot_base["locus"].tolist()
            for idx, (_, r_ens) in enumerate(plot_base.iterrows()):
                locus = r_ens["locus"]
                trait = r_ens["trait"]
                color = trait_colors.get(trait, "#333")

                # Get percentiles for all three models
                pct_ens = r_ens["null_percentile_signed"]
                pct_1b = None
                pct_7b = None
                if cal_1b is not None:
                    r1b = cal_1b[cal_1b["locus"] == locus]
                    if not r1b.empty:
                        pct_1b = r1b.iloc[0]["null_percentile_signed"]
                if cal_7b is not None:
                    r7b = cal_7b[cal_7b["locus"] == locus]
                    if not r7b.empty:
                        pct_7b = r7b.iloc[0]["null_percentile_signed"]

                # Connecting line across all available points
                all_pcts = [p for p in [pct_1b, pct_7b, pct_ens] if pd.notna(p)]
                if len(all_pcts) >= 2:
                    fig.add_trace(go.Scatter(
                        x=[min(all_pcts), max(all_pcts)],
                        y=[idx, idx],
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                # 1B (open circle)
                if pd.notna(pct_1b):
                    fig.add_trace(go.Scatter(
                        x=[pct_1b], y=[idx],
                        mode="markers",
                        marker=dict(
                            size=9, symbol="circle-open",
                            color=color, line=dict(width=2, color=color),
                        ),
                        text=f"{locus} 1B: {pct_1b:.1f}%",
                        hoverinfo="text",
                        showlegend=False,
                    ))

                # 7B (filled circle)
                if pd.notna(pct_7b):
                    fig.add_trace(go.Scatter(
                        x=[pct_7b], y=[idx],
                        mode="markers",
                        marker=dict(size=9, color=color),
                        text=f"{locus} 7B: {pct_7b:.1f}%",
                        hoverinfo="text",
                        showlegend=False,
                    ))

                # Ensemble (star)
                if pd.notna(pct_ens):
                    fig.add_trace(go.Scatter(
                        x=[pct_ens], y=[idx],
                        mode="markers",
                        marker=dict(
                            size=12, symbol="star",
                            color=color, line=dict(width=1, color=color),
                        ),
                        text=f"{locus} ensemble: {pct_ens:.1f}%",
                        hoverinfo="text",
                        showlegend=False,
                    ))

            # Legend traces
            for trait, color in trait_colors.items():
                label = "Mortality" if trait == "mortality" else "HY3+"
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=9, symbol="circle-open", color=color,
                                line=dict(width=2, color=color)),
                    name=f"{label} (1B)",
                ))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=9, color=color),
                    name=f"{label} (7B)",
                ))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=12, symbol="star", color=color,
                                line=dict(width=1, color=color)),
                    name=f"{label} (Ensemble)",
                ))

            fig.update_layout(
                title=(
                    "Where do the seven paper lead variants fall in their Evo 2 "
                    "score distributions? (1B, 7B, and ensemble)"
                ),
                xaxis_title="Signed percentile (0% = most disruptive)",
                xaxis_range=[-5, 105],
                yaxis=dict(
                    tickvals=list(range(len(loci_names))),
                    ticktext=loci_names,
                ),
                height=max(300, len(plot_base) * 60),
                margin=dict(l=80, r=20, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Headline observations ──────────────────────────────────────────
    st.markdown("### Headline observations")
    st.markdown("""
- **APOE** is the strongest signal (top 0.1% by |delta| in both models),
  confirming method calibration on coding variants
- **ASNS** (the variant LD-based fine-mapping cannot resolve) ranks at
  ensemble 11.0% against a MAF-matched null, providing independent
  functional support
- **TBXAS1** shows no signal in either 1B or 7B (ensemble 48.9%), consistent
  with its eQTL/regulatory mechanism
- Overall, signal magnitude correlates with variant class (coding >
  intronic-near-coding > regulatory) more than with GWAS p-value
- 1B/7B global Spearman rho = 0.695; the ensemble reduces per-model noise,
  especially in the 5%-25% percentile range where model choice matters most
""")


# ---------------------------------------------------------------------------
# Benchmark view (three-model Evo 2 columns)
# ---------------------------------------------------------------------------


def view_benchmark() -> None:
    """Render the benchmark comparison table with 1B, 7B, and ensemble Evo 2 columns."""
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

    # Load Evo 2 calibration for all three models
    all_matched = _load_all_matched()
    evo2_lookup: dict[str, dict[str, dict]] = {}  # locus -> model -> {delta_ll, matched_pct}
    for m in _MODELS:
        df = all_matched.get(m)
        if df is not None:
            for _, r in df.iterrows():
                locus = r["locus"]
                if locus not in evo2_lookup:
                    evo2_lookup[locus] = {}
                evo2_lookup[locus][m] = {
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

        locus_name = r["locus"]
        locus_evo2 = evo2_lookup.get(locus_name, {})

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
            "Ref MAF": maf_str,
        }

        for m in _MODELS:
            label = _MODEL_LABELS[m]
            info = locus_evo2.get(m, {})
            dll = info.get("delta_ll")
            mpct = info.get("matched_pct")
            row_dict[f"{label} delta"] = _fmt_dll(dll)
            row_dict[f"{label} %ile"] = _fmt_pct(mpct)

        row_dict["Status"] = status_badge
        rows.append(row_dict)

    bench_df = pd.DataFrame(rows)

    col_config = {
        "Status": st.column_config.TextColumn(width="medium"),
        "Paper p-value": st.column_config.TextColumn(width="small"),
        "Top PIP": st.column_config.TextColumn(width="small"),
        "Lead PIP": st.column_config.TextColumn(width="small"),
        "Lead in CS": st.column_config.TextColumn(width="small"),
        "CS Size": st.column_config.TextColumn(width="small"),
        "Ref MAF": st.column_config.TextColumn(width="small"),
    }
    for m in _MODELS:
        label = _MODEL_LABELS[m]
        col_config[f"{label} delta"] = st.column_config.TextColumn(width="small")
        col_config[f"{label} %ile"] = st.column_config.TextColumn(width="small")

    st.dataframe(
        bench_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
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
# eQTL probe view (prompt 8C)
# ---------------------------------------------------------------------------


def _tier_label(pct: float | None) -> str:
    if pct is None or pd.isna(pct):
        return "--"
    if pct >= 95:
        return "STRONG"
    if pct >= 80:
        return "MODERATE"
    if pct >= 50:
        return "WEAK"
    return "NONE"


def _pct_color(pct: float | None) -> str:
    """Background color for a percentile cell."""
    if pct is None or pd.isna(pct):
        return "#ffffff"
    if pct >= 90:
        return "#2e7d32"  # green
    if pct >= 75:
        return "#a5d6a7"  # light green
    if pct >= 50:
        return "#fff59d"  # yellow
    return "#ef9a9a"  # red


def _interpret(wb_pct: float, bc_pct: float) -> str:
    wb_t = _tier_label(wb_pct)
    bc_t = _tier_label(bc_pct)
    if wb_t in ("STRONG", "MODERATE") and bc_t in ("STRONG", "MODERATE"):
        return "Dual-tissue signal"
    if wb_t in ("STRONG", "MODERATE"):
        return "Blood-specific signal"
    if bc_t in ("STRONG", "MODERATE"):
        return "Brain-specific signal"
    return "No strong probe signal"


def view_eqtl_probe() -> None:
    """Render the eQTL Probe tab."""
    st.markdown(
        "Evo 2 7B embeddings at layer `blocks.26` are used as features for a "
        "logistic regression probe trained to distinguish fine-mapped eQTLs "
        "(GTEx v8 DAP-G, PIP > 0.5) from matched non-eQTL controls. Probes "
        "are trained per tissue on chromosomes **other than** the PD locus "
        "chromosomes, then applied to the seven PD progression lead variants "
        "as an out-of-distribution test. Higher locus percentiles indicate "
        "the probe thinks the variant is more likely to be a regulatory "
        "eQTL in that tissue."
    )

    scorecard = load_eqtl_probe_scorecard()
    if scorecard is None or scorecard.empty:
        st.error(
            "No probe scorecard found at `data/eqtl/probe_scorecard.csv`. "
            "Run `scripts/score_pd_loci_with_probes.py`."
        )
        return

    # --- Summary table with paper rsIDs substituted in ----------------------
    display = scorecard.copy()
    paper_rsid = {k: v.get("rsid", "") for k, v in _PAPER_LEADS.items()}
    display["lead rsID"] = display["locus"].map(paper_rsid)
    display["interpretation"] = [
        _interpret(r["wb_lead_percentile"], r["bc_lead_percentile"])
        for _, r in display.iterrows()
    ]

    summary_view = display[[
        "locus",
        "trait",
        "lead rsID",
        "paper_eqtl_status",
        "wb_lead_percentile",
        "bc_lead_percentile",
        "interpretation",
    ]].rename(columns={
        "paper_eqtl_status": "paper eQTL status",
        "wb_lead_percentile": "blood %ile",
        "bc_lead_percentile": "brain %ile",
    })

    def _row_style(row):
        styles = [""] * len(row)
        wb_idx = list(row.index).index("blood %ile")
        bc_idx = list(row.index).index("brain %ile")
        styles[wb_idx] = f"background-color: {_pct_color(row['blood %ile'])}"
        styles[bc_idx] = f"background-color: {_pct_color(row['brain %ile'])}"
        return styles

    styled = (
        summary_view.style
        .apply(_row_style, axis=1)
        .format({"blood %ile": "{:.1f}", "brain %ile": "{:.1f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Cell color: \U0001f7e2 \u226595 or \u226590 \u00b7 "
        "\U0001f7e2 (light) 75-90 \u00b7 \U0001f7e1 50-75 \u00b7 \U0001f534 <50"
    )

    st.divider()

    # --- Locus drilldown ----------------------------------------------------
    loci_available = list(display["locus"])
    default_idx = loci_available.index("TBXAS1") if "TBXAS1" in loci_available else 0
    selected = st.selectbox(
        "Locus drilldown", loci_available, index=default_idx, key="eqtl_drilldown"
    )
    sel_row = display[display["locus"] == selected].iloc[0]
    sel_trait = sel_row["trait"]

    locus_df = load_eqtl_probe_locus(selected, sel_trait)
    if locus_df is None or locus_df.empty:
        st.warning(
            f"No per-variant parquet found for {selected}_{sel_trait}. "
            "Run `scripts/score_pd_loci_with_probes.py`."
        )
        return

    lead = locus_df[locus_df["is_paper_lead"]]
    if lead.empty:
        st.warning("Paper lead row not found in this locus's parquet.")
        return
    lead_row = lead.iloc[0]
    paper_rsid_str = paper_rsid.get(selected, lead_row["rsid"])

    col1, col2 = st.columns(2)

    # Panel 1 — blood histogram
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=locus_df["wb_prob"],
            nbinsx=40,
            marker_color="#4a90d9",
            name="Locus variants",
        ))
        fig.add_vline(
            x=float(lead_row["wb_prob"]),
            line=dict(color="red", width=2),
            annotation_text=(
                f"{paper_rsid_str}<br>pct={lead_row['wb_percentile']:.1f}"
            ),
            annotation_position="top right",
        )
        fig.update_layout(
            title="Whole_Blood probe probability",
            xaxis_title="probe P(eQTL)",
            yaxis_title="n variants",
            height=360,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Panel 2 — brain histogram
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=locus_df["bc_prob"],
            nbinsx=40,
            marker_color="#9b59b6",
            name="Locus variants",
        ))
        fig.add_vline(
            x=float(lead_row["bc_prob"]),
            line=dict(color="red", width=2),
            annotation_text=(
                f"{paper_rsid_str}<br>pct={lead_row['bc_percentile']:.1f}"
            ),
            annotation_position="top right",
        )
        fig.update_layout(
            title="Brain_Cortex probe probability",
            xaxis_title="probe P(eQTL)",
            yaxis_title="n variants",
            height=360,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns([3, 2])

    # Panel 3 — scatter blood vs brain, colored by distance to lead
    with col3:
        bg = locus_df[~locus_df["is_paper_lead"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bg["wb_prob"],
            y=bg["bc_prob"],
            mode="markers",
            marker=dict(
                size=6,
                color=bg["distance_to_lead_hg38"].abs(),
                colorscale="Viridis",
                colorbar=dict(title="|dist to lead| (bp)", thickness=12, len=0.6),
                line=dict(width=0.3, color="white"),
            ),
            text=bg["rsid"],
            hovertemplate=(
                "%{text}<br>wb=%{x:.3f}<br>bc=%{y:.3f}<extra></extra>"
            ),
            name="Locus variants",
        ))
        fig.add_trace(go.Scatter(
            x=[float(lead_row["wb_prob"])],
            y=[float(lead_row["bc_prob"])],
            mode="markers",
            marker=dict(size=20, symbol="star", color="red",
                        line=dict(width=2, color="red")),
            text=[paper_rsid_str],
            hovertemplate="%{text}<br>wb=%{x:.3f}<br>bc=%{y:.3f}<extra></extra>",
            name="Paper lead",
        ))
        fig.update_layout(
            title="Blood probe vs Brain probe",
            xaxis_title="Whole_Blood P(eQTL)",
            yaxis_title="Brain_Cortex P(eQTL)",
            height=420,
            margin=dict(l=40, r=20, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Panel 4 — paper annotation + interpretation box
    with col4:
        st.markdown(f"### {selected} — {paper_rsid_str}")
        st.markdown(f"**Paper eQTL annotation:** {sel_row['paper_eqtl_status']}")
        st.markdown(
            f"**Blood probe:** P={lead_row['wb_prob']:.3f}, "
            f"percentile={lead_row['wb_percentile']:.1f} "
            f"(tier **{sel_row['wb_tier']}**)"
        )
        st.markdown(
            f"**Brain probe:** P={lead_row['bc_prob']:.3f}, "
            f"percentile={lead_row['bc_percentile']:.1f} "
            f"(tier **{sel_row['bc_tier']}**)"
        )
        st.markdown(f"**Verdict:** {sel_row['interpretation']}")
        st.caption(
            f"Locus {sel_row['chrom']}:{sel_row['hg38_bp']} (hg38) \u00b7 "
            f"{sel_row['n_variants']} variants scored"
        )

    st.divider()

    # --- Comparison vs 7B likelihood ensemble -------------------------------
    st.subheader("Comparison to likelihood scoring")

    matched_ens = load_calibration_matched("ensemble")
    if matched_ens is None or matched_ens.empty:
        st.info(
            "No `data/evo2/calibration_matched_ensemble.csv` found — showing "
            "probe results only."
        )
        cmp_view = display[[
            "locus", "lead rsID", "wb_lead_percentile", "bc_lead_percentile"
        ]].rename(columns={
            "wb_lead_percentile": "blood probe %ile",
            "bc_lead_percentile": "brain probe %ile",
        })
        st.dataframe(cmp_view, use_container_width=True, hide_index=True)
        return

    # Loci that are not eQTL test cases (coding variants etc.) — flag with
    # a footnote rather than counting "embeddings added info" normally.
    _CODING_FOOTNOTE_LOCI = {"APOE"}

    merged = display.merge(
        matched_ens[["locus", "null_percentile_signed", "null_percentile_abs"]],
        on="locus",
        how="left",
    ).rename(columns={
        "null_percentile_signed": "7B matched %ile",
        "null_percentile_abs": "7B |delta| %ile",
    })

    def _flag(row) -> str:
        """Compare probe percentiles against the MATCHED (signed) baseline.

        The signed/matched null is the metric that originally placed TBXAS1
        at ~47 pct and motivated the embeddings phase — so that's the right
        baseline for "did embeddings add information".
        """
        matched = row.get("7B matched %ile")
        if pd.isna(matched):
            return "--"
        wb = row.get("wb_lead_percentile")
        bc = row.get("bc_lead_percentile")
        diverges = (
            (pd.notna(wb) and abs(wb - matched) > 30)
            or (pd.notna(bc) and abs(bc - matched) > 30)
        )
        if row["locus"] in _CODING_FOOTNOTE_LOCI:
            return "yes\u2020" if diverges else "no\u2020"
        return "yes" if diverges else "no"

    merged["embeddings add info?"] = merged.apply(_flag, axis=1)

    # Decorate the locus label with a footnote marker for coding-variant loci
    merged["locus_display"] = merged["locus"].apply(
        lambda loc: f"{loc}\u2020" if loc in _CODING_FOOTNOTE_LOCI else loc
    )

    cmp_view = merged[[
        "locus_display",
        "lead rsID",
        "7B matched %ile",
        "7B |delta| %ile",
        "wb_lead_percentile",
        "bc_lead_percentile",
        "embeddings add info?",
    ]].rename(columns={
        "locus_display": "locus",
        "wb_lead_percentile": "blood probe %ile",
        "bc_lead_percentile": "brain probe %ile",
    })

    def _cmp_style(row):
        val = str(row["embeddings add info?"]).rstrip("\u2020")
        if val == "yes":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    styled_cmp = cmp_view.style.apply(_cmp_style, axis=1).format({
        "7B matched %ile": "{:.1f}",
        "7B |delta| %ile": "{:.1f}",
        "blood probe %ile": "{:.1f}",
        "brain probe %ile": "{:.1f}",
    })
    st.dataframe(styled_cmp, use_container_width=True, hide_index=True)
    st.caption(
        "**Matched %ile** is the signed test from the existing Evo 2 tab — the "
        "null metric that motivated the embeddings phase (TBXAS1 sits at ~49 "
        "pct here, which is why scalar likelihood scoring could not flag it). "
        "**|delta| %ile** is the magnitude test, shown for reference. Rows "
        "where either probe percentile differs from the matched %ile by more "
        "than 30 points are highlighted as cases where embeddings added "
        "information beyond scalar likelihood scoring. "
        "\u2020 APOE is a coding variant, not an eQTL test case; its "
        "matched/probe disagreement reflects that categorical difference "
        "rather than embeddings adding eQTL information."
    )


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
    (
        tab_detail,
        tab_evo2,
        tab_calibration,
        tab_benchmark,
        tab_eqtl,
    ) = st.tabs(
        ["Locus Detail", "Evo 2", "Calibration", "Benchmark", "eQTL Probe"]
    )

    with tab_detail:
        view_locus_detail(available)

    with tab_evo2:
        view_evo2()

    with tab_calibration:
        view_calibration()

    with tab_benchmark:
        view_benchmark()

    with tab_eqtl:
        view_eqtl_probe()


if __name__ == "__main__":
    main()
