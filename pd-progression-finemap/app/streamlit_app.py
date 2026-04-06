"""Streamlit dashboard for interactive exploration of PD fine-mapping results.

Run with:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

FINEMAP_DIR = Path(__file__).resolve().parent.parent / "data" / "finemap"
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
# Data loading
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
    # -- Sidebar locus selection (already rendered) --
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


def view_benchmark() -> None:
    """Render the benchmark comparison table."""
    summary = load_summary()
    if summary is None:
        st.error(
            "No baseline_summary.csv found. "
            "Run: `python -m src.evaluate`"
        )
        return

    st.subheader("Pipeline vs Paper — Benchmark Table")
    st.caption(
        "Comparing our SuSiE-RSS fine-mapping results (1000G EUR LD reference) "
        "against the lead variants reported in Tan et al. (2024)."
    )

    # Build display table
    rows: list[dict] = []
    for _, r in summary.iterrows():
        status_raw = str(r.get("locus_status", ""))
        status_badge = _STATUS_BADGES.get(status_raw, status_raw)

        # Paper p-value
        paper_pval = r.get("paper_pval")
        pval_str = f"{paper_pval:.2e}" if pd.notna(paper_pval) else "N/A"

        # Pipeline top SNP
        top_variant = str(r.get("top_variant", ""))

        # Top PIP
        top_pip = r.get("top_pip")
        pip_str = f"{top_pip:.4f}" if pd.notna(top_pip) else "N/A"

        # Paper lead in CS
        if status_raw == "NOT_TESTABLE":
            lead_cs = "N/A"
        elif r.get("paper_lead_in_cs"):
            lead_cs = "\u2713"
        else:
            lead_cs = "\u2717"

        # CS size
        cs_size = r.get("cs_size")
        if status_raw == "NOT_TESTABLE":
            cs_str = "N/A"
        elif pd.notna(cs_size):
            cs_str = str(int(cs_size))
        else:
            cs_str = "N/A"

        # Reference MAF
        ref_maf = r.get("reference_maf")
        if pd.notna(ref_maf):
            maf_str = f"{float(ref_maf) * 100:.2f}%"
        else:
            maf_str = "absent"

        # Paper lead PIP
        lead_pip = r.get("paper_lead_pip")
        lead_pip_str = f"{lead_pip:.4f}" if pd.notna(lead_pip) else "N/A"

        rows.append({
            "Locus": r["locus"],
            "Trait": r["trait"],
            "Paper Lead": str(r.get("paper_lead", "")),
            "Paper p-value": pval_str,
            "Top SNP": top_variant,
            "Top PIP": pip_str,
            "Lead in CS": lead_cs,
            "Lead PIP": lead_pip_str,
            "CS Size": cs_str,
            "Ref MAF (1KG EUR)": maf_str,
            "Status": status_badge,
        })

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
        },
    )

    # Status legend
    st.markdown(
        "**Status key:** "
        "\U0001f7e2 Paper lead replicated in credible set (PIP \u2265 0.1) · "
        "\U0001f7e1 Signal present but not fine-resolved · "
        "\U0001f534 Paper lead absent from reference (MAF filter) · "
        "\u26aa SuSiE did not converge"
    )

    # Expandable reason column
    reasons = summary[summary["reason"].notna() & (summary["reason"] != "")]
    if not reasons.empty:
        with st.expander("Status details"):
            for _, r in reasons.iterrows():
                status_badge = _STATUS_BADGES.get(str(r.get("locus_status", "")), "")
                st.markdown(f"**{r['locus']}** ({r['trait']}) — {status_badge}")
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
            "Tan et al. (2024) — Genome-wide determinants of mortality "
            "and motor progression in Parkinson's disease"
        )

    # -- Tabs ------------------------------------------------------------------
    tab_detail, tab_benchmark = st.tabs(["Locus Detail", "Benchmark"])

    with tab_detail:
        view_locus_detail(available)

    with tab_benchmark:
        view_benchmark()


if __name__ == "__main__":
    main()
