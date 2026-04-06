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

# Paper lead SNPs (locus_name → lead_rsid)
_PAPER_LEADS: dict[str, dict] = {
    "APOE": {"rsid": "rs429358", "trait": "mortality"},
    "TBXAS1": {"rsid": "rs4726467", "trait": "mortality"},
    "SYT10": {"rsid": "rs10437796", "trait": "mortality"},
    "MORN1": {"rsid": "rs115217673", "trait": "hy3"},
    "ASNS": {"rsid": "rs145274312", "trait": "hy3"},
    "PDE5A": {"rsid": "rs113120976", "trait": "hy3"},
    "XPO1": {"rsid": "rs141421624", "trait": "hy3"},
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def discover_loci() -> list[dict]:
    """Scan data/finemap/ for parquet files and return available (name, trait) pairs."""
    loci = []
    for p in sorted(FINEMAP_DIR.glob("*.parquet")):
        if p.name == "baseline_summary.csv":
            continue
        stem = p.stem  # e.g. "APOE_mortality"
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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_locuszoom(df: pd.DataFrame, paper_lead_rsid: str) -> go.Figure:
    """LocusZoom-style scatter: x=pos, y=-log10(p), color=PIP."""
    fig = go.Figure()

    # Main scatter — all variants
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

    # Highlight paper lead SNP with a star
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

    # Genome-wide significance line
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

    # Background bars (not in CS)
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

    # Highlighted bars (in CS)
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
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="PD Fine-Mapping Viewer",
        layout="wide",
    )

    st.title("PD Progression Fine-Mapping")

    # -- Discover available loci ----------------------------------------------
    available = discover_loci()
    if not available:
        st.error(
            "No finemap parquets found in `data/finemap/`. "
            "Run: `python -m src.finemap --all`"
        )
        return

    # -- Sidebar --------------------------------------------------------------
    with st.sidebar:
        st.header("Locus selection")

        # Locus names (unique)
        locus_names = sorted({loc["name"] for loc in available})
        selected_name = st.selectbox("Locus", locus_names)

        # Filter traits available for this locus
        traits_for_locus = sorted(
            {loc["trait"] for loc in available if loc["name"] == selected_name}
        )
        selected_trait = st.radio("Trait", traits_for_locus)

        st.divider()
        st.caption(
            "Tan et al. (2024) — Genome-wide determinants of mortality "
            "and motor progression in Parkinson's disease"
        )

    # -- Load data ------------------------------------------------------------
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

    # -- Layout: two columns --------------------------------------------------
    col_left, col_right = st.columns([3, 1])

    with col_left:
        # LocusZoom plot
        fig_lz = make_locuszoom(df, paper_lead_rsid)
        st.plotly_chart(fig_lz, use_container_width=True)

        # PIP track
        fig_pip = make_pip_track(df)
        st.plotly_chart(fig_pip, use_container_width=True)

    with col_right:
        # -- Metrics ----------------------------------------------------------
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
        lead_match = df[df["rsid"] == paper_lead_rsid]
        if not lead_match.empty:
            lr = lead_match.iloc[0]
            lead_in_cs = bool(lr["in_cs"])
            st.info(
                f"Paper lead: **{paper_lead_rsid}**\n\n"
                f"PIP = {lr['pip']:.4f} | "
                f"In CS: {'Yes' if lead_in_cs else 'No'}"
            )
        else:
            st.warning(f"Paper lead {paper_lead_rsid} not in locus data")

        # -- Credible set table -----------------------------------------------
        st.subheader("Credible set variants")
        if not cs_variants.empty:
            display_cols = ["rsid", "pos", "pip", "pval", "beta", "cs_id"]
            display_cols = [c for c in display_cols if c in cs_variants.columns]
            cs_display = cs_variants[display_cols].sort_values("pip", ascending=False)
            st.dataframe(cs_display, use_container_width=True, hide_index=True)
        else:
            st.caption("No credible set variants (purity filter may have removed all CS).")

        # -- Full table expander ----------------------------------------------
        with st.expander("Full locus table"):
            show_cols = ["rsid", "pos", "pval", "beta", "se", "pip", "in_cs", "cs_id"]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(
                df[show_cols].sort_values("pos"),
                use_container_width=True,
                hide_index=True,
                height=400,
            )


if __name__ == "__main__":
    main()
