
# inventory_streamlit_app.py
# Streamlit dashboard for Inventory Views (packs / non-packs) using Plotly
# - Drop this file next to your CSVs and run:  streamlit run inventory_streamlit_app.py

from typing import Optional, Iterable, Dict, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------- Helpers ----------------------------

def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _sorted_unique(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _validate_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

def _month_key(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.replace(r"^(\d{4})\D?(\d{2})$", r"\1-\2", regex=True)
    dt = pd.to_datetime(s2, errors="coerce")
    return dt.dt.to_period("M").astype(str)

def _order_availability(present_labels: List[str]) -> List[str]:
    priority = ["All100", "Core Instock", "Core OOS", "Fringe Available", "No_Sales"]
    canon_to_present = { _canon(x): x for x in present_labels }
    ordered = []
    for p in priority:
        k = _canon(p)
        if k in canon_to_present:
            ordered.append(canon_to_present[k])
    for x in present_labels:
        if x not in ordered:
            ordered.append(x)
    return ordered

def _exclusive_priority_lookup(present_labels: List[str]) -> Dict[str, int]:
    priority = ["All100", "Core Instock", "Core OOS", "Fringe Available", "No_Sales"]
    wanted = { _canon(p): i+1 for i, p in enumerate(priority) }
    return { lbl: wanted.get(_canon(lbl), 9999) for lbl in present_labels }

def _apply_filters(df: pd.DataFrame, channel=None, cat_l2=None, cat_l3=None) -> pd.DataFrame:
    def _mask(col, values):
        if col not in df.columns or values is None:
            return pd.Series(True, index=df.index)
        if isinstance(values, (str, int)):
            values = [values]
        vset = set(str(v).strip().lower() for v in values)
        return df[col].astype(str).str.strip().str.lower().isin(vset)
    m = _mask("Channel", channel) & _mask("cat_l2", cat_l2) & _mask("cat_l3", cat_l3)
    return df.loc[m].copy()

def _agg_counts(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    out = g.agg(
        option_count=("optioncode", lambda s: s.dropna().nunique()),
        revenue=("Revenue", "sum")
    ).reset_index()
    return out

def _make_100pct_stacked(fig_title: str, index_labels: List[str], series_labels: List[str],
                         matrix_pct: np.ndarray, matrix_count: np.ndarray, matrix_rev: np.ndarray,
                         matrix_rev_pct: np.ndarray) -> go.Figure:
    data_traces = []
    for j, series in enumerate(series_labels):
        custom = np.stack([matrix_count[:, j], matrix_pct[:, j], matrix_rev[:, j], matrix_rev_pct[:, j]], axis=-1)
        data_traces.append(
            go.Bar(
                name=series, x=index_labels, y=matrix_pct[:, j], customdata=custom,
                hovertemplate=(
                    "<b>%{x}</b><br>" + series + "<br>" +
                    "Count: %{customdata[0]}<br>" +
                    "Share: %{customdata[1]:.2f}%<br>" +
                    "Revenue: ₹%{customdata[2]:,.0f}<br>" +
                    "Revenue Share: %{customdata[3]:.2f}%<extra></extra>"
                )
            )
        )
    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title=fig_title, barmode="stack",
        yaxis=dict(title="% (100% split)", range=[0, 100]),
        bargap=0.15, legend_title="Availability", hovermode="x unified"
    )
    return fig

def _make_lines(fig_title: str, x_labels: List[str], series_labels: List[str],
                matrix_y: np.ndarray, matrix_count: np.ndarray, matrix_rev: np.ndarray,
                matrix_rev_pct: np.ndarray, ytitle: str) -> go.Figure:
    traces = []
    for j, series in enumerate(series_labels):
        custom = np.stack([matrix_count[:, j], matrix_y[:, j], matrix_rev[:, j], matrix_rev_pct[:, j]], axis=-1)
        traces.append(
            go.Scatter(
                name=series, x=x_labels, y=matrix_y[:, j],
                mode="lines+markers", customdata=custom,
                hovertemplate=(
                    "<b>%{x}</b><br>" + series + "<br>" +
                    "Count: %{customdata[0]}<br>" +
                    "Value: %{customdata[1]:.2f}<br>" +
                    "Revenue: ₹%{customdata[2]:,.0f}<br>" +
                    "Revenue Share: %{customdata[3]:.2f}%<extra></extra>"
                )
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(title=fig_title, yaxis_title=ytitle, legend_title="Availability", hovermode="x unified")
    return fig

def _grouped_bar(fig_title: str, x_labels: List[str], series_labels: List[str],
                 matrix_y: np.ndarray, matrix_count: np.ndarray, matrix_rev: np.ndarray,
                 matrix_rev_pct: np.ndarray, ytitle: str) -> go.Figure:
    traces = []
    for j, series in enumerate(series_labels):
        custom = np.stack([matrix_count[:, j], matrix_y[:, j], matrix_rev[:, j], matrix_rev_pct[:, j]], axis=-1)
        traces.append(
            go.Bar(
                name=series, x=x_labels, y=matrix_y[:, j], customdata=custom,
                hovertemplate=(
                    "<b>%{x}</b><br>" + series + "<br>" +
                    "Count: %{customdata[0]}<br>" +
                    ytitle + ": %{customdata[1]:.2f}<br>" +
                    "Revenue: ₹%{customdata[2]:,.0f}<br>" +
                    "Revenue Share: %{customdata[3]:.2f}%<extra></extra>"
                )
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(title=fig_title, barmode="group", yaxis_title=ytitle, legend_title="Availability", hovermode="x unified")
    return fig

def build_inventory_figures(
    df: pd.DataFrame,
    channel: Optional[Iterable[str]] = None,
    cat_l2: Optional[Iterable[str]] = None,
    cat_l3: Optional[Iterable[str]] = None,
    fixed_base_col: Optional[str] = None,
):
    _validate_columns(df, ["year_month", "DOH_bucket", "Availability", "optioncode", "Revenue"])
    work = df.copy()
    work["__ym"] = _month_key(work["year_month"])
    months = _sorted_unique(work["__ym"].dropna())

    avail_present = _sorted_unique(work["Availability"].astype(str))
    avail_order = _order_availability(avail_present)
    avail_rank = _exclusive_priority_lookup(avail_present)

    buckets = _sorted_unique(work["DOH_bucket"].astype(str))

    # Apply filters
    work = _apply_filters(work, channel=channel, cat_l2=cat_l2, cat_l3=cat_l3)

    # Base for constant denominator
    if fixed_base_col is not None and fixed_base_col in work.columns:
        base_options = set(work.loc[work[fixed_base_col].astype(bool), "optioncode"].dropna().unique())
    else:
        base_options = set(work["optioncode"].dropna().unique())
    base_n = max(len(base_options), 1)

    # 1) Faceted 100% Split — Month × DOH Bucket
    facets = {}
    for b in buckets:
        sub = work.loc[work["DOH_bucket"] == b]
        if sub.empty: 
            continue
        agg = _agg_counts(sub, ["__ym", "Availability"])
        pv_count = agg.pivot(index="__ym", columns="Availability", values="option_count").reindex(index=months, columns=avail_order, fill_value=0)
        pv_rev   = agg.pivot(index="__ym", columns="Availability", values="revenue").reindex(index=months, columns=avail_order, fill_value=0.0)
        row_sum = pv_count.sum(axis=1).replace(0, np.nan)
        pv_pct = (pv_count.div(row_sum, axis=0) * 100).fillna(0.0)
        rev_row_sum = pv_rev.sum(axis=1).replace(0, np.nan)
        pv_rev_pct = (pv_rev.div(rev_row_sum, axis=0) * 100).fillna(0.0)
        fig = _make_100pct_stacked(f"Faceted 100% Split — {b}: Month × Availability",
                                   months, avail_order,
                                   pv_pct.to_numpy(), pv_count.to_numpy(), pv_rev.to_numpy(), pv_rev_pct.to_numpy())
        facets[b] = fig

    # 2) DOH mix 100% by month
    agg_mix = _agg_counts(work, ["__ym", "DOH_bucket"])
    pv_mix = agg_mix.pivot(index="__ym", columns="DOH_bucket", values="option_count").reindex(index=months, columns=buckets, fill_value=0)
    row_sum = pv_mix.sum(axis=1).replace(0, np.nan)
    pv_mix_pct = (pv_mix.div(row_sum, axis=0) * 100).fillna(0.0)
    pv_mix_rev = agg_mix.pivot(index="__ym", columns="DOH_bucket", values="revenue").reindex(index=months, columns=buckets, fill_value=0.0)
    rev_row_sum = pv_mix_rev.sum(axis=1).replace(0, np.nan)
    pv_mix_rev_pct = (pv_mix_rev.div(rev_row_sum, axis=0) * 100).fillna(0.0)
    fig_mix = _make_100pct_stacked("DOH Mix 100% by Month",
                                   months, buckets,
                                   pv_mix_pct.to_numpy(), pv_mix.to_numpy(),
                                   pv_mix_rev.to_numpy(), pv_mix_rev_pct.to_numpy())

    # 3) Shares by Month with a Constant Denominator (buying lens)
    share_total_by_bucket = {}
    for b in buckets:
        sub = work.loc[work["DOH_bucket"] == b]
        agg = _agg_counts(sub, ["__ym", "Availability"])
        pv_count = agg.pivot(index="__ym", columns="Availability", values="option_count").reindex(index=months, columns=avail_order, fill_value=0)
        pv_rev   = agg.pivot(index="__ym", columns="Availability", values="revenue").reindex(index=months, columns=avail_order, fill_value=0.0)
        pv_pct_of_fixed = pv_count.applymap(lambda x: (x / base_n) * 100.0)
        rev_row_sum = pv_rev.sum(axis=1).replace(0, np.nan)
        pv_rev_pct  = (pv_rev.div(rev_row_sum, axis=0) * 100).fillna(0.0)
        fig = _make_lines(f"Share of Total by Month — {b} (constant base: {base_n} unique options)",
                          months, avail_order,
                          pv_pct_of_fixed.to_numpy(), pv_count.to_numpy(), pv_rev.to_numpy(), pv_rev_pct.to_numpy(),
                          ytitle="% of fixed base")
        share_total_by_bucket[b] = fig

    # 4) Shares by Month within a DOH bucket (100% split)
    share_in_bucket = {}
    for b in buckets:
        sub = work.loc[work["DOH_bucket"] == b]
        agg = _agg_counts(sub, ["__ym", "Availability"])
        pv_count = agg.pivot(index="__ym", columns="Availability", values="option_count").reindex(index=months, columns=avail_order, fill_value=0)
        pv_rev   = agg.pivot(index="__ym", columns="Availability", values="revenue").reindex(index=months, columns=avail_order, fill_value=0.0)
        row_sum = pv_count.sum(axis=1).replace(0, np.nan)
        pv_pct  = (pv_count.div(row_sum, axis=0) * 100.0).fillna(0.0)
        rev_row_sum = pv_rev.sum(axis=1).replace(0, np.nan)
        pv_rev_pct = (pv_rev.div(rev_row_sum, axis=0) * 100).fillna(0.0)
        fig = _make_100pct_stacked(f"Share in Bucket by Month — {b} (100% split)",
                                   months, avail_order,
                                   pv_pct.to_numpy(), pv_count.to_numpy(), pv_rev.to_numpy(), pv_rev_pct.to_numpy())
        share_in_bucket[b] = fig

    # 5) Fill-Rate vs DOH — Exclusive Categories
    tmp = work[["__ym", "DOH_bucket", "optioncode", "Availability", "Revenue"]].drop_duplicates()
    tmp["__rank"] = tmp["Availability"].map(avail_rank).fillna(9999).astype(int)
    pick = tmp.sort_values(["__ym", "DOH_bucket", "optioncode", "__rank"]).groupby(["__ym", "DOH_bucket", "optioncode"], as_index=False).first()
    excl = pick.groupby(["__ym", "DOH_bucket", "Availability"], dropna=False).agg(
        option_count=("optioncode", "nunique"),
        revenue=("Revenue", "sum")
    ).reset_index()

    exclusive_figs = {
        "all100_monthly": None,
        "core_instock_monthly": None,
        "core_oos_monthly": None,
        "fringe_available_monthly": None,
        "no_sales_monthly": None,
        "consolidated": None
    }
    label_map = {
        _canon("All100"): "all100_monthly",
        _canon("Core Instock"): "core_instock_monthly",
        _canon("Core OOS"): "core_oos_monthly",
        _canon("Fringe Available"): "fringe_available_monthly",
        _canon("No_Sales"): "no_sales_monthly"
    }

    for lbl in _order_availability(_sorted_unique(excl["Availability"].astype(str))) if not excl.empty else []:
        key = label_map.get(_canon(lbl))
        if key is None: 
            continue
        subset = excl.loc[excl["Availability"] == lbl]
        if subset.empty: 
            continue
        pv_count = subset.pivot(index="__ym", columns="DOH_bucket", values="option_count").reindex(index=months, columns=buckets, fill_value=0)
        pv_rev   = subset.pivot(index="__ym", columns="DOH_bucket", values="revenue").reindex(index=months, columns=buckets, fill_value=0.0)
        rev_row_sum = pv_rev.sum(axis=1).replace(0, np.nan)
        pv_rev_pct = (pv_rev.div(rev_row_sum, axis=0) * 100.0).fillna(0.0)
        fig = _make_lines(f"Exclusive Fill Count — {lbl} (monthly)",
                          months, buckets,
                          pv_count.to_numpy().astype(float), pv_count.to_numpy(), pv_rev.to_numpy(), pv_rev_pct.to_numpy(),
                          ytitle="# options")
        exclusive_figs[key] = fig

    cons = excl.groupby(["DOH_bucket", "Availability"], dropna=False).agg(
        option_count=("option_count", "sum"),
        revenue=("revenue", "sum")
    ).reset_index()
    if not cons.empty:
        pv_count = cons.pivot(index="DOH_bucket", columns="Availability", values="option_count").reindex(index=buckets, columns=_order_availability(_sorted_unique(cons["Availability"].astype(str))), fill_value=0)
        pv_rev   = cons.pivot(index="DOH_bucket", columns="Availability", values="revenue").reindex(index=buckets, columns=_order_availability(_sorted_unique(cons["Availability"].astype(str))), fill_value=0.0)
        rev_row_sum = pv_rev.sum(axis=1).replace(0, np.nan)
        pv_rev_pct = (pv_rev.div(rev_row_sum, axis=0) * 100.0).fillna(0.0)
        fig = _grouped_bar("Exclusive Fill Count — Consolidated (all months)",
                           buckets, _order_availability(_sorted_unique(cons["Availability"].astype(str))),
                           pv_count.to_numpy().astype(float), pv_count.to_numpy(), pv_rev.to_numpy(), pv_rev_pct.to_numpy(),
                           ytitle="# options")
        exclusive_figs["consolidated"] = fig

    return {
        "facets": facets,
        "doh_mix_100pct_by_month": fig_mix,
        "share_of_total_by_month": share_total_by_bucket,
        "share_in_bucket_by_month": share_in_bucket,
        "exclusive": exclusive_figs
    }

# ---------------------------- Data loading ----------------------------

@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    return df

def pick_base_column(df: pd.DataFrame) -> Optional[str]:
    # Propose boolean-like columns for fixed base
    candidates = []
    for c in df.columns:
        s = df[c].dropna()
        # Boolean or 0/1-ish
        if s.dtype == bool:
            candidates.append(c)
        else:
            uniq = set(pd.Series(s).astype(str).str.lower().str.strip().unique())
            if uniq.issubset({"0","1","true","false"}):
                candidates.append(c)
    return candidates

# ---------------------------- UI ----------------------------

st.set_page_config(page_title="Inventory Mix Dashboard", layout="wide")
st.title("Inventory Mix Dashboard (Packs / Non-packs)")

with st.sidebar:
    st.header("Data")
    uploaded_main = st.file_uploader("Upload mydata_mom_option.csv", type=["csv"], key="main_csv")
    uploaded_child = st.file_uploader("Upload mydata_mom_option_child.csv (optional)", type=["csv"], key="child_csv")

    data_mode = st.selectbox(
        "Row selection",
        ["All rows", "Packs only (pack_qty ≥ 2)", "Non-packs only (pack_qty = 1 or NaN)"],
        index=0
    )

    st.markdown("---")
    st.header("Filters")

if uploaded_main is not None:
    df = load_csv(uploaded_main)
else:
    # fallback to local file name if present
    local = Path("mydata_mom_option.csv")
    if local.exists():
        df = load_csv(local)
    else:
        st.warning("Please upload **mydata_mom_option.csv** to proceed.")
        st.stop()

# Basic validation
required = ["year_month", "DOH_bucket", "Availability", "optioncode", "Revenue"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Optional filters sourced from dataset
channels = sorted(pd.Series(df["Channel"].dropna().unique()).tolist()) if "Channel" in df.columns else []
cats2 = sorted(pd.Series(df["cat_l2"].dropna().unique()).tolist()) if "cat_l2" in df.columns else []
cats3 = sorted(pd.Series(df["cat_l3"].dropna().unique()).tolist()) if "cat_l3" in df.columns else []
buckets = _sorted_unique(df["DOH_bucket"].astype(str).tolist())

with st.sidebar:
    sel_channels = st.multiselect("Channel", channels, default=channels if channels else None)
    sel_cat_l2 = st.multiselect("cat_l2", cats2, default=cats2 if cats2 else None)
    sel_cat_l3 = st.multiselect("cat_l3", cats3, default=cats3 if cats3 else None)
    base_candidates = pick_base_column(df)
    use_base = st.selectbox(
        "Fixed base column (optional)",
        ["<None>"] + base_candidates if base_candidates else ["<None>"]
    )
    fixed_base_col = None if use_base == "<None>" else use_base

# Apply packs / non-packs mode
if "pack_qty" in df.columns:
    if data_mode == "Packs only (pack_qty ≥ 2)":
        df_view = df.loc[(df["pack_qty"] >= 2)]
    elif data_mode == "Non-packs only (pack_qty = 1 or NaN)":
        df_view = df.loc[(df["pack_qty"].fillna(0) == 1)]
    else:
        df_view = df.copy()
else:
    df_view = df.copy()

# Build figures
figs = build_inventory_figures(
    df_view,
    channel=sel_channels if sel_channels else None,
    cat_l2=sel_cat_l2 if sel_cat_l2 else None,
    cat_l3=sel_cat_l3 if sel_cat_l3 else None,
    fixed_base_col=fixed_base_col
)

# Layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "DOH Mix 100% by Month",
    "Faceted: Month × DOH Bucket",
    "Constant Base (Buying Lens)",
    "In-bucket 100% Split",
    "Exclusive Views",
    "Data Preview"
])

with tab1:
    st.subheader("DOH Mix 100% by Month")
    st.plotly_chart(figs["doh_mix_100pct_by_month"], use_container_width=True)

with tab2:
    st.subheader("Faceted 100% Split — Month × DOH Bucket")
    for b, f in figs["facets"].items():
        st.markdown(f"**DOH Bucket: {b}**")
        st.plotly_chart(f, use_container_width=True)

with tab3:
    st.subheader("Share of Total by Month — Constant Base")
    for b, f in figs["share_of_total_by_month"].items():
        st.markdown(f"**DOH Bucket: {b}**")
        st.plotly_chart(f, use_container_width=True)

with tab4:
    st.subheader("Share in Bucket by Month — 100% Split")
    for b, f in figs["share_in_bucket_by_month"].items():
        st.markdown(f"**DOH Bucket: {b}**")
        st.plotly_chart(f, use_container_width=True)

with tab5:
    st.subheader("Exclusive Fill-Rate vs DOH")
    excl = figs["exclusive"]
    for k, f in excl.items():
        if f is None: 
            continue
        st.markdown(f"**{k.replace('_',' ').title()}**")
        st.plotly_chart(f, use_container_width=True)

with tab6:
    st.subheader("Data Preview & Distincts")
    st.write("First 100 rows:")
    st.dataframe(df_view.head(100))
    if "Channel" in df_view.columns:
        st.write("Distinct Channels (first 50):", sorted(df_view["Channel"].dropna().unique().tolist())[:50])
    if "cat_l2" in df_view.columns:
        st.write("Distinct cat_l2 (first 50):", sorted(df_view["cat_l2"].dropna().unique().tolist())[:50])
    if "cat_l3" in df_view.columns:
        st.write("Distinct cat_l3 (first 50):", sorted(df_view["cat_l3"].dropna().unique().tolist())[:50])
    if "DOH_bucket" in df_view.columns:
        st.write("Distinct DOH_bucket:", _sorted_unique(df_view["DOH_bucket"].astype(str).tolist()))

# Optional child-level preview
if uploaded_child is not None:
    try:
        child_df = load_csv(uploaded_child)
        with st.expander("Child dataset preview"):
            st.dataframe(child_df.head(100))
            if "optioncode" in child_df.columns:
                st.write("Child distinct optioncode:", child_df["optioncode"].nunique())
            if "sku" in child_df.columns:
                st.write("Child distinct sku:", child_df["sku"].nunique())
    except Exception as e:
        st.warning(f"Could not read child file: {e}")

st.caption("Tip: Use the sidebar filters to slice the figures by Channel, cat_l2, and cat_l3. Toggle packs/non-packs in 'Row selection'.")

