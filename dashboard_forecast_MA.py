# -*- coding: utf-8 -*-
"""
dashboard_forecast_no_shock_pooled_resmerge_v13_option2_counts_as_regressor.py

✅ Correct residential merge:
Merge ONLY these building types into ONE building type called "Residential":
- Apartments
- Houses
- Retirement village
- Townhouses
- Temporary Accommodation

Everything else remains unchanged.

Includes methods:
- Ensemble (weighted)
- Robust (Huber)
- Pooled (type dummies)
- Optional ECM / GAM / SSM if present

No shock UI/logic.

Run:
  streamlit run dashboard_forecast_no_shock_pooled_resmerge_v13_option2_counts_as_regressor.py
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import streamlit as st
import altair as alt
import joblib
import statsmodels.api as sm

# Keep an alias to pandas.read_csv so our safe wrapper doesn't recurse if we patch calls below.
_pd_read_csv = pd.read_csv

def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV but return empty DataFrame if file is empty/corrupt."""
    try:
        return _pd_read_csv(path, **kwargs)
    except (EmptyDataError, ValueError):
        return pd.DataFrame()


# Option 2 feature sets
COUNT_FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1"]
# Value models include lagged count growth from the counts model
VALUE_FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1", "count_y_lag1"]

st.set_page_config(page_title="NZ Building Forecast Dashboard", layout="wide")
st.title("NZ Building Forecasts (Values & Counts) — Scenario Simulation (No Shock)")
st.caption("Simulates levels year-by-year from 2025 using macro scenario paths.")

FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1"]
ECM_FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1", "gap_lag1"]
GOVT_MAP = {"Conservative": 0.0, "Liberal": 1.0}

REGIONS = [
    "Auckland Region",
    "Waikato Region",
    "Wellington Region",
    "Rest of North Island",
    "Canterbury Region",
    "Rest of South Island",
]

# -----------------------------
# ✅ Same merge function as pipeline
# -----------------------------
def _norm_bt(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\u2013\u2014\-_/]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def merge_building_type(bt: str) -> str:
    n = _norm_bt(bt)
    if n == "apartments" or n == "apartment" or n.startswith("apartment "):
        return "Residential"
    if n == "houses" or n == "house" or n.startswith("house "):
        return "Residential"
    if n == "townhouses" or n == "townhouse" or n.startswith("townhouse "):
        return "Residential"
    if ("retirement" in n) and ("village" in n or "villages" in n):
        return "Residential"
    # domestic outbuilding(s)
    # merge into Residential
    if ('outbuilding' in n or 'outbuild' in n) and 'domestic' in n:
        return 'Residential'

    return str(bt).strip()

def has_file(path: str) -> bool:
    return Path(path).exists()

@st.cache_data(show_spinner=False)
def load_present_values(path: str = "present_values.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {path}. Run the pipeline in this folder first.")
    df = safe_read_csv(p)
    df.columns = df.columns.str.strip().str.lower()
    df["data type"] = df["data type"].astype(str).str.strip().str.lower()
    df["region"] = df["region"].astype(str)

    # Apply merge again (in case user has an old present_values.csv)
    df["building type"] = df["building type"].astype(str).map(merge_building_type)

    df["value"] = (
        df["value"].astype(str)
          .str.replace(",", "", regex=False)
          .str.replace("$", "", regex=False)
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # IMPORTANT: ensure duplicates like multiple 'Residential' rows are summed
    df = (
        df.groupby(["region", "building type", "data type"], as_index=False)["value"]
          .sum()
    )
    return df

@st.cache_data(show_spinner=False)
def load_weights(tag: str) -> pd.DataFrame:
    p = Path("model_store") / tag / f"ensemble_weights_{tag}.csv"
    if p.exists():
        w = safe_read_csv(p)
        if w is None or w.empty:
            return pd.DataFrame(columns=["Building Type", "method", "weight"])
        w.columns = [str(c).strip() for c in w.columns]
        if "method" not in w.columns or "weight" not in w.columns:
            return pd.DataFrame(columns=["Building Type", "method", "weight"])
        w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
        w = w[w["weight"] > 0]
        if "Building Type" in w.columns:
            w["Building Type"] = w["Building Type"].astype(str)
            s = w.groupby("Building Type")["weight"].transform("sum")
            w["weight"] = np.where(s > 0, w["weight"] / s, w["weight"])
        w["method"] = w["method"].astype(str)
        return w
    return pd.DataFrame(columns=["Building Type", "method", "weight"])


@st.cache_data(show_spinner=False)
def load_last_lag(tag: str) -> pd.DataFrame:
    p = Path("model_store") / tag / f"last_lag_growth_{tag}.csv"
    if p.exists():
        df = safe_read_csv(p)
        df["Region"] = df["Region"].astype(str)
        df["Building Type"] = df["Building Type"].astype(str)
        df["lag_growth_2024"] = pd.to_numeric(df["lag_growth_2024"], errors="coerce")
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_ecm_gap(tag: str) -> pd.DataFrame:
    p = Path("model_store") / tag / f"ecm_gap_{tag}.csv"
    if p.exists():
        df = safe_read_csv(p)
        df["Region"] = df["Region"].astype(str)
        df["Building Type"] = df["Building Type"].astype(str)
        df["gap_lag1"] = pd.to_numeric(df["gap_lag1"], errors="coerce")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        return df
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)

@st.cache_resource(show_spinner=False)
def load_lin_bundle(tag: str, fname: str):
    p = Path("model_store") / tag / fname
    if not p.exists():
        raise FileNotFoundError(str(p))
    return joblib.load(p)

@st.cache_resource(show_spinner=False)
def load_glm_bundle(tag: str, fname: str):
    p = Path("model_store") / tag / fname
    if not p.exists():
        raise FileNotFoundError(str(p))
    return joblib.load(p)

def load_model_obj(tag: str, model_key: str):
    p = Path("model_store") / tag / model_key
    if not p.exists():
        raise FileNotFoundError(str(p))
    return joblib.load(p)

def lag_seed(df: pd.DataFrame, region: str, bt: str) -> float:
    if df.empty:
        return 0.0
    sub = df[(df["Region"] == region) & (df["Building Type"] == bt)]
    if sub.empty:
        return 0.0
    v = float(sub.iloc[0]["lag_growth_2024"])
    return v if np.isfinite(v) else 0.0

def gap_seed(df_gap: pd.DataFrame, region: str, bt: str, base_year: int = 2024) -> float:
    if df_gap.empty:
        return 0.0
    sub = df_gap[(df_gap["Year"] == base_year) & (df_gap["Region"] == region) & (df_gap["Building Type"] == bt)]
    if sub.empty:
        return 0.0
    v = float(sub.iloc[0]["gap_lag1"])
    return v if np.isfinite(v) else 0.0

def build_path(start_pct: float, end_pct: float, years: np.ndarray, shape: str) -> np.ndarray:
    start = start_pct / 100.0
    end = end_pct / 100.0
    if shape == "Constant":
        return np.full_like(years, start, dtype=float)
    return np.linspace(start, end, len(years), dtype=float)

def predict_growth(
    tag: str,
    bt: str,
    gdp_chg: float,
    pop_chg: float,
    govt: float,
    y_lag1: float,
    gap_lag1: float,
    method: str,
    weights: pd.DataFrame,
    count_y_lag1: float | None = None,
) -> float:
    """Predict one-step-ahead growth for either counts or values.

    Option 2 logic: for VALUE models, include lagged count growth as an extra regressor *only*
    when the model expects it.
    """

    # --- Robust (Huber) ---
    if method == "Robust (Huber)":
        model = load_model_obj(tag, f"huber__{bt}.joblib")
        if tag == "values":
            X = np.array([[gdp_chg, pop_chg, govt, y_lag1, 0.0 if count_y_lag1 is None else float(count_y_lag1)]], dtype=float)
        else:
            X = np.array([[gdp_chg, pop_chg, govt, y_lag1]], dtype=float)
        return float(model.predict(X)[0])

    # --- ECM ---
    if method == "ECM (Error-correction)":
        bundle = load_model_obj(tag, f"ecm__{bt}.joblib")
        model = bundle["model"]
        features = bundle.get("features", [])
        row = [gdp_chg, pop_chg, govt, y_lag1]
        if "gap_lag1" in features:
            row.append(gap_lag1)
        if "count_y_lag1" in features:
            row.append(0.0 if count_y_lag1 is None else float(count_y_lag1))
        X_ecm = np.array([row], dtype=float)
        return float(model.predict(X_ecm)[0])

    # --- Pooled ---
    if method == "Pooled (type dummies)":
        bundle = load_model_obj(tag, f"pooled__ALL_{tag}.joblib")
        model = bundle["model"]
        dummy_cols = bundle["dummy_columns"]
        features = bundle.get("features", COUNT_FEATURES)

        row = [gdp_chg, pop_chg, govt, y_lag1]
        if "count_y_lag1" in features:
            row.append(0.0 if count_y_lag1 is None else float(count_y_lag1))

        X_base = pd.DataFrame([row], columns=features)
        dummies = pd.get_dummies(pd.Series([bt]), prefix="bt", drop_first=True)
        for col in dummy_cols:
            if col not in dummies.columns:
                dummies[col] = 0.0
        dummies = dummies[dummy_cols]
        X = pd.concat([X_base, dummies], axis=1).to_numpy(dtype=float)
        return float(model.predict(X)[0])

    # --- GAM ---
    if method == "GAM (pyGAM)":
        model = load_model_obj(tag, f"gam__{bt}.joblib")
        if tag == "values":
            X = np.array([[gdp_chg, pop_chg, govt, y_lag1, 0.0 if count_y_lag1 is None else float(count_y_lag1)]], dtype=float)
        else:
            X = np.array([[gdp_chg, pop_chg, govt, y_lag1]], dtype=float)
        return float(model.predict(X)[0])

    # --- Ensemble ---
    if method == "Ensemble (weighted)":
        # Guard: weights may be missing/empty or not include expected columns.
        if weights is None or len(weights) == 0 or ("method" not in weights.columns) or ("weight" not in weights.columns):
            base_methods = ["Robust (Huber)", "ECM (Error-correction)", "Pooled (type dummies)"]
            preds = []
            for mname in base_methods:
                try:
                    preds.append(
                        predict_growth(
                            tag, bt, gdp_chg, pop_chg, govt, y_lag1, gap_lag1, mname, weights,
                            count_y_lag1=count_y_lag1
                        )
                    )
                except Exception:
                    pass
            return float(np.mean(preds)) if preds else 0.0

        preds = []
        ws = []
        for mname in weights["method"].unique().tolist():
            w = float(weights.loc[weights["method"] == mname, "weight"].iloc[0])
            try:
                preds.append(
                    predict_growth(
                        tag, bt, gdp_chg, pop_chg, govt, y_lag1, gap_lag1, mname, weights,
                        count_y_lag1=count_y_lag1
                    )
                )
                ws.append(w)
            except Exception:
                pass

        if not ws:
            return float(np.mean(preds)) if preds else 0.0
        return float(np.dot(np.asarray(preds), np.asarray(ws)) / float(np.sum(ws)))

    return 0.0
@st.cache_data(show_spinner=False)
def simulate(region: str, btypes: list[str], years_sim: np.ndarray,
             base_values: dict[str, float], base_counts: dict[str, float],
             lag_values_seed: dict[str, float], lag_counts_seed: dict[str, float],
             gap_values_seed: dict[str, float], gap_counts_seed: dict[str, float],
             govt_val: float, method: str,
             gdp_path: tuple, pop_path: tuple, infl_path: tuple) -> pd.DataFrame:

    gdp_path = np.asarray(gdp_path, dtype=float)
    pop_path = np.asarray(pop_path, dtype=float)
    infl_path = np.asarray(infl_path, dtype=float)

    weights_values = load_weights("values")
    weights_counts = load_weights("counts")

    v = dict(base_values)
    c = dict(base_counts)
    lag_v = dict(lag_values_seed)
    lag_c = dict(lag_counts_seed)
    gap_v = dict(gap_values_seed)
    gap_c = dict(gap_counts_seed)

    rows = []
    for i, yy in enumerate(years_sim):
        inf = float(infl_path[i])

        for bt in btypes:
            
            # 1) Forecast counts first (so we have lagged count growth available for VALUE model)
            if bt in c:
                gc = predict_growth("counts", bt, float(gdp_path[i]), float(pop_path[i]), govt_val,
                                    float(lag_c.get(bt, 0.0)), float(gap_c.get(bt, 0.0)),
                                    method, weights_counts, count_y_lag1=None)
                prev_c = float(c.get(bt, np.nan))
                if np.isfinite(prev_c):
                    c[bt] = float(np.expm1(np.log1p(prev_c) + gc))
                lag_c[bt] = float(gc)

            # 2) Forecast values using lagged count growth as an input (Option 2)
            gv = predict_growth("values", bt, float(gdp_path[i]), float(pop_path[i]), govt_val,
                                float(lag_v.get(bt, 0.0)), float(gap_v.get(bt, 0.0)),
                                method, weights_values, count_y_lag1=float(lag_c.get(bt, 0.0)))
            prev_v = float(v.get(bt, np.nan))
            if np.isfinite(prev_v):
                v[bt] = prev_v * float(np.exp(gv)) * (1.0 + inf)
            lag_v[bt] = float(gv)

            rows.append({"Year": int(yy), "Region": region, "Building Type": bt,
                         "Value": v.get(bt, np.nan), "Count": c.get(bt, np.nan)})
    return pd.DataFrame(rows)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Scenario")
    target_year = st.selectbox("Forecast to year", options=list(range(2025, 2101)), index=0)
    region = st.selectbox("Region", options=REGIONS, index=0)

    st.divider()
    path_shape = st.selectbox("Macro path shape", options=["Constant", "Linear trend"], index=0)

    gdp_start = st.slider("GDP growth start (%)", min_value=-2.0, max_value=8.0, value=2.0, step=0.5)

    pop_start = st.slider("Population growth start (%)", min_value=-2.0, max_value=8.0, value=2.0, step=0.5)

    infl_start = st.slider("Inflation start (%)", min_value=0.0, max_value=8.0, value=2.0, step=0.5)

    st.divider()
    govt_label = st.radio("Fiscal policy", options=["Conservative", "Liberal"], index=0, horizontal=True)
    govt_val = GOVT_MAP[govt_label]

# ---------------------------
# Load base levels
# ---------------------------
try:
    pv = load_present_values("present_values.csv")
except Exception as e:
    st.error(str(e))
    st.stop()

pv_r = pv[pv["region"].str.lower() == region.lower()].copy()
rv = pv_r[pv_r["data type"] == "value"].copy()
rc = pv_r[pv_r["data type"] == "number"].copy()

if rv.empty:
    st.error(f"No Value rows found for region '{region}' in present_values.csv.")
    st.stop()

rv["bt"] = rv["building type"].astype(str)
rc["bt"] = rc["building type"].astype(str)
btypes = sorted(rv["bt"].dropna().unique().tolist())

method_options = [
    "Ensemble (weighted)",
    "Robust (Huber)",
    "GAM (pyGAM)",
    "ECM (Error-correction)",
    "Pooled (type dummies)",
]

with st.sidebar:
    method = st.selectbox("Forecast method", options=method_options, index=0)

base_values = rv.set_index("bt")["value"].to_dict()
base_counts = rc.set_index("bt")["value"].to_dict() if not rc.empty else {}

last_lag_values = load_last_lag("values")
last_lag_counts = load_last_lag("counts")
gap_values_df = load_ecm_gap("values")
gap_counts_df = load_ecm_gap("counts")

lag_values_seed = {bt: lag_seed(last_lag_values, region, bt) for bt in btypes}
lag_counts_seed = {bt: lag_seed(last_lag_counts, region, bt) for bt in btypes}
gap_values_seed = {bt: gap_seed(gap_values_df, region, bt, base_year=2024) for bt in btypes}
gap_counts_seed = {bt: gap_seed(gap_counts_df, region, bt, base_year=2024) for bt in btypes}

years_sim = np.arange(2025, target_year + 1, dtype=int)
gdp_path = build_path(gdp_start, gdp_start, years_sim, path_shape)
pop_path = build_path(pop_start, pop_start, years_sim, path_shape)
infl_path = build_path(infl_start, infl_start, years_sim, path_shape)

try:
    sim = simulate(region, btypes, years_sim, base_values, base_counts,
                   lag_values_seed, lag_counts_seed, gap_values_seed, gap_counts_seed,
                   govt_val, method,
                   tuple(gdp_path.tolist()), tuple(pop_path.tolist()), tuple(infl_path.tolist()))
except Exception as e:
    st.error(f"Simulation failed: {e}")
    st.stop()

snap = sim[sim["Year"] == target_year].copy()
snap["Value (NZD, millions)"] = (snap["Value"] / 1_000_000.0).round(0).astype("Int64")
snap["Count"] = snap["Count"].round(0).astype("Int64")
snap = snap.sort_values("Value (NZD, millions)", ascending=False)

st.subheader(f"Forecast summary — {region} to {target_year}")
st.caption("If you still see separate Apartments/Houses/etc., re-run the pipeline to regenerate present_values.csv.")
st.dataframe(snap[["Building Type", "Value (NZD, millions)", "Count"]].reset_index(drop=True),
             width="stretch", height=420)

st.divider()
colA, colB = st.columns([1, 1])
with colA:
    bt_focus = st.selectbox("Focus building type (for charts)", options=btypes, index=0)
with colB:
    show_kpi = st.selectbox("Chart KPI", options=["Counts", "Values"], index=0)

bt_df = sim[sim["Building Type"] == bt_focus].set_index("Year").sort_index()

k = bt_df["Value"] if show_kpi == "Values" else bt_df["Count"]
unit_label = "NZD" if show_kpi == "Values" else "units"
plot_df = pd.DataFrame({"Year": bt_df.index.astype(int), "KPI": k.values})

chart1 = (
    alt.Chart(plot_df)
    .mark_line()
    .encode(
        x=alt.X("Year:O", title="Year"),
        y=alt.Y("KPI:Q", title=f"{show_kpi} ({unit_label})"),
        tooltip=["Year:O", alt.Tooltip("KPI:Q", format=",.0f")],
    )
    .properties(height=320, title=f"{bt_focus}: {show_kpi} forecast")
)
st.altair_chart(chart1, use_container_width=True)