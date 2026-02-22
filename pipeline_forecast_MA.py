# -*- coding: utf-8 -*-
"""
pipeline_forecast_simulation_logdiff_ssm_ecm_no_shock_pooled_resmerge_v13_option2_counts_as_regressor.py

✅ Correct residential merge:
Merge ONLY these building types into ONE building type called "Residential":
- Apartments
- Houses
- Retirement village
- Townhouses
- Temporary Accommodation

Everything else remains unchanged.

Also:
- No shock logic (dashboard-only previously)
- Forecast methods included:
  - Robust (Huber) per type
  - ECM per type (+ gap_lag1)
  - Pooled (type dummies) across types ()
  - GAM (pyGAM) (optional if installed)
  - Optional State-space (SARIMAX)
- Removed:
  - Baseline() as standalone method
  - Boosted/GBR methods
  - Pooled ECM variants + residual variants

Run:
  python pipeline_forecast_simulation_logdiff_ssm_ecm_no_shock_pooled_resmerge_v13_option2_counts_as_regressor.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, QuantileRegressor, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from pygam import LinearGAM, s
    _HAS_PYGAM = True
except Exception:
    _HAS_PYGAM = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


# -----------------------------
# ✅ Residential merge (ONLY these 5)
# -----------------------------
def _norm_bt(s: str) -> str:
    s = str(s).strip().lower()
    # normalize punctuation/separators and whitespace
    s = re.sub(r"[\u2013\u2014\-_/]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def merge_building_type(bt: str) -> str:
    """
    Merge Apartments, Houses, Retirement village, Townhouses, Temporary Accommodation
    into a single Building Type: "Residential".
    """
    n = _norm_bt(bt)

    # apartments
    if n == "apartments" or n == "apartment" or n.startswith("apartment "):
        return "Residential"

    # houses
    if n == "houses" or n == "house" or n.startswith("house "):
        return "Residential"

    # townhouses
    if n == "townhouses" or n == "townhouse" or n.startswith("townhouse "):
        return "Residential"

    # retirement village
    # (your sheet often shows "Retirement village")
    if ("retirement" in n) and ("village" in n or "villages" in n):
        return "Residential"

    # temporary accommodation
    # accept both spelling variants accommodation/accomodation

    # everything else unchanged
    # domestic outbuilding(s)
    # merge into Residential
    if ('outbuilding' in n or 'outbuild' in n) and 'domestic' in n:
        return 'Residential'

    return str(bt).strip()


def read_res_nonres_long(res_path: str, nonres_path: str) -> pd.DataFrame:
    df = pd.read_excel(res_path, header=[0, 1, 2])
    df.columns = ["Year"] + list(df.columns[1:])
    m = pd.melt(df, id_vars="Year", var_name="Variable", value_name="Value")
    m[["Region", "Building Type", "Data Type"]] = m["Variable"].apply(lambda x: pd.Series(list(x)))
    m = m.drop(columns=["Variable"])
    m["Year"] = pd.to_numeric(m["Year"], errors="coerce").astype("Int64")
    m["Value"] = pd.to_numeric(m["Value"], errors="coerce")

    df2 = pd.read_excel(nonres_path, header=[0, 1, 2])
    df2.columns = ["Year"] + list(df2.columns[1:])
    m2 = pd.melt(df2, id_vars="Year", var_name="Variable", value_name="Value")
    m2[["Region", "Building Type", "Data Type"]] = m2["Variable"].apply(lambda x: pd.Series(list(x)))
    m2 = m2.drop(columns=["Variable"])
    m2["Year"] = pd.to_numeric(m2["Year"], errors="coerce").astype("Int64")
    m2["Value"] = pd.to_numeric(m2["Value"], errors="coerce")

    out = pd.concat([m, m2], ignore_index=True).dropna(subset=["Year"])

    # Standard label cleanups (kept from your earlier scripts)
    out["Building Type"] = out["Building Type"].astype(str).str.replace(
        "Social, cultural, and religious buildings", "Sociocultural", regex=False
    )
    out["Region"] = out["Region"].astype(str).str.replace(
        "North Island excluding Auckland, Waikato, and Wellington regions", "Rest of North Island", regex=False
    )
    out["Region"] = out["Region"].astype(str).str.replace(
        "South Island excluding Canterbury Region", "Rest of South Island", regex=False
    )
    out["Data Type"] = out["Data Type"].astype(str).str.strip()

    # ✅ Apply merge HERE (so pivots + present_values are merged)
    out["Building Type"] = out["Building Type"].map(merge_building_type)

    return out


def build_present_values(df_long: pd.DataFrame, base_year: int = 2024) -> pd.DataFrame:
    # IMPORTANT: after merging building types, multiple rows can map to the same
    # (Region, Building Type='Residential', Data Type). We MUST aggregate them.
    pv = df_long[df_long["Year"] == base_year].copy()
    pv = pv.rename(columns={"Value": "value"})
    pv = pv[["Region", "Building Type", "Data Type", "value"]]
    pv["value"] = pd.to_numeric(pv["value"], errors="coerce").fillna(0.0)

    pv = (
        pv.groupby(["Region", "Building Type", "Data Type"], as_index=False)["value"]
          .sum()
    )

    pv.to_csv("present_values.csv", index=False)
    return pv

def read_background(dataset2_path: str) -> pd.DataFrame:
    bg = pd.read_excel(dataset2_path)
    bg = bg[["Year", "CPI", "govt"]].copy()
    bg["Year"] = pd.to_numeric(bg["Year"], errors="coerce").astype(int)
    bg["CPI"] = pd.to_numeric(bg["CPI"], errors="coerce")
    bg["govt"] = pd.to_numeric(bg["govt"], errors="coerce")
    bg = bg.dropna(subset=["Year", "CPI"]).set_index("Year").sort_index()
    return bg


def read_pop(pop_path: str) -> pd.DataFrame:
    pop = pd.read_csv(pop_path).rename(columns={"Unnamed: 0": "Year"})
    pop["Year"] = pd.to_numeric(pop["Year"], errors="coerce").astype(int)
    for c in pop.columns:
        if c == "Year":
            continue
        pop[c] = pd.to_numeric(pop[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    pop = pop.set_index("Year").sort_index()
    pop["Rest of North Island"] = pop["North Island"] - pop["Auckland Region"] - pop["Waikato Region"] - pop["Wellington Region"]
    pop["Rest of South Island"] = pop["South Island"] - pop["Canterbury Region"]
    keep = ["Auckland Region", "Waikato Region", "Wellington Region", "Canterbury Region", "Rest of North Island", "Rest of South Island"]
    return pop[keep]


def read_gdp(gdp_path: str) -> pd.DataFrame:
    gdp = pd.read_csv(gdp_path).rename(columns={"Unnamed: 0": "Year"})
    gdp["Year"] = pd.to_numeric(gdp["Year"], errors="coerce").astype(int)
    for c in gdp.columns:
        if c == "Year":
            continue
        gdp[c] = pd.to_numeric(gdp[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    gdp = gdp.set_index("Year").sort_index()

    gdp["Rest of North Island"] = gdp["Total North Island"] - gdp["Auckland"] - gdp["Waikato"] - gdp["Wellington"]
    gdp["Rest of South Island"] = gdp["Total South Island"] - gdp["Canterbury"]
    keep = ["Auckland", "Waikato", "Wellington", "Canterbury", "Rest of North Island", "Rest of South Island"]
    gdp = gdp[keep].rename(columns={
        "Auckland": "Auckland Region",
        "Waikato": "Waikato Region",
        "Wellington": "Wellington Region",
        "Canterbury": "Canterbury Region",
    })
    return gdp


def pivot_values_counts(df_long: pd.DataFrame):
    df_values = df_long[df_long["Data Type"] == "Value"].drop(columns=["Data Type"]).copy()
    df_counts = df_long[df_long["Data Type"] == "Number"].drop(columns=["Data Type"]).copy()

    # Because we already merged those 5 types into "Residential", this pivot SUMS them automatically.
    values_p = df_values.pivot_table(index="Year", columns=["Region", "Building Type"], values="Value", aggfunc="sum").sort_index()
    counts_p = df_counts.pivot_table(index="Year", columns=["Region", "Building Type"], values="Value", aggfunc="sum").sort_index()
    return values_p, counts_p


def adjust_values_to_real(values_pivot: pd.DataFrame, background: pd.DataFrame, base_year: int = 2025) -> pd.DataFrame:
    if base_year not in background.index:
        raise ValueError(f"Base year {base_year} not present in dataset2 CPI series.")
    cpi_base = float(background.loc[base_year, "CPI"])

    def _adj_row(row):
        y = int(row.name)
        if y not in background.index or pd.isna(background.loc[y, "CPI"]):
            return row * np.nan
        return row * (cpi_base / float(background.loc[y, "CPI"]))

    return values_pivot.apply(_adj_row, axis=1)


def logdiff_long(df_pivot: pd.DataFrame, use_log1p: bool) -> pd.DataFrame:
    if use_log1p:
        z = np.log1p(df_pivot)
    else:
        z = np.log(df_pivot.where(df_pivot > 0))
    y = z.diff().dropna(how="all")
    return y.stack(["Region", "Building Type"]).to_frame("y").reset_index()


def join_features(chg_long: pd.DataFrame, gdp: pd.DataFrame, pop: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    gdp_chg = gdp.pct_change().dropna(how="all")
    pop_chg = pop.pct_change().dropna(how="all")

    def _get(m, y, r):
        try:
            return float(m.loc[int(y), r])
        except Exception:
            return np.nan

    chg_long["gdp_chg"] = [_get(gdp_chg, y, r) for y, r in zip(chg_long["Year"], chg_long["Region"])]
    chg_long["pop_chg"] = [_get(pop_chg, y, r) for y, r in zip(chg_long["Year"], chg_long["Region"])]
    chg_long["govt"] = [float(background.loc[int(y), "govt"]) if int(y) in background.index else np.nan for y in chg_long["Year"]]

    for c in ["y", "gdp_chg", "pop_chg", "govt"]:
        chg_long[c] = pd.to_numeric(chg_long[c], errors="coerce")

    chg_long = chg_long.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "gdp_chg", "pop_chg", "govt"])
    chg_long = chg_long.sort_values(["Region", "Building Type", "Year"])
    chg_long["y_lag1"] = chg_long.groupby(["Region", "Building Type"])["y"].shift(1)
    chg_long = chg_long.dropna(subset=["y_lag1"])
    chg_long["Building Type"] = chg_long["Building Type"].astype(str)
    chg_long["Region"] = chg_long["Region"].astype(str)
    return chg_long


def compute_ecm_gap(level_pivot: pd.DataFrame, gdp: pd.DataFrame, pop: pd.DataFrame, background: pd.DataFrame,
                    use_log1p_level: bool, out_name: str) -> pd.DataFrame:
    if use_log1p_level:
        z = np.log1p(level_pivot)
    else:
        z = np.log(level_pivot.where(level_pivot > 0))

    z_long = z.stack(["Region", "Building Type"]).to_frame("z").reset_index()
    z_long["Year"] = z_long["Year"].astype(int)
    z_long["Region"] = z_long["Region"].astype(str)
    z_long["Building Type"] = z_long["Building Type"].astype(str)

    gdp_l = np.log(gdp.replace(0, np.nan))
    pop_l = np.log(pop.replace(0, np.nan))

    def _get(m, y, r):
        try:
            return float(m.loc[int(y), r])
        except Exception:
            return np.nan

    z_long["log_gdp"] = [_get(gdp_l, y, r) for y, r in zip(z_long["Year"], z_long["Region"])]
    z_long["log_pop"] = [_get(pop_l, y, r) for y, r in zip(z_long["Year"], z_long["Region"])]
    z_long["govt"] = [float(background.loc[int(y), "govt"]) if int(y) in background.index else np.nan for y in z_long["Year"]]

    z_long = z_long.replace([np.inf, -np.inf], np.nan).dropna(subset=["z", "log_gdp", "log_pop", "govt"])

    X = np.column_stack([
        np.ones(len(z_long)),
        z_long["log_gdp"].to_numpy(dtype=float),
        z_long["log_pop"].to_numpy(dtype=float),
        z_long["govt"].to_numpy(dtype=float),
    ])
    y = z_long["z"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    z_hat = X @ beta
    z_long["gap"] = y - z_hat

    z_long = z_long.sort_values(["Region", "Building Type", "Year"])
    z_long["gap_lag1"] = z_long.groupby(["Region", "Building Type"])["gap"].shift(1)
    z_long = z_long.dropna(subset=["gap_lag1"])

    z_long[["Year", "Region", "Building Type", "gap_lag1"]].to_csv(out_name, index=False)
    return z_long[["Year", "Region", "Building Type", "gap_lag1"]]


def save_last_lag(df: pd.DataFrame, out_path: str, base_year: int = 2024):
    last = df[df["Year"] == base_year][["Region", "Building Type", "y"]].copy()
    last = last.rename(columns={"y": "lag_growth_2024"})
    last.to_csv(out_path, index=False)


COUNT_FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1"]
COUNT_ECM_FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1", "gap_lag1"]
# Option 2: VALUE models include lagged count growth as an extra regressor
VALUE_FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1", "count_y_lag1"]
VALUE_ECM_FEATURES = ["gdp_chg", "pop_chg", "govt", "y_lag1", "gap_lag1", "count_y_lag1"]


# Backwards-compatibility alias used in legacy evaluation blocks
FEATURES = COUNT_FEATURES

def ensure_dirs():
    os.makedirs("model_store/values", exist_ok=True)
    os.makedirs("model_store/counts", exist_ok=True)





def fit_save_quantile_models(df: pd.DataFrame, out_dir: str, features: list[str], quantile: float = 0.5):
    """Fit QuantileRegressor per building type (median by default) and save as lin_quantile50__{bt}.joblib.
    Uses StandardScaler to match other estimators.
    """
    os.makedirs(out_dir, exist_ok=True)
    qtag = int(round(quantile * 100))
    for bt in sorted(df["Building Type"].unique()):
        sub = df[df["Building Type"] == bt].dropna(subset=features + ["y"]).copy()
        if len(sub) < 12:
            continue
        X = sub[features].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)
        est = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
        model = Pipeline([("scaler", StandardScaler()), ("reg", est)])
        model.fit(X, y)
        joblib.dump({"model": model, "features": features},
                    os.path.join(out_dir, f"lin_quantile{qtag}__{bt}.joblib"))

def fit_save_huber(df: pd.DataFrame, out_dir: str, features: list[str]):
    for bt in sorted(df["Building Type"].unique()):
        sub = df[df["Building Type"] == bt].copy()
        if len(sub) < 25:
            continue
        X = sub[features].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)
        model = Pipeline([("scaler", StandardScaler()), ("reg", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500))])
        model.fit(X, y)
        joblib.dump(model, os.path.join(out_dir, f"huber__{bt}.joblib"))


def fit_save_ecm(df: pd.DataFrame, out_dir: str, ecm_features: list[str]):
    for bt in sorted(df["Building Type"].unique()):
        sub = df[df["Building Type"] == bt].copy()
        if len(sub) < 35:
            continue
        if "gap_lag1" not in sub.columns:
            continue
        sub = sub.dropna(subset=ecm_features + ["y"])
        if len(sub) < 35:
            continue
        X = sub[ecm_features].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)
        model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        model.fit(X, y)
        joblib.dump({"model": model, "features": ecm_features}, os.path.join(out_dir, f"ecm__{bt}.joblib"))


def fit_save_pooled(df: pd.DataFrame, out_dir: str, tag: str, features: list[str]):
    """Pooled model across building types with type dummies."""
    X_base = df[features].astype(float)
    dummies = pd.get_dummies(df["Building Type"].astype(str), prefix="bt", drop_first=True)
    X = pd.concat([X_base, dummies], axis=1).astype(float)
    y = df["y"].astype(float)

    model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    model.fit(X.to_numpy(), y.to_numpy())

    bundle = {"model": model, "dummy_columns": dummies.columns.tolist(), "prefix": "bt", "features": features}
    joblib.dump(bundle, os.path.join(out_dir, f"pooled__ALL_{tag}.joblib"))





def fit_save_gam(df: pd.DataFrame, out_dir: str, features: list[str]):
    if not _HAS_PYGAM:
        return
    for bt in sorted(df["Building Type"].unique()):
        sub = df[df["Building Type"] == bt].copy()
        if len(sub) < 25:
            continue
        X = sub[features].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)
        # build TermList safely (sum() would start with 0 and error)
        terms = s(0)
        for i in range(1, X.shape[1]):
            terms += s(i)
        model = LinearGAM(terms)
        model.gridsearch(X, y)
        joblib.dump(model, os.path.join(out_dir, f"gam__{bt}.joblib"))


def fit_save_statespace(df: pd.DataFrame, out_dir: str, features: list[str]):
    if not _HAS_STATSMODELS:
        return
    for bt in sorted(df["Building Type"].unique()):
        sub = df[df["Building Type"] == bt].sort_values("Year").copy()
        if len(sub) < 35:
            continue
        endog = sub["y"].to_numpy(dtype=float)
        exog = sub[features].to_numpy(dtype=float)
        model = SARIMAX(
            endog=endog,
            exog=exog,
            order=(0, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        bundle = {"results": res, "features": features}
        joblib.dump(bundle, os.path.join(out_dir, f"ssm__{bt}.joblib"))


def compute_weights(df: pd.DataFrame, methods: list, last_n_years: int, store_dir: str, tag: str) -> pd.DataFrame:
    years = np.sort(df["Year"].unique())
    test_years = years[-last_n_years:] if len(years) > last_n_years else years[1:]

    pooled_bundle = None
    pooled_path = os.path.join(store_dir, f"pooled__ALL_{tag}.joblib")
    if "pooled" in methods and os.path.exists(pooled_path):
        pooled_bundle = joblib.load(pooled_path)

    rows = []
    for bt in sorted(df["Building Type"].unique()):
        sub_bt = df[df["Building Type"] == bt].copy()
        if len(sub_bt) < 40:
            continue

        rmses = {}
        for m in methods:
            errs = []
            for ty in test_years:
                test_bt = sub_bt[sub_bt["Year"] == ty]
                if len(test_bt) < 1:
                    continue
                y_true = test_bt["y"].to_numpy(dtype=float)

                try:
                    if m == "huber":
                        p = os.path.join(store_dir, f"huber__{bt}.joblib")
                        if not os.path.exists(p):
                            continue
                        model = joblib.load(p)
                        X_test = test_bt[FEATURES].to_numpy(dtype=float)
                        y_hat = model.predict(X_test)

                    elif m == "ecm":
                        p = os.path.join(store_dir, f"ecm__{bt}.joblib")
                        if not os.path.exists(p):
                            continue
                        if "gap_lag1" not in test_bt.columns:
                            continue
                        bundle = joblib.load(p)
                        model = bundle["model"]
                        X_test = test_bt[ECM_FEATURES].to_numpy(dtype=float)
                        y_hat = model.predict(X_test)

                    elif m == "pooled":
                        if pooled_bundle is None:
                            continue
                        model = pooled_bundle["model"]
                        dummy_cols = pooled_bundle["dummy_columns"]
                        Xg = pd.DataFrame(test_bt[FEATURES].to_numpy(dtype=float), columns=FEATURES)
                        dum = pd.get_dummies(pd.Series([bt] * len(Xg)), prefix=pooled_bundle["prefix"], drop_first=True)
                        for col in dummy_cols:
                            if col not in dum.columns:
                                dum[col] = 0.0
                        dum = dum[dummy_cols]
                        X_full = pd.concat([Xg, dum], axis=1).to_numpy(dtype=float)
                        y_hat = model.predict(X_full)

                    elif m == "gam":
                        p = os.path.join(store_dir, f"gam__{bt}.joblib")
                        if not os.path.exists(p):
                            continue
                        model = joblib.load(p)
                        X_test = test_bt[FEATURES].to_numpy(dtype=float)
                        y_hat = model.predict(X_test)

                    elif m == "ssm":
                        p = os.path.join(store_dir, f"ssm__{bt}.joblib")
                        if not os.path.exists(p):
                            continue
                        bundle = joblib.load(p)
                        res = bundle["results"]
                        X_test = test_bt[FEATURES].to_numpy(dtype=float)
                        y_hat = res.get_forecast(steps=len(X_test), exog=X_test).predicted_mean

                    else:
                        continue

                    errs.append((y_true - np.asarray(y_hat, dtype=float)) ** 2)
                except Exception:
                    continue

            rmse = float(np.sqrt(np.mean(np.concatenate(errs)))) if errs else float("inf")
            rmses[m] = rmse

        eps = 1e-9
        inv = {m: 1.0 / (rmses[m] + eps) for m in rmses if np.isfinite(rmses[m])}
        if not inv:
            continue
        s = sum(inv.values())
        for m in inv:
            rows.append({"Building Type": bt, "method": m, "rmse": rmses[m], "weight": inv[m] / s})

    w = pd.DataFrame(rows)
    w.to_csv(os.path.join(store_dir, f"ensemble_weights_{tag}.csv"), index=False)
    return w


def main():
    # Resolve data files relative to this script so the pipeline runs no matter
    # what the current working directory is.
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent

    res_path = str(base_dir / "Final Residential Regional MA.xlsx")
    nonres_path = str(base_dir / "Final Non-Residential Regional MA.xlsx")
    pop_path = str(base_dir / "regional population.csv")
    gdp_path = str(base_dir / "regional gdp prod.csv")
    dataset2_path = str(base_dir / "dataset2.xlsx")

    df_long = read_res_nonres_long(res_path, nonres_path)
    build_present_values(df_long, base_year=2024)

    values_pivot, counts_pivot = pivot_values_counts(df_long)
    background = read_background(dataset2_path)
    pop = read_pop(pop_path)
    gdp = read_gdp(gdp_path)

    real_values = adjust_values_to_real(values_pivot, background, base_year=2025)

    ensure_dirs()

    gap_values = compute_ecm_gap(real_values, gdp, pop, background, use_log1p_level=False,
                                 out_name="model_store/values/ecm_gap_values.csv")
    gap_counts = compute_ecm_gap(counts_pivot, gdp, pop, background, use_log1p_level=True,
                                 out_name="model_store/counts/ecm_gap_counts.csv")

    values_y = logdiff_long(real_values, use_log1p=False)
    counts_y = logdiff_long(counts_pivot, use_log1p=True)

    values_df = join_features(values_y, gdp, pop, background)
    counts_df = join_features(counts_y, gdp, pop, background)

    values_df = values_df.merge(gap_values, on=["Year", "Region", "Building Type"], how="left")
    counts_df = counts_df.merge(gap_counts, on=["Year", "Region", "Building Type"], how="left")

    # Option 2: add lagged count growth (count_y_lag1) into VALUE panel
    # counts_df already contains y_lag1 = lagged Δlog1p(counts)
    counts_lag = counts_df[["Year", "Region", "Building Type", "y_lag1"]].copy()
    counts_lag = counts_lag.rename(columns={"y_lag1": "count_y_lag1"})
    values_df = values_df.merge(counts_lag, on=["Year", "Region", "Building Type"], how="left")
    # For VALUE models that require count_y_lag1, drop rows where it is missing
    values_df = values_df.dropna(subset=["count_y_lag1"])

    save_last_lag(values_df, "model_store/values/last_lag_growth_values.csv", base_year=2024)
    save_last_lag(counts_df, "model_store/counts/last_lag_growth_counts.csv", base_year=2024)

    # Fit methods
    fit_save_huber(values_df, "model_store/values", VALUE_FEATURES)
    fit_save_ecm(values_df, "model_store/values", VALUE_ECM_FEATURES)
    fit_save_pooled(values_df, "model_store/values", tag="values", features=VALUE_FEATURES)
    fit_save_gam(values_df, "model_store/values", VALUE_FEATURES)
    fit_save_huber(counts_df, "model_store/counts", COUNT_FEATURES)
    fit_save_ecm(counts_df, "model_store/counts", COUNT_ECM_FEATURES)
    fit_save_pooled(counts_df, "model_store/counts", tag="counts", features=COUNT_FEATURES)
    fit_save_gam(counts_df, "model_store/counts", COUNT_FEATURES)
    methods = ["huber", "ecm", "pooled"]
    if _HAS_PYGAM:
        methods.append("gam")

    compute_weights(values_df, methods, last_n_years=5, store_dir="model_store/values", tag="values")
    compute_weights(counts_df, methods, last_n_years=5, store_dir="model_store/counts", tag="counts")

    print("✅ Done. Residential merge applied: Apartments/Houses/Retirement village/Townhouses/Temporary Accommodation → Residential.")


if __name__ == "__main__":
    main()