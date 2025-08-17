from __future__ import annotations
import warnings

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.getcwd()
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "processed")
DBG = os.path.join(OUT, "debug")

BUFFER_DAYS = 85
UPTAKE_WINDOW_DAYS = 90
POST_WINDOW_DAYS = 90
MIN_LAG_CASES = 100
ALPHA = 0.05


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


STATE_NAME_TO_CODE = {
    "andhra pradesh": "AP", "arunachal pradesh": "AR", "assam": "AS", "bihar": "BR",
    "chhattisgarh": "CT", "goa": "GA", "gujarat": "GJ", "haryana": "HR", "himachal pradesh": "HP",
    "jharkhand": "JH", "karnataka": "KA", "kerala": "KL", "madhya pradesh": "MP", "maharashtra": "MH",
    "manipur": "MN", "meghalaya": "ML", "mizoram": "MZ", "nagaland": "NL", "odisha": "OR",
    "orissa": "OR", "punjab": "PB", "rajasthan": "RJ", "sikkim": "SK", "tamil nadu": "TN",
    "telangana": "TS", "tripura": "TR", "uttar pradesh": "UP", "uttarakhand": "UT",
    "west bengal": "WB", "chattisgarh": "CT",
    "andaman and nicobar islands": "AN", "andaman & nicobar islands": "AN", "chandigarh": "CH",
    "dadra and nagar haveli and daman and diu": "DN", "daman and diu": "DN", "dadra and nagar haveli": "DN",
    "delhi": "DL", "jammu and kashmir": "JK", "ladakh": "LA", "lakshadweep": "LD",
    "puducherry": "PY", "pondicherry": "PY",
}


def to_code(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    mask_code = s.str.len() <= 3
    s_loc = s.copy()
    s_loc.loc[mask_code] = s_loc.loc[mask_code].str.upper().replace({"OD": "OR", "UK": "UT", "DD": "DN", "CG": "CT"})
    name_mask = ~mask_code
    s_loc.loc[name_mask] = s_loc.loc[name_mask].str.lower().map(STATE_NAME_TO_CODE).fillna(s_loc.loc[name_mask])
    return s_loc


def load_cases_tidy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if "Date_YMD" in df.columns:
        dt = pd.to_datetime(df["Date_YMD"], dayfirst=True, errors="coerce")
        df["Date"] = dt
    df = df.dropna(subset=["Date"]).copy()
    value_cols = [c for c in df.columns if c not in ["Date", "Date_YMD", "Status"]]
    long = df.melt(id_vars=["Date", "Status"], value_vars=value_cols, var_name="State", value_name="value")
    long = long[long["State"] != "TT"].copy()
    long["value"] = to_num(long["value"]).fillna(0)
    tidy = long.pivot_table(index=["Date", "State"], columns="Status", values="value", aggfunc="sum").reset_index()
    tidy.columns.name = None
    tidy = tidy.rename(columns={"Confirmed": "New_Confirmed", "Deceased": "New_Deceased"})
    for c in ["New_Confirmed", "New_Deceased"]:
        if c not in tidy.columns:
            tidy[c] = 0
        tidy[c] = tidy[c].fillna(0)
        tidy.loc[tidy[c] < 0, c] = 0
    return tidy.sort_values(["State", "Date"]).reset_index(drop=True)


def make_weekly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["WeekStart"] = d["Date"] - pd.to_timedelta(d["Date"].dt.weekday, unit="D")
    g = (
        d.groupby(["State", "WeekStart"], as_index=False)
        .agg(New_Confirmed_wk=("New_Confirmed", "sum"), New_Deceased_wk=("New_Deceased", "sum"))
        .sort_values(["State", "WeekStart"])
    )
    return g


def add_cfr_lagged(weekly: pd.DataFrame, lag_weeks: int = 2) -> pd.DataFrame:
    d = weekly.copy()
    d["Cases_lag"] = d.groupby("State")["New_Confirmed_wk"].shift(lag_weeks)
    d = d[d["Cases_lag"] >= MIN_LAG_CASES].copy()
    d["CFR"] = d["New_Deceased_wk"] / d["Cases_lag"].replace({0: np.nan})
    def _win(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        return s.clip(lower=lo, upper=hi)
    d["CFR_wins"] = d.groupby("State")["CFR"].transform(_win)
    d["InfectionWeek"] = d["WeekStart"] - pd.to_timedelta(7 * lag_weeks, unit="D")
    return d


def load_vacc_timeseries(path: str) -> pd.DataFrame:
    v = pd.read_csv(path)
    date_col = "Updated On" if "Updated On" in v.columns else ("Date" if "Date" in v.columns else None)
    if not date_col:
        raise ValueError("Vaccination file missing date column")
    v["Date"] = pd.to_datetime(v[date_col], dayfirst=True, errors="coerce")
    v = v.dropna(subset=["Date"]).copy()
    col_fd = "First Dose Administered"
    if col_fd not in v.columns:
        raise ValueError("Vaccination file missing 'First Dose Administered'")
    v[col_fd] = to_num(v[col_fd]).fillna(0)
    vv = v[["Date", "State", col_fd]].rename(columns={col_fd: "FirstDose"})
    vv = vv[vv["State"].notna()].copy()
    vv["State"] = to_code(vv["State"])
    vv["FirstDose"] = vv["FirstDose"].fillna(0)
    vv = (
        vv.groupby(["State", "Date"], as_index=False)["FirstDose"].sum()
          .sort_values(["State", "Date"]).reset_index(drop=True)
    )
    vv["Cum_FirstDose"] = vv.groupby("State")["FirstDose"].cumsum()
    return vv


def load_population(path: str) -> pd.DataFrame:
    p = pd.read_csv(path)
    tcol = None
    for c in p.columns:
        if c.strip().lower() == "total 2021":
            tcol = c
            break
    if not tcol:
        raise ValueError("Population file must contain 'Total 2021'")
    p[tcol] = to_num(p[tcol])
    p = p.rename(columns={tcol: "Population"})
    p["State"] = to_code(p["State"]) 
    return p[["State", "Population"]]


def load_testing_timeseries(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    t = pd.read_csv(path)
    date_col = None
    for c in ["Date", "Updated On", "Tested As Of", "date"]:
        if c in t.columns:
            date_col = c
            break
    if not date_col:
        raise ValueError("Testing file missing a date column (e.g., 'Date')")
    t["Date"] = pd.to_datetime(t[date_col], dayfirst=True, errors="coerce")
    t = t.dropna(subset=["Date"]).copy()
    if "State" not in t.columns:
        for alt in ["State/UT", "State_Name", "State Code", "StateCode"]:
            if alt in t.columns:
                t = t.rename(columns={alt: "State"})
                break
    if "State" not in t.columns:
        raise ValueError("Testing file missing 'State' column")
    t["State"] = to_code(t["State"]).astype(str)
    daily_candidates = [
        "Daily_Tests", "Tests", "Daily Samples Tested", "Daily Samples", "Samples Tested - Daily",
    ]
    cum_candidates = [
        "Total Samples Tested", "Total Samples", "Samples Tested", "Cumulative Tests",
    ]
    daily_col = next((c for c in daily_candidates if c in t.columns), None)
    cum_col = next((c for c in cum_candidates if c in t.columns), None)
    if daily_col:
        t[daily_col] = to_num(t[daily_col]).fillna(0)
        out = t[["Date", "State", daily_col]].rename(columns={daily_col: "Daily_Tests"}).copy()
    elif cum_col:
        t[cum_col] = to_num(t[cum_col]).fillna(0)
        t = t.sort_values(["State", "Date"])
        t["Daily_Tests"] = t.groupby("State")[cum_col].diff().fillna(0)
        t.loc[t["Daily_Tests"] < 0, "Daily_Tests"] = 0
        out = t[["Date", "State", "Daily_Tests"]].copy()
    else:
        raise ValueError("Testing file must contain a daily or cumulative tests column")
    return out


def compute_uptake60(vacc: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    pop_map = pop.set_index("State")["Population"].to_dict()
    rows = []
    for st, g in vacc.groupby("State"):
        g = g.sort_values("Date")
        nz = g[g["Cum_FirstDose"] > 0]
        if nz.empty:
            continue
        rollout = nz["Date"].iloc[0]
        exp_end = rollout + pd.Timedelta(days=UPTAKE_WINDOW_DAYS)
        full_idx = pd.date_range(start=g["Date"].min(), end=exp_end, freq="D")
        daily_cum = g.set_index("Date")["Cum_FirstDose"].sort_index()
        if not daily_cum.index.is_unique:
            daily_cum = daily_cum.groupby(level=0).max()
        s = daily_cum.reindex(full_idx).ffill().fillna(0.0)
        base = s.loc[rollout - pd.Timedelta(days=1)] if (rollout - pd.Timedelta(days=1)) in s.index else 0.0
        inc = float(s.loc[exp_end] if exp_end in s.index else s.iloc[-1]) - float(base)
        popv = pop_map.get(st)
        if not popv or popv <= 0:
            continue
        uptake = (inc / float(popv)) * 100.0
        post_start = exp_end + pd.Timedelta(days=BUFFER_DAYS)
        post_end = post_start + pd.Timedelta(days=POST_WINDOW_DAYS)
        rows.append({
            "State": st,
            "rollout_date": rollout.normalize(),
            "post_start": post_start.normalize(),
            "post_end": post_end.normalize(),
            "Uptake60_per100": uptake,
        })
    return pd.DataFrame(rows)

def compute_uptake_window(vacc: pd.DataFrame, pop: pd.DataFrame, window_days: int) -> pd.DataFrame:
    pop_map = pop.set_index("State")["Population"].to_dict()
    rows = []
    for st, g in vacc.groupby("State"):
        g = g.sort_values("Date")
        nz = g[g["Cum_FirstDose"] > 0]
        if nz.empty:
            continue
        rollout = nz["Date"].iloc[0]
        exp_end = rollout + pd.Timedelta(days=window_days)
        full_idx = pd.date_range(start=g["Date"].min(), end=exp_end, freq="D")
        daily_cum = g.set_index("Date")["Cum_FirstDose"].sort_index()
        if not daily_cum.index.is_unique:
            daily_cum = daily_cum.groupby(level=0).max()
        s = daily_cum.reindex(full_idx).ffill().fillna(0.0)
        base = s.loc[rollout - pd.Timedelta(days=1)] if (rollout - pd.Timedelta(days=1)) in s.index else 0.0
        inc = float(s.loc[exp_end] if exp_end in s.index else s.iloc[-1]) - float(base)
        popv = pop_map.get(st)
        if not popv or popv <= 0:
            continue
        uptake = (inc / float(popv)) * 100.0
        post_start = exp_end + pd.Timedelta(days=BUFFER_DAYS)
        post_end = post_start + pd.Timedelta(days=POST_WINDOW_DAYS)
        rows.append({
            "State": st,
            "rollout_date": rollout.normalize(),
            "post_start": post_start.normalize(),
            "post_end": post_end.normalize(),
            f"Uptake{window_days}_per100": uptake,
        })
    return pd.DataFrame(rows)


def build_panel(weekly_cfr: pd.DataFrame, pop: pd.DataFrame, uptake: pd.DataFrame, tests_weekly: pd.DataFrame | None = None, uptake_var: str = "Uptake_per100") -> pd.DataFrame:
    df = weekly_cfr.copy()
    pop_map = pop.set_index("State")["Population"].to_dict()
    df["Population"] = df["State"].map(pop_map)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Cases_per100k"] = (df["New_Confirmed_wk"] / df["Population"]) * 1e5
    if tests_weekly is not None and not tests_weekly.empty:
        df = df.merge(tests_weekly, on=["State", "WeekStart"], how="left")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["Tests_per100k"] = (df["Tests_wk"] / df["Population"]) * 1e5
            df["Positivity"] = df["New_Confirmed_wk"] / df["Tests_wk"].replace({0: np.nan})
    df = df.merge(uptake, on="State", how="inner")
    try:
        df["CalendarWeek"] = df["WeekStart"].dt.to_period("W-MON").astype(str)
    except Exception:
        df["CalendarWeek"] = df["WeekStart"].astype(str)
    if "InfectionWeek" not in df.columns:
        warnings.warn("InfectionWeek not found; falling back to WeekStart for event time.")
        df["InfectionWeek"] = df["WeekStart"]
    df["EventWeek"] = ((df["InfectionWeek"] - df["rollout_date"]) / np.timedelta64(1, "W")).apply(np.floor).astype(int)
    df["post_start_w"] = ((df["post_start"] - df["rollout_date"]) / np.timedelta64(1, "W")).apply(np.floor).astype(int)
    df["post_end_w"] = ((df["post_end"] - df["rollout_date"]) / np.timedelta64(1, "W")).apply(np.floor).astype(int)
    df["Post"] = (df["EventWeek"] >= df["post_start_w"]) & (df["EventWeek"] <= df["post_end_w"]) 
    min_pre = df["post_start_w"] - 12
    df = df[(df["EventWeek"] >= min_pre) & (df["EventWeek"] <= df["post_end_w"])].copy()
    df["PostWeek"] = 0
    mask = df["Post"].astype(bool)
    df.loc[mask, "PostWeek"] = (df.loc[mask, "EventWeek"] - df.loc[mask, "post_start_w"]).astype(int)
    df["PreWeek"] = np.nan
    pre_mask = df["EventWeek"] < df["post_start_w"]
    df.loc[pre_mask, "PreWeek"] = (df.loc[pre_mask, "EventWeek"] - (df.loc[pre_mask, "post_start_w"] - 12)).astype(float)
    return df.dropna(subset=["CFR_wins", uptake_var, "Cases_per100k"]) 


def run_models(panel: pd.DataFrame, dep: str = "CFR_wins", uptake_var: str = "Uptake_per100"):
    controls = ["Cases_per100k"]
    if "Tests_per100k" in panel.columns:
        controls.append("Tests_per100k")
    if "Positivity" in panel.columns:
        controls.append("Positivity")
    controls_term = " + ".join(controls) if controls else "1"
    fe_terms = "C(State) + C(CalendarWeek)"
    f_lvl = f"{dep} ~ Post*{uptake_var} + {controls_term} + {fe_terms}"
    f_slp = f"{dep} ~ PostWeek*{uptake_var} + {controls_term} + {fe_terms}"
    m1 = smf.ols(formula=f_lvl, data=panel).fit()
    m2 = smf.ols(formula=f_slp, data=panel).fit()
    m1 = m1.get_robustcov_results(cov_type="cluster", groups=panel["State"]) 
    m2 = m2.get_robustcov_results(cov_type="cluster", groups=panel["State"]) 
    return m1, m2


def run_pretrend_model(panel: pd.DataFrame, dep: str = "CFR_wins", uptake_var: str = "Uptake_per100"):
    pre = panel[(panel["Post"] == False) & (panel["PreWeek"].notna())].copy()
    if pre.empty:
        return None
    controls = ["Cases_per100k"]
    if "Tests_per100k" in pre.columns:
        controls.append("Tests_per100k")
    if "Positivity" in pre.columns:
        controls.append("Positivity")
    controls_term = " + ".join(controls) if controls else "1"
    fe_terms = "C(State) + C(CalendarWeek)"
    f = f"{dep} ~ PreWeek*{uptake_var} + {controls_term} + {fe_terms}"
    m = smf.ols(formula=f, data=pre).fit()
    m = m.get_robustcov_results(cov_type="cluster", groups=pre["State"]) 
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DBG, exist_ok=True)

    cases = load_cases_tidy(os.path.join(RAW, "state_wise_daily.csv"))
    weekly = make_weekly(cases)
    weekly2 = add_cfr_lagged(weekly, lag_weeks=2)
    weekly3 = add_cfr_lagged(weekly, lag_weeks=3)
    weekly4 = add_cfr_lagged(weekly, lag_weeks=4)
    weekly.to_csv(os.path.join(DBG, "weekly_cases_deaths.csv"), index=False)

    vacc = load_vacc_timeseries(os.path.join(RAW, "cowin_vaccine_data_statewise.csv"))
    pop = load_population(os.path.join(RAW, "state_wise_population.csv"))
    windows = [180, 270]
    uptake_map: dict[int, pd.DataFrame] = {}
    for w in windows:
        up = compute_uptake_window(vacc, pop, w)
        uptake_map[w] = up
        up.to_csv(os.path.join(DBG, f"uptake{w}.csv"), index=False)

    tests_weekly = None
    testing_path = os.path.join(RAW, "state_wise_testing.csv")
    if os.path.exists(testing_path):
        try:
            tests = load_testing_timeseries(testing_path)
            tt = tests.copy()
            tt["WeekStart"] = tt["Date"] - pd.to_timedelta(tt["Date"].dt.weekday, unit="D")
            tests_weekly = (
                tt.groupby(["State", "WeekStart"], as_index=False)["Daily_Tests"].sum()
                  .rename(columns={"Daily_Tests": "Tests_wk"})
            )
            tests_weekly = tests_weekly.sort_values(["State", "WeekStart"]) 
        except Exception as e:
            print(f"Warning: failed to load testing data: {e}")
            tests_weekly = None
    else:
        print("Info: testing data not found; models will run without testing controls.")

    lines = [f"Hypothesis summary (alpha={ALPHA:.2f})"]

    def save_results(name: str, model):
        params = np.asarray(model.params)
        pvals = np.asarray(model.pvalues)
        names = None
        try:
            idx = getattr(model.params, "index", None)
            if idx is not None:
                names = list(idx)
        except Exception:
            names = None
        if names is None or len(names) != len(params):
            try:
                names = list(model.model.exog_names)
            except Exception:
                names = [f"x{i}" for i in range(len(params))]
        df = pd.DataFrame({
            "term": names,
            "estimate": params,
            "pvalue": pvals,
        })
        df.to_csv(os.path.join(OUT, f"model_results_{name}.csv"), index=False)
        return df

    def pick(df, keys):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            row = df[df["term"] == k]
            if not row.empty:
                return float(row["estimate"].iloc[0]), float(row["pvalue"].iloc[0])
        return None

    for w, uptake in uptake_map.items():
        uptake_var = f"Uptake{w}_per100"
        p2 = build_panel(weekly2, pop, uptake, tests_weekly, uptake_var=uptake_var)
        p3 = build_panel(weekly3, pop, uptake, tests_weekly, uptake_var=uptake_var)
        p4 = build_panel(weekly4, pop, uptake, tests_weekly, uptake_var=uptake_var)

        p2.to_csv(os.path.join(OUT, f"panel_cfr_lag2_w{w}.csv"), index=False)
        p3.to_csv(os.path.join(OUT, f"panel_cfr_lag3_w{w}.csv"), index=False)
        p4.to_csv(os.path.join(OUT, f"panel_cfr_lag4_w{w}.csv"), index=False)

        m2_lvl, m2_slp = run_models(p2, dep="CFR_wins", uptake_var=uptake_var)
        m3_lvl, m3_slp = run_models(p3, dep="CFR_wins", uptake_var=uptake_var)
        m4_lvl, m4_slp = run_models(p4, dep="CFR_wins", uptake_var=uptake_var)

        pd.DataFrame(m2_lvl.params, columns=["estimate"]).to_csv(os.path.join(OUT, f"model_coefs_lag2_w{w}.csv"))
        pd.DataFrame(m3_lvl.params, columns=["estimate"]).to_csv(os.path.join(OUT, f"model_coefs_lag3_w{w}.csv"))
        pd.DataFrame(m4_lvl.params, columns=["estimate"]).to_csv(os.path.join(OUT, f"model_coefs_lag4_w{w}.csv"))
        pd.DataFrame(m2_slp.params, columns=["estimate"]).to_csv(os.path.join(OUT, f"model_slope_coefs_lag2_w{w}.csv"))
        pd.DataFrame(m3_slp.params, columns=["estimate"]).to_csv(os.path.join(OUT, f"model_slope_coefs_lag3_w{w}.csv"))
        pd.DataFrame(m4_slp.params, columns=["estimate"]).to_csv(os.path.join(OUT, f"model_slope_coefs_lag4_w{w}.csv"))

        r2 = save_results(f"lag2_level_w{w}", m2_lvl)
        r2s = save_results(f"lag2_slope_w{w}", m2_slp)
        r3 = save_results(f"lag3_level_w{w}", m3_lvl)
        r3s = save_results(f"lag3_slope_w{w}", m3_slp)
        r4 = save_results(f"lag4_level_w{w}", m4_lvl)
        r4s = save_results(f"lag4_slope_w{w}", m4_slp)

        post_uptake_alts = [
            f"Post:{uptake_var}",
            f"Post[T.True]:{uptake_var}",
            f"Post[T.1]:{uptake_var}",
            f"{uptake_var}:Post[T.True]",
            f"{uptake_var}:Post[T.1]",
        ]
        keys = {
            f"w{w}_lag2_level": (r2, post_uptake_alts),
            f"w{w}_lag2_slope": (r2s, f"PostWeek:{uptake_var}"),
            f"w{w}_lag3_level": (r3, post_uptake_alts),
            f"w{w}_lag3_slope": (r3s, f"PostWeek:{uptake_var}"),
            f"w{w}_lag4_level": (r4, post_uptake_alts),
            f"w{w}_lag4_slope": (r4s, f"PostWeek:{uptake_var}"),
        }
        for name, (dfres, term) in keys.items():
            val = pick(dfres, term)
            if val is None:
                lines.append(f"{name}: term '{term}' not found")
                continue
            est, pv = val
            concl = "REJECT H0 (effect)" if pv < ALPHA else "FAIL TO REJECT H0"
            lines.append(f"{name}: {term}: est={est:.6f}, p={pv:.4g} -> {concl}")

        pre_m = run_pretrend_model(p2, dep="CFR_wins", uptake_var=uptake_var)
        if pre_m is not None:
            try:
                est = float(getattr(pre_m.params, "get", lambda k, d=None: np.nan)(f"PreWeek:{uptake_var}") or np.nan)
                pv = float(getattr(pre_m.pvalues, "get", lambda k, d=None: np.nan)(f"PreWeek:{uptake_var}") or np.nan)
                if not np.isnan(est) and not np.isnan(pv):
                    concl = "NO pre-trend" if pv >= ALPHA else "Potential pre-trend"
                    lines.append(f"pretrend_w{w}_lag2: PreWeek:{uptake_var}: est={est:.6f}, p={pv:.4g} -> {concl}")
            except Exception:
                pass
    summary_path = os.path.join(BASE, "hypothesis_result.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Done. Outputs in data/processed/")


if __name__ == "__main__":
    main()
