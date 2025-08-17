
# app.py — Team A vs Team B Betting Prediction (Advanced Forecasts, No Monte Carlo)
#
# Run locally:
#   pip install streamlit pandas numpy scikit-learn matplotlib statsmodels beautifulsoup4 lxml
#   streamlit run app.py
#
# Highlights:
# - Input last 10 games for each team (A,B), last 10 home for A, last 10 away for B
# - Multiple forecasting models:
#     1) Exponential Regression (baseline; constraints use this)
#     2) EWMA (Simple Exponential Smoothing)
#     3) Holt's Linear Trend (damped) — via statsmodels if available; fallback to manual Holt
#     4) Time‑Weighted Linear Regression (recent games weighted higher)
#     5) Polynomial (Quadratic) Ridge Regression
# - R² and trend (rising/falling) per model, one‑step‑ahead forecast
# - Constraints (as specified) checked using Exponential Regression
# - Improved UI with tabs, tables, and plots
#
# DISCLAIMER: For education only. Small samples (10 games) limit statistical power.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

st.set_page_config(page_title="Team A vs Team B — Advanced Predictor", layout="wide")

OUTCOME_MAP = {
    "W": 1.0, "w": 1.0, 1: 1.0, 1.0: 1.0, "1": 1.0, "win": 1.0, "Win": 1.0,
    "D": 0.5, "d": 0.5, 0.5: 0.5, "0.5": 0.5, "draw": 0.5, "Draw": 0.5,
    "L": 0.0, "l": 0.0, 0: 0.0, 0.0: 0.0, "0": 0.0, "loss": 0.0, "Loss": 0.0
}

def to_numeric_outcomes(seq, length=10):
    vals = []
    for x in seq:
        if pd.isna(x):
            continue
        if isinstance(x, (int, float)):
            v = float(x); v = max(0.0, min(1.0, v)); vals.append(v)
        else:
            x = str(x).strip()
            v = OUTCOME_MAP.get(x, None)
            if v is None:
                try:
                    v = float(x); v = max(0.0, min(1.0, v))
                except:
                    continue
            vals.append(v)
    vals = vals[-length:]
    return vals

def r2_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def clip01(a):
    return np.clip(a, 0.0, 1.0)

def exp_regression_fit(y, eps=1e-6):
    y = np.array(y, dtype=float)
    n = len(y)
    if n < 2:
        return {"name":"Exponential", "yhat":y, "r2":0.0, "rising":False, "slope":0.0, "next":y[-1] if n else 0.5}
    x = np.arange(n).reshape(-1,1)
    y_log = np.log(y + eps).reshape(-1,1)
    model = LinearRegression().fit(x, y_log)
    b = float(model.coef_[0][0]); ln_a = float(model.intercept_[0]); a = np.exp(ln_a)
    x_flat = x.flatten()
    yhat = a * np.exp(b * x_flat); yhat = clip01(yhat)
    r2 = r2_score(y, yhat); rising = b > 0; slope = b
    nxt = float(a * np.exp(b * n))
    return {"name":"Exponential","yhat":yhat,"r2":r2,"rising":rising,"slope":slope,"next":clip01(nxt)}

def ewma_fit(y, alpha=0.5):
    y = np.array(y, dtype=float)
    if len(y) == 0:
        return {"name":"EWMA","yhat":y,"r2":0.0,"rising":False,"slope":0.0,"next":0.5}
    s = pd.Series(y).ewm(alpha=alpha, adjust=False).mean().values
    r2 = r2_score(y, s); rising = np.polyfit(np.arange(len(s)), s, 1)[0] > 0; nxt = s[-1]
    return {"name":"EWMA","yhat":clip01(s),"r2":r2,"rising":rising,"slope":float(np.polyfit(np.arange(len(s)), s, 1)[0]),"next":clip01(nxt)}

def manual_holt(y, alpha=0.6, beta=0.2, damp=0.98):
    y = np.array(y, dtype=float)
    L = y[0] if len(y) else 0.0
    T = y[1] - y[0] if len(y) > 1 else 0.0
    fitted = []
    for t in range(len(y)):
        if t == 0:
            fitted.append(L); continue
        prev_L, prev_T = L, T
        L = alpha * y[t] + (1-alpha) * (prev_L + prev_T)
        T = beta * (L - prev_L) + (1-beta) * (damp * prev_T)
        fitted.append(L + damp * T)
    next_forecast = L + damp * T
    return np.array(fitted), float(next_forecast)

def holt_fit(y, damped=True):
    y = np.array(y, dtype=float)
    if len(y) < 3:
        return {"name":"Holt","yhat":y,"r2":0.0,"rising":False,"slope":0.0,"next":y[-1] if len(y) else 0.5}
    if HAS_STATSMODELS:
        try:
            model = HWES(y, trend='add', damped_trend=damped, initialization_method='estimated')
            res = model.fit(optimized=True)
            s = res.fittedvalues; nxt = res.forecast(1)[0]
        except Exception:
            s, nxt = manual_holt(y)
    else:
        s, nxt = manual_holt(y)
    r2 = r2_score(y, s); slope = float(np.polyfit(np.arange(len(s)), s, 1)[0]); rising = slope > 0
    return {"name":"Holt","yhat":clip01(s),"r2":r2,"rising":rising,"slope":slope,"next":clip01(nxt)}

def weighted_linear_fit(y, gamma=0.85):
    y = np.array(y, dtype=float); n = len(y)
    if n < 2:
        return {"name":"TWLR","yhat":y,"r2":0.0,"rising":False,"slope":0.0,"next":y[-1] if n else 0.5}
    x = np.arange(n).reshape(-1,1)
    w = np.array([gamma**(n-1-i) for i in range(n)], dtype=float)
    model = LinearRegression(); model.fit(x, y, sample_weight=w)
    yhat = model.predict(x); r2 = r2_score(y, yhat); slope = float(model.coef_[0]); rising = slope > 0
    nxt = float(model.predict(np.array([[n]]))[0])
    return {"name":"TWLR","yhat":clip01(yhat),"r2":r2,"rising":rising,"slope":slope,"next":clip01(nxt)}

def poly_ridge_fit(y, degree=2, alpha=1.0):
    y = np.array(y, dtype=float); n = len(y)
    if n < 3:
        return {"name":"Poly2","yhat":y,"r2":0.0,"rising":False,"slope":0.0,"next":y[-1] if n else 0.5}
    x = np.arange(n).reshape(-1,1)
    model = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                      ('ridge', Ridge(alpha=alpha))])
    model.fit(x, y)
    yhat = model.predict(x); r2 = r2_score(y, yhat)
    y_next = model.predict(np.array([[n]]))[0]; slope = float(y_next - yhat[-1]); rising = slope > 0
    return {"name":"Poly2","yhat":clip01(yhat),"r2":r2,"rising":rising,"slope":slope,"next":clip01(y_next)}

def ensemble_next_prob(next_vals):
    vals = np.array(list(next_vals.values()), dtype=float)
    return float(np.nanmean(vals)) if len(vals) else 0.5

def win_rate(y):
    y = np.array(y, dtype=float)
    return float(np.mean(y)) * 100.0 if len(y) else 0.0

def momentum(y):
    y = np.array(y, dtype=float)
    if len(y) < 3:
        return 0.0
    k = min(3, len(y))
    first = y[:-k] if len(y) > k else y
    last = y[-k:]
    return (float(np.mean(last)) - float(np.mean(first))) * 100.0

# Odds helpers (manual only)
def american_to_decimal(odds_amer):
    o = float(odds_amer)
    if o > 0:
        return 1 + o/100.0
    else:
        return 1 + 100.0/abs(o)

def parse_odds(value, fmt="Decimal"):
    if value is None or value == "":
        return None
    s = str(value).strip()
    if fmt == "Decimal":
        try: return float(s)
        except: return None
    elif fmt == "American":
        try: return american_to_decimal(float(s))
        except: return None
    else:
        try: return float(s)
        except: return None

def implied_prob_from_decimal(d):
    return 0.0 if (d is None or d <= 1.0) else 1.0/d

def normalize_implied_probs(pA, pD, pB):
    probs = np.array([pA, pD, pB], dtype=float)
    probs = np.where(probs < 0, 0, probs)
    total = np.nansum(probs)
    if total == 0 or np.isnan(total):
        return probs, 0.0
    overround = total - 1.0
    norm = probs / total
    return norm, overround

def kelly_fraction(p, dec_odds):
    if dec_odds is None or dec_odds <= 1.0:
        return 0.0
    b = dec_odds - 1.0
    f = (b*p - (1-p)) / b
    return max(0.0, f)

# UI controls
st.sidebar.title("Settings")
teamA_name = st.sidebar.text_input("Team A name", "Team A")
teamB_name = st.sidebar.text_input("Team B name", "Team B")
order = st.sidebar.radio("Game order in lists", ["Oldest → Newest", "Newest → Oldest"], index=0)
st.sidebar.markdown("---")
st.sidebar.subheader("Forecasting controls")
alpha_ewma = st.sidebar.slider("EWMA alpha (SES)", 0.05, 0.95, 0.5, 0.05)
gamma_wlin = st.sidebar.slider("Time‑Weighted LR gamma", 0.6, 0.98, 0.85, 0.01)
ridge_alpha = st.sidebar.slider("Poly Ridge α", 0.1, 10.0, 1.0, 0.1)
primary_model = st.sidebar.selectbox("Primary trend model (for plots)", ["Exponential","EWMA","Holt","TWLR","Poly2"], index=0)
st.sidebar.markdown("---")
st.sidebar.subheader("Scoring weights (defaults tuned)")
w_overall = st.sidebar.slider("Weight: Overall win%", 0.0, 1.0, 0.35, 0.05)
w_context = st.sidebar.slider("Weight: Home(A)/Away(B)", 0.0, 1.0, 0.25, 0.05)
w_r2trend = st.sidebar.slider("Weight: R² (exp) × Trend", 0.0, 1.0, 0.20, 0.05)
w_momentum = st.sidebar.slider("Weight: Momentum", 0.0, 1.0, 0.20, 0.05)
falling_penalty = st.sidebar.slider("Penalty if trend falling", 0.5, 2.0, 1.0, 0.1)

st.title("Advanced Team A vs Team B Predictor")
st.caption("Multiple forecasting models, improved UI, no Monte Carlo.")

tab_data, tab_models, tab_odds, tab_reco, tab_export = st.tabs(["Data","Models & Fits","Odds","Recommendation","Export"])

with tab_data:
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Upload CSV (optional)")
        st.write("Long format: columns = team {A,B}, venue {H,A}, result {W/D/L or 1/0.5/0}, optional date.")
        st.write("Wide format: columns A_last10, B_last10, A_home_last10, B_away_last10.")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        data = None; parse_error = None
        def parse_long_csv(df):
            required = {"team","venue","result"}
            if not required.issubset(set(df.columns)):
                return None, "CSV (long) missing columns: team, venue, result"
            if "date" in df.columns:
                try: df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
                except: pass
            A_all = to_numeric_outcomes(df.loc[df["team"].astype(str).str.upper()=="A","result"])
            B_all = to_numeric_outcomes(df.loc[df["team"].astype(str).str.upper()=="B","result"])
            A_home = to_numeric_outcomes(df.loc[(df["team"].astype(str).str.upper()=="A") & (df["venue"].astype(str).str.upper()=="H"),"result"])
            B_away = to_numeric_outcomes(df.loc[(df["team"].astype(str).str.upper()=="B") & (df["venue"].astype(str).str.upper()=="A"),"result"])
            return {"A_last10":A_all,"B_last10":B_all,"A_home_last10":A_home,"B_away_last10":B_away}, None
        def parse_wide_csv(df):
            cols = {"A_last10","B_last10","A_home_last10","B_away_last10"}
            if not cols.issubset(df.columns):
                return None, "CSV (wide) missing one of: " + ", ".join(cols)
            data = {}
            for c in cols:
                data[c] = to_numeric_outcomes(df[c].tolist())
            return data, None

        csv_mode = st.radio("CSV layout", ["Long","Wide"], horizontal=True)
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            data, parse_error = (parse_long_csv(df_up) if csv_mode=="Long" else parse_wide_csv(df_up))
            if parse_error: st.error(parse_error)

    with col2:
        st.subheader("Manual input (10 values each)")
        use_demo = st.checkbox("Use demo values", value=True)
        def demo_data(seed=7):
            rng = np.random.default_rng(seed)
            A_last10 = list(np.clip(rng.normal(0.7, 0.15, 10), 0, 1))
            A_home_last10 = list(np.clip(rng.normal(0.78, 0.1, 10), 0, 1))
            base = np.linspace(0.6, 0.3, 10); noise = rng.normal(0, 0.08, 10)
            B_last10 = list(np.clip(base + noise, 0, 1)); B_away_last10 = list(np.clip(rng.normal(0.35, 0.12, 10), 0, 1))
            return {"A_last10":A_last10,"B_last10":B_last10,"A_home_last10":A_home_last10,"B_away_last10":B_away_last10}
        if use_demo and (uploaded is None):
            demo = demo_data(); A_last10_in = demo["A_last10"]; B_last10_in = demo["B_last10"]
            A_home10_in = demo["A_home_last10"]; B_away10_in = demo["B_away_last10"]
        else:
            A_last10_in = [""]*10; B_last10_in = [""]*10; A_home10_in = [""]*10; B_away10_in = [""]*10

        edf1 = st.data_editor(pd.DataFrame({f"{teamA_name}_last10":A_last10_in}), num_rows="fixed", key="edf1")
        edf2 = st.data_editor(pd.DataFrame({f"{teamB_name}_last10":B_last10_in}), num_rows="fixed", key="edf2")
        edf3 = st.data_editor(pd.DataFrame({f"{teamA_name}_home_last10":A_home10_in}), num_rows="fixed", key="edf3")
        edf4 = st.data_editor(pd.DataFrame({f"{teamB_name}_away_last10":B_away10_in}), num_rows="fixed", key="edf4")

    if ('data' not in locals()) or (data is None):
        data = {"A_last10": to_numeric_outcomes(edf1.iloc[:,0].tolist()),
                "B_last10": to_numeric_outcomes(edf2.iloc[:,0].tolist()),
                "A_home_last10": to_numeric_outcomes(edf3.iloc[:,0].tolist()),
                "B_away_last10": to_numeric_outcomes(edf4.iloc[:,0].tolist())}

    if order == "Newest → Oldest":
        for k in list(data.keys()): data[k] = list(reversed(data[k]))

    ok = True; missing = []
    for k in ["A_last10","B_last10","A_home_last10","B_away_last10"]:
        if len(data.get(k, [])) < 10: ok = False; missing.append(k)
    if not ok: st.warning(f"Please provide at least 10 results for: {', '.join(missing)}")

    if st.button("Compute models", disabled=not ok, type="primary"):
        A_last10 = data["A_last10"]; B_last10 = data["B_last10"]
        A_home10 = data["A_home_last10"]; B_away10 = data["B_away_last10"]
        wr_A = win_rate(A_last10); wr_B = win_rate(B_last10)
        home_A = win_rate(A_home10); away_B = win_rate(B_away10)
        mom_A = momentum(A_last10); mom_B = momentum(B_last10)

        fits_A = {"Exponential": exp_regression_fit(A_last10),
                  "EWMA": ewma_fit(A_last10, alpha=alpha_ewma),
                  "Holt": holt_fit(A_last10),
                  "TWLR": weighted_linear_fit(A_last10, gamma=gamma_wlin),
                  "Poly2": poly_ridge_fit(A_last10, degree=2, alpha=ridge_alpha)}
        fits_B = {"Exponential": exp_regression_fit(B_last10),
                  "EWMA": ewma_fit(B_last10, alpha=alpha_ewma),
                  "Holt": holt_fit(B_last10),
                  "TWLR": weighted_linear_fit(B_last10, gamma=gamma_wlin),
                  "Poly2": poly_ridge_fit(B_last10, degree=2, alpha=ridge_alpha)}

        next_A = {k: v["next"] for k,v in fits_A.items()}
        next_B = {k: v["next"] for k,v in fits_B.items()}
        next_A_ens = ensemble_next_prob(next_A); next_B_ens = ensemble_next_prob(next_B)

        expA = fits_A["Exponential"]; expB = fits_B["Exponential"]
        constraints = {f"A R² (exp) > B R² (exp)": expA["r2"] > expB["r2"],
                       f"A trend rising (exp)": expA["rising"],
                       f"B trend falling (exp)": (not expB["rising"]),
                       f"A overall win% > B overall win%": wr_A > wr_B,
                       f"A home win% > B away win%": home_A > away_B}

        def weighted_score_local(win_overall, context_rate, r2_exp, trend_up, momentum_pts):
            trend_sign = 1.0 if trend_up else -1.0 * falling_penalty
            return (w_overall * win_overall + w_context * context_rate + w_r2trend * (r2_exp * 100.0 * trend_sign) + w_momentum * momentum_pts)

        score_A = weighted_score_local(wr_A, home_A, expA["r2"], expA["rising"], mom_A)
        score_B = weighted_score_local(wr_B, away_B, expB["r2"], expB["rising"], mom_B)

        st.session_state["core"] = dict(A_last10=A_last10, B_last10=B_last10, A_home10=A_home10, B_away10=B_away10,
                                        wr_A=wr_A, wr_B=wr_B, home_A=home_A, away_B=away_B, mom_A=mom_A, mom_B=mom_B,
                                        fits_A=fits_A, fits_B=fits_B, expA=expA, expB=expB,
                                        next_A=next_A, next_B=next_B, next_A_ens=next_A_ens, next_B_ens=next_B_ens,
                                        constraints=constraints, score_A=score_A, score_B=score_B,
                                        teamA=teamA_name, teamB=teamB_name, primary_model=primary_model)
        st.success("Models computed. Switch to Models & Fits / Odds / Recommendation tabs.")

with tab_models:
    if "core" not in st.session_state:
        st.info("Compute models first in the Data tab.")
    else:
        c = st.session_state["core"]
        st.subheader("Key Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(f"{c['teamA']} — Win% (last 10)", f"{c['wr_A']:.1f}%")
        mc2.metric(f"{c['teamB']} — Win% (last 10)", f"{c['wr_B']:.1f}%")
        mc3.metric(f"{c['teamA']} — Home Win% (last 10)", f"{c['home_A']:.1f}%")
        mc4.metric(f"{c['teamB']} — Away Win% (last 10)", f"{c['away_B']:.1f}%")

        st.subheader("Model Comparison (R² & trend)")
        rows = []
        for name, fa in c["fits_A"].items():
            fb = c["fits_B"][name]
            rows.append([name, round(fa["r2"],3), "↑" if fa["rising"] else "↓", round(fb["r2"],3), "↑" if fb["rising"] else "↓"])
        df_summary = pd.DataFrame(rows, columns=["Model","A_R2","A_trend","B_R2","B_trend"])
        st.dataframe(df_summary, use_container_width=True)

        fig_r2, ax = plt.subplots(figsize=(8,4))
        labels = [r[0] for r in rows]; A_vals = [r[1] for r in rows]; B_vals = [r[3] for r in rows]
        x = np.arange(len(labels)); width = 0.35
        ax.bar(x - width/2, A_vals, width, label=c['teamA']); ax.bar(x + width/2, B_vals, width, label=c['teamB'])
        ax.set_title("R² by Model"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("R²"); ax.legend()
        st.pyplot(fig_r2)

        pm = st.session_state.get("primary_model", "Exponential")
        st.subheader(f"Primary Model Plots — {pm}")
        fA = c["fits_A"][pm]; fB = c["fits_B"][pm]
        figA, axA = plt.subplots(); axA.plot(range(1, len(c["A_last10"])+1), c["A_last10"], marker="o", label="Actual")
        axA.plot(range(1, len(fA["yhat"])+1), fA["yhat"], linestyle="--", label=f"{pm} fit"); axA.set_title(f"{c['teamA']}: Actual vs {pm} Fit")
        axA.set_xlabel("Game Index (oldest → newest)"); axA.set_ylabel("Performance"); axA.legend(); st.pyplot(figA)

        figB, axB = plt.subplots(); axB.plot(range(1, len(c["B_last10"])+1), c["B_last10"], marker="o", label="Actual")
        axB.plot(range(1, len(fB["yhat"])+1), fB["yhat"], linestyle="--", label=f"{pm} fit"); axB.set_title(f"{c['teamB']}: Actual vs {pm} Fit")
        axB.set_xlabel("Game Index (oldest → newest)"); axB.set_ylabel("Performance"); axB.legend(); st.pyplot(figB)

with tab_odds:
    if "core" not in st.session_state:
        st.info("Compute models first in the Data tab.")
    else:
        c = st.session_state["core"]
        st.subheader("Odds Analysis (manual input)")
        fmt = st.selectbox("Odds format", ["Decimal","American"], index=0)
        dA = parse_odds(st.text_input(f"Odds for {c['teamA']} win", "1.80"), fmt)
        dD = parse_odds(st.text_input("Odds for Draw", "3.40"), fmt)
        dB = parse_odds(st.text_input(f"Odds for {c['teamB']} win", "4.50"), fmt)

        pA_imp = implied_prob_from_decimal(dA) if dA else 0.0; pD_imp = implied_prob_from_decimal(dD) if dD else 0.0; pB_imp = implied_prob_from_decimal(dB) if dB else 0.0
        probs_norm, overround = normalize_implied_probs(pA_imp, pD_imp, pB_imp)
        st.write("Implied probabilities (normalized):", {c['teamA']:float(probs_norm[0]),"Draw":float(probs_norm[1]),c['teamB']:float(probs_norm[2])})
        st.write(f"Overround: {overround:.3f}")

        p_draw = np.clip(0.1, 0.05, 0.35)  # simple placeholder
        diff = (c["next_A_ens"] - c["next_B_ens"]) * 100.0
        pA_model = 1.0 / (1.0 + np.exp(-diff/10.0))
        pB_model = max(0.0, 1.0 - pA_model - p_draw); tot = pA_model + p_draw + pB_model
        if tot>0: pA_model, p_draw, pB_model = pA_model/tot, p_draw/tot, pB_model/tot
        st.write("Model probabilities (ensemble):", {c['teamA']:round(pA_model,4),"Draw":round(p_draw,4),c['teamB']:round(pB_model,4)})

        EV_A = pA_model * (dA - 1.0) - (1 - pA_model) if dA else None
        EV_D = p_draw   * (dD - 1.0) - (1 - p_draw) if dD else None
        EV_B = pB_model * (dB - 1.0) - (1 - pB_model) if dB else None
        st.write("EV per unit (model-based):", {"Bet A":EV_A,"Bet Draw":EV_D,"Bet B":EV_B})

        kA_model = kelly_fraction(pA_model, dA) if dA else 0.0; kB_model = kelly_fraction(pB_model, dB) if dB else 0.0
        st.write("Kelly fractions (model-based):", {c['teamA']:round(kA_model,3), c['teamB']:round(kB_model,3)})

with tab_reco:
    if "core" not in st.session_state:
        st.info("Compute models first in the Data tab.")
    else:
        c = st.session_state["core"]
        st.subheader("Constraint Checks (Exponential)")
        for k,v in c["constraints"].items():
            st.write(f"- {k}: {'✅' if v else '❌'}")
        st.subheader("Scores & Decision")
        sc1, sc2 = st.columns(2)
        sc1.metric(f"Score — {c['teamA']}", f"{c['score_A']:.2f}"); sc2.metric(f"Score — {c['teamB']}", f"{c['score_B']:.2f}")
        rules_pass = all(c["constraints"].values())
        if (c['score_A'] > c['score_B']) and rules_pass:
            st.success(f"Prediction: {c['teamA']} favored (meets constraints).")
        else:
            st.warning("Inconclusive or constraints not satisfied.")

        st.subheader("One-step-ahead forecasts")
        rows = [[name, c["next_A"][name], c["next_B"][name]] for name in c["next_A"].keys()]
        st.dataframe(pd.DataFrame(rows, columns=["Model",f"{c['teamA']} next",f"{c['teamB']} next"]), use_container_width=True)

with tab_export:
    if "core" not in st.session_state:
        st.info("Compute models first in the Data tab.")
    else:
        c = st.session_state["core"]
        results_df = pd.DataFrame({
            "metric":[ "A_win_rate","B_win_rate","A_home_win_rate","B_away_win_rate","A_momentum","B_momentum",
                       "A_R2_exp","B_R2_exp","A_trend_exp","B_trend_exp","Score_A","Score_B","A_next_exp","B_next_exp","A_next_ensemble","B_next_ensemble"],
            "value":[ c["wr_A"],c["wr_B"],c["home_A"],c["away_B"],c["mom_A"],c["mom_B"],
                      c["expA"]["r2"],c["expB"]["r2"],c["expA"]["rising"],c["expB"]["rising"],
                      c["score_A"],c["score_B"], c["next_A"]["Exponential"], c["next_B"]["Exponential"], c["next_A_ens"], c["next_B_ens"]]
        })
        csv = results_df.to_csv(index=False).encode()
        st.download_button("Download metrics CSV", csv, file_name="advanced_prediction_metrics.csv", mime="text/csv")
