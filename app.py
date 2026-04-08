# app.py
# AI Hedge Fund / Index Builder - PRO MVP
# Single-file Streamlit app with:
# - Indices + decomposition
# - comparison module
# - stock analysis + ML scoring
# - auto-rebalancing backtest
# - UI with dark/light toggle, menu, logo (demo)
# Notes: This is a demo MVP. Replace holdings with authoritative lists for production.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io

st.set_page_config(page_title="HedgeLab — AI Index Builder", layout="wide", initial_sidebar_state="expanded")

# --------------------
# UTILS & THEME
# --------------------
LOGO = """
<div style="display:flex;align-items:center">
  <div style="width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,#ff7a18,#af002d);display:flex;align-items:center;justify-content:center;margin-right:10px;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M4 12h16v2H4z"/><path d="M7 7h10v2H7z"/><path d="M10 2h4v2h-4z"/></svg>
  </div>
  <div>
    <div style="font-weight:700;color:var(--text-color);">HedgeLab</div>
    <div style="font-size:11px;color:var(--muted-color);">AI Index Builder • PRO MVP</div>
  </div>
</div>
"""
def local_css():
    st.markdown(
        """
<style>
:root { --bg: #0e1117; --text-color: #E6EDF3; --muted-color: #9aa8b2; }
[data-theme="light"] { --bg: #ffffff; --text-color: #0b1220; --muted-color: #6c757d; }
.main-header { padding: 8px 0; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:10px; padding:12px; }
.small { font-size:12px; color:var(--muted-color); }
</style>
""", unsafe_allow_html=True)

local_css()

# Dark/light toggle using session_state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
theme_col1, theme_col2 = st.columns([1,8])
with theme_col1:
    if st.button("Toggle Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
with theme_col2:
    st.markdown(f'<div class="main-header">{LOGO}</div>', unsafe_allow_html=True)
# apply data-theme attribute for CSS variables
if st.session_state.theme == "light":
    st.markdown('<div data-theme="light"></div>', unsafe_allow_html=True)

# --------------------
# Index Definitions & sample holdings
# --------------------
# Use ETF tickers for index historical price, and example constituents (top holdings sample).
INDEX_ETFS = {
    "S&P 500": "SPY",
    "NASDAQ 100": "QQQ",
    "MSCI World (ish)": "URTH",   # iShares MSCI World ETF (approx)
    "Emerging Markets": "EEM",
    "Russell 2000": "IWM"
}

# Example holdings by index (sample top names; replace with authoritative lists for production)
INDEX_HOLDINGS = {
    "S&P 500": ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","V"],
    "NASDAQ 100": ["NVDA","AAPL","MSFT","AMZN","TSLA","GOOGL","META","ADBE","PYPL","CMCSA"],
    "MSCI World (ish)": ["AAPL","MSFT","NVDA","TSLA","GOOGL","AMZN","JPM","V","UNH","MA"],
    "Emerging Markets": ["TCEHY","TSM","BABA","NIO","PDD","JD","LI","KWEFY","HDB","HMMY"],  # mix of ADRs
    "Russell 2000": ["CRUS","LSPD","HAE","BHE","SRPT","ICUI","GHLD","VCTR","VSAT","MTSI"]
}

ALL_TICKERS = sorted(list({t for vals in INDEX_HOLDINGS.values() for t in vals} | set(INDEX_ETFS.values())))

# --------------------
# DATA LOADER (safe)
# --------------------
@st.cache_data(ttl=60*60)
def load_close(tickers, period="2y"):
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        return pd.DataFrame()
    # normalize columns if MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            df = raw["Close"]
        else:
            df = raw.iloc[:, raw.columns.get_level_values(1) == "Close"]
            df.columns = df.columns.get_level_values(0)
    else:
        df = raw
    # if single ticker -> pd.Series -> convert
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all")
    return df

# --------------------
# SIMPLE DEMO AUTH (session store) -- DEMO ONLY
# --------------------
if "users" not in st.session_state:
    st.session_state["users"] = {"demo@demo.com":"demo123"}
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

def login_widget():
    st.sidebar.header("Account")
    if not st.session_state.logged_in:
        email = st.sidebar.text_input("E-Mail")
        pwd = st.sidebar.text_input("Passwort", type="password")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Login"):
            if email in st.session_state.users and st.session_state.users[email] == pwd:
                st.session_state.logged_in = True
                st.session_state.user = email
                st.sidebar.success(f"angemeldet als {email}")
            else:
                st.sidebar.error("Ungültig.")
        if col2.button("Register"):
            if email and pwd:
                st.session_state.users[email] = pwd
                st.sidebar.success("Account erstellt. Jetzt anmelden.")
            else:
                st.sidebar.warning("E-Mail + Passwort eingeben")
    else:
        st.sidebar.write(f"Angemeldet: **{st.session_state.user}**")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.sidebar.success("Abgemeldet")

login_widget()

# --------------------
# SIDEBAR MENU
# --------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Dashboard","Compare Indices","Stock Lab","Smart Index Builder","Backtest & Rebalance","Research / Exports","Settings"])

# small quick selector
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Indices")
for k in INDEX_ETFS.keys():
    if st.sidebar.button(k):
        st.experimental_set_query_params(view=k)

# --------------------
# Dashboard (landing)
# --------------------
def show_dashboard():
    st.header("Dashboard")
    st.markdown("**Live Übersicht über Indizes & Smart Index.** Wähle ein Modul links.")
    # fetch etf closes
    etf_df = load_close(list(INDEX_ETFS.values()), period="1y")
    if etf_df.empty:
        st.warning("Keine Daten geladen. Prüfe Internet/Yahoo.")
        return
    fig = go.Figure()
    for col in etf_df.columns:
        fig.add_trace(go.Scatter(x=etf_df.index, y=etf_df[col]/etf_df[col].iloc[0], mode="lines", name=col))
    fig.update_layout(height=420, template="plotly_dark" if st.session_state.theme=="dark" else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Top Holdings (Beispiel)**")
    cols = st.columns(2)
    for i,(k,v) in enumerate(INDEX_HOLDINGS.items()):
        cols[i%2].write(f"**{k}**: " + ", ".join(v[:10]))

# --------------------
# Compare Indices
# --------------------
def show_compare_indices():
    st.header("Index Vergleich")
    st.markdown("Wähle Indizes, zeige/hide Indizes, normiere auf 100, zeige Performance.")
    choices = st.multiselect("Wähle Indizes (ETF-Ticker werden geladen):", list(INDEX_ETFS.keys()), default=list(INDEX_ETFS.keys())[:3])
    period = st.selectbox("Zeitraum:", ["6mo","1y","2y","5y"], index=1)
    if not choices:
        st.info("Wähle mindestens einen Index.")
        return
    tickers = [INDEX_ETFS[c] for c in choices]
    df = load_close(tickers, period=period)
    if df.empty:
        st.error("Fehler beim Laden der Zeitreihe.")
        return
    # normalized
    norm = df / df.iloc[0] * 100
    hide = st.multiselect("Ausgeblendete Indizes (optional):", choices, default=[])
    fig = go.Figure()
    for c in choices:
        if c in hide:
            continue
        t = INDEX_ETFS[c]
        fig.add_trace(go.Scatter(x=norm.index, y=norm[t], name=c))
    fig.update_layout(title="Index-Performance (normiert)", height=500, template="plotly_dark" if st.session_state.theme=="dark" else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    # show table metrics
    perf = (df.iloc[-1]/df.iloc[0]-1).T
    perf_df = pd.DataFrame({"Ticker":[INDEX_ETFS[k] for k in choices],"Return":perf.values})
    st.dataframe(perf_df.style.format({"Return":"{:.2%}"}))

# --------------------
# Stock Lab (single stock analysis)
# --------------------
def show_stock_lab():
    st.header("Stock Lab — Einzeltitel Analyse")
    universe = sorted(list(set(ALL_TICKERS)))
    sel = st.selectbox("Wähle Aktie:", universe, index=universe.index("AAPL") if "AAPL" in universe else 0)
    period = st.selectbox("Zeitraum:", ["6mo","1y","2y","5y"], index=2)
    df = load_close(sel, period=period)
    if df.empty:
        st.warning("Keine Daten.")
        return
    price = df[sel]
    st.subheader(f"{sel} — Letzter Preis: {price.iloc[-1]:.2f}")
    # Price chart
    fig = px.line(price, x=price.index, y=price.values, labels={"x":"Date","y":"Price"}, title=f"{sel} Price")
    st.plotly_chart(fig, use_container_width=True)
    # indicators
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=price.index, y=price.values, name="Price"))
    fig2.add_trace(go.Scatter(x=ma50.index, y=ma50.values, name="MA50"))
    fig2.add_trace(go.Scatter(x=ma200.index, y=ma200.values, name="MA200"))
    fig2.update_layout(title=f"{sel} Price + MAs", height=420, template="plotly_dark" if st.session_state.theme=="dark" else "plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    # returns
    ret = price.pct_change().dropna()
    st.metric("Volatilität (ann.)", f"{ret.std()*np.sqrt(252):.2%}")
# YTD sauber berechnen
current_year = datetime.now().year
ytd_data = price[price.index.year == current_year]

if len(ytd_data) > 0:
    ytd_return = price.iloc[-1] / ytd_data.iloc[0] - 1
else:
    ytd_return = price.iloc[-1] / price.iloc[0] - 1

st.metric("YTD", f"{ytd_return:.2%}")
    csv = price.to_csv().encode()
    st.download_button(
      label="Download Preis CSV",
      data=csv,
      file_name=f"{sel}_prices.csv",
      mime="text/csv"
    )

# --------------------
# Smart Index Builder (decompose & build index of top performers)
# --------------------
def show_smart_index_builder():
    st.header("Smart Index Builder")
    st.markdown("Zerlege Indizes, wähle Top-Performer und baue eigenen Index. ML-Scoring optional.")
    base_index = st.selectbox("Index zum Aufdröseln:", list(INDEX_HOLDINGS.keys()))
    holdings = INDEX_HOLDINGS.get(base_index, [])
    st.markdown(f"Top-Holdings (Beispiel): {', '.join(holdings[:20])}")
    period = st.selectbox("Datenzeitraum:", ["1y","2y","5y"], index=1)
    df = load_close(holdings, period=period)
    if df.empty:
        st.warning("Keine Daten für Holdings.")
        return
    # compute total returns and momentum
    perf = df.pct_change().iloc[-1]
    mom3 = df.pct_change(60).iloc[-1]
    vol = df.pct_change().rolling(20).std().iloc[-1]
    score = mom3 - vol  # simple
    score = score.sort_values(ascending=False)
    st.subheader("Scoring (Momentum - Volatility)")
    st.table(score.to_frame("Score"))
    n = st.slider("Wie viele Top-Titel in Smart Index?", 1, min(20, len(score)), value=min(5, len(score)))
    top = score.head(n).index.tolist()
    st.write("Top ausgewählt:", top)
    smart_df = df[top]
    smart_index = smart_df.mean(axis=1)  # equal weighted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=smart_index.index, y=smart_index/smart_index.iloc[0]*100, name="Smart Index (eq)"))
    for k,t in INDEX_ETFS.items():
        if k==base_index: continue
    fig.update_layout(title="Smart Index (Normiert)", template="plotly_dark" if st.session_state.theme=="dark" else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    # show constituents performance and allow include/exclude
    st.subheader("Constituent Performance")
    show_table = st.checkbox("Show constituents", value=False)
    if show_table:
        cons_perf = (smart_df.iloc[-1]/smart_df.iloc[0]-1).sort_values(ascending=False)
        st.dataframe(cons_perf.to_frame("Return").style.format({"Return":"{:.2%}"}))
    # export
    out_csv = smart_index.to_frame("value").to_csv().encode()
    st.download_button("Download Smart Index CSV", data=out_csv, file_name="smart_index.csv", mime="text/csv")

# --------------------
# Smart ML engine + Backtest & Auto-Rebalance
# --------------------
def show_backtest_rebalance():
    st.header("Backtest & Auto Rebalancing")
    universe = st.multiselect("Universe (manuell/tickern):", ALL_TICKERS, default=ALL_TICKERS[:10])
    if not universe:
        st.info("Wähle Universe.")
        return
    lookback = st.slider("Lookback (days) for momentum:", 20, 240, 60)
    rebalance = st.slider("Rebalance every (days):", 7, 90, 30)
    topk = st.slider("Top K to hold:", 1, min(20, len(universe)), 5)
    period = st.selectbox("Backtest Period:", ["1y","2y","5y"], index=2)
    df = load_close(universe, period=period)
    if df.empty:
        st.error("Keine Daten.")
        return
    # feature building (vectorized)
    returns = df.pct_change()
    momentum = df.pct_change(lookback)
    vol = returns.rolling(20).std()
    # simple ML: train RandomForest on historical windows (per stock)
    window = 200
    # prepare training rows by stacking per stock
    rows = []
    for stock in df.columns:
        s = df[stock].dropna()
        if len(s) < window + 10:
            continue
        for i in range(window, len(s)-5):
            price_window = s.iloc[i-window:i+1]
            mom = price_window.pct_change(lookback).iloc[-1] if i>=lookback else np.nan
            vol_i = price_window.pct_change().rolling(20).std().iloc[-1]
            target = price_window.pct_change().shift(-5).iloc[-1] if (i+5)<len(s) else np.nan
            if np.isnan(mom) or np.isnan(vol_i) or np.isnan(target):
                continue
            rows.append({"stock":stock,"mom":mom,"vol":vol_i,"target":target})
    train = pd.DataFrame(rows).dropna()
    if train.empty:
        st.error("Zu wenig Trainingsdaten.")
        return
    X = train[["mom","vol"]]
    y = train["target"]
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    st.success("ML Model trained (RF) on universe history.")
    # scoring latest
    latest_df = []
    for s in df.columns:
        if len(df[s].dropna()) < lookback+10:
            continue
        mom = df[s].pct_change(lookback).iloc[-1]
        volv = df[s].pct_change().rolling(20).std().iloc[-1]
        latest_df.append({"stock":s,"mom":mom,"vol":volv})
    latest_df = pd.DataFrame(latest_df).dropna().set_index("stock")
    latest_df["pred"] = model.predict(latest_df[["mom","vol"]])
    ranking = latest_df["pred"].sort_values(ascending=False)
    st.subheader("ML Ranking (pred next return)")
    st.dataframe(ranking.head(20).to_frame("predicted"))
    # auto rebalance backtest simulation
    start_cap = st.number_input("Start Kapital", value=10000)
    dates = df.index
    cash = start_cap
    history = []
    for i in range(200, len(dates), rebalance):
        window_df = df.iloc[:i]
        # score by momentum - vol or model predict on latest window
        scores = {}
        for s in window_df.columns:
            mom = window_df[s].pct_change(lookback).iloc[-1] if len(window_df[s].dropna())>lookback else np.nan
            volv = window_df[s].pct_change().rolling(20).std().iloc[-1]
            if np.isnan(mom) or np.isnan(volv):
                continue
            scores[s] = mom - volv
        if len(scores) < topk: continue
        chosen = pd.Series(scores).sort_values(ascending=False).head(topk).index
        # assume equal weight, compute return next rebalance period
        next_ret = df[chosen].iloc[i:i+rebalance].pct_change().mean().mean()  # approx
        cash = cash * (1 + next_ret)
        history.append((dates[i], cash))
    if len(history)==0:
        st.warning("Nicht genug History für Backtest.")
        return
    equity = pd.DataFrame(history, columns=["date","value"]).set_index("date")
    fig = px.line(equity, y="value", title="Backtest Equity Curve")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Return", f"{equity['value'].iloc[-1]/start_cap-1:.2%}")
    # allow download
    csv = equity.to_csv().encode()
    st.download_button("Download Equity CSV", data=csv, file_name="equity.csv", mime="text/csv")

# --------------------
# Research / Exports
# --------------------
def show_research():
    st.header("Research Engine & Exports")
    st.markdown("Zeige Top-Faktoren, erkläre Picks, exportiere reports.")
    # simple explanation for picks
    st.subheader("Explain Top Picks (breakdown)")
    sample_pick = st.selectbox("Pick stock to explain:", sorted(list(ALL_TICKERS)))
    period = st.selectbox("History window for explanation:", ["1y","2y","5y"], index=1)
    df = load_close(sample_pick, period=period)
    if df.empty:
        st.warning("Keine Daten.")
        return
    price = df[sample_pick]
    mom1 = price.pct_change(20).iloc[-1]
    mom3 = price.pct_change(60).iloc[-1]
    vol = price.pct_change().rolling(20).std().iloc[-1]
    st.write(f"Momentum 1M: {mom1:.2%}, 3M: {mom3:.2%}, Vol: {vol:.2%}")
    st.markdown("**Narrative (automated)**")
    # ultra-simple narrative generator (placeholder for real NLP model)
    narrative = f"{sample_pick} hat in den letzten 3 Monaten Momentum von {mom3:.2%} gezeigt. Volatilität ist {vol:.2%}. Trend: {'bullish' if mom3>0 else 'bearish'}."
    st.info(narrative)
    st.subheader("Export Report")
    buf = io.StringIO()
    buf.write("HedgeLab Report\n")
    buf.write(f"Stock: {sample_pick}\n")
    buf.write(narrative + "\n")
    b = buf.getvalue().encode()
    st.download_button("Download Report (txt)", data=b, file_name=f"{sample_pick}_report.txt", mime="text/plain")

# --------------------
# Settings & About
# --------------------
def show_settings():
    st.header("Settings")
    st.markdown("App settings and info.")
    st.markdown("**Note**: This is a demo MVP. For production use, secure authentication, proper holdings API, DB, and hosted model infra are required.")
    st.markdown("**Theme**: " + st.session_state.theme)
    st.markdown("Contact: demo@hedgelab.local")

# --------------------
# Router
# --------------------
if menu == "Dashboard":
    show_dashboard()
elif menu == "Compare Indices":
    show_compare_indices()
elif menu == "Stock Lab":
    show_stock_lab()
elif menu == "Smart Index Builder":
    show_smart_index_builder()
elif menu == "Backtest & Rebalance":
    show_backtest_rebalance()
elif menu == "Research / Exports":
    show_research()
elif menu == "Settings":
    show_settings()
else:
    st.write("Not implemented")

# --------------------
# Footer
# --------------------
st.markdown("---")
st.markdown("Built with ❤️ • HedgeLab PRO MVP — Demo. Replace sample holdings with real ones for production.")
