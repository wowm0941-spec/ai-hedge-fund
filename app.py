import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI Hedge Fund", layout="wide")

# ================================
# HEADER
# ================================
st.title("🚀 AI Hedge Fund Dashboard")

# ================================
# INDEX DEFINITIONS
# ================================
INDEX_ETFS = {
    "S&P 500": "SPY",
    "NASDAQ 100": "QQQ",
    "MSCI World": "URTH",
    "Emerging Markets": "EEM",
    "Russell 2000": "IWM"
}

INDEX_HOLDINGS = {
    "S&P 500": ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","V"],
    "NASDAQ 100": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"],
    "MSCI World": ["AAPL","MSFT","NVDA","TSLA","GOOGL","AMZN"],
    "Emerging Markets": ["TSM","BABA","PDD","JD"],
    "Russell 2000": ["CRUS","SRPT","MTSI"]
}

ALL_TICKERS = sorted(list(set([t for sub in INDEX_HOLDINGS.values() for t in sub])))

# ================================
# DATA LOADER
# ================================
@st.cache_data
def load_data(tickers, period="1y"):
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data["Close"]
        return data.dropna()
    except:
        return pd.DataFrame()

# ================================
# MENU
# ================================
menu = st.sidebar.selectbox("Menü", ["Dashboard","Index Vergleich","Stock Analyse","Smart Index"])

# ================================
# DASHBOARD
# ================================
if menu == "Dashboard":

    st.subheader("📊 Indizes Übersicht")

    df = load_data(list(INDEX_ETFS.values()), "1y")

    if df.empty:
        st.warning("Keine Daten")
    else:
        fig = go.Figure()

        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col] / df[col].iloc[0],
                name=col
            ))

        st.plotly_chart(fig, use_container_width=True)

# ================================
# INDEX VERGLEICH
# ================================
elif menu == "Index Vergleich":

    st.subheader("📈 Index Vergleich")

    selected = st.multiselect(
        "Indizes wählen",
        list(INDEX_ETFS.keys()),
        default=["S&P 500","NASDAQ 100"]
    )

    period = st.selectbox("Zeitraum", ["6mo","1y","2y","5y"])

    if selected:

        tickers = [INDEX_ETFS[s] for s in selected]
        df = load_data(tickers, period)

        if df.empty:
            st.warning("Keine Daten")
        else:
            fig = go.Figure()

            for col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col] / df[col].iloc[0],
                    name=col
                ))

            st.plotly_chart(fig, use_container_width=True)

# ================================
# STOCK ANALYSE (STABIL)
# ================================
elif menu == "Stock Analyse":

    st.subheader("📊 Aktien Analyse")

    sel = st.selectbox("Aktie wählen", ALL_TICKERS)

    period = st.selectbox("Zeitraum", ["6mo","1y","2y","5y"])

    df = load_data(sel, period)

    if df.empty:
        st.warning("Keine Daten")
    else:
        price = df[sel]

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price, name="Preis"))
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        returns = price.pct_change().dropna()

        st.metric("Volatilität", f"{returns.std()*np.sqrt(252):.2%}")

        # YTD FIX
        current_year = datetime.now().year
        ytd_data = price[price.index.year == current_year]

        if len(ytd_data) > 0:
            ytd_return = price.iloc[-1] / ytd_data.iloc[0] - 1
        else:
            ytd_return = price.iloc[-1] / price.iloc[0] - 1

        st.metric("YTD", f"{ytd_return:.2%}")

        total_return = price.iloc[-1] / price.iloc[0] - 1
        st.metric("Total Return", f"{total_return:.2%}")

        # Download
        csv = price.to_csv().encode()

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{sel}.csv",
            mime="text/csv"
        )

# ================================
# SMART INDEX
# ================================
elif menu == "Smart Index":

    st.subheader("🤖 Smart Index Builder")

    index_choice = st.selectbox("Index auswählen", list(INDEX_HOLDINGS.keys()))

    stocks = INDEX_HOLDINGS[index_choice]

    df = load_data(stocks, "1y")

    if df.empty:
        st.warning("Keine Daten")
    else:
        scores = {}

        for s in stocks:
            p = df[s]
            mom = p.pct_change(60).iloc[-1]
            vol = p.pct_change().rolling(20).std().iloc[-1]

            if not np.isnan(mom) and not np.isnan(vol):
                scores[s] = mom - vol

        if len(scores) == 0:
            st.warning("Nicht genug Daten")
        else:
            ranking = pd.Series(scores).sort_values(ascending=False)

            st.write("Top Aktien:")
            st.dataframe(ranking)

            top_n = st.slider("Top auswählen", 1, len(ranking), 3)

            selected = ranking.head(top_n).index

            smart = df[selected].mean(axis=1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=smart.index,
                y=smart / smart.iloc[0],
                name="Smart Index"
            ))

            st.plotly_chart(fig, use_container_width=True)
