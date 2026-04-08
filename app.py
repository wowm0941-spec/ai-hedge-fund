import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Hedge Fund", layout="wide")

# ================================
# TITLE
# ================================
st.title("🚀 AI Hedge Fund Dashboard")

# ================================
# SETTINGS
# ================================
stocks = [
"AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V"
]

top_n = st.slider("Top Aktien", 3, 10, 5)
capital = st.number_input("Startkapital", value=10000)
period = st.selectbox("Zeitraum", ["1y","2y","5y"])

# ================================
# DATA
# ================================
@st.cache_data
def load_data(tickers, period):
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna()

data = load_data(stocks, period)

# ================================
# FEATURES
# ================================
def create_features(df):
    frames = []
    for stock in df.columns:
        d = pd.DataFrame()
        d["price"] = df[stock]
        d["ret"] = d["price"].pct_change()

        d["mom1"] = d["price"].pct_change(20)
        d["mom3"] = d["price"].pct_change(60)
        d["mom6"] = d["price"].pct_change(120)

        d["vol"] = d["ret"].rolling(20).std()

        d["ma50"] = d["price"].rolling(50).mean()
        d["ma200"] = d["price"].rolling(200).mean()
        d["trend"] = (d["ma50"] > d["ma200"]).astype(int)

        d["target"] = d["ret"].shift(-5)
        d["stock"] = stock

        frames.append(d)

    return pd.concat(frames).dropna()

df = create_features(data)

# ================================
# MODEL
# ================================
features = ["mom1","mom3","mom6","vol","trend"]

model = RandomForestRegressor(n_estimators=150)
model.fit(df[features], df["target"])

latest = df.groupby("stock").last()
preds = model.predict(latest[features])

ranking = pd.Series(preds, index=latest.index).sort_values(ascending=False)

# ================================
# DISPLAY
# ================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 AI Ranking")
    st.dataframe(ranking)

selected = ranking.head(top_n).index.tolist()

with col2:
    st.subheader("🔥 Auswahl")
    st.write(selected)

# ================================
# BACKTEST
# ================================
cash = capital
values = []

for i in range(200, len(data), 30):
    window = data.iloc[:i]

    scores = {}
    for s in stocks:
        p = window[s]

        mom = p.pct_change(60).iloc[-1]
        vol = p.pct_change().rolling(20).std().iloc[-1]

        if pd.isna(mom) or pd.isna(vol):
            continue

        scores[s] = mom - vol

    if len(scores) < top_n:
        continue

    scores = pd.Series(scores).sort_values(ascending=False)
    chosen = scores.head(top_n).index

    ret = window[chosen].pct_change().iloc[-1].mean()
    cash *= (1 + ret)

    values.append(cash)

equity = pd.Series(values)

# ================================
# CHART
# ================================
st.subheader("📈 Performance")

fig, ax = plt.subplots()
ax.plot(equity)
st.pyplot(fig)

# ================================
# METRICS
# ================================
if len(equity) > 0:
    ret = equity.iloc[-1] / capital - 1
    st.metric("Return", f"{ret:.2%}")
else:
    st.warning("Nicht genug Daten für Backtest")
