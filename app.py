  import streamlit as st
import pandas as pd
import numpy as np
import ta
import yfinance as yf
import cv2
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import requests

# ================= CONFIG =================
st.set_page_config(layout="wide")
st.title("🚀 ULTIMATE NEXT CANDLE AI")

DB = "market_data.db"

mode = st.selectbox("Select Mode", ["Live Market", "Upload CSV", "Upload Image"])

# ================= FEATURE FUNCTION =================
def add_features(df):
    df = df.copy()

    # Ensure numeric
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    df["EMA9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["ATR"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )
    df["Body"] = df["Close"] - df["Open"]

    df.dropna(inplace=True)
    return df


def train_predict(df):
    df = df.copy()
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    features = ["EMA9", "EMA21", "RSI", "MACD", "ATR", "Body"]

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    prob = model.predict_proba(X.iloc[-1:])[0][1]

    return acc, prob


# ================= LIVE MARKET =================
if mode == "Live Market":
    symbol = st.text_input("Symbol", "BTC-USD")
    interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=1)

    if st.button("Predict"):
        df = yf.download(symbol, interval=interval, period="5d")

        if df.empty:
            st.error("No data found.")
        else:
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = add_features(df)

            acc, prob = train_predict(df)

            signal = (
                "BUY 🔥" if prob > 0.6 else
                "SELL ❌" if prob < 0.4 else
                "HOLD ⚠️"
            )

            st.subheader(f"Accuracy: {round(acc*100,2)}%")
            st.subheader(f"Signal: {signal}")
            st.subheader(f"Bullish Probability: {round(prob*100,2)}%")

            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            )])

            st.plotly_chart(fig, use_container_width=True)


# ================= CSV MODE =================
elif mode == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        required = ["Open","High","Low","Close","Volume"]
        if not all(col in df.columns for col in required):
            st.error("CSV must contain Open, High, Low, Close, Volume columns")
        else:
            df = df[required]
            df = add_features(df)

            acc, prob = train_predict(df)

            signal = (
                "BUY 🔥" if prob > 0.6 else
                "SELL ❌" if prob < 0.4 else
                "HOLD ⚠️"
            )

            st.subheader(f"Accuracy: {round(acc*100,2)}%")
            st.subheader(f"Signal: {signal}")
            st.subheader(f"Bullish Probability: {round(prob*100,2)}%")


# ================= IMAGE MODE =================
else:
    img_file = st.file_uploader("Upload JPG/JPEG/PNG", type=["jpg","jpeg","png"])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        h, w = edges.shape
        left = np.sum(edges[:, :w//2])
        right = np.sum(edges[:, w//2:])

        if right > left:
            st.subheader("Predicted Trend: UP 📈")
        else:
            st.subheader("Predicted Trend: DOWN 📉")

        st.image(img, channels="BGR")
        st.warning("Image-based prediction is approximate only.")
