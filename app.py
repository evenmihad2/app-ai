
# ================== IMPORT ==================

import subprocess, time, os


# ================== STREAMLIT APP ==================
app_code = '''
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
import time
import sqlite3
import requests


DB = "market_data.db"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

def add_indicators(df):
    # Ensure columns are numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'
st.set_page_config(layout="wide")
st.title("🚀 ULTIMATE NEXT CANDLE AI")

mode = st.selectbox("Select Mode", ["Live Market","Upload CSV","Upload Image"])

# ================= FEATURE FUNCTION =================
def add_features(df):
    # যদি squeeze() কাজ না করে, তবে এটি ব্যবহার করতে পারেন:
df['EMA9'] = ta.trend.ema_indicator(df['Close'].values.flatten(), 9)

    df['EMA21'] = ta.trend.ema_indicator(df['Close'],21)
    df['RSI'] = ta.momentum.rsi(df['Close'],14)
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['ATR'] = ta.volatility.average_true_range(df['High'],df['Low'],df['Close'],14)
    df['Body'] = df['Close'] - df['Open']
    df.dropna(inplace=True)
    return df

def train_predict(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'],1,0)
    features = ['EMA9','EMA21','RSI','MACD','ATR','Body']
    X = df[features]
    y = df['Target']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    model = RandomForestClassifier(n_estimators=400)
    model.fit(X_train,y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    prob = model.predict_proba(X.iloc[-1:])[0][1]
    return acc, prob

# ================= LIVE MARKET =================
if mode == "Live Market":
    symbol = st.text_input("Symbol","BTC-USD")
    interval = st.selectbox("Interval",["1m","5m","15m"], index=1)

    if st.button("Predict"):
        df = yf.download(symbol, interval=interval, period="5d")
        df = df[['Open','High','Low','Close','Volume']].dropna()
        df = add_features(df)

        acc, prob = train_predict(df)

        signal = "BUY 🔥" if prob>0.6 else "SELL ❌" if prob<0.4 else "HOLD ⚠️"

        st.subheader(f"Accuracy: {round(acc*100,2)}%")
        st.subheader(f"Signal: {signal}")
        st.subheader(f"Bullish Probability: {round(prob*100,2)}%")

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        st.plotly_chart(fig, use_container_width=True)

# ================= CSV MODE =================
elif mode == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = df[['Open','High','Low','Close','Volume']]
        df = add_features(df)
        acc, prob = train_predict(df)

        signal = "BUY 🔥" if prob>0.6 else "SELL ❌" if prob<0.4 else "HOLD ⚠️"

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
        edges = cv2.Canny(gray,50,150)

        h,w = edges.shape
        left = np.sum(edges[:,:w//2])
        right = np.sum(edges[:,w//2:])

        if right > left:
            st.subheader("Predicted Trend: UP 📈")
        else:
            st.subheader("Predicted Trend: DOWN 📉")

        st.image(img, channels="BGR")
        st.write("⚠ Image-based prediction is approximate only.")
,,,




# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS candles(
        time TEXT PRIMARY KEY,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL
    )
    """)
    conn.commit()
    conn.close()

# ================= FETCH LIVE DATA =================
def fetch_live_data():
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit=100"
    data = requests.get(url).json()

    df = pd.DataFrame(data, columns=[
        'time','open','high','low','close','volume',
        '_','_','_','_','_','_'
    ])

    df = df[['time','open','high','low','close','volume']]
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.astype({
        'open':float,
        'high':float,
        'low':float,
        'close':float,
        'volume':float
    })

    return df

# ================= STORE DATA =================
def store_data(df):
    conn = sqlite3.connect(DB)
    df.to_sql("candles", conn, if_exists="replace", index=False)
    conn.close()

# ================= FEATURE ENGINEERING =================
def add_features(df):
    df['EMA9'] = ta.trend.ema_indicator(df['Close'].squeeze(), 9)

    df['RSI'] = ta.momentum.rsi(df['close'],14)
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    df['ATR'] = ta.volatility.average_true_range(df['high'],df['low'],df['close'],14)
    df.dropna(inplace=True)
    return df

# ================= TRAIN MODEL =================
def train_predict(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'],1,0)

    X = df[['EMA9','RSI','MACD','ATR']]
    y = df['target']

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X[:-1], y[:-1])

    pred = model.predict(X.iloc[-1:])[0]
    prob = model.predict_proba(X.iloc[-1:])[0].max()

    return pred, prob

# ================= MAIN LOOP =================
def run_live():
    init_db()
    print("🚀 Live Auto Candle Predictor Started...\n")

    while True:
        df = fetch_live_data()
        store_data(df)

        df = add_features(df)
        pred, prob = train_predict(df)

        signal = "BUY" if pred==1 else "SELL"
        print(f"Signal: {signal} | Confidence: {prob*100:.2f}%")

        time.sleep(60)

run_live()

subprocess.Popen(["streamlit","run","app.py","--server.port","8501"])

time.sleep(6)
app_code = """
import streamlit as st
st.title("🚀 Candle Predictor")
"""

with open("app.py","w") as f:
    f.write(app_code)


'''
# ================== RUN STREAMLIT ==================
subprocess.Popen(["streamlit","run","app.py","--server.port","8501"])

time.sleep(6)

