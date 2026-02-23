# =======================  Streamlit + Localtunnel =========================

import subprocess, time, os
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import cv2
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import requests

# ================== STREAMLIT APP ==================
app_code = """
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
import time

DB = "market_data.db"

st.set_page_config(layout="wide")
st.title("🚀 ULTIMATE NEXT CANDLE AI")

mode = st.selectbox("Select Mode", ["Live Market","Upload CSV","Upload Image"])

# ================= FEATURE FUNCTION =================
def add_features(df):
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['EMA9'] = ta.trend.ema_indicator(df['Close'].values.flatten(), 9)
    df['SMA20'] = ta.trend.sma_indicator(df['Close'].values.flatten(), 20)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'].values.flatten(),21)
    df['RSI'] = ta.momentum.rsi(df['Close'].values.flatten(),14)
    df['MACD'] = ta.trend.MACD(df['Close'].values.flatten()).macd()
    df['ATR'] = ta.volatility.average_true_range(df['High'].values.flatten(),
                                                 df['Low'].values.flatten(),
                                                 df['Close'].values.flatten(),14)
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

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(\"\"\"
    CREATE TABLE IF NOT EXISTS candles(
        time TEXT PRIMARY KEY,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL
    )
    \"\"\")
    conn.commit()
    conn.close()

# ================= FETCH LIVE DATA =================
def fetch_live_data():
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'time','open','high','low','close','volume','ignore1','ignore2','ignore3','ignore4','ignore5','ignore6'
    ])
    df = df[['time','open','high','low','close','volume']]
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.astype({'open':float,'high':float,'low':float,'close':float,'volume':float})
    return df

def store_data(df):
    conn = sqlite3.connect(DB)
    df.to_sql("candles", conn, if_exists="replace", index=False)
    conn.close()

# ================= TRAIN MODEL =================
def train_live_model(df):
    df['EMA9'] = ta.trend.ema_indicator(df['close'].values.flatten(), 9)
    df['RSI'] = ta.momentum.rsi(df['close'].values.flatten(),14)
    df['MACD'] = ta.trend.MACD(df['close'].values.flatten()).macd()
    df['ATR'] = ta.volatility.average_true_range(df['high'].values.flatten(),
                                                 df['low'].values.flatten(),
                                                 df['close'].values.flatten(),14)
    df['target'] = np.where(df['close'].shift(-1) > df['close'],1,0)
    X = df[['EMA9','RSI','MACD','ATR']]
    y = df['target']
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X[:-1], y[:-1])
    pred = model.predict(X.iloc[-1:])[0]
    prob = model.predict_proba(X.iloc[-1:])[0].max()
    return pred, prob

def run_live():
    init_db()
    print("🚀 Live Auto Candle Predictor Started...")
    while True:
        df = fetch_live_data()
        store_data(df)
        pred, prob = train_live_model(df)
        signal = "BUY" if pred==1 else "SELL"
        print(f"Signal: {signal} | Confidence: {prob*100:.2f}%")
        time.sleep(60)

# Uncomment below line to run live in background
# run_live()
"""

# Save Streamlit app
with open("app.py", "w") as f:
    f.write(app_code)

# Run Streamlit server in background
subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port=8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(5)

# Start localtunnel
lt_process = subprocess.Popen(
    ["lt", "--port", "8501", "--subdomain=myapp", "--auth", "user:1234"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(5)

# Public URL capture
while True:
    line = lt_process.stdout.readline()
    if "https://" in line.lower():
        print("🌐 Open this URL in your Kodular app or browser:")
        print(line.strip())
        break
