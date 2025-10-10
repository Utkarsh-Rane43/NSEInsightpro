import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import os
from transformers import pipeline
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
import numpy as np
from urllib.parse import quote
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Config ---
NEWSAPI_KEY = "9d01ca71d0114b77ae22e01d1d230f1f"
CACHE_FILE = "symbol_name_cache.json"
PORTFOLIO_FILE = "user_portfolio.json"
TRIGGER_FILE = "price_triggers.json"

# Load stock list
stocks = pd.read_csv("stocks.csv")["symbol"].tolist()

st.title("📈 NSEInsightPro - Stock Analysis & Portfolio Manager")

# --- Cache handling ---
def load_json(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {}

def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f)

# --- Company name cache ---
def get_company_name(symbol):
    cache = load_json(CACHE_FILE)
    if symbol in cache:
        return cache[symbol]
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        name = info.get("longName") or info.get("shortName") or symbol
        cache[symbol] = name
        save_json(cache, CACHE_FILE)
        return name
    except Exception:
        return symbol

# --- Sentiment Model ---
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

sentiment_analyzer = load_sentiment_model()

# --- Fetch News ---
def fetch_news(symbol):
    company_name = get_company_name(symbol)
    query = f'"{company_name}" AND (stock OR share OR NSE)'
    encoded_query = quote(query)
    url = (f"https://newsapi.org/v2/everything?"
           f"q={encoded_query}&language=en&apiKey={NEWSAPI_KEY}&sortBy=publishedAt&pageSize=8")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return [(a['title'], a['url']) for a in resp.json().get("articles", [])]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# --- Safe fetch stock data ---
def safe_yf_ticker(symbol):
    ticker = yf.Ticker(symbol + ".NS")
    try:
        info = ticker.info
        hist = ticker.history(period="1y", interval="1d")
        if hist.empty:
            st.warning("No data found for this stock.")
            return {}, pd.DataFrame()
        return info, hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return {}, pd.DataFrame()

# --- Portfolio Management ---
def load_portfolio():
    return load_json(PORTFOLIO_FILE)

def save_portfolio(portfolio):
    save_json(portfolio, PORTFOLIO_FILE)

# --- Display Portfolio ---
def display_portfolio(portfolio):
    st.header("💼 Portfolio Summary")
    if not portfolio:
        st.info("Portfolio is empty.")
        return
    df = []
    for sym, data in portfolio.items():
        try:
            ticker = yf.Ticker(sym + ".NS")
            current_price = ticker.history(period="1d")['Close'][0]
            qty = data['quantity']
            avg_price = data['avg_price']
            market_value = current_price * qty
            invested = avg_price * qty
            pl = market_value - invested
            df.append([sym, qty, avg_price, current_price, market_value, pl])
        except:
            continue
    df = pd.DataFrame(df, columns=["Stock", "Qty", "Buy Price", "Current Price", "Market Value", "P/L"])
    st.dataframe(df)
    st.write("**Total Portfolio Value:** ₹", df["Market Value"].sum())
    st.write("**Total P/L:** ₹", df["P/L"].sum())

    # Export portfolio
    if st.button("📤 Export Portfolio to CSV"):
        df.to_csv("portfolio_report.csv", index=False)
        st.success("Portfolio exported as 'portfolio_report.csv'.")

# --- Export as PDF ---
def export_to_pdf(stock, info, sentiment_summary, recommendation):
    pdf_file = f"{stock}_report.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 800, "NSEInsightPro Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Stock: {stock}")
    c.drawString(50, 750, f"Company: {get_company_name(stock)}")
    c.drawString(50, 730, f"Current Price: ₹{info.get('currentPrice', 'N/A')}")
    c.drawString(50, 710, f"Day High/Low: ₹{info.get('dayHigh', 'N/A')} / ₹{info.get('dayLow', 'N/A')}")
    c.drawString(50, 690, f"Market Cap: ₹{info.get('marketCap', 'N/A')}")
    c.drawString(50, 670, f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    c.drawString(50, 650, f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
    c.drawString(50, 620, "Sentiment Summary:")
    y = 600
    for label, count in sentiment_summary.items():
        c.drawString(70, y, f"{label}: {count}")
        y -= 20
    c.drawString(50, y - 10, f"Final Recommendation: {recommendation}")
    c.showPage()
    c.save()
    st.success(f"Report exported as {pdf_file}")

# --- Price Trigger System ---
def price_trigger_ui(stock, current_price):
    triggers = load_json(TRIGGER_FILE)
    st.subheader("📊 Price Trigger Alerts")
    trigger_price = st.number_input("Set a Target Price (₹):", min_value=1.0, value=float(current_price))
    if st.button("Set Trigger"):
        triggers[stock] = trigger_price
        save_json(triggers, TRIGGER_FILE)
        st.success(f"Trigger set for {stock} at ₹{trigger_price:.2f}")

    if stock in triggers:
        target = triggers[stock]
        st.info(f"Trigger active: ₹{target:.2f}")
        if current_price >= target:
            st.success(f"🎯 {stock} reached target ₹{target:.2f}!")
        elif current_price <= target * 0.98:
            st.error(f"⚠️ {stock} dropped below ₹{target * 0.98:.2f}")

# --- Main ---
stock = st.selectbox("Select Stock for Analysis", stocks)
if stock:
    info, hist = safe_yf_ticker(stock)
    if not hist.empty:
        company_name = get_company_name(stock)
        st.subheader(f"{stock} - {company_name}")
        current_price = info.get('currentPrice', 0.0)
        price_trigger_ui(stock, current_price)

        # --- Sentiment Analysis ---
        st.header("📰 News Sentiment")
        news_articles = fetch_news(stock)
        pos, neg, neu = 0, 0, 0
        for title, url in news_articles:
            s = sentiment_analyzer(title)[0]
            label = s["label"].lower()
            if label == "positive": pos += 1
            elif label == "negative": neg += 1
            else: neu += 1
            color = "green" if label == "positive" else "red" if label == "negative" else "gray"
            st.markdown(f"- [{title}]({url}) — <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

        sentiment_summary = {"Positive": pos, "Negative": neg, "Neutral": neu}
        labels = list(sentiment_summary.keys())
        values = list(sentiment_summary.values())
        st.plotly_chart(go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)]))

        # Recommendation
        if pos > neg:
            recommendation = "BUY"
        elif neg > pos:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        st.subheader(f"💡 Recommendation: {recommendation}")

        # Export
        if st.button("📄 Export Report as PDF"):
            export_to_pdf(stock, info, sentiment_summary, recommendation)

# --- Portfolio Section ---
display_portfolio(load_portfolio())
