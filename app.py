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
from io import BytesIO  # For Export
import base64
import io
from fpdf import FPDF
import datetime
# --- Config ---
NEWSAPI_KEY = "9d01ca71d0114b77ae22e01d1d230f1f"
CACHE_FILE = "symbol_name_cache.json"
PORTFOLIO_FILE = "user_portfolio.json"  # To persist user portfolio data
TRIGGER_FILE = "price_triggers.json"    # To persist trigger notifications

# Load stocks list from CSV
stocks = pd.read_csv("stocks.csv")["symbol"].tolist()

st.title("NSEInsightPRo - NSE Stock Analysis & Portfolio Manager    ")

# --- NEW: Price Trigger Feature ---
def load_triggers():
    if os.path.exists(TRIGGER_FILE):
        with open(TRIGGER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_triggers(triggers):
    with open(TRIGGER_FILE, "w") as f:
        json.dump(triggers, f)

def check_trigger(stock, current_price):
    triggers = load_triggers()
    trig = triggers.get(stock)
    if trig:
        above = trig.get("above")
        below = trig.get("below")
        if above and current_price > above:
            st.warning(f"{stock} moved **above** ₹{above:.2f} (Current: ₹{current_price:.2f})")
        if below and current_price < below:
            st.warning(f"{stock} moved **below** ₹{below:.2f} (Current: ₹{current_price:.2f})")

def trigger_inputs(stock):
    st.sidebar.header("Price Trigger Alerts")
    triggers = load_triggers()
    above_value = st.sidebar.number_input(f"Notify if {stock} goes **above**", min_value=0.0, value=triggers.get(stock, {}).get("above", 0.0), format="%.2f")
    below_value = st.sidebar.number_input(f"Notify if {stock} goes **below**", min_value=0.0, value=triggers.get(stock, {}).get("below", 0.0), format="%.2f")
    if st.sidebar.button("Set Price Triggers"):
        triggers[stock] = {"above": above_value if above_value > 0 else None, "below": below_value if below_value > 0 else None}
        save_triggers(triggers)
        st.sidebar.success("Triggers updated.")

st.sidebar.header("Analysis Date Range")
default_start = (datetime.date.today() - datetime.timedelta(days=365))
start_date = st.sidebar.date_input('Start Date', default_start)
end_date = st.sidebar.date_input('End Date', datetime.date.today())

stock = st.selectbox("Select a Stock for Analysis", stocks)

if stock:
    trigger_inputs(stock)  # --- Existing feature
    # Updated: pass selected date range to yfinance
    ticker = yf.Ticker(stock + ".NS")
    try:
        info = ticker.info
        hist = ticker.history(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date + datetime.timedelta(days=1)),
            interval="1d")
        if hist.empty or info is None:
            st.warning("No data found for this stock. Try another.")
            info, hist = {}, pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        info, hist = {}, pd.DataFrame()

    if info and not hist.empty:
        company_name = get_company_name(stock)
        st.subheader(f"{stock} - {company_name}")

        # Existing: Price & KPIs
        price = float(info.get('currentPrice', 'nan'))
        check_trigger(stock, price)

        # Existing: KPIs
        prev_close = float(info.get('previousClose', 'nan'))
        price_change = price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price (₹)", f"{price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
        col2.metric("Day High (₹)", info.get('dayHigh', '-'))
        col3.metric("Day Low (₹)", info.get('dayLow', '-'))

        # [52-week Range Visualization - EASY ADDITION]
        wk52_high = info.get('fiftyTwoWeekHigh', None)
        wk52_low = info.get('fiftyTwoWeekLow', None)
        if wk52_high and wk52_low and not pd.isna(wk52_high) and not pd.isna(wk52_low):
            pct = 100 * (price - wk52_low) / (wk52_high - wk52_low)
            st.progress(float(pct) / 100)
            st.caption(f"52-Week Range: ₹{wk52_low:.2f} – ₹{wk52_high:.2f} (Current: ₹{price:.2f})")
        # If not available, this block is skipped.

        # Fundamentals (unchanged)
        # ... same as before ...

        # Existing: Technicals
        hist["MA20"] = hist["Close"].rolling(window=20).mean()
        hist["MA50"] = hist["Close"].rolling(window=50).mean()
        hist["MA200"] = hist["Close"].rolling(window=200).mean()
        hist["RSI"] = ta.momentum.rsi(hist["Close"], window=14)
        # [Bollinger Bands - EASY ADDITION]
        bb = ta.volatility.BollingerBands(hist['Close'], window=20)
        hist["BB_High"] = bb.bollinger_hband()
        hist["BB_Low"]  = bb.bollinger_lband()

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=["Price with Moving Averages (Candlestick)", "Volume", "RSI (14-day)"])

        fig.add_trace(go.Candlestick(x=hist.index,
                                     open=hist['Open'], high=hist['High'],
                                     low=hist['Low'], close=hist['Close'],
                                     name="OHLC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='blue', width=1), name='MA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], line=dict(color='orange', width=1), name='MA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='green', width=1), name='MA 200'), row=1, col=1)
        # Bollinger Bands Overlay
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_High"], line=dict(color='gray', width=1, dash='dash'), 
                                 name='Bollinger High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_Low"], line=dict(color='gray', width=1, dash='dash'), 
                                 name='Bollinger Low'), row=1, col=1)

        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color='lightblue', name='Volume'), row=2, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # ... [rest of your analysis code remains unchanged] ...


# --- Remaining Existing Logic unchanged ---

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def get_company_name(symbol):
    cache = load_cache()
    if symbol in cache:
        return cache[symbol]
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        name = info.get("longName") or info.get("shortName") or symbol
        cache[symbol] = name
        save_cache(cache)
        return name
    except Exception:
        return symbol

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

sentiment_analyzer = load_sentiment_model()

def fetch_news(symbol):
    company_name = get_company_name(symbol)
    query = f'"{company_name}" AND (stock OR share OR NSE)'
    encoded_query = quote(query)  # Properly URL encode
    url = (f"https://newsapi.org/v2/everything?"
           f"q={encoded_query}&"
           "language=en&"
           f"apiKey={NEWSAPI_KEY}&"
           "sortBy=publishedAt&"
           "pageSize=8")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [(a['title'], a['url']) for a in articles]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def fetch_earnings_and_dividends(symbol):
    ticker = yf.Ticker(symbol + ".NS")
    try:
        calendar = ticker.calendar
        dividends = ticker.dividends
        upcoming_div = dividends[dividends.index > pd.Timestamp.today()]
        return calendar, upcoming_div
    except Exception:
        return None, None

def safe_yf_ticker(symbol):
    ticker = yf.Ticker(symbol + ".NS")
    try:
        info = ticker.info
        hist = ticker.history(period="1y", interval="1d")
        if hist.empty or info is None:
            st.warning("No data found for this stock. Try another.")
            return {}, pd.DataFrame()
        return info, hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return {}, pd.DataFrame()

# User Portfolio management
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)

def display_portfolio(portfolio):
    st.header("Your Portfolio Summary")
    if not portfolio:
        st.info("Your portfolio is empty. Add stocks below.")
        return
    portfolio_items = []
    total_value = 0
    total_invested = 0
    st.write("| Stock | Qty | Avg Buy Price (₹) | Current Price (₹) | Market Value (₹) | P/L (₹) |")
    st.write("|-------|-----|-------------------|-------------------|-----------------|---------|")
    for sym, data in portfolio.items():
        qty = data['quantity']
        avg_price = data['avg_price']
        try:
            ticker = yf.Ticker(sym + ".NS")
            current_price = ticker.history(period="1d")['Close'][0]
            market_value = current_price * qty
            invested = avg_price * qty
            pl = market_value - invested
            pl_color = "green" if pl >= 0 else "red"
            st.write(f"| {sym} | {qty} | {avg_price:.2f} | {current_price:.2f} | {market_value:.2f} | <span style='color:{pl_color}'>{pl:.2f}</span> |", unsafe_allow_html=True)
            total_value += market_value
            total_invested += invested
        except:
            st.write(f"| {sym} | {qty} | {avg_price:.2f} | N/A | N/A | N/A |")
    st.markdown(f"**Total Market Value:** ₹{total_value:.2f}")
    st.markdown(f"**Total Invested:** ₹{total_invested:.2f}")
    st.markdown(f"**Total P/L:** ₹{(total_value - total_invested):.2f}")

# Add stock to portfolio UI
def portfolio_management_ui():
    st.header("Manage Your Portfolio")
    portfolio = load_portfolio()
    with st.form("Add Stock to Portfolio"):
        add_symbol = st.selectbox("Select Stock to Add", stocks)
        qty = st.number_input("Quantity", min_value=1, value=1)
        avg_price = st.number_input("Average Buy Price (₹)", format="%.2f", min_value=0.01)
        submitted = st.form_submit_button("Add / Update Stock")

    if submitted:
        if add_symbol in portfolio:
            # Update avg price weighted by existing qty & new qty
            total_qty = portfolio[add_symbol]['quantity'] + qty
            total_cost = (portfolio[add_symbol]['avg_price']*portfolio[add_symbol]['quantity']) + (avg_price*qty)
            portfolio[add_symbol]['quantity'] = total_qty
            portfolio[add_symbol]['avg_price'] = total_cost / total_qty
        else:
            portfolio[add_symbol] = {"quantity": qty, "avg_price": avg_price}

        save_portfolio(portfolio)
        st.success(f"{add_symbol} added/updated in portfolio!")

    # Optionally remove stocks
    if portfolio:
        st.subheader("Remove Stock From Portfolio")
        to_remove = st.selectbox("Select Stock to Remove", list(portfolio.keys()))
        if st.button("Remove Selected Stock"):
            portfolio.pop(to_remove, None)
            save_portfolio(portfolio)
            st.success(f"{to_remove} removed from portfolio!")

    display_portfolio(portfolio)

# --- NEW: Export Analysis to CSV/PDF ---
def export_analysis_to_csv(info, hist, stock, company_name, portfolio):
    # Basic CSV export: info, last 5 data points, and portfolio summary
    buffer = BytesIO()
    # Stock info
    info_df = pd.DataFrame([info])
    info_df.to_csv(buffer, index=False)
    buffer.write('\n'.encode())
    # Historical data (last 5 rows)
    hist.tail(5).to_csv(buffer)
    buffer.write('\n'.encode())
    # Portfolio data
    port_df = pd.DataFrame([{**v, "symbol": k} for k, v in portfolio.items()])
    port_df.to_csv(buffer, index=False)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="analysis_{stock}.csv">Download Analysis as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def export_analysis_to_pdf(info, hist, stock, company_name, portfolio_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"{stock} - {company_name} Analysis", ln=True)
    
    # Example: add some portfolio data
    pdf.set_font("Arial", "", 12)
    for sym, data in portfolio_data.items():
        pdf.cell(0, 8, f"{sym}: Qty={data['quantity']}, Avg Price={data['avg_price']}", ln=True)
    
    # Save PDF to file
    pdf_path = f"{stock}_analysis.pdf"
    pdf.output(pdf_path)  # ✅ must provide filename string
    
    # Provide download button
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF", f, file_name=pdf_path)
# UI Main
stock = st.selectbox("Select a Stock for Analysis", stocks)

if stock:
    trigger_inputs(stock)  # --- NEW: Trigger input sidebar
    info, hist = safe_yf_ticker(stock)

    if info and not hist.empty:
        company_name = get_company_name(stock)
        st.subheader(f"{stock} - {company_name}")

        # NEW: Check triggers on price
        price = float(info.get('currentPrice', 'nan'))
        check_trigger(stock, price)

        # Price & KPIs
        col1, col2, col3 = st.columns(3)
        try:
            prev_close = float(info.get('previousClose', 'nan'))
            price_change = price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            col1.metric("Current Price (₹)", f"{price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
        except:
            col1.write("Price data unavailable.")
        col2.metric("Day High (₹)", info.get('dayHigh', '-'))
        col3.metric("Day Low (₹)", info.get('dayLow', '-'))

        # Fundamentals
        mcap = info.get('marketCap', None)
        pe = info.get('trailingPE', None)
        div_yield = info.get('dividendYield', None)
        st.markdown("### Fundamental Metrics")
        col4, col5, col6 = st.columns(3)
        col4.write(f"Market Cap: ₹{mcap:,}" if mcap else "Market Cap: N/A")
        col5.write(f"P/E Ratio: {pe:.2f}" if pe else "P/E Ratio: N/A")
        col6.write(f"Dividend Yield: {div_yield*100:.2f}%" if div_yield else "Dividend Yield: N/A")

        try:
            buy_strike = price * 0.98
            sell_strike = price * 1.02
            stoploss = price * 0.95
            st.info(f"Suggested Buy Price: ₹{buy_strike:.2f} | Sell Price: ₹{sell_strike:.2f} | Stoploss Price: ₹{stoploss:.2f}")
        except Exception:
            st.warning("Not enough data for buy/sell recommendations.")

        # Moving averages & RSI
        hist["MA20"] = hist["Close"].rolling(window=20).mean()
        hist["MA50"] = hist["Close"].rolling(window=50).mean()
        hist["MA200"] = hist["Close"].rolling(window=200).mean()
        hist["RSI"] = ta.momentum.rsi(hist["Close"], window=14)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=["Price with Moving Averages (Candlestick)", "Volume", "RSI (14-day)"])

        fig.add_trace(go.Candlestick(x=hist.index,
                                     open=hist['Open'], high=hist['High'],
                                     low=hist['Low'], close=hist['Close'],
                                     name="OHLC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='blue', width=1), name='MA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], line=dict(color='orange', width=1), name='MA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='green', width=1), name='MA 200'), row=1, col=1)

        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color='lightblue', name='Volume'), row=2, col=1)

        fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple', width=1), name='RSI'), row=3, col=1)

        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Price gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=price,
            delta={'reference': prev_close},
            gauge={'axis': {'range': [info.get('dayLow', 0) or 0, info.get('dayHigh', 0) or 1]}},
            title={'text': f"{stock} Price Gauge"}))
        st.plotly_chart(gauge)

        # Earnings & dividends calendar
        earnings_cal, dividend_data = fetch_earnings_and_dividends(stock)
        st.header("Earnings / Events Calendar")
        if earnings_cal is not None and not earnings_cal.empty:
            st.table(earnings_cal)
        else:
            st.info("No upcoming earnings/events found.")

        st.header("Upcoming Dividends")
        if dividend_data is not None and not dividend_data.empty:
            dividend_df = pd.DataFrame(dividend_data)
            dividend_df.reset_index(inplace=True)
            dividend_df = dividend_df.rename(columns={"Dividends":"Dividend"})
            # Show next 5 dividends
            st.table(dividend_df.tail(5)[['Date', 'Dividend']])
        else:
            st.info("No upcoming dividend data found.")

        # News + Sentiment
        st.header(f"{stock} News & Sentiment")
        news_articles = fetch_news(stock)
        pos_count = neg_count = neu_count = 0

        if news_articles:
            for title, url in news_articles:
                sentiment = sentiment_analyzer(title)[0]
                label = sentiment["label"].lower()
                score = sentiment["score"]
                color = "green" if label == "positive" else "red" if label == "negative" else "gray"
                st.markdown(f"- [{title}]({url}) — <span style='color:{color}'>{label.capitalize()} ({score:.2f})</span>", unsafe_allow_html=True)

                if label == "positive":
                    pos_count += 1
                elif label == "negative":
                    neg_count += 1
                else:
                    neu_count += 1

            # Sentiment pie chart
            labels = ["Positive", "Negative", "Neutral"]
            values = [pos_count, neg_count, neu_count]
            pie_fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4,
                                             marker_colors=["green", "red", "gray"])])
            pie_fig.update_layout(title="News Sentiment Distribution")
            st.plotly_chart(pie_fig)

            # Recommendation logic
            if pos_count > neg_count and price >= prev_close:
                rec = "BUY"
            elif neg_count > pos_count and price < prev_close:
                rec = "SELL"
            else:
                rec = "HOLD"
            st.subheader(f"Overall Recommendation: {rec}")
        else:
            st.info("No recent news found.")

        # --- NEW: Export to CSV/PDF ---
        st.markdown("---")
        st.subheader("Export Analysis")
        portfolio_data = load_portfolio()
        if st.button("Export as CSV"):
            export_analysis_to_csv(info, hist, stock, company_name, portfolio_data)
        if st.button("Export as PDF"):
            export_analysis_to_pdf(info, hist, stock, company_name, portfolio_data)

    else:
        st.warning("No price data found for this stock.")

# User portfolio management section
portfolio_management_ui()
