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
from io import BytesIO
import base64
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# --- Config ---
NEWSAPI_KEY = "9d01ca71d0114b77ae22e01d1d230f1f"
CACHE_FILE = "symbol_name_cache.json"
PORTFOLIO_FILE = "user_portfolio.json"
TRIGGER_FILE = "price_triggers.json"
MODEL_CACHE = "ml_models_cache.json"

# Load stocks
stocks = pd.read_csv("stocks.csv")["symbol"].tolist()

st.set_page_config(page_title="NSEInsightPro", layout="wide")
st.title("🚀 NSEInsightPro - Advanced NSE Stock Analysis & Portfolio Manager")

# --- Utility Functions ---
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
            st.warning(f"🔔 {stock} moved **above** ₹{above:.2f} (Current: ₹{current_price:.2f})")
        if below and current_price < below:
            st.warning(f"🔔 {stock} moved **below** ₹{below:.2f} (Current: ₹{current_price:.2f})")

def trigger_inputs(stock):
    st.sidebar.header("📊 Price Trigger Alerts")
    triggers = load_triggers()
    above_value = st.sidebar.number_input(
        f"Notify if {stock} goes **above**", 
        min_value=0.0, 
        value=triggers.get(stock, {}).get("above", 0.0), 
        format="%.2f"
    )
    below_value = st.sidebar.number_input(
        f"Notify if {stock} goes **below**", 
        min_value=0.0, 
        value=triggers.get(stock, {}).get("below", 0.0), 
        format="%.2f"
    )
    if st.sidebar.button("Set Price Triggers"):
        triggers[stock] = {
            "above": above_value if above_value > 0 else None, 
            "below": below_value if below_value > 0 else None
        }
        save_triggers(triggers)
        st.sidebar.success("✅ Triggers updated.")

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

sentiment_analyzer = load_sentiment_model()

def fetch_news(symbol):
    company_name = get_company_name(symbol)
    query = f'"{company_name}" AND (stock OR share OR NSE)'
    encoded_query = quote(query)
    url = (f"https://newsapi.org/v2/everything?"
           f"q={encoded_query}&"
           "language=en&"
           f"apiKey={NEWSAPI_KEY}&"
           "sortBy=publishedAt&"
           "pageSize=10")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [(a['title'], a['url'], a.get('publishedAt', '')) for a in articles]
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

def safe_yf_ticker(symbol, period="1y", interval="1d", start=None, end=None):
    ticker = yf.Ticker(symbol + ".NS")
    try:
        if start and end:
            hist = ticker.history(start=start, end=end, interval=interval)
        else:
            hist = ticker.history(period=period, interval=interval)
        info = ticker.info
        if hist.empty or info is None:
            st.warning("⚠️ No data found for this stock. Try another.")
            return {}, pd.DataFrame()
        return info, hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return {}, pd.DataFrame()

def get_latest_price(stock):
    ticker = yf.Ticker(stock + ".NS")
    try:
        price = ticker.history(period="1d")['Close'].iloc[-1]
        return price
    except:
        return None

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)

# --- ML Price Prediction ---
def prepare_features_for_ml(hist):
    """Prepare features for ML model"""
    df = hist.copy()
    
    # Technical indicators as features
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
    
    # Price changes
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Diff'] = df['High'] - df['Low']
    
    # Lag features
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    
    df = df.dropna()
    return df

def train_price_prediction_model(hist):
    """Train ML model to predict next day price"""
    try:
        df = prepare_features_for_ml(hist)
        
        if len(df) < 50:
            return None, None
        
        # Features and target
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 
                       'RSI', 'MACD', 'Volume_MA', 'Price_Change', 'High_Low_Diff',
                       'Lag1', 'Lag2', 'Lag3']
        
        X = df[feature_cols]
        y = df['Close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        latest_features = X.iloc[-1:].values
        latest_scaled = scaler.transform(latest_features)
        predicted_price = model.predict(latest_scaled)[0]
        
        # Model accuracy on test set
        from sklearn.metrics import r2_score, mean_absolute_error
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return predicted_price, {'r2_score': r2, 'mae': mae, 'model': model, 'scaler': scaler}
    
    except Exception as e:
        st.error(f"Model training error: {e}")
        return None, None

def predict_future_prices(hist, days=7):
    """Predict prices for next N days"""
    try:
        df = prepare_features_for_ml(hist)
        
        if len(df) < 50:
            return None
        
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 
                       'RSI', 'MACD', 'Volume_MA', 'Price_Change', 'High_Low_Diff',
                       'Lag1', 'Lag2', 'Lag3']
        
        X = df[feature_cols]
        y = df['Close']
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        # Predict iteratively
        predictions = []
        last_data = df.iloc[-1:].copy()
        
        for i in range(days):
            features = last_data[feature_cols].values
            features_scaled = scaler.transform(features)
            pred_price = model.predict(features_scaled)[0]
            predictions.append(pred_price)
            
            # Update for next prediction (simplified)
            last_data['Close'] = pred_price
            last_data['Lag3'] = last_data['Lag2'].values[0]
            last_data['Lag2'] = last_data['Lag1'].values[0]
            last_data['Lag1'] = pred_price
            last_data['MA5'] = (last_data['MA5'].values[0] * 4 + pred_price) / 5
            last_data['MA10'] = (last_data['MA10'].values[0] * 9 + pred_price) / 10
        
        return predictions
    except Exception as e:
        return None

# --- Advanced Technical Indicators ---
def calculate_advanced_indicators(hist):
    """Calculate MACD, Stochastic, Support/Resistance"""
    df = hist.copy()
    
    # MACD
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_diff'] = ta.trend.macd_diff(df['Close'])
    
    # Stochastic Oscillator
    df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_signal'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    
    # Support and Resistance (using recent highs/lows)
    recent_high = df['High'].tail(20).max()
    recent_low = df['Low'].tail(20).min()
    
    return df, recent_high, recent_low

# --- Risk Metrics ---
def calculate_risk_metrics(hist, market_data=None):
    """Calculate Volatility, Beta, Sharpe Ratio"""
    returns = hist['Close'].pct_change().dropna()
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming risk-free rate = 6%)
    risk_free_rate = 0.06
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
    
    # Beta (if market data available)
    beta = None
    if market_data is not None:
        market_returns = market_data['Close'].pct_change().dropna()
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance if market_variance != 0 else None
    
    return {
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'beta': beta
    }

# --- Backtesting ---
def simple_backtest(hist, strategy='MA_crossover'):
    """Simple backtesting strategy"""
    df = hist.copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    if strategy == 'MA_crossover':
        df['Signal'] = 0
        df.loc[df['MA20'] > df['MA50'], 'Signal'] = 1  # Buy signal
        df.loc[df['MA20'] < df['MA50'], 'Signal'] = -1  # Sell signal
        
        df['Position'] = df['Signal'].shift()
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Position'] * df['Returns']
        
        total_return = (1 + df['Strategy_Returns']).prod() - 1
        buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        return {
            'strategy_return': total_return,
            'buy_hold_return': buy_hold_return,
            'signals': df[df['Signal'] != 0][['Signal', 'Close']].tail(10)
        }
    
    return None

# --- Enhanced Export Functions ---
def export_comprehensive_csv(stock, company_name, info, hist, predictions, news_sentiment, portfolio, risk_metrics, backtest_results):
    """Export comprehensive analysis to CSV"""
    buffer = BytesIO()
    
    # Stock Info
    info_df = pd.DataFrame([{
        'Symbol': stock,
        'Company': company_name,
        'Current Price': info.get('currentPrice', 'N/A'),
        'Market Cap': info.get('marketCap', 'N/A'),
        'P/E Ratio': info.get('trailingPE', 'N/A'),
        'Dividend Yield': info.get('dividendYield', 'N/A'),
        '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52W Low': info.get('fiftyTwoWeekLow', 'N/A'),
        'Day High': info.get('dayHigh', 'N/A'),
        'Day Low': info.get('dayLow', 'N/A'),
        'Volume': info.get('volume', 'N/A'),
        'Avg Volume': info.get('averageVolume', 'N/A')
    }])
    
    # Write to buffer
    info_df.to_csv(buffer, index=False)
    buffer.write(b'\n\n--- PREDICTIONS ---\n')
    
    if predictions:
        pred_df = pd.DataFrame({
            'Predicted Next Day Price': [predictions.get('next_day', 'N/A')],
            'Model Accuracy (R2)': [predictions.get('r2_score', 'N/A')],
            'Mean Absolute Error': [predictions.get('mae', 'N/A')],
            'Suggested Buy Price': [predictions.get('buy_strike', 'N/A')],
            'Suggested Sell Price': [predictions.get('sell_strike', 'N/A')],
            'Suggested Stop Loss': [predictions.get('stop_loss', 'N/A')]
        })
        pred_df.to_csv(buffer, index=False)
    
    buffer.write(b'\n\n--- RISK METRICS ---\n')
    if risk_metrics:
        risk_df = pd.DataFrame([risk_metrics])
        risk_df.to_csv(buffer, index=False)
    
    buffer.write(b'\n\n--- HISTORICAL DATA (Last 30 Days) ---\n')
    hist.tail(30).to_csv(buffer)
    
    buffer.write(b'\n\n--- NEWS SENTIMENT ---\n')
    if news_sentiment:
        news_df = pd.DataFrame(news_sentiment)
        news_df.to_csv(buffer, index=False)
    
    buffer.write(b'\n\n--- BACKTEST RESULTS ---\n')
    if backtest_results:
        bt_df = pd.DataFrame([{
            'Strategy Return': backtest_results.get('strategy_return', 'N/A'),
            'Buy & Hold Return': backtest_results.get('buy_hold_return', 'N/A')
        }])
        bt_df.to_csv(buffer, index=False)
    
    buffer.write(b'\n\n--- PORTFOLIO ---\n')
    port_df = pd.DataFrame([{**v, "symbol": k} for k, v in portfolio.items()])
    if not port_df.empty:
        port_df.to_csv(buffer, index=False)
    
    buffer.seek(0)
    return buffer

def export_comprehensive_pdf(stock, company_name, info, hist, predictions, news_sentiment, portfolio, risk_metrics, backtest_results):
    """Export comprehensive analysis to PDF using matplotlib"""
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            # Page 1: Stock Info
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            title_text = f"{stock} - {company_name}\nComprehensive Stock Analysis Report"
            ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=16, fontweight='bold')
            
            info_text = f"""
STOCK INFORMATION:
Current Price: ₹{info.get('currentPrice', 'N/A')}
Market Cap: ₹{info.get('marketCap', 'N/A'):,} 
P/E Ratio: {info.get('trailingPE', 'N/A')}
Dividend Yield: {info.get('dividendYield', 'N/A')}
52-Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}
52-Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}
Day High: ₹{info.get('dayHigh', 'N/A')}
Day Low: ₹{info.get('dayLow', 'N/A')}
Volume: {info.get('volume', 'N/A'):,}
            """
            ax.text(0.1, 0.85, info_text, ha='left', va='top', fontsize=10, family='monospace')
            
            if predictions:
                pred_text = f"""
PRICE PREDICTIONS (ML Model):
Next Day Predicted Price: ₹{predictions.get('next_day', 'N/A'):.2f}
Model Accuracy (R²): {predictions.get('r2_score', 0):.3f}
Mean Absolute Error: ₹{predictions.get('mae', 0):.2f}

SUGGESTED TRADING LEVELS:
Buy Price: ₹{predictions.get('buy_strike', 'N/A'):.2f}
Sell Price: ₹{predictions.get('sell_strike', 'N/A'):.2f}
Stop Loss: ₹{predictions.get('stop_loss', 'N/A'):.2f}
                """
                ax.text(0.1, 0.55, pred_text, ha='left', va='top', fontsize=10, family='monospace', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            if risk_metrics:
                risk_text = f"""
RISK METRICS:
Volatility (Annual): {risk_metrics.get('volatility', 'N/A'):.2%}
Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 'N/A'):.3f}
Beta: {risk_metrics.get('beta', 'N/A')}
                """
                ax.text(0.1, 0.30, risk_text, ha='left', va='top', fontsize=10, family='monospace')
            
            ax.text(0.5, 0.05, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   ha='center', va='bottom', fontsize=8, style='italic')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Price Chart
            fig, ax = plt.subplots(figsize=(11, 8.5))
            hist['Close'].tail(90).plot(ax=ax, label='Close Price', linewidth=2)
            hist['MA20'].tail(90).plot(ax=ax, label='MA20', linestyle='--')
            hist['MA50'].tail(90).plot(ax=ax, label='MA50', linestyle='--')
            ax.set_title(f'{stock} - Price Movement (Last 90 Days)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: News Sentiment
            if news_sentiment:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                ax.text(0.5, 0.95, 'News Sentiment Analysis', ha='center', va='top', 
                       fontsize=14, fontweight='bold')
                
                sentiment_counts = pd.DataFrame(news_sentiment)['sentiment'].value_counts()
                
                y_pos = 0.85
                for idx, (title, sent) in enumerate([(n['title'][:80], n['sentiment']) for n in news_sentiment[:10]]):
                    color = 'green' if sent == 'positive' else 'red' if sent == 'negative' else 'gray'
                    ax.text(0.05, y_pos, f"{idx+1}. {title}... [{sent.upper()}]", 
                           ha='left', va='top', fontsize=8, color=color, wrap=True)
                    y_pos -= 0.08
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None

# --- Portfolio Management ---
def display_portfolio(portfolio):
    st.header("💼 Your Portfolio Summary")
    if not portfolio:
        st.info("Your portfolio is empty. Add stocks below.")
        return
    
    total_value = 0
    total_invested = 0
    portfolio_data = []
    
    for sym, data in portfolio.items():
        qty = data['quantity']
        avg_price = data['avg_price']
        try:
            ticker = yf.Ticker(sym + ".NS")
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            market_value = current_price * qty
            invested = avg_price * qty
            pl = market_value - invested
            pl_pct = (pl / invested) * 100 if invested > 0 else 0
            
            portfolio_data.append({
                'Stock': sym,
                'Quantity': qty,
                'Avg Buy Price (₹)': f"{avg_price:.2f}",
                'Current Price (₹)': f"{current_price:.2f}",
                'Market Value (₹)': f"{market_value:.2f}",
                'P/L (₹)': f"{pl:.2f}",
                'P/L (%)': f"{pl_pct:.2f}%"
            })
            
            total_value += market_value
            total_invested += invested
        except Exception as e:
            st.error(f"Error fetching data for {sym}: {e}")
    
    if portfolio_data:
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Market Value", f"₹{total_value:.2f}")
        col2.metric("Total Invested", f"₹{total_invested:.2f}")
        total_pl = total_value - total_invested
        col3.metric("Total P/L", f"₹{total_pl:.2f}", f"{(total_pl/total_invested)*100:.2f}%" if total_invested > 0 else "0%")

def portfolio_management_ui():
    st.header("📈 Manage Your Portfolio")
    portfolio = load_portfolio()
    
    with st.form("Add Stock to Portfolio"):
        col1, col2, col3 = st.columns(3)
        add_symbol = col1.selectbox("Select Stock to Add", stocks)
        qty = col2.number_input("Quantity", min_value=1, value=1)
        avg_price = col3.number_input("Average Buy Price (₹)", format="%.2f", min_value=0.01)
        submitted = st.form_submit_button("Add / Update Stock")
    
    if submitted:
        if add_symbol in portfolio:
            total_qty = portfolio[add_symbol]['quantity'] + qty
            total_cost = (portfolio[add_symbol]['avg_price']*portfolio[add_symbol]['quantity']) + (avg_price*qty)
            portfolio[add_symbol]['quantity'] = total_qty
            portfolio[add_symbol]['avg_price'] = total_cost / total_qty
        else:
            portfolio[add_symbol] = {"quantity": qty, "avg_price": avg_price}
        save_portfolio(portfolio)
        st.success(f"✅ {add_symbol} added/updated in portfolio!")
        st.rerun()
    
    if portfolio:
        st.subheader("Remove Stock From Portfolio")
        col1, col2 = st.columns([3, 1])
        to_remove = col1.selectbox("Select Stock to Remove", list(portfolio.keys()))
        if col2.button("🗑️ Remove", use_container_width=True):
            portfolio.pop(to_remove, None)
            save_portfolio(portfolio)
            st.success(f"✅ {to_remove} removed from portfolio!")
            st.rerun()
    
    display_portfolio(portfolio)

# --- Main Analysis Section ---
st.sidebar.header("📅 Analysis Date Range")
default_start = (datetime.date.today() - datetime.timedelta(days=365))
start_date = st.sidebar.date_input('Start Date', default_start)
end_date = st.sidebar.date_input('End Date', datetime.date.today())

stock = st.selectbox("🔍 Select a Stock for Analysis", stocks, index=0)

if stock:
    trigger_inputs(stock)
    
    with st.spinner('Fetching data and running analysis...'):
        info, hist = safe_yf_ticker(stock, start=pd.Timestamp(start_date), end=pd.Timestamp(end_date + datetime.timedelta(days=1)))
        
        if info and not hist.empty:
            company_name = get_company_name(stock)
            st.subheader(f"📊 {stock} - {company_name}")
            
            # Get latest price
            refresh = st.button("🔄 Refresh Latest Price")
            try:
                price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
            except:
                price = hist['Close'].iloc[-1]
            
            if refresh:
                price = get_latest_price(stock)
                if price:
                    st.success(f"✅ Latest price: ₹{price:.2f}")
                check_trigger(stock, price)
            else:
                check_trigger(stock, price)
            
            # 52-week progress bar
            wk52_high = info.get('fiftyTwoWeekHigh')
            wk52_low = info.get('fiftyTwoWeekLow')
            if wk52_high and wk52_low and not pd.isna(wk52_high) and not pd.isna(wk52_low):
                pct = 100 * (price - wk52_low) / (wk52_high - wk52_low)
                st.progress(float(pct) / 100)
                st.caption(f"52-Week Range: ₹{wk52_low:.2f} – ₹{wk52_high:.2f} (Current: ₹{price:.2f})")
            
            # Price metrics
            col1, col2, col3, col4 = st.columns(4)
            try:
                prev_close = float(info.get('previousClose', hist['Close'].iloc[-2]))
                price_change = price - prev_close
                price_change_pct = (price_change / prev_close) * 100
                col1.metric("Current Price (₹)", f"{price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
            except:
                col1.metric("Current Price (₹)", f"{price:.2f}")
            
            col2.metric("Day High (₹)", f"{info.get('dayHigh', 'N/A')}")
            col3.metric("Day Low (₹)", f"{info.get('dayLow', 'N/A')}")
            col4.metric("Volume", f"{info.get('volume', 'N/A'):,}")
            
            # Fundamental Metrics
            st.markdown("### 📈 Fundamental Metrics")
            col4, col5, col6, col7 = st.columns(4)
            mcap = info.get('marketCap')
            pe = info.get('trailingPE')
            div_yield = info.get('dividendYield')
            eps = info.get('trailingEps')
            
            col4.metric("Market Cap", f"₹{mcap:,}" if mcap else "N/A")
            col5.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
            col6.metric("Dividend Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")
            col7.metric("EPS", f"₹{eps:.2f}" if eps else "N/A")
            
            # ML Price Prediction
            st.markdown("### 🤖 ML-Based Price Prediction & Strike Price Suggestion")
            with st.spinner('Training ML model...'):
                predicted_price, model_info = train_price_prediction_model(hist)
                
                if predicted_price and model_info:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Predicted Next Day Price", f"₹{predicted_price:.2f}")
                    col2.metric("Model Accuracy (R²)", f"{model_info['r2_score']:.3f}")
                    col3.metric("Mean Absolute Error", f"₹{model_info['mae']:.2f}")
                    
                    # Dynamic Strike Prices
                    buy_strike = predicted_price * 0.98
                    sell_strike = predicted_price * 1.02
                    stop_loss = predicted_price * 0.95
                    
                    st.info(f"""
                    **ML-Based Trading Recommendations:**
                    - 🟢 **Suggested Buy Price**: ₹{buy_strike:.2f}
                    - 🔴 **Suggested Sell Price**: ₹{sell_strike:.2f}
                    - ⛔ **Suggested Stop Loss**: ₹{stop_loss:.2f}
                    """)
                    
                    predictions_dict = {
                        'next_day': predicted_price,
                        'r2_score': model_info['r2_score'],
                        'mae': model_info['mae'],
                        'buy_strike': buy_strike,
                        'sell_strike': sell_strike,
                        'stop_loss': stop_loss
                    }
                else:
                    st.warning("⚠️ Not enough data for ML prediction.")
                    predictions_dict = None
            
            # Future price predictions
            st.markdown("#### 📅 7-Day Price Forecast")
            future_preds = predict_future_prices(hist, days=7)
            if future_preds:
                dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=7)
                forecast_df = pd.DataFrame({'Date': dates, 'Predicted Price (₹)': future_preds})
                st.dataframe(forecast_df, use_container_width=True)
            
            # Technical Indicators
            hist["MA20"] = hist["Close"].rolling(window=20).mean()
            hist["MA50"] = hist["Close"].rolling(window=50).mean()
            hist["MA200"] = hist["Close"].rolling(window=200).mean()
            hist["RSI"] = ta.momentum.rsi(hist["Close"], window=14)
            bb = ta.volatility.BollingerBands(hist['Close'], window=20)
            hist["BB_High"] = bb.bollinger_hband()
            hist["BB_Low"] = bb.bollinger_lband()
            
            # Advanced indicators
            hist, support, resistance = calculate_advanced_indicators(hist)
            
            # Risk Metrics
            st.markdown("### ⚠️ Risk Metrics")
            # Fetch Nifty 50 for beta calculation
            try:
                nifty = yf.Ticker("^NSEI")
                nifty_hist = nifty.history(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date + datetime.timedelta(days=1)))
                risk_metrics = calculate_risk_metrics(hist, nifty_hist)
            except:
                risk_metrics = calculate_risk_metrics(hist)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Volatility (Annual)", f"{risk_metrics['volatility']:.2%}")
            col2.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.3f}")
            col3.metric("Beta (vs Nifty)", f"{risk_metrics['beta']:.2f}" if risk_metrics['beta'] else "N/A")
            
            # Support & Resistance
            st.markdown("### 📊 Support & Resistance Levels")
            col1, col2 = st.columns(2)
            col1.metric("Resistance Level", f"₹{resistance:.2f}")
            col2.metric("Support Level", f"₹{support:.2f}")
            
            # Charts
            st.markdown("### 📉 Technical Analysis Charts")
            
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=["Price with MAs & Bollinger Bands", "Volume", "RSI (14-day)", "MACD"],
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Candlestick + MAs + Bollinger Bands
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'],
                name="OHLC"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], 
                                    line=dict(color='blue', width=1), name='MA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], 
                                    line=dict(color='orange', width=1), name='MA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], 
                                    line=dict(color='green', width=1), name='MA 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_High"], 
                                    line=dict(color='gray', width=1, dash='dash'), name='BB High'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_Low"], 
                                    line=dict(color='gray', width=1, dash='dash'), name='BB Low'), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], 
                                marker_color='lightblue', name='Volume'), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], 
                                    line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], 
                                    line=dict(color='blue', width=1), name='MACD'), row=4, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], 
                                    line=dict(color='orange', width=1), name='Signal'), row=4, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_diff'], 
                                marker_color='gray', name='MACD Diff'), row=4, col=1)
            
            fig.update_layout(height=1200, showlegend=True, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price Gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=price,
                delta={'reference': prev_close},
                gauge={'axis': {'range': [info.get('dayLow', 0) or 0, info.get('dayHigh', 0) or 1]}},
                title={'text': f"{stock} Price Gauge"}
            ))
            st.plotly_chart(gauge, use_container_width=True)
            
            # Stochastic Oscillator Chart
            st.markdown("### 📊 Stochastic Oscillator")
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=hist.index, y=hist['Stoch'], 
                                          mode='lines', name='Stochastic', line=dict(color='blue')))
            fig_stoch.add_trace(go.Scatter(x=hist.index, y=hist['Stoch_signal'], 
                                          mode='lines', name='Signal', line=dict(color='red')))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
            fig_stoch.update_layout(height=400)
            st.plotly_chart(fig_stoch, use_container_width=True)
            
            # Backtesting
            st.markdown("### 🔄 Backtesting Results (MA Crossover Strategy)")
            backtest_results = simple_backtest(hist)
            if backtest_results:
                col1, col2 = st.columns(2)
                col1.metric("Strategy Return", f"{backtest_results['strategy_return']:.2%}")
                col2.metric("Buy & Hold Return", f"{backtest_results['buy_hold_return']:.2%}")
                
                if not backtest_results['signals'].empty:
                    st.write("**Recent Signals:**")
                    st.dataframe(backtest_results['signals'], use_container_width=True)
            
            # Earnings & Dividends
            st.markdown("### 📅 Earnings / Events Calendar")
            earnings_cal, dividend_data = fetch_earnings_and_dividends(stock)
            if earnings_cal is not None and not earnings_cal.empty:
                st.table(earnings_cal)
            else:
                st.info("No upcoming earnings/events found.")
            
            st.markdown("### 💰 Upcoming Dividends")
            if dividend_data is not None and not dividend_data.empty:
                dividend_df = pd.DataFrame(dividend_data).reset_index()
                dividend_df = dividend_df.rename(columns={"Dividends": "Dividend"})
                st.table(dividend_df.tail(5)[['Date', 'Dividend']])
            else:
                st.info("No upcoming dividend data found.")
            
            # News & Sentiment
            st.markdown("### 📰 News & Sentiment Analysis")
            news_articles = fetch_news(stock)
            pos_count = neg_count = neu_count = 0
            news_sentiment_data = []
            
            if news_articles:
                for title, url, pub_date in news_articles:
                    try:
                        sentiment = sentiment_analyzer(title)[0]
                        label = sentiment["label"].lower()
                        score = sentiment["score"]
                        color = "green" if label == "positive" else "red" if label == "negative" else "gray"
                        
                        st.markdown(f"- [{title}]({url}) — <span style='color:{color}'>{label.capitalize()} ({score:.2f})</span> *({pub_date[:10]})*", 
                                   unsafe_allow_html=True)
                        
                        news_sentiment_data.append({
                            'title': title,
                            'sentiment': label,
                            'score': score,
                            'date': pub_date
                        })
                        
                        if label == "positive":
                            pos_count += 1
                        elif label == "negative":
                            neg_count += 1
                        else:
                            neu_count += 1
                    except:
                        pass
                
                # Sentiment Pie Chart
                labels = ["Positive", "Negative", "Neutral"]
                values = [pos_count, neg_count, neu_count]
                pie_fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=.4,
                    marker_colors=["green", "red", "gray"]
                )])
                pie_fig.update_layout(title="News Sentiment Distribution")
                st.plotly_chart(pie_fig, use_container_width=True)
                
                # Overall Recommendation
                if pos_count > neg_count and price >= prev_close:
                    rec = "🟢 BUY"
                elif neg_count > pos_count and price < prev_close:
                    rec = "🔴 SELL"
                else:
                    rec = "🟡 HOLD"
                st.subheader(f"Overall Recommendation: {rec}")
            else:
                st.info("No recent news found.")
                news_sentiment_data = []
            
            # Export Section
            st.markdown("---")
            st.markdown("### 📥 Export Comprehensive Analysis")
            
            col1, col2 = st.columns(2)
            
            portfolio_data = load_portfolio()
            
            if col1.button("📊 Export as CSV", use_container_width=True):
                csv_buffer = export_comprehensive_csv(
                    stock, company_name, info, hist, predictions_dict, 
                    news_sentiment_data, portfolio_data, risk_metrics, backtest_results
                )
                b64 = base64.b64encode(csv_buffer.read()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="NSEInsightPro_{stock}_Analysis.csv">📥 Download CSV Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ CSV report generated successfully!")
            
            if col2.button("📄 Export as PDF", use_container_width=True):
                pdf_buffer = export_comprehensive_pdf(
                    stock, company_name, info, hist, predictions_dict, 
                    news_sentiment_data, portfolio_data, risk_metrics, backtest_results
                )
                if pdf_buffer:
                    b64 = base64.b64encode(pdf_buffer.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="NSEInsightPro_{stock}_Analysis.pdf">📥 Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ PDF report generated successfully!")
        else:
            st.warning("⚠️ No price data found for this stock.")

# Portfolio Management Section
st.markdown("---")
portfolio_management_ui()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>NSEInsightPro - Advanced Stock Analysis & Portfolio Management</p>
    <p>Powered by ML, Technical Analysis & Real-time Data</p>
</div>
""", unsafe_allow_html=True)
