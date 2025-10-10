# --- existing imports ---
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

# --- Added Feature ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Config ---
NEWSAPI_KEY = "9d01ca71d0114b77ae22e01d1d230f1f"
CACHE_FILE = "symbol_name_cache.json"
PORTFOLIO_FILE = "user_portfolio.json"  # To persist user portfolio data

# Load stocks list from CSV
stocks = pd.read_csv("stocks.csv")["symbol"].tolist()

st.title("NSEInsightPRo - NSE Stock Analysis & Portfolio Manager")

# ... (keep your full existing code unchanged above) ...


# --- EXISTING portfolio_management_ui FUNCTION ---
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

    # --- Added Feature 1: Export Buttons ---
    if portfolio:
        st.subheader("Export Portfolio")
        if st.button("Export as CSV"):
            df = pd.DataFrame([
                {"Stock": sym, "Quantity": data["quantity"], "Average Price (₹)": data["avg_price"]}
                for sym, data in portfolio.items()
            ])
            csv_path = "portfolio_export.csv"
            df.to_csv(csv_path, index=False)
            st.success(f"Portfolio exported as {csv_path}")
            st.download_button("Download CSV", open(csv_path, "rb"), file_name="portfolio.csv")

        if st.button("Export as PDF"):
            pdf_path = "portfolio_export.pdf"
            c = canvas.Canvas(pdf_path, pagesize=A4)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(100, 800, "Portfolio Summary - NSEInsightPro")
            c.setFont("Helvetica", 12)
            y = 770
            for sym, data in portfolio.items():
                text = f"{sym} | Qty: {data['quantity']} | Avg Price: ₹{data['avg_price']:.2f}"
                c.drawString(100, y, text)
                y -= 20
            c.save()
            st.success(f"Portfolio exported as {pdf_path}")
            st.download_button("Download PDF", open(pdf_path, "rb"), file_name="portfolio.pdf")

# --- END OF portfolio_management_ui ---


# --- Added Feature 2: Price Trigger ---
def price_trigger_section():
    st.header("Price Trigger Alert")
    selected_stock = st.selectbox("Select Stock to Set Alert", stocks)
    trigger_price = st.number_input("Enter Trigger Price (₹)", min_value=0.01)
    trigger_type = st.radio("Alert When Price:", ["Goes Above", "Goes Below"])
    set_trigger = st.button("Set Price Trigger")

    if set_trigger:
        info, _ = safe_yf_ticker(selected_stock)
        if info and "currentPrice" in info:
            current_price = info["currentPrice"]
            condition_met = (current_price >= trigger_price if trigger_type == "Goes Above"
                             else current_price <= trigger_price)
            if condition_met:
                st.success(f"✅ Trigger Condition Met! {selected_stock} current price ₹{current_price:.2f}")
            else:
                st.info(f"Trigger not met yet. Current price ₹{current_price:.2f}")
        else:
            st.warning("Could not fetch current price. Try again later.")


# --- MAIN EXECUTION ---
if stock:
    info, hist = safe_yf_ticker(stock)

    if info and not hist.empty:
        # (keep all your existing logic for graphs, indicators, sentiment, etc. exactly the same)
        ...
    else:
        st.warning("No price data found for this stock.")

# --- Added new section at the end ---
price_trigger_section()
portfolio_management_ui()
