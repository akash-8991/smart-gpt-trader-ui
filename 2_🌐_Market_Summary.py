import streamlit as st
import yfinance as yf
import pandas as pd

def get_market_summary():
    tickers = {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Brent Crude": "BZ=F"
    }

    data = []
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="2d", progress=False)
        if len(df) >= 2:
            open_price = df['Open'][-1]
            close_price = df['Close'][-2]
            pct_change = ((open_price - close_price) / close_price) * 100
            data.append([name, round(open_price, 2), round(close_price, 2), f"{pct_change:.2f}%"])
    return pd.DataFrame(data, columns=["Index", "Open", "Previous Close", "% Change"])

st.title("🌐 Market Summary")

with st.spinner("Fetching market summary..."):
    df = get_market_summary()
st.dataframe(df, use_container_width=True)
