{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ff17b18f-9104-4ea5-a756-996589c11d49",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SmartGPT Trading Tool Script\n",
    "\n",
    "import time\n",
    "import pandas as pd\n",
    "import yfinance as yf\n",
    "from textblob import TextBlob\n",
    "from bs4 import BeautifulSoup\n",
    "from selenium import webdriver\n",
    "from selenium.webdriver.chrome.options import Options\n",
    "from selenium.webdriver.common.by import By\n",
    "import streamlit as st\n",
    "import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ff886317-f7e7-4b3c-91c3-985b84081089",
   "metadata": {},
   "outputs": [],
   "source": [
    "# User-defined symbols (e.g. top NIFTY 50 stocks + BANKNIFTY)\n",
    "symbols = [\"RELIANCE\", \"TCS\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6c21aba4-d512-4500-b2bd-cae9d7f0a8f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup Selenium for logged-in Moneycontrol scraping\n",
    "chrome_options = Options()\n",
    "chrome_options.add_argument(\"--user-data-dir=/Users/akash/Library/Application Support/Google/Chrome/Default\")  # Change path\n",
    "chrome_options.add_argument(\"--profile-directory=Default\")\n",
    "driver = webdriver.Chrome(options=chrome_options)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "386ef5c1-f9a0-478f-84dc-fc64fcf9b732",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_market_summary():\n",
    "    tickers = {\n",
    "        \"Nifty 50\": \"^NSEI\",\n",
    "        \"Sensex\": \"^BSESN\",\n",
    "        \"Gold\": \"GC=F\",\n",
    "        \"Silver\": \"SI=F\",\n",
    "        \"Brent Crude\": \"BZ=F\",\n",
    "        \"S&P 500\": \"^GSPC\",\n",
    "        \"Dow Jones\": \"^DJI\",\n",
    "        \"Nasdaq\": \"^IXIC\"\n",
    "    }\n",
    "\n",
    "    summary = []\n",
    "    for name, ticker in tickers.items():\n",
    "        try:\n",
    "            df = yf.download(ticker, period=\"5d\")\n",
    "            df.dropna(inplace=True)\n",
    "\n",
    "            if len(df) >= 2:\n",
    "                # Get last row's open and second last row's close\n",
    "                open_price = float(df['Open'].iloc[-1])\n",
    "                close_price = float(df['Close'].iloc[-2])\n",
    "                pct_change = ((open_price - close_price) / close_price) * 100\n",
    "                summary.append([name, open_price, close_price, round(pct_change, 2)])\n",
    "            else:\n",
    "                summary.append([name, None, None, None])\n",
    "        except Exception as e:\n",
    "            summary.append([name, None, None, f\"Error: {str(e)}\"])\n",
    "\n",
    "    df_summary = pd.DataFrame(summary, columns=[\"Index\", \"Open\", \"Previous Close\", \"% Change\"])\n",
    "    \n",
    "    # Ensure all columns are streamlit-friendly\n",
    "    df_summary = df_summary.astype({\n",
    "        \"Index\": str,\n",
    "        \"Open\": float,\n",
    "        \"Previous Close\": float,\n",
    "        \"% Change\": float\n",
    "    }, errors='ignore')\n",
    "\n",
    "    return df_summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6efed1ac-39c3-41cd-8425-c7a5489086ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "def fetch_technical_indicators(symbol):\n",
    "    df = yf.download(f\"{symbol}.NS\", period=\"6mo\", interval=\"1d\", auto_adjust=True)\n",
    "    if df.empty or len(df) < 50:\n",
    "        raise ValueError(f\"Not enough data to calculate indicators for {symbol}\")\n",
    "\n",
    "    close = df['Close']\n",
    "\n",
    "    rsi_series = compute_rsi(close)\n",
    "    rsi = rsi_series.iloc[-1] if not rsi_series.empty else None\n",
    "    macd_series, signal_series = compute_macd(close)\n",
    "    macd_diff = macd_series.iloc[-1] - signal_series.iloc[-1]\n",
    "    dma_50 = close.rolling(window=50).mean().iloc[-1]\n",
    "    dma_200 = close.rolling(window=200).mean().iloc[-1]\n",
    "    cmp = float(close.iloc[-1])\n",
    "    \n",
    "    signals = {}\n",
    "\n",
    "    signals['cmp'] = cmp\n",
    "    \n",
    "    # RSI Signal\n",
    "    if float(rsi) > 70:\n",
    "        signals['RSI'] = 'SELL'\n",
    "    elif float(rsi) < 30:\n",
    "        signals['RSI'] = 'BUY'\n",
    "    else:\n",
    "        signals['RSI'] = 'HOLD'\n",
    "\n",
    "    # MACD Signal\n",
    "    signals['MACD'] = 'BUY' if float(macd_diff) > 0 else 'SELL' if float(macd_diff) < 0 else 'HOLD'\n",
    "\n",
    "    # DMA Signal\n",
    "    if float(df['Close'].iloc[-1]) > float(dma_50.iloc[-1]) > float(dma_200.iloc[-1]):\n",
    "        signals['DMA'] = 'BUY'\n",
    "    elif float(df['Close'].iloc[-1]) < float(dma_50.iloc[-1]) < float(dma_200.iloc[-1]):\n",
    "        signals['DMA'] = 'SELL'\n",
    "    else:\n",
    "        signals['DMA'] = 'HOLD'\n",
    "\n",
    "    return signals"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1f73449b-d8cd-4ce2-bd6e-20ed9844567a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_rsi(series, period=14):\n",
    "    delta = series.diff().dropna()\n",
    "    gain = delta.where(delta > 0, 0)\n",
    "    loss = -delta.where(delta < 0, 0)\n",
    "\n",
    "    avg_gain = gain.rolling(window=period).mean()\n",
    "    avg_loss = loss.rolling(window=period).mean()\n",
    "\n",
    "    rs = avg_gain / avg_loss\n",
    "    rsi = 100 - (100 / (1 + rs))\n",
    "    return rsi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "059ae3b5-9293-44da-bdc6-7c6e4544a71c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_macd(series, fast=12, slow=26, signal=9):\n",
    "    ema_fast = series.ewm(span=fast, adjust=False).mean()\n",
    "    ema_slow = series.ewm(span=slow, adjust=False).mean()\n",
    "    macd = ema_fast - ema_slow\n",
    "    signal_line = macd.ewm(span=signal, adjust=False).mean()\n",
    "    return macd, signal_line"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1ae32c53-a1ed-43a6-90ea-cd43ebddbb63",
   "metadata": {},
   "outputs": [],
   "source": [
    "def fetch_analyst_rating(symbol):\n",
    "    try:\n",
    "        url = f\"https://www.moneycontrol.com/india/stockpricequote/{symbol.lower()}\"\n",
    "        driver.get(url)\n",
    "        time.sleep(3)\n",
    "        page = driver.page_source\n",
    "        soup = BeautifulSoup(page, 'html.parser')\n",
    "        analyst_rating = soup.find(\"span\", class_=\"mctext\")\n",
    "        return analyst_rating.text.strip() if analyst_rating else \"N/A\"\n",
    "    except Exception as e:\n",
    "        return \"N/A\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "bb64f2a9-dca5-464d-be28-5610d6965172",
   "metadata": {},
   "outputs": [],
   "source": [
    "def fetch_news_sentiment(symbol):\n",
    "    try:\n",
    "        query = f\"https://www.moneycontrol.com/news/tags/{symbol.lower()}.html\"\n",
    "        driver.get(query)\n",
    "        time.sleep(3)\n",
    "        page = driver.page_source\n",
    "        soup = BeautifulSoup(page, 'html.parser')\n",
    "        headlines = soup.select(\".clearfix h2\")[:5]\n",
    "        sentiments = []\n",
    "        for h in headlines:\n",
    "            text = h.text.strip()\n",
    "            polarity = TextBlob(text).sentiment.polarity\n",
    "            sentiments.append(polarity)\n",
    "        if not sentiments:\n",
    "            return 0, \"Neutral\"\n",
    "        avg_score = sum(sentiments) / len(sentiments)\n",
    "        sentiment = \"Positive\" if avg_score > 0.1 else \"Negative\" if avg_score < -0.1 else \"Neutral\"\n",
    "        return avg_score, sentiment\n",
    "    except:\n",
    "        return 0, \"Neutral\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7e1b1f32-14b0-4c53-97e1-5297de8de66d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_recommendation(rating, sentiment, signals):\n",
    "    if \"Buy\" in rating and sentiment == \"Positive\" and signals['RSI'] == \"BUY\" and signals['MACD'] == \"BUY\" and signals['DMA'] == \"BUY\":\n",
    "        return \"BUY\"\n",
    "    elif \"Sell\" in rating and sentiment == \"Negative\" and signals['RSI'] == \"SELL\" and signals['MACD'] == \"SELL\" and signals['DMA'] == \"SELL\":\n",
    "        return \"SELL\"\n",
    "    else:\n",
    "        return \"HOLD\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "07d2db5b-ae95-441f-bf00-2a62ec0263ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing RELIANCE...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%***********************]  1 of 1 completed\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing TCS...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%***********************]  1 of 1 completed\n"
     ]
    }
   ],
   "source": [
    "# Output Data\n",
    "stock_data = []\n",
    "\n",
    "for symbol in symbols:\n",
    "    print(f\"Processing {symbol}...\")\n",
    "    #cmp, rsi, macd, dma = fetch_technical_indicators(symbol)\n",
    "    signals = fetch_technical_indicators(symbol)\n",
    "    rating = fetch_analyst_rating(symbol)\n",
    "    news_score, news_sentiment = fetch_news_sentiment(symbol)\n",
    "    reco = generate_recommendation(rating, news_sentiment, signals)\n",
    "\n",
    "    target = signals['cmp'] * 1.05 if reco == \"BUY\" else signals['cmp'] * 0.95 if reco == \"SELL\" else signals['cmp']\n",
    "    sl = signals['cmp'] * 0.97 if reco == \"BUY\" else signals['cmp'] * 1.03 if reco == \"SELL\" else signals['cmp']\n",
    "\n",
    "    stock_data.append([\n",
    "        symbol, signals['cmp'], signals['RSI'], signals['MACD'], signals['DMA'],\n",
    "        rating, target, news_sentiment,\n",
    "        reco, signals['cmp'], target, sl\n",
    "    ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "10a32448-d1c4-4095-8708-b5d6db5f33b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Analysis saved to SmartGPT_Stock_Analysis_Output.xlsx\n"
     ]
    }
   ],
   "source": [
    "# Save to Excel\n",
    "output = pd.DataFrame(stock_data, columns=[\n",
    "    'Symbol', 'CMP', 'RSI', 'MACD Signal', 'DMA',\n",
    "    'Analyst Rating', 'Target Price', 'News Sentiment',\n",
    "    'Recommendation', 'Entry Price', 'Target', 'Stop Loss'\n",
    "])\n",
    "output.to_excel(\"SmartGPT_Stock_Analysis_Output.xlsx\", index=False)\n",
    "\n",
    "print(\"✅ Analysis saved to SmartGPT_Stock_Analysis_Output.xlsx\")\n",
    "driver.quit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "57cd42da-1062-4049-bc08-3468ad78c9fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-18 16:44:29.146 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Users/akash/anaconda3/lib/python3.11/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "[*********************100%***********************]  1 of 1 completed\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "YF.download() has changed argument auto_adjust default to True\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%***********************]  1 of 1 completed\n",
      "[*********************100%***********************]  1 of 1 completed\n",
      "[*********************100%***********************]  1 of 1 completed\n",
      "[*********************100%***********************]  1 of 1 completed\n",
      "[*********************100%***********************]  1 of 1 completed\n",
      "[*********************100%***********************]  1 of 1 completed\n",
      "[*********************100%***********************]  1 of 1 completed\n"
     ]
    }
   ],
   "source": [
    "# Streamlit App UI\n",
    "st.set_page_config(page_title=\"Smart GPT Traders Dashboard\", layout=\"wide\")\n",
    "st.title(\"📈 Smart GPT Traders Dashboard\")\n",
    "\n",
    "username = st.text_input(\"Enter your unique username to proceed:\")\n",
    "\n",
    "if username:\n",
    "    st.success(f\"Welcome, {username}! Here's your personalized dashboard:\")\n",
    "    \n",
    "    with st.spinner(\"Fetching market summary...\"):\n",
    "        market_df = get_market_summary()\n",
    "    st.subheader(\"📊 Market Summary\")\n",
    "    st.dataframe(market_df)\n",
    "\n",
    "    st.markdown(\"---\")\n",
    "\n",
    "    st.subheader(\"🔍 Today's Trading Recommendations\")\n",
    "\n",
    "    if st.button(\"🛒 Run Today's Trade\"):\n",
    "        with st.spinner(\"Analyzing stocks...\"):\n",
    "            stock_data = []\n",
    "        \n",
    "            for symbol in symbols:\n",
    "                print(f\"Processing {symbol}...\")\n",
    "                #cmp, rsi, macd, dma = fetch_technical_indicators(symbol)\n",
    "                signals = fetch_technical_indicators(symbol)\n",
    "                rating = fetch_analyst_rating(symbol)\n",
    "                news_score, news_sentiment = fetch_news_sentiment(symbol)\n",
    "                reco = generate_recommendation(rating, news_sentiment, signals)\n",
    "\n",
    "                target = signals['cmp'] * 1.05 if reco == \"BUY\" else signals['cmp'] * 0.95 if reco == \"SELL\" else signals['cmp']\n",
    "                sl = signals['cmp'] * 0.97 if reco == \"BUY\" else signals['cmp'] * 1.03 if reco == \"SELL\" else signals['cmp']\n",
    "\n",
    "                stock_data.append([\n",
    "                    symbol, signals['cmp'], signals['RSI'], signals['MACD'], signals['DMA'],\n",
    "                    rating, target, news_sentiment,\n",
    "                    reco, signals['cmp'], target, sl\n",
    "                ])\n",
    "\n",
    "            st.dataframe(pd.DataFrame(stock_data))\n",
    "\n",
    "else:\n",
    "    st.warning(\"Please enter your username to access the dashboard.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
