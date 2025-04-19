# Smart GPT Traders - Multi Page Streamlit App

## Overview
This app has 3 pages:
1. **Login Page** - Enter a unique login ID
2. **Market Summary Page** - View Indian, global indices and commodities
3. **Trading Analysis Page** - Run external trading logic from `trade.ipynb` and display the output Excel as a table.

## How to Run
```bash
streamlit run app.py
```

## Requirements
- Streamlit
- yfinance
- pandas
- openpyxl
- Jupyter

Make sure `trade.ipynb` exists and generates `trade_output.xlsx`.
