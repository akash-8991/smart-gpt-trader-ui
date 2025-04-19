import streamlit as st
import pandas as pd
import subprocess
import os

st.title("📈 Trade Analysis")

# Run the trade.py file when button is clicked
if st.button("Run Trade Analysis"):
    try:
        # Run the script
        result = subprocess.run(
            ["python", "trade.py"],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("Trade analysis completed successfully.")

        # Check if output.xlsx was created
        if os.path.exists("output.xlsx"):
            df = pd.read_excel("output.xlsx")
            st.dataframe(df)
        else:
            st.error("❌ output.xlsx not found. Please ensure trade.py generates it.")

    except subprocess.CalledProcessError as e:
        st.error(f"🚨 Error running trade.py:\n{e.stderr}")
