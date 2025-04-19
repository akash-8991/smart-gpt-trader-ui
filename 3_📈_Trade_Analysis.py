import streamlit as st
import subprocess
import pandas as pd
import os

st.title("📉 Trading Analysis")

if st.button("Run Today's Trade Analysis"):
    with st.spinner("Running trade analysis..."):
        # Run the external trade.py file
        try:
            subprocess.run(["python", "trade.py"], check=True)
            st.success("Trade analysis completed successfully.")
            
            # Check if Excel file exists and load it
            if os.path.exists("trade_output.xlsx"):
                df = pd.read_excel("trade_output.xlsx")
                st.dataframe(df)
            else:
                st.error("Output Excel not found.")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to run trade analysis: {e}")

