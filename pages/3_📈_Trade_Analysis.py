import streamlit as st
import subprocess
import os
import pandas as pd

st.set_page_config(page_title="Smart GPT Traders - Trade Analysis", layout="wide")
st.title("📊 Trading Analysis")

st.markdown("Click the button below to run the trade logic and get today’s trading opportunities.")

if st.button("Run Today's Trade Analysis"):
    with st.spinner("Running trading logic from trade.ipynb..."):
        try:
            # Execute the notebook and output to new notebook (output will still be saved as output.xlsx)
            subprocess.run([
                "jupyter", "nbconvert", "--to", "notebook", 
                "--execute", "trade.ipynb", 
                "--output", "trade_output.ipynb"
            ], check=True)

            # Check for Excel file
            if os.path.exists("output.xlsx"):
                df = pd.read_excel("output.xlsx")
                st.success("✅ Trade analysis completed successfully!")
                st.dataframe(df, use_container_width=True)
            else:
                st.error("❌ output.xlsx not found. Please ensure trade.ipynb creates the file.")

        except subprocess.CalledProcessError as e:
            st.error("⚠️ Error while executing trade.ipynb")
            st.exception(e)

# Optional: debug
# st.write("Files in current directory:", os.listdir("."))