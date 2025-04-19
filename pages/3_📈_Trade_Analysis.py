import streamlit as st
import pandas as pd
import papermill as pm
import os

st.set_page_config(page_title="Trading Analysis", layout="wide")
st.title("📈 Smart GPT Traders - Trade Analysis")

st.markdown("Click the button below to run trading logic and view the output:")

if st.button("Run Trade Analysis"):
    with st.spinner("Executing trade notebook..."):
        try:
            # Run the notebook
            pm.execute_notebook(
                input_path='trade.ipynb',
                output_path='trade_output.ipynb'
            )

            # Load the output Excel file
            if os.path.exists("output.xlsx"):
                df = pd.read_excel("output.xlsx")
                st.success("✅ Trade analysis complete. Here's the result:")
                st.dataframe(df, use_container_width=True)
            else:
                st.error("❌ output.xlsx not found. Please ensure the notebook generates the Excel output.")
        except Exception as e:
            st.error(f"Execution error: {e}")