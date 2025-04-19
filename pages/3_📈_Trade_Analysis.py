import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="Trading Analysis", layout="wide")
st.title("📈 Smart GPT Traders - Trade Analysis")

st.markdown("Click the button below to run trading logic and view the output:")

if st.button("Run Trade Analysis"):
    with st.spinner("Running trade logic..."):
        try:
            # Execute trade.py instead of a notebook
            subprocess.run(["python", "trade.py"], check=True)

            if os.path.exists("output.xlsx"):
                df = pd.read_excel("output.xlsx")
                st.success("✅ Trade analysis complete. Here's the result:")
                st.dataframe(df, use_container_width=True)
            else:
                st.error("❌ output.xlsx not found. Please ensure trade.py generates it.")
        except subprocess.CalledProcessError as e:
            st.error(f"Script execution failed with return code {e.returncode}")
        except Exception as e:
            st.error(f"Execution error: {e}")
