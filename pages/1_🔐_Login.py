import streamlit as st

def login():
    st.title("🔐 Login")
    user_id = st.text_input("Enter your unique login ID:")
    if user_id:
        st.success(f"Welcome, {user_id}!")
        st.session_state["user_id"] = user_id

if __name__ == "__main__":
    login()
