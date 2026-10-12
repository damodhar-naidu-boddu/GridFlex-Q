
import streamlit as st

st.set_page_config(
    page_title="GridFlex-Q Login",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ GridFlex-Q")
st.subheader("Smart Emergency Power Scheduling")
st.write("Welcome! Please choose an option below.")

login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

with login_tab:
    st.subheader("Welcome Back")
    username = st.text_input("Username", key="login_username")
    password = st.text_input(
        "Password",
        type="password",
        key="login_password"
    )

    if st.button("Login"):
        st.warning(
            "Authentication is not configured yet. "
            "Please complete the secure login setup."
        )

with signup_tab:
    st.subheader("Create an Account")
    new_username = st.text_input(
        "Choose Username",
        key="new_username"
    )
    new_password = st.text_input(
        "Choose Password",
        type="password",
        key="new_password"
    )
    confirm_password = st.text_input(
        "Confirm Password",
        type="password",
        key="confirm_password"
    )

    if st.button("Sign Up"):
        if not new_username or not new_password:
            st.error("Please fill in all fields.")
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            st.info(
                "The account system is not configured yet."
            )

st.divider()
st.caption("GridFlex-Q | Educational Prototype")
