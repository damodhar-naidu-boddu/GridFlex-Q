"""
GridFlex-Q Authentication & Account Management
==============================================
Secure local authentication system utilizing PBKDF2-HMAC-SHA256 salted hashes,
parameterized SQLite storage, and session-state synchronization across Streamlit reruns.

Educational Prototype Notice:
Credentials are stored locally in an embedded SQLite database using cryptographically
salted PBKDF2 hashes. This illustrates credential hygiene but is intended for
local demonstration purposes rather than enterprise production grid identity infrastructure.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from typing import Optional, Tuple
import streamlit as st

import data_store

HASH_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 100_000
SALT_BYTES = 16


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using PBKDF2-HMAC-SHA256 with a random cryptographic salt.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt = secrets.token_hex(SALT_BYTES)
    salt_bytes = bytes.fromhex(salt)
    dk = hashlib.pbkdf2_hmac(
        hash_name=HASH_ALGORITHM,
        password=password.encode("utf-8"),
        salt=salt_bytes,
        iterations=PBKDF2_ITERATIONS
    )
    return dk.hex(), salt


def verify_password(plain_password: str, stored_hash: str, salt: str) -> bool:
    """Verify password in constant time using hmac.compare_digest."""
    candidate_hash, _ = hash_password(plain_password, salt=salt)
    return hmac.compare_digest(candidate_hash, stored_hash)


def validate_username(username: str) -> Tuple[bool, str]:
    """Validate username length and allowed characters."""
    username = username.strip()
    if len(username) < 3 or len(username) > 30:
        return False, "Username must be between 3 and 30 characters long."
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", username):
        return False, "Username may only contain letters, numbers, hyphens, periods, and underscores."
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password minimum complexity."""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, ""


def register_account(username: str, password: str, role: str = "operator") -> Tuple[bool, str]:
    """Validate and persist a new user account."""
    u_valid, u_err = validate_username(username)
    if not u_valid:
        return False, u_err

    p_valid, p_err = validate_password(password)
    if not p_valid:
        return False, p_err

    existing = data_store.get_user_by_username(username)
    if existing is not None:
        return False, f"Username '{username}' is already registered. Please choose another."

    pwd_hash, salt = hash_password(password)
    created = data_store.create_user(username=username, password_hash=pwd_hash, salt=salt, role=role)
    if created:
        return True, "Account registered successfully! You may now log in."
    return False, "Database error: unable to register account."


def authenticate(username: str, password: str) -> Tuple[bool, str, Optional[dict]]:
    """Authenticate username and password against SQLite store."""
    if not username.strip() or not password:
        return False, "Please enter both username and password.", None

    user = data_store.get_user_by_username(username)
    if user is None:
        return False, "Invalid username or password.", None

    if verify_password(password, user["password_hash"], user["salt"]):
        return True, "Authentication successful.", user
    return False, "Invalid username or password.", None


def init_auth_session() -> None:
    """Ensure authentication keys exist in Streamlit session state."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = "Guest"
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = "guest"


def is_authenticated() -> bool:
    """Return whether current session is authenticated."""
    init_auth_session()
    return bool(st.session_state.get("authenticated", False))


def get_current_user() -> str:
    """Return currently active username."""
    init_auth_session()
    return str(st.session_state.get("username", "Guest"))


def logout() -> None:
    """Clear authenticated session variables."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = "Guest"
    st.session_state["user_role"] = "guest"


def seed_default_admin_if_empty() -> None:
    """Seed initial operator account if the local database has 0 users."""
    data_store.init_db()
    if data_store.list_users_count() == 0:
        register_account("operator", "GridFlex2026!", role="grid_operator")


def render_auth_page() -> None:
    """Render the full Authentication & User Profile page."""
    init_auth_session()
    seed_default_admin_if_empty()

    st.markdown("## 🔐 Authentication & Access Management")
    st.caption("Secure local session management with salted PBKDF2-HMAC-SHA256 credential hashing.")

    st.info(
        "🛡️ **Local Demonstration Security Disclaimer:**\n\n"
        "Passwords are cryptographically hashed with a per-user 16-byte random salt and "
        "100,000 PBKDF2 iterations before storage in SQLite. All queries are parameterized. "
        "This local prototype simulates access control and does not replace enterprise identity providers (e.g. OAuth2, SAML, LDAP).",
        icon="ℹ️"
    )

    if is_authenticated():
        st.success(f"✅ Currently authenticated as **{st.session_state['username']}** (Role: `{st.session_state['user_role']}`)")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚪 Log Out", use_container_width=True, type="primary"):
                logout()
                st.rerun()
        with col2:
            st.write("You have active privileges to run simulated quantum optimizations, edit loads, and export telemetry.")

        st.markdown("---")
        st.subheader("Account Details")
        st.json({
            "User": st.session_state["username"],
            "Role": st.session_state["user_role"],
            "Session Active": True,
            "Storage Engine": "SQLite3 (Parameterized)",
            "Hash Scheme": "PBKDF2-HMAC-SHA256 (100k rounds)"
        })
        return

    # Tabs for Login vs Register
    tab_login, tab_register, tab_quick = st.tabs(["🔑 Sign In", "📝 Create Account", "⚡ Quick Demo Access"])

    with tab_login:
        st.subheader("Operator Login")
        with st.form("login_form"):
            login_user = st.text_input("Username", placeholder="e.g. operator", key="auth_login_user")
            login_pass = st.text_input("Password", type="password", placeholder="Enter your password", key="auth_login_pass")
            submit_login = st.form_submit_button("Sign In", use_container_width=True)

            if submit_login:
                ok, msg, udata = authenticate(login_user, login_pass)
                if ok and udata:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = udata["username"]
                    st.session_state["user_role"] = udata["role"]
                    st.success(f"Welcome back, {udata['username']}!")
                    st.rerun()
                else:
                    st.error(msg)

    with tab_register:
        st.subheader("Register New Account")
        with st.form("register_form"):
            reg_user = st.text_input("Desired Username", placeholder="3-30 letters, numbers, or dashes", key="auth_reg_user")
            reg_role = st.selectbox("Role", ["operator", "researcher", "auditor"], key="auth_reg_role")
            reg_pass1 = st.text_input("Password", type="password", placeholder="Minimum 6 characters", key="auth_reg_pass1")
            reg_pass2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="auth_reg_pass2")
            submit_reg = st.form_submit_button("Register Account", use_container_width=True)

            if submit_reg:
                if reg_pass1 != reg_pass2:
                    st.error("Passwords do not match.")
                else:
                    reg_ok, reg_msg = register_account(reg_user, reg_pass1, role=reg_role)
                    if reg_ok:
                        st.success(reg_msg)
                    else:
                        st.error(reg_msg)

    with tab_quick:
        st.subheader("Hackathon Quick Access")
        st.write("For rapid evaluation during judging and demonstration, you can log in with the preloaded demonstration account:")
        st.code("Username: operator\nPassword: GridFlex2026!", language="text")
        if st.button("🚀 One-Click Demo Login (as 'operator')", key="quick_demo_login_btn"):
            ok, msg, udata = authenticate("operator", "GridFlex2026!")
            if ok and udata:
                st.session_state["authenticated"] = True
                st.session_state["username"] = udata["username"]
                st.session_state["user_role"] = udata["role"]
                st.success("Authenticated as operator!")
                st.rerun()
            else:
                st.warning("Could not auto-login. Please register or sign in manually.")
