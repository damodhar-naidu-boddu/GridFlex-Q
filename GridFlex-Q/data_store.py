"""
GridFlex-Q Local Data Store
===========================
SQLite-backed persistent data management for authentication records,
synthetic simulation telemetry, and optimization experiment benchmarks.

All queries use parameterized SQL to guarantee safe data execution.
Telemetry exported or stored is explicitly marked as simulated data.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", "gridflex.db")


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Return the database path to use, prioritizing custom overrides."""
    return custom_path or DEFAULT_DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create a thread-safe connection with row factory enabled."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize SQLite schema for user accounts, telemetry, and optimizer runs."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()

        # Users table with salted hash storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'operator',
                created_at TEXT NOT NULL
            )
        """)

        # Simulation telemetry history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mode TEXT NOT NULL,
                supply_capacity_kw REAL NOT NULL,
                total_demand_kw REAL NOT NULL,
                scheduled_power_kw REAL NOT NULL,
                operating_state TEXT NOT NULL,
                voltage_v REAL NOT NULL,
                current_a REAL NOT NULL,
                frequency_hz REAL NOT NULL,
                power_factor REAL NOT NULL,
                details_json TEXT
            )
        """)

        # Optimization runs benchmark history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                method TEXT NOT NULL,
                capacity_kw REAL NOT NULL,
                objective_score REAL NOT NULL,
                total_power_kw REAL NOT NULL,
                remaining_capacity_kw REAL NOT NULL,
                feasible INTEGER NOT NULL,
                execution_time_ms REAL NOT NULL,
                selected_loads_json TEXT NOT NULL,
                metadata_json TEXT
            )
        """)

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Authentication Store Helpers
# ---------------------------------------------------------------------------

def create_user(
    username: str,
    password_hash: str,
    salt: str,
    role: str = "operator",
    db_path: Optional[str] = None
) -> bool:
    """Create a new user account. Returns True if successful, False if username exists."""
    init_db(db_path)
    now_iso = datetime.utcnow().isoformat()
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, salt, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username.strip(), password_hash, salt, role, now_iso)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_username(username: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user credentials and role by username."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash, salt, role, created_at FROM users WHERE username = ?",
            (username.strip(),)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def list_users_count(db_path: Optional[str] = None) -> int:
    """Return count of registered user accounts."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM users")
        row = cursor.fetchone()
        return int(row["total"]) if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Simulation Telemetry Store Helpers
# ---------------------------------------------------------------------------

def record_simulation_telemetry(record: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Record a single simulated electrical telemetry data point."""
    init_db(db_path)
    ts = record.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    details = json.dumps(record.get("details", {}))
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO simulation_history (
                timestamp, mode, supply_capacity_kw, total_demand_kw, scheduled_power_kw,
                operating_state, voltage_v, current_a, frequency_hz, power_factor, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                str(record.get("mode", "Normal Operation")),
                float(record.get("supply_capacity_kw", 0.0)),
                float(record.get("total_demand_kw", 0.0)),
                float(record.get("scheduled_power_kw", 0.0)),
                str(record.get("operating_state", "Normal")),
                float(record.get("voltage_v", 230.0)),
                float(record.get("current_a", 0.0)),
                float(record.get("frequency_hz", 50.0)),
                float(record.get("power_factor", 0.95)),
                details
            )
        )
        conn.commit()
        return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def get_simulation_history_df(limit: int = 100, db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve recent simulation telemetry as a Pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = """
            SELECT id, timestamp, mode, supply_capacity_kw, total_demand_kw,
                   scheduled_power_kw, operating_state, voltage_v, current_a,
                   frequency_hz, power_factor
            FROM simulation_history
            ORDER BY id DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        if not df.empty:
            df = df.sort_values(by="id", ascending=True).reset_index(drop=True)
        return df
    finally:
        conn.close()


def clear_simulation_history(db_path: Optional[str] = None) -> None:
    """Clear all simulation telemetry points."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM simulation_history")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Optimization Records Store Helpers
# ---------------------------------------------------------------------------

def record_optimization_run(result_dict: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Save an optimization solution benchmark into the database."""
    init_db(db_path)
    ts = result_dict.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    selected_json = json.dumps(result_dict.get("selected_loads", []))
    meta_json = json.dumps(result_dict.get("metadata", {}))

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO optimization_records (
                timestamp, method, capacity_kw, objective_score, total_power_kw,
                remaining_capacity_kw, feasible, execution_time_ms,
                selected_loads_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                str(result_dict.get("method", "Unknown")),
                float(result_dict.get("capacity_kw", 0.0)),
                float(result_dict.get("objective_score", 0.0)),
                float(result_dict.get("total_power_kw", 0.0)),
                float(result_dict.get("remaining_capacity_kw", 0.0)),
                1 if result_dict.get("feasible", False) else 0,
                float(result_dict.get("execution_time_ms", 0.0)),
                selected_json,
                meta_json
            )
        )
        conn.commit()
        return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def get_optimization_history_df(limit: int = 50, db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve optimization history as a Pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = """
            SELECT id, timestamp, method, capacity_kw, objective_score,
                   total_power_kw, remaining_capacity_kw, feasible, execution_time_ms,
                   selected_loads_json, metadata_json
            FROM optimization_records
            ORDER BY id DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        if not df.empty:
            df = df.sort_values(by="id", ascending=True).reset_index(drop=True)
        return df
    finally:
        conn.close()


def clear_optimization_history(db_path: Optional[str] = None) -> None:
    """Clear all saved optimization benchmark runs."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM optimization_records")
        conn.commit()
    finally:
        conn.close()
