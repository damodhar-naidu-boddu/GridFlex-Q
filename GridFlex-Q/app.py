"""
GridFlex-Q: Quantum-Assisted Emergency Power Scheduling & Intelligent Energy Monitoring
========================================================================================
Interactive Streamlit Web Application

Visual Styling: Dark navy & charcoal, electric blue/cyan accents, restrained status alerts.
Engine: Real-time synthetic telemetry generator with physics-grounded electrical modeling.
Optimizers: Classical exact powerset baseline and executable Qiskit Aer QAOA quantum circuit.
"""

from __future__ import annotations

import copy
import io
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import auth
import classical_optimizer
import data_store
import quantum_optimizer
import simulation

load_dotenv()

# ---------------------------------------------------------------------------
# Streamlit Application Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GridFlex-Q | Emergency Power Scheduling",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS Design System (Dark Navy, Charcoal, Electric Cyan)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Base Dark Theme Colors */
:root {
    --bg-primary: #0A0F1D;
    --bg-card: #111827;
    --bg-card-hover: #1F2937;
    --border-color: #1E293B;
    --cyan-accent: #00F0FF;
    --blue-accent: #38BDF8;
    --green-status: #10B981;
    --amber-status: #F59E0B;
    --red-status: #EF4444;
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
}

/* App Background & Typography */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Persistent Header Top Banner */
.top-status-bar {
    background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 18px;
    margin-bottom: 20px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
}

.sim-badge {
    background: #0284C7;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 9999px;
    letter-spacing: 0.05em;
    display: inline-block;
}

.sim-badge-alert {
    background: #EF4444;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 9999px;
    animation: pulse 2s infinite;
}

/* Metric Card Containers */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.metric-card:hover {
    border-color: #38BDF8;
}

.metric-title {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.metric-value {
    color: var(--text-primary);
    font-size: 1.85rem;
    font-weight: 700;
    font-family: 'SF Mono', Monaco, Inconsolata, monospace;
}

.metric-sub {
    color: var(--cyan-accent);
    font-size: 0.82rem;
    margin-top: 4px;
}

/* Callout Alert Boxes */
.alert-box {
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    border-left: 4px solid;
}
.alert-normal {
    background: rgba(16, 185, 129, 0.12);
    border-color: var(--green-status);
    color: #A7F3D0;
}
.alert-warning {
    background: rgba(245, 158, 11, 0.12);
    border-color: var(--amber-status);
    color: #FDE68A;
}
.alert-emergency {
    background: rgba(239, 68, 68, 0.15);
    border-color: var(--red-status);
    color: #FECACA;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B;
}

/* Table and dataframe borders */
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B;
    border-radius: 8px;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_app_state() -> None:
    """Initialize persistent Streamlit session variables."""
    auth.init_auth_session()
    auth.seed_default_admin_if_empty()
    data_store.init_db()

    if "loads" not in st.session_state:
        st.session_state["loads"] = simulation.get_default_loads()

    if "supply_capacity_kw" not in st.session_state:
        st.session_state["supply_capacity_kw"] = simulation.DEFAULT_SUPPLY_CAPACITY_KW

    if "simulation_mode" not in st.session_state:
        st.session_state["simulation_mode"] = "Critical Shortage"

    if "sim_engine" not in st.session_state:
        st.session_state["sim_engine"] = simulation.SimulationEngine(
            loads=st.session_state["loads"],
            supply_capacity_kw=st.session_state["supply_capacity_kw"],
            mode=st.session_state["simulation_mode"],
            seed=42
        )

    if "telemetry_history" not in st.session_state:
        # Pre-seed history with 5 initial historical readings
        eng: simulation.SimulationEngine = st.session_state["sim_engine"]
        init_history = []
        for _ in range(5):
            rec = eng.generate_step()
            init_history.append(rec)
            data_store.record_simulation_telemetry(rec)
        st.session_state["telemetry_history"] = init_history

    if "last_classical_res" not in st.session_state:
        # Precompute initial classical solution
        res = classical_optimizer.solve_classical(
            st.session_state["loads"],
            st.session_state["supply_capacity_kw"],
            only_enabled=True
        )
        st.session_state["last_classical_res"] = res

    if "last_quantum_res" not in st.session_state:
        st.session_state["last_quantum_res"] = None

    if "is_simulation_running" not in st.session_state:
        st.session_state["is_simulation_running"] = True


init_app_state()


# ---------------------------------------------------------------------------
# Helper Telemetry Functions
# ---------------------------------------------------------------------------

def get_current_reading() -> Dict[str, Any]:
    """Retrieve most recent simulated telemetry reading."""
    hist: List[Dict[str, Any]] = st.session_state["telemetry_history"]
    if hist:
        return hist[-1]
    # Fallback
    eng: simulation.SimulationEngine = st.session_state["sim_engine"]
    rec = eng.generate_step()
    hist.append(rec)
    return rec


def trigger_simulation_step() -> Dict[str, Any]:
    """Advance simulation by 1 step and update session history and database."""
    eng: simulation.SimulationEngine = st.session_state["sim_engine"]
    eng.loads = copy.deepcopy(st.session_state["loads"])
    eng.supply_capacity_kw = float(st.session_state["supply_capacity_kw"])
    eng.mode = st.session_state["simulation_mode"]

    new_reading = eng.generate_step()

    # Append to history, maintaining configurable window (default 100)
    max_history = int(os.getenv("HISTORY_WINDOW_SIZE", "100"))
    st.session_state["telemetry_history"].append(new_reading)
    if len(st.session_state["telemetry_history"]) > max_history:
        st.session_state["telemetry_history"] = st.session_state["telemetry_history"][-max_history:]

    # Persist in SQLite
    data_store.record_simulation_telemetry(new_reading)

    # Update classical schedule
    c_res = classical_optimizer.solve_classical(
        st.session_state["loads"],
        new_reading["supply_capacity_kw"],
        only_enabled=True
    )
    st.session_state["last_classical_res"] = c_res

    return new_reading


# ---------------------------------------------------------------------------
# Shared UI Components
# ---------------------------------------------------------------------------

def render_top_banner() -> None:
    """Render persistent simulation badge and operating state banner."""
    curr = get_current_reading()
    state = curr.get("operating_state", "Normal")
    ts = curr.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    supply = curr.get("supply_capacity_kw", 7.0)
    demand = curr.get("total_demand_kw", 17.0)
    mode = curr.get("mode", "Critical Shortage")

    if state == "Emergency":
        state_pill = f'<span class="sim-badge-alert">🚨 EMERGENCY ({demand:.1f} kW > {supply:.1f} kW)</span>'
    elif state == "Warning":
        state_pill = f'<span class="sim-badge" style="background:#F59E0B;">⚠️ WARNING ({demand:.1f} kW > {supply:.1f} kW)</span>'
    else:
        state_pill = f'<span class="sim-badge" style="background:#10B981;">✅ NORMAL ({demand:.1f} kW &le; {supply:.1f} kW)</span>'

    banner_html = f"""
    <div class="top-status-bar">
        <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-weight:800; font-size:1.1rem; color:#00F0FF; letter-spacing:0.04em;">⚡ GridFlex-Q</span>
            {state_pill}
            <span style="font-size:0.85rem; color:#94A3B8;">Mode: <strong style="color:#F8FAFC;">{mode}</strong></span>
            <span style="font-size:0.85rem; color:#94A3B8;">Sim Time: <code style="color:#38BDF8;">{ts}</code></span>
        </div>
        <div>
            <span class="sim-badge">{simulation.DATA_SOURCE_BADGE}</span>
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, sub: str = "", border_color: str = "#1E293B") -> None:
    """Render a styled dark-theme metric card."""
    card_html = f"""
    <div class="metric-card" style="border-left: 4px solid {border_color};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation Sidebar
# ---------------------------------------------------------------------------

PAGES = [
    "1. Overview Dashboard",
    "2. Live Simulation",
    "3. Energy Monitoring",
    "4. Emergency Load Scheduling",
    "5. Quantum Lab",
    "6. Classical vs Quantum Comparison",
    "7. Analytics and Reports",
    "8. Authentication",
    "9. Project Information & Limitations"
]

with st.sidebar:
    st.markdown("### ⚡ GridFlex-Q Navigation")
    current_page = st.radio("Select View:", PAGES, index=0)

    st.markdown("---")
    st.markdown("#### 🎮 Simulation Controls")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶️ Step (+1)", use_container_width=True, help="Advance simulation telemetry by 1 step"):
            trigger_simulation_step()
            st.rerun()
    with col_s2:
        if st.button("🎲 Randomize", use_container_width=True, help="Randomize generation scenario"):
            eng: simulation.SimulationEngine = st.session_state["sim_engine"]
            simulation.randomize_scenario(eng)
            st.session_state["simulation_mode"] = eng.mode
            st.session_state["supply_capacity_kw"] = eng.supply_capacity_kw
            trigger_simulation_step()
            st.rerun()

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        if st.button("🔄 Reset Sim", use_container_width=True, help="Reset step counter"):
            st.session_state["sim_engine"].reset()
            trigger_simulation_step()
            st.rerun()
    with col_s4:
        if st.button("↩️ Reset Data", use_container_width=True, help="Restore default sample loads and 7.0 kW"):
            st.session_state["loads"] = simulation.get_default_loads()
            st.session_state["supply_capacity_kw"] = simulation.DEFAULT_SUPPLY_CAPACITY_KW
            st.session_state["simulation_mode"] = "Critical Shortage"
            st.session_state["sim_engine"] = simulation.SimulationEngine(
                loads=st.session_state["loads"],
                supply_capacity_kw=7.0,
                mode="Critical Shortage",
                seed=42
            )
            trigger_simulation_step()
            st.rerun()

    st.markdown("---")
    # Capacity Adjuster: Slider and Exact Manual Number Input
    st.markdown("##### ⚡ Emergency Supply Capacity (kW)")
    col_cap1, col_cap2 = st.columns([1, 1])
    with col_cap1:
        manual_cap_num = st.number_input(
            "Enter kW",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state["supply_capacity_kw"]),
            step=0.5,
            key="sidebar_manual_cap_input",
            help="Type exact power capacity in kW"
        )
    with col_cap2:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        if st.button("Apply kW", use_container_width=True, help="Update supply capacity"):
            st.session_state["supply_capacity_kw"] = manual_cap_num
            st.session_state["sim_engine"].supply_capacity_kw = manual_cap_num
            trigger_simulation_step()
            st.rerun()

    new_cap = st.slider(
        "Slide to Adjust (kW)",
        min_value=0.0,
        max_value=25.0,
        value=float(st.session_state["supply_capacity_kw"]),
        step=0.5,
        key="sidebar_cap_slider",
        help="Drag to adjust capacity"
    )
    if abs(new_cap - st.session_state["supply_capacity_kw"]) > 1e-4:
        st.session_state["supply_capacity_kw"] = new_cap
        st.session_state["sim_engine"].supply_capacity_kw = new_cap
        trigger_simulation_step()
        st.rerun()

    # Simulation Mode Selector
    new_mode = st.selectbox(
        "Simulation Mode",
        simulation.SIMULATION_MODES,
        index=simulation.SIMULATION_MODES.index(st.session_state["simulation_mode"]),
        help="Choose supply deficit and variability profile"
    )
    if new_mode != st.session_state["simulation_mode"]:
        st.session_state["simulation_mode"] = new_mode
        st.session_state["sim_engine"].set_mode(new_mode)
        trigger_simulation_step()
        st.rerun()

    st.markdown("---")
    # Current User Profile Pill
    curr_user = auth.get_current_user()
    user_role = st.session_state.get("user_role", "guest")
    is_auth = auth.is_authenticated()
    auth_badge = "🟢 Operator" if is_auth else "⚪ Guest"
    st.markdown(f"**Session:** `{curr_user}` ({auth_badge})")
    st.caption("GridFlex-Q Educational Prototype v1.0")


# ---------------------------------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# ---------------------------------------------------------------------------

if current_page == "1. Overview Dashboard":
    render_top_banner()

    st.markdown("## 📊 Overview Dashboard")
    st.caption("High-level executive overview of campus microgrid emergency power status, loads, and scheduling.")

    curr = get_current_reading()
    supply = curr.get("supply_capacity_kw", 7.0)
    demand = curr.get("total_demand_kw", 17.0)
    scheduled = curr.get("scheduled_power_kw", 7.0)
    reserve = max(0.0, round(supply - scheduled, 2))
    state = curr.get("operating_state", "Normal")

    # Status Alert Callout
    if state == "Emergency":
        st.markdown(
            f"""
            <div class="alert-box alert-emergency">
                <strong>🚨 CRITICAL SHORTAGE DETECTED:</strong> Total connected load demand ({demand:.1f} kW) exceeds
                available supply capacity ({supply:.1f} kW) by {(demand - supply):.1f} kW.
                Emergency optimization has scheduled <strong>{scheduled:.1f} kW</strong> to avoid cascading system trip.
            </div>
            """,
            unsafe_allow_html=True
        )
    elif state == "Warning":
        st.markdown(
            f"""
            <div class="alert-box alert-warning">
                <strong>⚠️ MODERATE POWER SHORTAGE:</strong> Connected demand ({demand:.1f} kW) exceeds capacity ({supply:.1f} kW).
                Optimization active.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="alert-box alert-normal">
                <strong>✅ NORMAL OPERATION:</strong> Supply capacity ({supply:.1f} kW) adequately powers all active loads ({demand:.1f} kW).
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4 Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Available Supply", f"{supply:.1f} kW", f"Mode: {curr.get('mode', 'N/A')}", "#00F0FF")
    with m2:
        render_metric_card("Connected Demand", f"{demand:.1f} kW", f"{len(st.session_state['loads'])} loads monitored", "#38BDF8")
    with m3:
        render_metric_card("Scheduled Power", f"{scheduled:.1f} kW", f"Reserve Margin: {reserve:.1f} kW", "#10B981" if reserve >= 0 else "#EF4444")
    with m4:
        c_res = st.session_state.get("last_classical_res")
        score = c_res.objective_score if c_res else 0.0
        render_metric_card("Objective Score", f"{score:.1f}", "Maximized Priority Sum", "#F59E0B")

    st.markdown("###")

    # Two column overview: Live chart and Active Schedule Breakdown
    col_chart, col_loads = st.columns([3, 2])

    with col_chart:
        st.subheader("📈 Real-Time Supply vs Demand Telemetry")
        hist = st.session_state["telemetry_history"]
        if hist:
            df_hist = pd.DataFrame(hist)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist.index,
                y=df_hist["supply_capacity_kw"],
                mode="lines+markers",
                name="Supply Capacity (kW)",
                line=dict(color="#00F0FF", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=df_hist.index,
                y=df_hist["total_demand_kw"],
                mode="lines+markers",
                name="Total Demand (kW)",
                line=dict(color="#EF4444", width=2, dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=df_hist.index,
                y=df_hist["scheduled_power_kw"],
                mode="lines+markers",
                name="Scheduled Load (kW)",
                line=dict(color="#10B981", width=2)
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#0A0F1D",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Simulation Step",
                yaxis_title="Power (kW)",
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_loads:
        st.subheader("⚡ Recommended Load Schedule")
        c_res = st.session_state.get("last_classical_res")
        if c_res and c_res.selected_loads:
            sel_df = pd.DataFrame(c_res.selected_loads)[["name", "power", "priority", "criticality"]]
            sel_df.columns = ["Load", "Power (kW)", "Priority", "Criticality"]
            st.dataframe(sel_df, use_container_width=True, hide_index=True)
            st.caption(f"✅ {len(c_res.selected_loads)} loads scheduled | ⏸️ {len(c_res.deferred_loads)} loads deferred")
        else:
            st.warning("No loads could be scheduled within the current capacity limit.")

    st.markdown("---")
    # Quick Navigation Cards
    st.subheader("🚀 Platform Modules")
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        st.info("**2. Live Simulation**\n\nMonitor voltage sags, grid frequency droop, and synthetic telemetry.")
    with nav2:
        st.info("**4. Emergency Scheduling**\n\nEdit connected electrical loads, tweak priorities, and test capacity limits.")
    with nav3:
        st.info("**5. Quantum Lab**\n\nExecute the real QAOA circuit on local Qiskit Aer simulator and view bitstrings.")
    with nav4:
        st.info("**6. Classical vs Quantum**\n\nHead-to-head comparison of execution times, circuit depth, and objective scores.")


# ---------------------------------------------------------------------------
# PAGE 2: LIVE SIMULATION
# ---------------------------------------------------------------------------

elif current_page == "2. Live Simulation":
    render_top_banner()

    st.markdown("## 📡 Live Simulation Engine")
    st.caption("Physics-grounded synthetic telemetry generator modeling voltage sag, frequency droop, and current dynamics.")

    curr = get_current_reading()

    # Controls Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 2, 3])
    with ctrl_col1:
        if st.button("⚡ Generate Next Step", type="primary", use_container_width=True):
            curr = trigger_simulation_step()
            st.rerun()
    with ctrl_col2:
        if st.button("⏩ Advance 5 Steps", use_container_width=True):
            for _ in range(5):
                trigger_simulation_step()
            st.rerun()
    with ctrl_col3:
        sim_seed = st.number_input("RNG Seed", value=42, step=1, help="Seed for reproducible demonstration")
        if st.button("Set Seed", use_container_width=True):
            st.session_state["sim_engine"].set_seed(int(sim_seed))
            st.success(f"Simulation RNG seeded with {sim_seed}")
    with ctrl_col4:
        st.markdown(
            f"""
            <div style="background:#111827; padding:10px 14px; border-radius:8px; border:1px solid #1E293B;">
                <span style="color:#94A3B8; font-size:0.85rem;">Operating State Reason:</span><br/>
                <strong style="color:{curr.get('state_color', '#10B981')};">{curr.get('state_reason', 'Normal conditions.')}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("###")

    # 4 Gauges / Telemetry metrics
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        v = curr.get("voltage_v", 230.0)
        v_color = "#10B981" if v >= 225.0 else ("#F59E0B" if v >= 215.0 else "#EF4444")
        render_metric_card("AC Voltage", f"{v:.1f} V", "Nominal 230.0 V (Single Phase)", v_color)
    with g2:
        f = curr.get("frequency_hz", 50.00)
        f_color = "#10B981" if f >= 49.85 else ("#F59E0B" if f >= 49.60 else "#EF4444")
        render_metric_card("Grid Frequency", f"{f:.2f} Hz", "Nominal 50.00 Hz (Governor Droop)", f_color)
    with g3:
        i = curr.get("current_a", 0.0)
        render_metric_card("Simulated Current", f"{i:.1f} A", f"I = P / (V * PF)", "#38BDF8")
    with g4:
        pf = curr.get("power_factor", 0.95)
        render_metric_card("Power Factor", f"{pf:.3f}", "Apparent vs Real Ratio", "#00F0FF")

    st.markdown("###")

    # Historical Telemetry Visualizations
    hist = st.session_state["telemetry_history"]
    if hist:
        df_hist = pd.DataFrame(hist)

        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.subheader("⚡ Voltage Sag & Regulation (V)")
            fig_v = px.line(df_hist, x=df_hist.index, y="voltage_v", markers=True, template="plotly_dark")
            fig_v.add_hline(y=230.0, line_dash="dash", line_color="#38BDF8", annotation_text="Nominal (230V)")
            fig_v.add_hline(y=218.0, line_dash="dot", line_color="#EF4444", annotation_text="Under-Voltage Alert (218V)")
            fig_v.update_traces(line_color="#00F0FF", line_width=2)
            fig_v.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0A0F1D",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                xaxis_title="Step",
                yaxis_title="Voltage (V)"
            )
            st.plotly_chart(fig_v, use_container_width=True)

        with t_col2:
            st.subheader("🌀 Grid Frequency Droop (Hz)")
            fig_f = px.line(df_hist, x=df_hist.index, y="frequency_hz", markers=True, template="plotly_dark")
            fig_f.add_hline(y=50.00, line_dash="dash", line_color="#10B981", annotation_text="Nominal (50Hz)")
            fig_f.add_hline(y=49.60, line_dash="dot", line_color="#EF4444", annotation_text="Emergency Droop (49.6Hz)")
            fig_f.update_traces(line_color="#F59E0B", line_width=2)
            fig_f.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0A0F1D",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                xaxis_title="Step",
                yaxis_title="Frequency (Hz)"
            )
            st.plotly_chart(fig_f, use_container_width=True)

    # Telemetry Table View
    with st.expander("📋 View Raw Synthetic Telemetry Log (Most Recent Steps)", expanded=False):
        if hist:
            raw_df = pd.DataFrame(hist)[["timestamp", "mode", "supply_capacity_kw", "total_demand_kw", "scheduled_power_kw", "operating_state", "voltage_v", "frequency_hz", "power_factor"]]
            st.dataframe(raw_df.iloc[::-1], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 3: ENERGY MONITORING
# ---------------------------------------------------------------------------

elif current_page == "3. Energy Monitoring":
    render_top_banner()

    st.markdown("## ⚡ Energy & Load Monitoring")
    st.caption("Individual electrical load profiles, real-time consumption breakdown, and priority rankings.")

    curr = get_current_reading()
    loads_telemetry = curr.get("loads_telemetry", [])

    if loads_telemetry:
        df_loads = pd.DataFrame(loads_telemetry)

        c1, c2 = st.columns([3, 2])

        with c1:
            st.subheader("📊 Individual Load Power Ratings vs Fluctuation")
            fig_bar = px.bar(
                df_loads,
                x="name",
                y="current_kw",
                color="priority",
                color_continuous_scale="Viridis",
                labels={"current_kw": "Live Demand (kW)", "name": "Load", "priority": "Priority"},
                template="plotly_dark"
            )
            fig_bar.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0A0F1D",
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                xaxis_tickangle=-25
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            st.subheader("🥧 Load Demand Distribution")
            fig_pie = px.pie(
                df_loads,
                names="name",
                values="current_kw",
                hole=0.45,
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_pie.update_layout(
                paper_bgcolor="#111827",
                margin=dict(l=20, r=20, t=20, b=20),
                height=320
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Connected Load Inventory & Criticality Matrix")
        table_df = df_loads[["name", "nominal_kw", "current_kw", "priority", "criticality"]]
        table_df.columns = ["Equipment / Subsystem", "Nominal Power (kW)", "Simulated Power (kW)", "Priority (1-10)", "Criticality Level"]
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    else:
        st.info("No load telemetry available. Please advance simulation step.")


# ---------------------------------------------------------------------------
# PAGE 4: EMERGENCY LOAD SCHEDULING
# ---------------------------------------------------------------------------

elif current_page == "4. Emergency Load Scheduling":
    render_top_banner()

    st.markdown("## 🚨 Emergency Load Scheduling & Management")
    st.caption("Edit connected loads, toggle subsystems, test supply capacity constraints, and inspect the classical exact schedule.")

    curr = get_current_reading()
    curr_cap = float(st.session_state["supply_capacity_kw"])

    # Manual Supply Capacity Bar with Direct Number Input & Quick Presets
    st.markdown("### ⚡ Supply Capacity Controller")
    cap_c1, cap_c2, cap_c3 = st.columns([2, 2, 3])
    with cap_c1:
        entered_cap = st.number_input(
            "Emergency Supply Capacity (kW)",
            min_value=0.0,
            max_value=50.0,
            value=curr_cap,
            step=0.5,
            key="page4_manual_cap_entry",
            help="Type exact power capacity in kW"
        )
    with cap_c2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Apply Capacity (kW)", type="primary", use_container_width=True, key="btn_apply_cap"):
            st.session_state["supply_capacity_kw"] = entered_cap
            st.session_state["sim_engine"].supply_capacity_kw = entered_cap
            trigger_simulation_step()
            st.success(f"Capacity updated to {entered_cap:.1f} kW!")
            st.rerun()
    with cap_c3:
        st.markdown("<span style='font-size:0.8rem; color:#94A3B8;'>Quick Scenario Presets:</span>", unsafe_allow_html=True)
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        if p_col1.button("0 kW", help="Test total blackout", use_container_width=True):
            st.session_state["supply_capacity_kw"] = 0.0
            st.session_state["sim_engine"].supply_capacity_kw = 0.0
            trigger_simulation_step()
            st.rerun()
        if p_col2.button("5 kW", help="Severe shortage", use_container_width=True):
            st.session_state["supply_capacity_kw"] = 5.0
            st.session_state["sim_engine"].supply_capacity_kw = 5.0
            trigger_simulation_step()
            st.rerun()
        if p_col3.button("7 kW", help="Canonical demo default", use_container_width=True):
            st.session_state["supply_capacity_kw"] = 7.0
            st.session_state["sim_engine"].supply_capacity_kw = 7.0
            trigger_simulation_step()
            st.rerun()
        if p_col4.button("15 kW", help="Plentiful capacity", use_container_width=True):
            st.session_state["supply_capacity_kw"] = 15.0
            st.session_state["sim_engine"].supply_capacity_kw = 15.0
            trigger_simulation_step()
            st.rerun()

    st.markdown("---")

    # Load Editor Section with Multiple Input Methods
    st.subheader("✏️ Electrical Load Management & Manual Entry")
    st.caption("You can either edit the spreadsheet table directly, or use the dedicated input form to add equipment.")

    load_tab_table, load_tab_form, load_tab_restore = st.tabs([
        "📊 Spreadsheet Table Editor",
        "➕ Manual Add Load Form",
        "🔄 Reset Default Loads"
    ])

    with load_tab_table:
        st.write("Double-click any cell below to edit names, power ratings, priorities, or enable/disable loads. You can also add or delete rows directly.")

        # Convert current loads to DataFrame for data_editor
        df_editor_input = pd.DataFrame([
            {
                "Name": l["name"],
                "Power (kW)": float(l["power"]),
                "Priority (1-10)": float(l["priority"]),
                "Criticality": l.get("criticality", "Medium"),
                "Active": bool(l.get("enabled", True))
            }
            for l in st.session_state["loads"]
        ])

        edited_df = st.data_editor(
            df_editor_input,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Equipment Name", required=True),
                "Power (kW)": st.column_config.NumberColumn("Power (kW)", min_value=0.1, max_value=50.0, step=0.1, format="%.2f kW", required=True),
                "Priority (1-10)": st.column_config.NumberColumn("Priority (1-10)", min_value=1.0, max_value=10.0, step=1.0, format="%.1f", required=True),
                "Criticality": st.column_config.SelectboxColumn("Criticality", options=["Critical", "High", "Medium", "Low"], required=True),
                "Active": st.column_config.CheckboxColumn("Active / Monitored", default=True)
            },
            key="loads_data_spreadsheet"
        )

        col_save_btn, col_save_info = st.columns([2, 5])
        with col_save_btn:
            if st.button("💾 Apply & Save Table Changes", type="primary", use_container_width=True):
                # Validate and commit edits
                new_loads_list = []
                has_error = False
                for idx, row in edited_df.iterrows():
                    name_str = str(row["Name"]).strip()
                    if not name_str:
                        st.error(f"Row {idx+1}: Equipment Name cannot be empty.")
                        has_error = True
                        break
                    pwr_val = float(row["Power (kW)"])
                    if pwr_val <= 0:
                        st.error(f"Row {idx+1}: Power rating must be positive (> 0 kW).")
                        has_error = True
                        break
                    prio_val = float(row["Priority (1-10)"])
                    if prio_val < 1.0 or prio_val > 10.0:
                        st.error(f"Row {idx+1}: Priority must be between 1.0 and 10.0.")
                        has_error = True
                        break
                    new_loads_list.append({
                        "id": f"load-row-{idx}",
                        "name": name_str,
                        "power": round(pwr_val, 2),
                        "priority": round(prio_val, 1),
                        "criticality": str(row["Criticality"]),
                        "enabled": bool(row["Active"]),
                        "is_default": False
                    })
                if not has_error:
                    st.session_state["loads"] = new_loads_list
                    trigger_simulation_step()
                    st.success(f"Successfully updated {len(new_loads_list)} electrical loads!")
                    st.rerun()
        with col_save_info:
            st.info("Tip: After editing cells or adding rows in the table above, click 'Apply & Save Table Changes' to re-run optimization.")

    with load_tab_form:
        st.write("Fill in the fields below to append a new electrical load to the campus microgrid inventory.")
        with st.form("manual_add_load_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                form_name = st.text_input("Equipment / Load Name", placeholder="e.g. ICU Life Support / Server UPS")
                form_power = st.number_input("Power Demand (kW)", min_value=0.1, max_value=50.0, value=3.0, step=0.5)
            with f_col2:
                form_prio = st.number_input("Priority Ranking (1.0 to 10.0)", min_value=1.0, max_value=10.0, value=8.0, step=1.0)
                form_crit = st.selectbox("Criticality Classification", ["Critical", "High", "Medium", "Low"], index=1)

            submit_load = st.form_submit_button("➕ Add This Equipment Load", type="primary", use_container_width=True)
            if submit_load:
                if form_name.strip():
                    new_item = {
                        "id": f"load-manual-{int(time.time()*1000)}",
                        "name": form_name.strip(),
                        "power": round(float(form_power), 2),
                        "priority": round(float(form_prio), 1),
                        "criticality": form_crit,
                        "enabled": True,
                        "is_default": False
                    }
                    st.session_state["loads"].append(new_item)
                    trigger_simulation_step()
                    st.success(f"Added '{form_name.strip()}' ({form_power:.1f} kW, Priority {form_prio:.1f})!")
                    st.rerun()
                else:
                    st.error("Please enter a valid Equipment / Load Name.")

    with load_tab_restore:
        st.write("Restore the canonical 6 sample loads (17.0 kW total connected demand).")
        if st.button("↩️ Reset Loads to Default (6 Loads)", type="secondary"):
            st.session_state["loads"] = simulation.get_default_loads()
            st.session_state["supply_capacity_kw"] = 7.0
            trigger_simulation_step()
            st.success("Restored default 6 loads and 7.0 kW capacity!")
            st.rerun()

    st.markdown("---")

    # Classical Optimizer Execution & Explanation
    st.subheader("💡 Classical Exact Optimization Result")

    # Run solver
    c_res = classical_optimizer.solve_classical(st.session_state["loads"], curr_cap, only_enabled=True)
    st.session_state["last_classical_res"] = c_res

    # Metric summaries
    res_m1, res_m2, res_m3, res_m4 = st.columns(4)
    with res_m1:
        render_metric_card("Selected Power", f"{c_res.total_selected_power:.2f} kW", f"Capacity: {curr_cap:.2f} kW", "#10B981")
    with res_m2:
        render_metric_card("Unused Capacity", f"{c_res.remaining_capacity:.2f} kW", "Safety Reserve Margin", "#00F0FF")
    with res_m3:
        render_metric_card("Objective Score", f"{c_res.objective_score:.1f}", "Max Priority Achieved", "#F59E0B")
    with res_m4:
        render_metric_card("Combinations Evaluated", f"{c_res.candidates_evaluated}", f"{c_res.execution_time_ms:.2f} ms runtime", "#38BDF8")

    st.markdown("###")
    st.info(f"📋 **Optimization Decision Explanation:** {c_res.explanation}")

    # Side by Side: Retained vs Deferred Loads
    col_sel, col_def = st.columns(2)

    with col_sel:
        st.markdown(f"#### 🟢 Scheduled Loads ({len(c_res.selected_loads)})")
        if c_res.selected_loads:
            df_sel = pd.DataFrame(c_res.selected_loads)[["name", "power", "priority", "criticality"]]
            df_sel.columns = ["Load", "Power (kW)", "Priority", "Criticality"]
            st.dataframe(df_sel, use_container_width=True, hide_index=True)
        else:
            st.warning("No loads could be scheduled within available capacity.")

    with col_def:
        st.markdown(f"#### 🔴 Deferred Loads ({len(c_res.deferred_loads)})")
        if c_res.deferred_loads:
            df_def = pd.DataFrame(c_res.deferred_loads)[["name", "power", "priority", "criticality"]]
            df_def.columns = ["Load", "Power (kW)", "Priority", "Criticality"]
            st.dataframe(df_def, use_container_width=True, hide_index=True)
        else:
            st.success("All loads are successfully powered!")


# ---------------------------------------------------------------------------
# PAGE 5: QUANTUM LAB
# ---------------------------------------------------------------------------

elif current_page == "5. Quantum Lab":
    render_top_banner()

    st.markdown("## 🔬 Quantum Lab — QAOA Simulation")
    st.caption("Construct, parameterize, and execute a genuine Quantum Approximate Optimization Algorithm (QAOA) circuit using Qiskit Aer.")

    st.info(
        "💡 **Quantum Circuit Execution Notice:**\n\n"
        "This module constructs an authentic Qiskit parameterized quantum circuit ($p$ alternating layers of cost unitary $U(H_C, \\gamma)$ "
        "and mixer unitary $U(H_M, \\beta)$) and executes it on the local **Qiskit Aer** simulator. "
        "Measured bitstrings are directly decoded from simulated quantum state collapse. "
        "No actual physical quantum computer (QPU) is connected, and no false quantum speedup is claimed.",
        icon="ℹ️"
    )

    curr_cap = float(st.session_state["supply_capacity_kw"])
    active_loads = [l for l in st.session_state["loads"] if l.get("enabled", True)]
    n_qubits = len(active_loads)

    # QAOA Configuration Controls Form
    with st.form("qaoa_config_form"):
        st.subheader("⚙️ Quantum Circuit Hyperparameters")
        q_col1, q_col2, q_col3 = st.columns(3)

        with q_col1:
            qaoa_p = st.slider("QAOA Depth (p layers)", min_value=1, max_value=3, value=1, help="Number of alternating cost and mixer unitary layers")
            shots = st.select_slider("Measurement Shots", options=[100, 256, 512, 1024, 2048, 4096], value=1024, help="Number of circuit executions / measurement samples")

        with q_col2:
            penalty_lam = st.number_input("Penalty Multiplier (λ)", min_value=0.5, max_value=20.0, value=4.0, step=0.5, help="Constraint penalty weight in QUBO formulation")
            optimize_params = st.checkbox("Optimize Angles via COBYLA", value=False, help="Run classical outer loop (COBYLA, 15 iterations) to tune gamma and beta angles")

        with q_col3:
            st.markdown("**Initial Rotation Angles:**")
            gamma_init = st.number_input("Cost Angle γ₁", min_value=0.01, max_value=3.14, value=0.35, step=0.05)
            beta_init = st.number_input("Mixer Angle β₁", min_value=0.01, max_value=3.14, value=0.45, step=0.05)

        run_quantum_btn = st.form_submit_button("🚀 Run QAOA Circuit on Qiskit Aer", type="primary", use_container_width=True)

    if run_quantum_btn:
        with st.spinner("Compiling QAOA circuit and executing on Qiskit Aer simulator..."):
            gammas = [gamma_init] + [round(gamma_init * (i + 1), 3) for i in range(1, qaoa_p)]
            betas = [beta_init] + [round(beta_init / (i + 1), 3) for i in range(1, qaoa_p)]

            q_res = quantum_optimizer.run_qaoa_simulation(
                loads=st.session_state["loads"],
                capacity_kw=curr_cap,
                depth_p=qaoa_p,
                shots=shots,
                penalty_multiplier=penalty_lam,
                gamma_values=gammas,
                beta_values=betas,
                optimize_parameters=optimize_params,
                only_enabled=True
            )
            st.session_state["last_quantum_res"] = q_res

            # Record in SQLite optimization history
            data_store.record_optimization_run(q_res.to_dict())

    # Display Quantum Results if available
    q_res: Optional[quantum_optimizer.QuantumOptimizationResult] = st.session_state.get("last_quantum_res")

    if q_res is not None:
        st.markdown("---")
        st.subheader("🎯 QAOA Simulation Results & Telemetry")

        qm1, qm2, qm3, qm4, qm5 = st.columns(5)
        with qm1:
            render_metric_card("Qubits Monitored", f"{q_res.num_qubits}", f"Active Loads: {q_res.num_qubits}", "#00F0FF")
        with qm2:
            render_metric_card("Circuit Depth", f"{q_res.circuit_depth}", f"QAOA Layers: p={q_res.qaoa_depth}", "#38BDF8")
        with qm3:
            render_metric_card("Aer Sim Time", f"{q_res.simulator_time_ms:.1f} ms", f"Total: {q_res.total_workflow_time_ms:.1f} ms", "#10B981")
        with qm4:
            f_color = "#10B981" if q_res.feasible_sample_percentage >= 30.0 else "#F59E0B"
            render_metric_card("Feasible Samples", f"{q_res.feasible_sample_percentage:.1f}%", f"{q_res.shots} Total Shots", f_color)
        with qm5:
            render_metric_card("Sampled Score", f"{q_res.objective_score:.1f}", f"Bitstring: {q_res.best_bitstring}", "#F59E0B")

        st.markdown("###")
        st.info(f"🔍 **Quantum Solution Summary:** {q_res.explanation}")

        # Gate Breakdown and Outcome Distribution Tabs
        tab_dist, tab_sched, tab_circ = st.tabs(["📊 Sampled Outcome Distribution", "📋 Best Feasible Schedule", "🧩 Circuit Architecture & ASCII"])

        with tab_dist:
            dist = q_res.sampled_distribution
            if dist:
                df_dist = pd.DataFrame(dist)
                top_20 = df_dist.head(20)

                fig_dist = px.bar(
                    top_20,
                    x="bitstring",
                    y="count",
                    color="is_feasible",
                    color_discrete_map={True: "#10B981", False: "#EF4444"},
                    labels={"bitstring": "Measured Bitstring", "count": "Sample Count", "is_feasible": "Capacity Feasible"},
                    title="Top 20 Measured Bitstrings (Green = Within Capacity, Red = Exceeded Capacity)",
                    template="plotly_dark"
                )
                fig_dist.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#0A0F1D",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=340
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                st.dataframe(
                    top_20[["bitstring", "count", "percentage", "total_power_kw", "objective_score", "is_feasible"]].rename(columns={
                        "bitstring": "Bitstring",
                        "count": "Counts",
                        "percentage": "Frequency (%)",
                        "total_power_kw": "Power (kW)",
                        "objective_score": "Score",
                        "is_feasible": "Feasible"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

        with tab_sched:
            if q_res.feasible and q_res.selected_loads:
                st.markdown(f"#### Best Feasible Quantum Schedule (Objective Score: {q_res.objective_score:.1f})")
                q_df = pd.DataFrame(q_res.selected_loads)[["name", "power", "priority", "criticality"]]
                q_df.columns = ["Scheduled Load", "Power (kW)", "Priority", "Criticality"]
                st.dataframe(q_df, use_container_width=True, hide_index=True)
            else:
                st.error("No feasible load schedule was found within sampled quantum circuit measurements.")

        with tab_circ:
            st.markdown("#### Quantum Circuit Gate Breakdown")
            st.json(q_res.gate_counts)
            st.markdown("#### Qiskit Text Circuit Diagram")
            st.code(q_res.ascii_circuit, language="text")

    else:
        st.write("Click **Run QAOA Circuit** above to execute the quantum simulation.")


# ---------------------------------------------------------------------------
# PAGE 6: CLASSICAL VS QUANTUM COMPARISON
# ---------------------------------------------------------------------------

elif current_page == "6. Classical vs Quantum Comparison":
    render_top_banner()

    st.markdown("## ⚖️ Classical vs Quantum Comparison")
    st.caption("Rigorous, side-by-side benchmarking of exact classical combinatorial search against local QAOA simulation.")

    c_res: Optional[classical_optimizer.ClassicalOptimizationResult] = st.session_state.get("last_classical_res")
    q_res: Optional[quantum_optimizer.QuantumOptimizationResult] = st.session_state.get("last_quantum_res")

    if q_res is None:
        st.warning("⚠️ Quantum simulation has not been executed yet. Running QAOA automatically for comparison...")
        with st.spinner("Executing QAOA circuit on Qiskit Aer..."):
            q_res = quantum_optimizer.run_qaoa_simulation(
                loads=st.session_state["loads"],
                capacity_kw=float(st.session_state["supply_capacity_kw"]),
                depth_p=1,
                shots=1024,
                penalty_multiplier=4.0
            )
            st.session_state["last_quantum_res"] = q_res

    curr_cap = float(st.session_state["supply_capacity_kw"])

    # Fair Comparison Table
    st.subheader("📊 Benchmark Metric Matrix")

    comp_data = {
        "Evaluation Dimension": [
            "Optimization Methodology",
            "Objective Score (Total Priority)",
            "Total Power Selected (kW)",
            "Remaining Power Reserve (kW)",
            "Feasibility Status",
            "Optimal Bitstring Representation",
            "Candidates Evaluated / Shots",
            "Execution Runtime (ms)",
            "Circuit Depth / Gates",
            "Hardware Environment"
        ],
        "Classical Exact Optimizer": [
            c_res.method if c_res else "N/A",
            f"{c_res.objective_score:.1f}" if c_res else "N/A",
            f"{c_res.total_selected_power:.2f} kW" if c_res else "N/A",
            f"{c_res.remaining_capacity:.2f} kW" if c_res else "N/A",
            "✅ 100% Feasible" if (c_res and c_res.feasible) else "❌ Infeasible",
            c_res.bitstring if c_res else "N/A",
            f"{c_res.candidates_evaluated} combinatorial subsets" if c_res else "N/A",
            f"{c_res.execution_time_ms:.2f} ms" if c_res else "N/A",
            "N/A (CPU Branch/Powerset)",
            "Local CPU (x86_64 / Windows)"
        ],
        "Quantum QAOA Simulator (Aer)": [
            f"QAOA (Depth p={q_res.qaoa_depth})" if q_res else "N/A",
            f"{q_res.objective_score:.1f}" if q_res else "N/A",
            f"{q_res.total_selected_power:.2f} kW" if q_res else "N/A",
            f"{q_res.remaining_capacity:.2f} kW" if q_res else "N/A",
            f"✅ Feasible ({q_res.feasible_sample_percentage:.1f}% shots)" if (q_res and q_res.feasible) else "❌ No Feasible Sample Found",
            q_res.best_bitstring if q_res else "N/A",
            f"{q_res.shots} circuit shots ({len(q_res.sampled_distribution)} unique bitstrings)" if q_res else "N/A",
            f"{q_res.simulator_time_ms:.2f} ms (Aer Sim)" if q_res else "N/A",
            f"Depth: {q_res.circuit_depth} | {sum(q_res.gate_counts.values())} gates" if q_res else "N/A",
            "Qiskit Aer (Classical CPU Statevector/Clifford)"
        ]
    }

    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("###")

    # Side-by-side Bar Visualizations
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.subheader("🎯 Objective Score Comparison")
        score_df = pd.DataFrame({
            "Method": ["Classical Exact", "Quantum QAOA (Aer)"],
            "Score": [c_res.objective_score if c_res else 0.0, q_res.objective_score if q_res else 0.0]
        })
        fig_score = px.bar(
            score_df,
            x="Method",
            y="Score",
            color="Method",
            color_discrete_map={"Classical Exact": "#00F0FF", "Quantum QAOA (Aer)": "#38BDF8"},
            template="plotly_dark"
        )
        fig_score.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0A0F1D",
            margin=dict(l=20, r=20, t=20, b=20),
            height=280
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with b_col2:
        st.subheader("⚡ Selected Power vs Capacity Limit")
        pwr_df = pd.DataFrame({
            "Method": ["Classical Exact", "Quantum QAOA (Aer)"],
            "Selected Power (kW)": [c_res.total_selected_power if c_res else 0.0, q_res.total_selected_power if q_res else 0.0]
        })
        fig_pwr = px.bar(
            pwr_df,
            x="Method",
            y="Selected Power (kW)",
            color="Method",
            color_discrete_map={"Classical Exact": "#10B981", "Quantum QAOA (Aer)": "#F59E0B"},
            template="plotly_dark"
        )
        fig_pwr.add_hline(y=curr_cap, line_dash="dash", line_color="#EF4444", annotation_text=f"Capacity Limit ({curr_cap:.1f} kW)")
        fig_pwr.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0A0F1D",
            margin=dict(l=20, r=20, t=20, b=20),
            height=280
        )
        st.plotly_chart(fig_pwr, use_container_width=True)

    st.markdown("---")
    st.subheader("🎓 Research Discussion & Quantum Reality Check")
    st.markdown("""
    **Honest Scientific Analysis:**
    - **Optimality:** For small problem instances ($N \\le 20$), the classical combinatorial solver finds the guaranteed global optimum in under 5 milliseconds.
    - **QAOA Sampling:** QAOA does not directly guarantee the optimal bitstring with 100% probability; it prepares a quantum state whose measurement distribution is biased toward low-energy (high-priority, feasible) states.
    - **Simulation Overhead:** Qiskit Aer simulates quantum matrix multiplications on a classical CPU. Aer execution times include simulator initialization, circuit compilation, and pseudo-random measurement sampling.
    - **Quantum Advantage Status:** Quantum advantage is **not demonstrated** on small toy problem sizes. Practical quantum utility would require large-scale fault-tolerant quantum hardware with hundreds of logical qubits solving non-linear microgrid dispatch equations beyond classical dynamic programming reach.
    """)


# ---------------------------------------------------------------------------
# PAGE 7: ANALYTICS AND REPORTS
# ---------------------------------------------------------------------------

elif current_page == "7. Analytics and Reports":
    render_top_banner()

    st.markdown("## 📈 Analytics & Benchmark Reports")
    st.caption("Deep-dive telemetry analytics, historical operating states, and CSV / Markdown export utilities.")

    hist = st.session_state["telemetry_history"]
    df_hist = pd.DataFrame(hist) if hist else pd.DataFrame()

    tab_exp, tab_charts, tab_history = st.tabs(["📥 Data & Report Export", "📊 Historical Visualizations", "🗄️ Database Records"])

    with tab_exp:
        st.subheader("Generate & Export Reports")

        exp_c1, exp_c2, exp_c3 = st.columns(3)

        with exp_c1:
            st.markdown("#### 📄 Telemetry CSV Export")
            if not df_hist.empty:
                csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Telemetry CSV",
                    data=csv_bytes,
                    file_name=f"gridflex_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"{len(df_hist)} simulated measurement points ready.")
            else:
                st.info("No telemetry points to export.")

        with exp_c2:
            st.markdown("#### ⚡ Optimization History CSV")
            opt_df = data_store.get_optimization_history_df(limit=50)
            if not opt_df.empty:
                opt_csv_bytes = opt_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Optimization Runs CSV",
                    data=opt_csv_bytes,
                    file_name=f"gridflex_optimization_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"{len(opt_df)} optimizer benchmark runs.")
            else:
                st.info("No optimization runs recorded.")

        with exp_c3:
            st.markdown("#### 📋 Comprehensive Text Report")
            c_res = st.session_state.get("last_classical_res")
            q_res = st.session_state.get("last_quantum_res")
            curr = get_current_reading()

            summary_text = f"""================================================================================
GRIDFLEX-Q EMERGENCY POWER SCHEDULING BENCHMARK REPORT
================================================================================
Generated Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Data Classification: SIMULATED TELEMETRY - NOT CONNECTED TO PHYSICAL HARDWARE

1. OPERATIONAL CONTEXT
----------------------
Simulation Mode: {curr.get('mode', 'N/A')}
Grid Operating State: {curr.get('operating_state', 'N/A')}
Available Supply Capacity: {curr.get('supply_capacity_kw', 0.0):.2f} kW
Total Connected Demand: {curr.get('total_demand_kw', 0.0):.2f} kW
Monitored Loads Count: {len(st.session_state['loads'])}

2. ELECTRICAL TELEMETRY AT REPORT GENERATION
---------------------------------------------
AC Voltage: {curr.get('voltage_v', 0.0):.1f} V
Simulated Current: {curr.get('current_a', 0.0):.1f} A
Grid Frequency: {curr.get('frequency_hz', 0.0):.2f} Hz
Power Factor: {curr.get('power_factor', 0.0):.3f}

3. CLASSICAL OPTIMIZATION RESULT
---------------------------------
Method: {c_res.method if c_res else 'N/A'}
Objective Score: {c_res.objective_score if c_res else 0.0}
Scheduled Power: {c_res.total_selected_power if c_res else 0.0} kW
Reserve Capacity: {c_res.remaining_capacity if c_res else 0.0} kW
Runtime: {c_res.execution_time_ms if c_res else 0.0} ms
Feasible: {c_res.feasible if c_res else False}

4. QUANTUM QAOA SIMULATION RESULT
----------------------------------
Method: {q_res.optimization_mode if q_res else 'N/A'}
Qubits: {q_res.num_qubits if q_res else 0}
Circuit Depth: {q_res.circuit_depth if q_res else 0}
Shots: {q_res.shots if q_res else 0}
Feasible Sample %: {q_res.feasible_sample_percentage if q_res else 0.0}%
Simulator Runtime: {q_res.simulator_time_ms if q_res else 0.0} ms
Best Objective Score: {q_res.objective_score if q_res else 0.0}
Best Bitstring: {q_res.best_bitstring if q_res else 'N/A'}

================================================================================
END OF SIMULATION REPORT
================================================================================
"""
            st.download_button(
                label="💾 Download Summary Report (.txt)",
                data=summary_text.encode("utf-8"),
                file_name=f"gridflex_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    with tab_charts:
        if not df_hist.empty:
            st.subheader("Historical Grid Operating States")
            state_counts = df_hist["operating_state"].value_counts().reset_index()
            state_counts.columns = ["Operating State", "Count"]

            fig_states = px.bar(
                state_counts,
                x="Operating State",
                y="Count",
                color="Operating State",
                color_discrete_map={"Normal": "#10B981", "Warning": "#F59E0B", "Emergency": "#EF4444", "Critical Shortage": "#EF4444"},
                template="plotly_dark"
            )
            fig_states.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0A0F1D",
                margin=dict(l=20, r=20, t=20, b=20),
                height=280
            )
            st.plotly_chart(fig_states, use_container_width=True)

    with tab_history:
        st.subheader("SQLite Persisted Optimization Records")
        opt_df = data_store.get_optimization_history_df(limit=25)
        if not opt_df.empty:
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Optimization History Table"):
                data_store.clear_optimization_history()
                st.success("Optimization history cleared.")
                st.rerun()
        else:
            st.info("No recorded optimization history in SQLite database yet.")


# ---------------------------------------------------------------------------
# PAGE 8: AUTHENTICATION
# ---------------------------------------------------------------------------

elif current_page == "8. Authentication":
    render_top_banner()
    auth.render_auth_page()


# ---------------------------------------------------------------------------
# PAGE 9: PROJECT INFORMATION AND LIMITATIONS
# ---------------------------------------------------------------------------

elif current_page == "9. Project Information & Limitations":
    render_top_banner()

    st.markdown("## 📘 Project Information, Architecture & Limitations")
    st.caption("Theoretical framework, mathematical formulations, software architecture, and real-world boundaries.")

    st.markdown("""
    ### 1. Project Vision
    **GridFlex-Q** is an educational, interactive energy optimization prototype designed to demonstrate how
    combinatorial quantum algorithms (specifically QAOA) can be formulated and simulated for emergency load shedding
    in microgrids and smart facilities.

    ### 2. Theoretical Formulation

    #### A. Classical Knapsack Formulation
    Emergency load selection is modeled as a 0/1 Knapsack Problem:
    $$\\text{Maximize } \\sum_{i=0}^{n-1} p_i x_i \\quad \\text{subject to } \\sum_{i=0}^{n-1} w_i x_i \\le C, \\quad x_i \\in \\{0, 1\\}$$
    where $p_i$ is priority (1–10), $w_i$ is power demand (kW), and $C$ is available emergency power capacity (kW).

    #### B. QUBO & Cost Hamiltonian
    To map the constrained problem to quantum circuits, the capacity constraint is converted to a quadratic penalty:
    $$C(x) = - \\sum_{i=0}^{n-1} p_i x_i + \\lambda \\left( \\sum_{i=0}^{n-1} w_i x_i - C \\right)^2$$
    Using the standard qubit spin transformation $x_i = \\frac{I - Z_i}{2}$, the cost Hamiltonian is derived as:
    $$H_C = \\sum_{i=0}^{n-1} h_i Z_i + \\sum_{0 \\le i < j < n} J_{ij} Z_i Z_j + \\text{const} \\cdot I$$

    #### C. QAOA Circuit Architecture
    1. **Initialization:** Prepare equal superposition state $|+\\rangle^{\\otimes n} = H^{\\otimes n} |0\\rangle^{\\otimes n}$.
    2. **Alternating Unitaries:** For each layer $l = 1, \\dots, p$:
       - Cost Unitary: $U(H_C, \\gamma_l) = e^{-i \\gamma_l H_C}$ implemented via single-qubit $R_z$ and two-qubit $R_{zz}$ gates.
       - Mixer Unitary: $U(H_M, \\beta_l) = e^{-i \\beta_l \\sum_i X_i}$ implemented via single-qubit $R_x$ gates.
    3. **Measurement:** Collapses quantum state into computational basis bitstrings.
    4. **Decoding & Verification:** Every sampled bitstring is checked against capacity constraint $\\sum w_i x_i \\le C$.

    ---

    ### 3. Transparent Technical Limitations

    > [!IMPORTANT]
    > **Educational Prototype Boundary:**
    > - **Simulated Data:** All voltage, frequency, and current values are synthesized by an internal physics-grounded engine. They are NOT readings from live sensors.
    > - **No Physical Control:** GridFlex-Q cannot actuate physical circuit breakers, relays, or electrical switchgear.
    > - **Quantum Simulation:** Circuits run locally on **Qiskit Aer** CPU/GPU simulation rather than cryogenically cooled QPUs.
    > - **No False Advantage:** Classical branch-and-bound solves small knapsack instances in $<5\\text{ ms}$. Quantum advantage is not claimed.

    ---

    ### 4. System Architecture
    ```
    GridFlex-Q/
    ├── app.py                  <-- Streamlit 9-Page UI & Navigation
    ├── auth.py                 <-- Salted PBKDF2-HMAC-SHA256 Auth
    ├── simulation.py           <-- Synthetic Electrical Engine
    ├── classical_optimizer.py  <-- Exact 0/1 Knapsack Solver
    ├── quantum_optimizer.py    <-- Qiskit 2.x QAOA Circuit & Aer
    ├── data_store.py           <-- SQLite Persistent Telemetry
    ├── requirements.txt        <-- Pinned Verified Dependencies
    └── tests/                  <-- Pytest Unit Test Suite (19/19)
    ```
    """)
