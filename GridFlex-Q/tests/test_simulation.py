"""
Unit Tests for Simulation Engine
================================
Tests telemetry generation, bounded variations, deterministic seeding,
load manipulations, and state classifications.
"""

import pytest
import simulation


def test_default_sample_loads_structure():
    """Verify default sample loads match the specification."""
    loads = simulation.get_default_loads()
    assert len(loads) == 6
    names = [l["name"] for l in loads]
    assert "Emergency Lighting" in names
    assert "Network Equipment" in names
    assert "Water Pump" in names
    assert "Classroom AC" in names
    assert "Laboratory Equipment" in names
    assert "General Lighting" in names

    total_demand = simulation.calculate_connected_demand(loads)
    assert total_demand == 17.0  # 2 + 3 + 4 + 5 + 2 + 1 = 17 kW


def test_deterministic_simulation_seed():
    """Same seed must produce identical telemetry sequences."""
    eng1 = simulation.SimulationEngine(seed=123)
    eng2 = simulation.SimulationEngine(seed=123)

    r1_step1 = eng1.generate_step()
    r2_step1 = eng2.generate_step()

    assert r1_step1["voltage_v"] == r2_step1["voltage_v"]
    assert r1_step1["frequency_hz"] == r2_step1["frequency_hz"]
    assert r1_step1["power_factor"] == r2_step1["power_factor"]
    assert r1_step1["supply_capacity_kw"] == r2_step1["supply_capacity_kw"]

    r1_step2 = eng1.generate_step()
    r2_step2 = eng2.generate_step()

    assert r1_step2["voltage_v"] == r2_step2["voltage_v"]
    assert r1_step2["current_a"] == r2_step2["current_a"]


def test_grid_operating_state_classification():
    """Ensure transparent thresholding for Normal, Warning, and Emergency states."""
    # Normal: supply 20 kW, demand 15 kW (ratio <= 1.0)
    state, color, _ = simulation.evaluate_grid_operating_state(supply_capacity_kw=20.0, demand_kw=15.0)
    assert state == "Normal"

    # Warning: supply 10 kW, demand 13 kW (ratio 1.3, <= 1.4)
    state, color, _ = simulation.evaluate_grid_operating_state(supply_capacity_kw=10.0, demand_kw=13.0)
    assert state == "Warning"

    # Emergency: supply 7 kW, demand 17 kW (ratio 2.4, > 1.4)
    state, color, _ = simulation.evaluate_grid_operating_state(supply_capacity_kw=7.0, demand_kw=17.0)
    assert state == "Emergency"

    # Zero supply: blackout emergency
    state, _, _ = simulation.evaluate_grid_operating_state(supply_capacity_kw=0.0, demand_kw=10.0)
    assert state == "Critical Shortage"


def test_telemetry_physical_bounds():
    """Verify all electrical measurements remain physically realistic."""
    engine = simulation.SimulationEngine(seed=999)
    for _ in range(25):
        reading = engine.generate_step()

        # Voltage around 230V nominal (between 200V and 245V)
        assert 200.0 <= reading["voltage_v"] <= 245.0

        # Frequency around 50Hz nominal (between 48.5Hz and 51.5Hz)
        assert 48.5 <= reading["frequency_hz"] <= 51.5

        # Power factor between 0.85 and 1.00
        assert 0.85 <= reading["power_factor"] <= 1.00

        # Current is non-negative
        assert reading["current_a"] >= 0.0

        # Data source badge present
        assert "SIMULATED" in reading["data_source_badge"]


def test_load_manipulation():
    """Test dynamic addition, disabling, and demand calculation."""
    loads = simulation.get_default_loads()
    initial_demand = simulation.calculate_connected_demand(loads)
    assert initial_demand == 17.0

    # Add new load
    loads.append({
        "id": "custom-1",
        "name": "Solar Inverter Auxiliary",
        "power": 3.5,
        "priority": 5.0,
        "criticality": "Medium",
        "enabled": True,
        "is_default": False
    })
    new_demand = simulation.calculate_connected_demand(loads)
    assert new_demand == 20.5

    # Disable one load
    loads[0]["enabled"] = False  # 2.0 kW disabled
    active_demand = simulation.calculate_connected_demand(loads, only_enabled=True)
    assert active_demand == 18.5
