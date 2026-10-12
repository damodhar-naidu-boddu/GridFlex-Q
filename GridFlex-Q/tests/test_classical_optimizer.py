"""
Unit Tests for Classical Optimizer
==================================
Tests capacity constraints, priority calculations, deterministic tie handling,
edge cases (zero capacity, empty list, all loads exceeding), and known manual benchmarks.
"""

import pytest
import classical_optimizer


@pytest.fixture
def sample_loads():
    return [
        {"name": "Emergency Lighting", "power": 2.0, "priority": 10.0, "criticality": "Critical", "enabled": True},
        {"name": "Network Equipment", "power": 3.0, "priority": 8.0, "criticality": "High", "enabled": True},
        {"name": "Water Pump", "power": 4.0, "priority": 7.0, "criticality": "High", "enabled": True},
        {"name": "Classroom AC", "power": 5.0, "priority": 3.0, "criticality": "Low", "enabled": True},
        {"name": "Laboratory Equipment", "power": 2.0, "priority": 6.0, "criticality": "Medium", "enabled": True},
        {"name": "General Lighting", "power": 1.0, "priority": 5.0, "criticality": "Medium", "enabled": True}
    ]


def test_known_benchmark_7kw(sample_loads):
    """
    Known Benchmark Test Case:
    At 7.0 kW capacity with the canonical 6 sample loads:
    - Emergency Lighting (2 kW, P10) + Network (3 kW, P8) + Lab (2 kW, P6) = 7.0 kW, Priority = 24.0.
    This is mathematically verified to be the unique maximal priority subset.
    """
    result = classical_optimizer.solve_classical(sample_loads, capacity_kw=7.0)

    assert result.feasible is True
    assert result.objective_score == 24.0
    assert result.total_selected_power == 7.0
    assert result.remaining_capacity == 0.0

    selected_names = [l["name"] for l in result.selected_loads]
    assert "Emergency Lighting" in selected_names
    assert "Network Equipment" in selected_names
    assert "Laboratory Equipment" in selected_names
    assert len(result.selected_loads) == 3
    assert len(result.deferred_loads) == 3


def test_capacity_strictly_enforced(sample_loads):
    """Ensure total power of selected loads never exceeds capacity."""
    capacities = [1.5, 3.0, 5.5, 9.0, 15.0]
    for cap in capacities:
        result = classical_optimizer.solve_classical(sample_loads, capacity_kw=cap)
        assert result.total_selected_power <= cap + 1e-6
        assert result.remaining_capacity >= -1e-6
        assert result.feasible is True


def test_zero_capacity(sample_loads):
    """Zero capacity must safely defer all loads and score 0.0."""
    result = classical_optimizer.solve_classical(sample_loads, capacity_kw=0.0)
    assert result.total_selected_power == 0.0
    assert result.objective_score == 0.0
    assert len(result.selected_loads) == 0
    assert len(result.deferred_loads) == len(sample_loads)
    assert result.feasible is True


def test_negative_capacity(sample_loads):
    """Negative capacity is treated as 0 kW safely."""
    result = classical_optimizer.solve_classical(sample_loads, capacity_kw=-5.0)
    assert result.total_selected_power == 0.0
    assert result.objective_score == 0.0
    assert len(result.selected_loads) == 0


def test_empty_loads_list():
    """Empty list must yield 0 loads and not raise errors."""
    result = classical_optimizer.solve_classical([], capacity_kw=10.0)
    assert result.total_selected_power == 0.0
    assert result.objective_score == 0.0
    assert len(result.selected_loads) == 0
    assert result.feasible is True


def test_all_loads_exceed_capacity():
    """When every single load exceeds capacity, no loads should be selected."""
    heavy_loads = [
        {"name": "Gigantic Compressor", "power": 12.0, "priority": 9.0},
        {"name": "Industrial Furnace", "power": 25.0, "priority": 10.0}
    ]
    result = classical_optimizer.solve_classical(heavy_loads, capacity_kw=5.0)
    assert len(result.selected_loads) == 0
    assert result.total_selected_power == 0.0
    assert result.objective_score == 0.0
    assert result.remaining_capacity == 5.0


def test_deterministic_tie_breaking():
    """
    When two options have identical priority, prefer lower power (higher safety reserve).
    Option A: Load 1 (Power 3, Priority 8)
    Option B: Load 2 (Power 4, Priority 8)
    Capacity: 5
    """
    tied_loads = [
        {"name": "Efficient Server", "power": 3.0, "priority": 8.0},
        {"name": "Hungry Server", "power": 4.0, "priority": 8.0}
    ]
    result = classical_optimizer.solve_classical(tied_loads, capacity_kw=5.0)
    assert len(result.selected_loads) == 1
    assert result.selected_loads[0]["name"] == "Efficient Server"
    assert result.total_selected_power == 3.0


def test_invalid_input_validation():
    """Invalid input types, negative power, or out-of-bound priorities must be caught gracefully."""
    invalid_loads = [
        {"name": "Bad Power", "power": -2.0, "priority": 5.0}
    ]
    result = classical_optimizer.solve_classical(invalid_loads, capacity_kw=10.0)
    assert result.feasible is False
    assert "negative" in result.explanation.lower()

    bad_priority = [
        {"name": "High Priority", "power": 2.0, "priority": 15.0}
    ]
    result2 = classical_optimizer.solve_classical(bad_priority, capacity_kw=10.0)
    assert result2.feasible is False
    assert "between 1 and 10" in result2.explanation.lower()


def test_disabled_loads_filtering(sample_loads):
    """Loads with enabled=False should be excluded from selection."""
    loads_copy = [dict(l) for l in sample_loads]
    # Disable Emergency Lighting (the highest priority load)
    loads_copy[0]["enabled"] = False

    result = classical_optimizer.solve_classical(loads_copy, capacity_kw=7.0, only_enabled=True)
    selected_names = [l["name"] for l in result.selected_loads]
    assert "Emergency Lighting" not in selected_names
    # Disabled load must appear in deferred list
    deferred_names = [l["name"] for l in result.deferred_loads]
    assert "Emergency Lighting" in deferred_names
