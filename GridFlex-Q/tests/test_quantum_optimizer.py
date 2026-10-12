"""
Unit Tests for Quantum Optimizer (QAOA)
=======================================
Tests QUBO formulation, Qiskit Aer circuit execution, bitstring decoding,
capacity feasibility checking, edge case handling, and objective score agreement.
"""

import numpy as np
import pytest
import simulation
import classical_optimizer
import quantum_optimizer


@pytest.fixture
def sample_loads():
    return simulation.get_default_loads()


def test_qubo_hamiltonian_formulation(sample_loads):
    """Verify Hamiltonian arrays dimension and offset scalar."""
    h, J, offset = quantum_optimizer.formulate_qubo_hamiltonian(
        sample_loads,
        capacity_kw=7.0,
        penalty_multiplier=4.0
    )
    n = len(sample_loads)
    assert len(h) == n
    assert J.shape == (n, n)
    assert isinstance(offset, float)
    # Off-diagonal symmetry J_ij == J_ji
    assert np.allclose(J, J.T)


def test_qaoa_circuit_execution_on_aer(sample_loads):
    """Verify QAOA circuit runs on Qiskit Aer and returns valid measurement outcomes."""
    res = quantum_optimizer.run_qaoa_simulation(
        loads=sample_loads,
        capacity_kw=7.0,
        depth_p=1,
        shots=300,
        penalty_multiplier=4.0
    )

    assert res.num_qubits == 6
    assert res.shots == 300
    assert res.circuit_depth > 0
    assert "rzz" in res.gate_counts
    assert "rx" in res.gate_counts
    assert "rz" in res.gate_counts
    assert res.simulator_time_ms > 0
    assert len(res.sampled_distribution) > 0

    # Ensure best candidate satisfies feasibility
    if res.feasible:
        assert res.total_selected_power <= 7.0 + 1e-6
        assert res.remaining_capacity >= 0.0
        assert res.objective_score > 0.0


def test_objective_score_agreement_classical_vs_quantum(sample_loads):
    """
    Given the optimal schedule '110010' (loads 0, 1, 4):
    Verify that both classical and quantum objective functions calculate
    the exact same total power (7.0 kW) and priority score (24.0).
    """
    bitstring = "110010"
    selected_indices = [i for i, bit in enumerate(bitstring) if bit == "1"]
    selected = [sample_loads[i] for i in selected_indices]

    quantum_power = round(sum(l["power"] for l in selected), 2)
    quantum_score = round(sum(l["priority"] for l in selected), 2)

    classical_res = classical_optimizer.solve_classical(sample_loads, capacity_kw=7.0)

    assert quantum_power == classical_res.total_selected_power == 7.0
    assert quantum_score == classical_res.objective_score == 24.0


def test_quantum_zero_capacity_behavior(sample_loads):
    """Zero capacity must not crash and should safely return 0 loads."""
    res = quantum_optimizer.run_qaoa_simulation(
        loads=sample_loads,
        capacity_kw=0.0,
        depth_p=1,
        shots=100
    )
    assert res.total_selected_power == 0.0
    assert res.objective_score == 0.0
    assert len(res.selected_loads) == 0
    assert res.feasible is True


def test_quantum_impossible_capacity_infeasible_reporting():
    """
    If capacity is ridiculously small (0.1 kW) such that no combination is feasible,
    the quantum optimizer must honestly report feasible=False and NOT fabricate a valid schedule.
    """
    big_loads = [
        {"name": "Heavy Machine 1", "power": 10.0, "priority": 5.0, "enabled": True},
        {"name": "Heavy Machine 2", "power": 15.0, "priority": 8.0, "enabled": True}
    ]
    # Synthetic counts that have only '1' bits
    fake_counts = {"11": 100}
    distribution, best_candidate, feas_pct = quantum_optimizer.decode_qiskit_counts(
        counts=fake_counts,
        loads=big_loads,
        capacity_kw=0.1,  # 0.1 kW capacity, but both loads are >= 10 kW
        total_shots=100
    )

    # 11 requires 25 kW, which is > 0.1 kW -> Infeasible!
    assert feas_pct == 0.0
    assert best_candidate is None
