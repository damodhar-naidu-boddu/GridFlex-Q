"""
GridFlex-Q Quantum Optimization Engine (QAOA)
==============================================
Executable Quantum Approximate Optimization Algorithm (QAOA) circuit
workflow implemented with Qiskit 2.x and executed locally on Qiskit Aer.

Mathematical QUBO / Ising Formulation:
---------------------------------------
For N electrical loads, binary decision variables x_i in {0, 1} represent:
    x_i = 1 (load i scheduled / powered)
    x_i = 0 (load i deferred / shed)

Objective:
    Maximize sum(p_i * x_i)  ==>  Minimize -sum(p_i * x_i)

Capacity Constraint via Quadratic Penalty:
    Penalty = lambda * (sum(w_i * x_i) - C)^2
    Expanding: (sum w_i x_i - C)^2 = sum_i (w_i^2 - 2*C*w_i) x_i + 2 * sum_{i < j} w_i * w_j * x_i * x_j + C^2

Cost Hamiltonian H_C mapping (x_i = (I - Z_i) / 2):
    H_C = sum_i h_i Z_i + sum_{i < j} J_ij Z_i Z_j + constant * I

QAOA Circuit Construction:
    1. Initial state: |+>^N via Hadamard gates on all qubits.
    2. For layer l = 1..p:
       - Phase separation unitary: U(H_C, gamma_l) via Rz and Rzz gates.
       - Mixer unitary: U(H_M, beta_l) via Rx gates on all qubits.
    3. Measurement of all qubits in computational basis.
    4. Execution on local Qiskit AerSimulator.
    5. Decoding bitstrings into candidate schedules and rigorous feasibility verification.

Simulation Distinction:
    Executed on local Qiskit Aer CPU/GPU simulator, NOT on physical quantum processing units (QPUs).
    No claims of quantum supremacy or advantage are asserted.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import classical_optimizer


@dataclass
class QuantumOptimizationResult:
    """Structured result returned by the QAOA quantum circuit simulation."""
    selected_loads: List[Dict[str, Any]]
    deferred_loads: List[Dict[str, Any]]
    total_selected_power: float
    remaining_capacity: float
    capacity_limit: float
    objective_score: float
    feasible: bool
    qaoa_depth: int
    num_qubits: int
    shots: int
    circuit_depth: int
    gate_counts: Dict[str, int]
    feasible_sample_percentage: float
    best_bitstring: str
    sampled_distribution: List[Dict[str, Any]]
    simulator_time_ms: float
    total_workflow_time_ms: float
    penalty_multiplier: float
    optimization_mode: str
    optimizer_details: Dict[str, Any]
    explanation: str
    ascii_circuit: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


def formulate_qubo_hamiltonian(
    loads: List[Dict[str, Any]],
    capacity_kw: float,
    penalty_multiplier: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Formulate the Cost Hamiltonian H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset.

    Returns:
    --------
    (h_vector, J_matrix, offset)
    """
    n = len(loads)
    if n == 0:
        return np.zeros(0), np.zeros((0, 0)), 0.0

    p = np.array([float(l.get("priority", 1.0)) for l in loads], dtype=float)
    w = np.array([float(l.get("power", 0.0)) for l in loads], dtype=float)
    C = max(0.0, float(capacity_kw))
    lam = float(penalty_multiplier)

    # Linear QUBO terms: alpha_i = - p_i + lam * (w_i^2 - 2 * C * w_i)
    alpha = -p + lam * (w**2 - 2.0 * C * w)

    # Quadratic QUBO terms: beta_ij = 2 * lam * w_i * w_j (for i < j)
    # Mapping x_i = (I - Z_i)/2 into QUBO:
    # x_i = 0.5 * I - 0.5 * Z_i
    # x_i * x_j = 0.25 * (I - Z_i - Z_j + Z_i * Z_j)

    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)
    offset = lam * (C**2)

    # Linear term contributions
    for i in range(n):
        offset += alpha[i] * 0.5
        h[i] -= alpha[i] * 0.5

    # Quadratic term contributions
    for i in range(n):
        for j in range(i + 1, n):
            beta_ij = 2.0 * lam * w[i] * w[j]
            offset += beta_ij * 0.25
            h[i] -= beta_ij * 0.25
            h[j] -= beta_ij * 0.25
            J[i, j] = beta_ij * 0.25
            J[j, i] = J[i, j]

    return h, J, offset


def build_qaoa_circuit(
    n_qubits: int,
    h_vector: np.ndarray,
    J_matrix: np.ndarray,
    gammas: List[float],
    betas: List[float],
    depth_p: int
) -> QuantumCircuit:
    """
    Construct the parameterized QAOA quantum circuit.

    Architecture:
    -------------
    1. Layer 0: Superposition via H gates.
    2. For layer l = 0 .. p-1:
       - Phase separation unitary: U(H_C, gamma_l)
         * Rz(2 * h_i * gamma) on each qubit i
         * Rzz(2 * J_ij * gamma) on each pair (i, j) with J_ij != 0
       - Mixer unitary: U(H_M, beta_l)
         * Rx(2 * beta_l) on each qubit i
    3. Measure all qubits into classical bits.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    if n_qubits == 0:
        return qc

    # 1. Initialization: Equal superposition
    for i in range(n_qubits):
        qc.h(i)

    # 2. Alternating QAOA operator layers
    for l in range(depth_p):
        gamma_l = gammas[l]
        beta_l = betas[l]

        # --- Phase Separation Unitary: U(H_C, gamma_l) ---
        # Single-qubit Z rotations
        for i in range(n_qubits):
            theta_z = 2.0 * h_vector[i] * gamma_l
            if abs(theta_z) > 1e-8:
                qc.rz(theta_z, i)

        # Two-qubit ZZ interactions
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                theta_zz = 2.0 * J_matrix[i, j] * gamma_l
                if abs(theta_zz) > 1e-8:
                    qc.rzz(theta_zz, i, j)

        # --- Mixer Unitary: U(H_M, beta_l) ---
        for i in range(n_qubits):
            theta_x = 2.0 * beta_l
            qc.rx(theta_x, i)

    # 3. Measurement in computational Z basis
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


def decode_qiskit_counts(
    counts: Dict[str, int],
    loads: List[Dict[str, Any]],
    capacity_kw: float,
    total_shots: int
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    """
    Decode raw Qiskit bitstring counts into load combinations, evaluate feasibility,
    and find the best feasible sampled schedule.

    Endianness Handling:
    --------------------
    In Qiskit's output dictionary, bitstrings are ordered little-endian:
    string 'b_{n-1}...b_1 b_0', where qubit 0 is the rightmost character.
    Reversing the string (bitstring[::-1]) maps character index i directly to load i.
    """
    distribution: List[Dict[str, Any]] = []
    feasible_shots = 0
    best_candidate: Optional[Dict[str, Any]] = None

    n = len(loads)

    for bitstring_raw, count in counts.items():
        # Clean bitstring and reverse to align with load index i
        clean_str = bitstring_raw.replace(" ", "")
        aligned_bits = clean_str[::-1]

        selected_indices = [i for i in range(min(n, len(aligned_bits))) if aligned_bits[i] == "1"]
        selected_loads = [loads[i] for i in selected_indices]

        total_power = round(sum(l["power"] for l in selected_loads), 2)
        total_score = round(sum(l["priority"] for l in selected_loads), 2)
        is_feasible = (total_power <= capacity_kw + 1e-6)

        probability = count / total_shots if total_shots > 0 else 0.0

        item = {
            "bitstring": aligned_bits,
            "raw_qiskit_bitstring": bitstring_raw,
            "count": count,
            "probability": round(probability, 4),
            "percentage": round(probability * 100.0, 2),
            "total_power_kw": total_power,
            "objective_score": total_score,
            "is_feasible": is_feasible,
            "selected_loads": selected_loads,
            "selected_indices": selected_indices
        }
        distribution.append(item)

        if is_feasible:
            feasible_shots += count
            if best_candidate is None:
                best_candidate = item
            else:
                # Deterministic comparison: higher score, then lower power
                if item["objective_score"] > best_candidate["objective_score"] + 1e-6:
                    best_candidate = item
                elif abs(item["objective_score"] - best_candidate["objective_score"]) <= 1e-6:
                    if item["total_power_kw"] < best_candidate["total_power_kw"] - 1e-6:
                        best_candidate = item

    # Sort distribution by count descending
    distribution.sort(key=lambda x: x["count"], reverse=True)
    feasible_pct = round((feasible_shots / total_shots) * 100.0, 2) if total_shots > 0 else 0.0

    return distribution, best_candidate, feasible_pct


def run_qaoa_simulation(
    loads: List[Dict[str, Any]],
    capacity_kw: float,
    depth_p: int = 1,
    shots: int = 1024,
    penalty_multiplier: float = 4.0,
    gamma_values: Optional[List[float]] = None,
    beta_values: Optional[List[float]] = None,
    optimize_parameters: bool = False,
    only_enabled: bool = True
) -> QuantumOptimizationResult:
    """
    Execute a real QAOA circuit on the local Qiskit Aer simulator.

    Parameters:
    -----------
    loads: List of electrical loads.
    capacity_kw: Power supply capacity in kW.
    depth_p: QAOA circuit depth / layers (typically 1 to 3).
    shots: Number of quantum circuit measurement repetitions.
    penalty_multiplier: Constraint penalty weight lambda in QUBO.
    gamma_values: Fixed or initial cost unitary rotation angles.
    beta_values: Fixed or initial mixer unitary rotation angles.
    optimize_parameters: If True, uses local classical optimizer outer-loop (COBYLA).
    only_enabled: Optimize only active loads.

    Returns:
    --------
    QuantumOptimizationResult containing sampled distribution, feasibility, and benchmarks.
    """
    total_start = time.perf_counter()

    # Filter active loads
    active_loads: List[Dict[str, Any]] = []
    disabled_loads: List[Dict[str, Any]] = []

    for l in loads:
        if only_enabled and not l.get("enabled", True):
            disabled_loads.append(l)
        else:
            active_loads.append(l)

    n_qubits = len(active_loads)
    cap = max(0.0, float(capacity_kw))
    depth_p = max(1, min(depth_p, 5))
    shots = max(100, min(shots, 8192))

    # Fallback default heuristic angles if not provided
    if gamma_values is None or len(gamma_values) < depth_p:
        # Standard heuristic initial angles for depth p
        gamma_values = [round(0.35 * (l + 1), 3) for l in range(depth_p)]
    else:
        gamma_values = gamma_values[:depth_p]

    if beta_values is None or len(beta_values) < depth_p:
        beta_values = [round(0.45 / (l + 1), 3) for l in range(depth_p)]
    else:
        beta_values = beta_values[:depth_p]

    # Special edge case: zero capacity or no active loads
    if n_qubits == 0 or cap <= 0.0:
        elapsed_total = (time.perf_counter() - total_start) * 1000.0
        all_deferred = active_loads + disabled_loads
        explanation = (
            "Zero capacity available (0 kW). Quantum optimizer safely defers all loads."
            if cap <= 0.0 else
            "No active loads available to schedule."
        )
        return QuantumOptimizationResult(
            selected_loads=[],
            deferred_loads=all_deferred,
            total_selected_power=0.0,
            remaining_capacity=0.0,
            capacity_limit=cap,
            objective_score=0.0,
            feasible=True,
            qaoa_depth=depth_p,
            num_qubits=n_qubits,
            shots=shots,
            circuit_depth=0,
            gate_counts={},
            feasible_sample_percentage=100.0 if n_qubits == 0 else 0.0,
            best_bitstring="0" * n_qubits if n_qubits > 0 else "0",
            sampled_distribution=[],
            simulator_time_ms=0.0,
            total_workflow_time_ms=round(elapsed_total, 3),
            penalty_multiplier=penalty_multiplier,
            optimization_mode="Edge Case Handling",
            optimizer_details={"status": "Immediate termination on edge condition"},
            explanation=explanation,
            ascii_circuit="No quantum circuit required for empty or zero-capacity scenario."
        )

    # 1. Formulate QUBO Hamiltonian
    h_vec, J_mat, offset = formulate_qubo_hamiltonian(
        loads=active_loads,
        capacity_kw=cap,
        penalty_multiplier=penalty_multiplier
    )

    optimizer_details: Dict[str, Any] = {
        "method": "Fixed-parameter QAOA demonstration (heuristic angles)" if not optimize_parameters else "COBYLA local optimizer",
        "iterations": 0,
        "initial_gammas": gamma_values,
        "initial_betas": beta_values
    }

    # 2. Parameter optimization (if enabled)
    final_gammas = list(gamma_values)
    final_betas = list(beta_values)

    if optimize_parameters and n_qubits <= 10:
        # Use simple scipy COBYLA optimizer with low evaluation count (15 iterations)
        # to find tuned angles without causing excessive laptop lag
        from scipy.optimize import minimize

        eval_sim = AerSimulator()
        eval_shots = 256  # Fewer shots during optimization loop

        def objective_fn(params: np.ndarray) -> float:
            cur_g = params[:depth_p].tolist()
            cur_b = params[depth_p:].tolist()
            qc_eval = build_qaoa_circuit(n_qubits, h_vec, J_mat, cur_g, cur_b, depth_p)
            counts_eval = eval_sim.run(qc_eval, shots=eval_shots).result().get_counts()

            # Calculate expected cost <H_C>
            total_cost = 0.0
            for b_str, cnt in counts_eval.items():
                bits = b_str.replace(" ", "")[::-1]
                x = np.array([1.0 if i < len(bits) and bits[i] == "1" else 0.0 for i in range(n_qubits)])
                # Cost function: -sum(p*x) + lam * (sum(w*x) - C)^2
                p_sum = sum(active_loads[i]["priority"] * x[i] for i in range(n_qubits))
                w_sum = sum(active_loads[i]["power"] * x[i] for i in range(n_qubits))
                cost = -p_sum + penalty_multiplier * ((w_sum - cap) ** 2)
                total_cost += cost * cnt
            return total_cost / eval_shots

        initial_x = np.array(gamma_values + beta_values, dtype=float)
        opt_start = time.perf_counter()
        opt_res = minimize(objective_fn, initial_x, method="COBYLA", options={"maxiter": 15, "tol": 1e-2})
        opt_time = (time.perf_counter() - opt_start) * 1000.0

        final_gammas = opt_res.x[:depth_p].tolist()
        final_betas = opt_res.x[depth_p:].tolist()

        optimizer_details.update({
            "iterations": int(opt_res.nfev),
            "converged": bool(opt_res.success),
            "optimized_gammas": [round(g, 4) for g in final_gammas],
            "optimized_betas": [round(b, 4) for b in final_betas],
            "optimization_time_ms": round(opt_time, 2),
            "stopping_condition": "Max evaluations (15) or tolerance reached"
        })

    # 3. Build Final QAOA Circuit
    qc = build_qaoa_circuit(
        n_qubits=n_qubits,
        h_vector=h_vec,
        J_matrix=J_mat,
        gammas=final_gammas,
        betas=final_betas,
        depth_p=depth_p
    )

    # Circuit metrics
    circuit_depth = qc.depth()
    ops = qc.count_ops()
    gate_counts = {str(k): int(v) for k, v in ops.items()}

    # Render circuit diagram as ASCII string
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            circuit_diagram_str = str(qc.draw(output="text"))
    except Exception:
        circuit_diagram_str = "ASCII circuit diagram omitted (console encoding limitation)."

    # 4. Execute on Local Qiskit Aer Simulator
    sim_start = time.perf_counter()
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    simulator_time_ms = round((time.perf_counter() - sim_start) * 1000.0, 3)

    # 5. Extract Measurement Counts
    counts = result.get_counts()

    # 6. Decode Bitstrings and Check Feasibility
    distribution, best_candidate, feasible_pct = decode_qiskit_counts(
        counts=counts,
        loads=active_loads,
        capacity_kw=cap,
        total_shots=shots
    )

    elapsed_total_ms = round((time.perf_counter() - total_start) * 1000.0, 3)

    # 7. Formulate Result
    if best_candidate is not None:
        sel_loads = best_candidate["selected_loads"]
        selected_set = set(best_candidate["selected_indices"])
        deferred_loads = [active_loads[i] for i in range(n_qubits) if i not in selected_set] + disabled_loads

        total_power = best_candidate["total_power_kw"]
        rem_cap = round(max(0.0, cap - total_power), 2)
        score = best_candidate["objective_score"]
        bitstr = best_candidate["bitstring"]

        names_str = ", ".join(f"{l['name']} (P{l['priority']}, {l['power']} kW)" for l in sel_loads)
        explanation = (
            f"QAOA circuit (depth p={depth_p}, {shots} shots) sampled {len(distribution)} unique configurations, "
            f"with {feasible_pct:.1f}% feasible shots. The best feasible schedule selects: {names_str}. "
            f"Power demand: {total_power:.2f} kW / {cap:.2f} kW (Reserve: {rem_cap:.2f} kW, Objective: {score:.1f})."
        )
        is_feasible = True
    else:
        # Honest reporting: circuit produced no feasible samples
        sel_loads = []
        deferred_loads = active_loads + disabled_loads
        total_power = 0.0
        rem_cap = cap
        score = 0.0
        bitstr = "None"
        is_feasible = False
        explanation = (
            f"No feasible schedule was sampled across all {shots} circuit measurements. "
            f"All candidate bitstrings exceeded the capacity of {cap:.2f} kW. "
            "Consider increasing penalty multiplier lambda, increasing QAOA depth p, or tuning rotation angles."
        )

    mode_label = (
        "Parameter-Optimized QAOA (COBYLA)" if optimize_parameters else
        f"Fixed-parameter QAOA demonstration (γ={[round(g, 3) for g in final_gammas]}, β={[round(b, 3) for b in final_betas]})"
    )

    return QuantumOptimizationResult(
        selected_loads=sel_loads,
        deferred_loads=deferred_loads,
        total_selected_power=total_power,
        remaining_capacity=rem_cap,
        capacity_limit=cap,
        objective_score=score,
        feasible=is_feasible,
        qaoa_depth=depth_p,
        num_qubits=n_qubits,
        shots=shots,
        circuit_depth=circuit_depth,
        gate_counts=gate_counts,
        feasible_sample_percentage=feasible_pct,
        best_bitstring=bitstr,
        sampled_distribution=distribution,
        simulator_time_ms=simulator_time_ms,
        total_workflow_time_ms=elapsed_total_ms,
        penalty_multiplier=penalty_multiplier,
        optimization_mode=mode_label,
        optimizer_details=optimizer_details,
        explanation=explanation,
        ascii_circuit=circuit_diagram_str
    )
