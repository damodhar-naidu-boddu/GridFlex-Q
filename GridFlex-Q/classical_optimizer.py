"""
GridFlex-Q Classical Optimization Engine
========================================
Exact baseline solver for emergency electrical load scheduling.
Solves the 0/1 Knapsack Problem formulated as:

    Maximize:   sum(p_i * x_i)
    Subject to: sum(w_i * x_i) <= C
                x_i in {0, 1}

where:
    w_i = power demand of load i (kW)
    p_i = priority score of load i (1 to 10)
    C   = available emergency supply capacity (kW)
    x_i = binary decision variable (1 = retain/power, 0 = defer/shed)

Deterministic tie-breaking:
    1. Higher objective score sum(p_i * x_i).
    2. Lower total power consumption sum(w_i * x_i) to maximize power reserve.
    3. Lexicographical order of selected load names.

Educational Disclaimer:
Maximizing priority score is the chosen educational objective function for
this demonstration and does not constitute a universally optimal policy for
all real-world critical infrastructure.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ClassicalOptimizationResult:
    """Structured result returned by the exact classical optimizer."""
    selected_loads: List[Dict[str, Any]]
    deferred_loads: List[Dict[str, Any]]
    total_selected_power: float
    remaining_capacity: float
    capacity_limit: float
    objective_score: float
    feasible: bool
    execution_time_ms: float
    candidates_evaluated: int
    method: str
    explanation: str
    bitstring: str
    selected_indices: List[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to dictionary."""
        return asdict(self)


MAX_BRUTE_FORCE_SIZE = 20


def validate_loads(loads: List[Dict[str, Any]]) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Validate load definitions.
    Returns (is_valid, error_message, cleaned_loads).
    """
    if not isinstance(loads, list):
        return False, "Loads input must be a list of dictionaries.", []

    cleaned = []
    for idx, load in enumerate(loads):
        if not isinstance(load, dict):
            return False, f"Load at index {idx} is not a valid dictionary.", []

        name = str(load.get("name", f"Load_{idx}")).strip()
        if not name:
            name = f"Load_{idx}"

        try:
            power = float(load.get("power", load.get("power_kw", 0.0)))
        except (ValueError, TypeError):
            return False, f"Invalid power value for '{name}'. Must be a positive number.", []

        if power < 0:
            return False, f"Power rating for '{name}' cannot be negative ({power} kW).", []

        try:
            priority = float(load.get("priority", 1.0))
        except (ValueError, TypeError):
            return False, f"Invalid priority for '{name}'. Must be a number between 1 and 10.", []

        if priority < 1 or priority > 10:
            return False, f"Priority for '{name}' must be between 1 and 10 (got {priority}).", []

        criticality = str(load.get("criticality", "Medium"))
        enabled = bool(load.get("enabled", True))

        cleaned.append({
            "name": name,
            "power": round(power, 2),
            "priority": round(priority, 1),
            "criticality": criticality,
            "enabled": enabled
        })

    return True, "", cleaned


def solve_classical(
    loads: List[Dict[str, Any]],
    capacity_kw: float,
    only_enabled: bool = True
) -> ClassicalOptimizationResult:
    """
    Find the exact optimal load combination via combinatorial search.

    Parameters:
    -----------
    loads: List of load dictionaries containing 'name', 'power', 'priority'.
    capacity_kw: Maximum available power capacity in kW.
    only_enabled: If True, filters out loads with enabled=False.

    Returns:
    --------
    ClassicalOptimizationResult containing selected/deferred loads, scores, and timings.
    """
    start_time = time.perf_counter()

    is_valid, err_msg, validated_loads = validate_loads(loads)
    if not is_valid:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ClassicalOptimizationResult(
            selected_loads=[],
            deferred_loads=loads if isinstance(loads, list) else [],
            total_selected_power=0.0,
            remaining_capacity=max(0.0, float(capacity_kw)),
            capacity_limit=float(capacity_kw),
            objective_score=0.0,
            feasible=False,
            execution_time_ms=round(elapsed_ms, 3),
            candidates_evaluated=0,
            method="Classical Exact (Failed Validation)",
            explanation=f"Validation error: {err_msg}",
            bitstring="",
            selected_indices=[]
        )

    # Active subset based on enabled flag
    active_loads: List[Tuple[int, Dict[str, Any]]] = []
    disabled_loads: List[Dict[str, Any]] = []

    for orig_idx, load in enumerate(validated_loads):
        if only_enabled and not load.get("enabled", True):
            disabled_loads.append(load)
        else:
            active_loads.append((orig_idx, load))

    n = len(active_loads)
    cap = max(0.0, float(capacity_kw))

    # Edge Case: Zero or negative capacity
    if cap <= 0.0 or n == 0:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        all_deferred = [l for _, l in active_loads] + disabled_loads
        explanation = (
            "Available capacity is 0 kW. All loads are safely deferred to avoid grid overload."
            if cap <= 0.0 else
            "No active loads were provided for scheduling."
        )
        return ClassicalOptimizationResult(
            selected_loads=[],
            deferred_loads=all_deferred,
            total_selected_power=0.0,
            remaining_capacity=0.0,
            capacity_limit=cap,
            objective_score=0.0,
            feasible=True,
            execution_time_ms=round(elapsed_ms, 3),
            candidates_evaluated=1 if n == 0 else 2**n,
            method="Classical Exact (Edge Case)",
            explanation=explanation,
            bitstring="0" * n if n > 0 else "0",
            selected_indices=[]
        )

    # Safety boundary for combinatorial brute force
    if n > MAX_BRUTE_FORCE_SIZE:
        # Fallback to exact dynamic programming if problem size exceeds brute-force limit
        return _solve_knapsack_dp(
            active_loads=active_loads,
            disabled_loads=disabled_loads,
            capacity_kw=cap,
            start_time=start_time
        )

    best_score = -1.0
    best_power = float("inf")
    best_indices: Tuple[int, ...] = ()
    best_names_tuple: Tuple[str, ...] = ()
    candidates_count = 0

    indices = list(range(n))

    # Evaluate all 2^n possible subsets
    for r in range(n + 1):
        for combo in itertools.combinations(indices, r):
            candidates_count += 1
            combo_power = sum(active_loads[i][1]["power"] for i in combo)

            if combo_power <= cap + 1e-9:  # Float tolerance
                combo_score = sum(active_loads[i][1]["priority"] for i in combo)
                combo_names = tuple(sorted(active_loads[i][1]["name"] for i in combo))

                # Deterministic tie-breaking rules:
                # 1. Higher score
                # 2. Lower power (more reserve margin)
                # 3. Lexicographical tie-breaker
                is_better = False
                if combo_score > best_score + 1e-9:
                    is_better = True
                elif abs(combo_score - best_score) <= 1e-9:
                    if combo_power < best_power - 1e-9:
                        is_better = True
                    elif abs(combo_power - best_power) <= 1e-9:
                        if combo_names < best_names_tuple:
                            is_better = True

                if is_better:
                    best_score = combo_score
                    best_power = combo_power
                    best_indices = combo
                    best_names_tuple = combo_names

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    selected_set = set(best_indices)
    selected_loads = [active_loads[i][1] for i in best_indices]
    deferred_loads = [active_loads[i][1] for i in range(n) if i not in selected_set] + disabled_loads

    # Build bitstring representation (length n, 1 if active load selected, 0 otherwise)
    bitstring = "".join("1" if i in selected_set else "0" for i in range(n))
    selected_orig_indices = [active_loads[i][0] for i in best_indices]

    total_power = round(sum(l["power"] for l in selected_loads), 2)
    rem_cap = round(max(0.0, cap - total_power), 2)
    final_score = round(sum(l["priority"] for l in selected_loads), 2)

    if selected_loads:
        names_str = ", ".join(f"{l['name']} ({l['power']} kW, P{l['priority']})" for l in selected_loads)
        explanation = (
            f"Selected {len(selected_loads)} of {n} active loads: {names_str}. "
            f"Total power demand is {total_power:.2f} kW with a remaining safety reserve of {rem_cap:.2f} kW "
            f"(Capacity: {cap:.2f} kW, Objective Score: {final_score:.1f})."
        )
    else:
        explanation = (
            f"No loads could be safely scheduled within the available capacity of {cap:.2f} kW. "
            "All loads deferred."
        )

    return ClassicalOptimizationResult(
        selected_loads=selected_loads,
        deferred_loads=deferred_loads,
        total_selected_power=total_power,
        remaining_capacity=rem_cap,
        capacity_limit=cap,
        objective_score=final_score,
        feasible=True,
        execution_time_ms=round(elapsed_ms, 3),
        candidates_evaluated=candidates_count,
        method="Classical Exact (Exhaustive Powerset)",
        explanation=explanation,
        bitstring=bitstring,
        selected_indices=selected_orig_indices
    )


def _solve_knapsack_dp(
    active_loads: List[Tuple[int, Dict[str, Any]]],
    disabled_loads: List[Dict[str, Any]],
    capacity_kw: float,
    start_time: float
) -> ClassicalOptimizationResult:
    """
    Exact 0/1 knapsack dynamic programming solver for larger load subsets.
    Uses discretized scaling (0.1 kW granularity).
    """
    scale = 10  # 0.1 kW granularity
    int_cap = int(round(capacity_kw * scale))
    n = len(active_loads)

    # dp[i][w] = max priority
    dp = [[0.0] * (int_cap + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        _, l = active_loads[i - 1]
        w_i = int(round(l["power"] * scale))
        p_i = l["priority"]
        for w in range(int_cap + 1):
            if w_i <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + p_i)
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack solution
    w = int_cap
    chosen_indices = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_indices.append(i - 1)
            _, l = active_loads[i - 1]
            w -= int(round(l["power"] * scale))

    chosen_indices.reverse()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    selected_set = set(chosen_indices)
    selected_loads = [active_loads[i][1] for i in chosen_indices]
    deferred_loads = [active_loads[i][1] for i in range(n) if i not in selected_set] + disabled_loads

    bitstring = "".join("1" if i in selected_set else "0" for i in range(n))
    selected_orig_indices = [active_loads[i][0] for i in chosen_indices]

    total_power = round(sum(l["power"] for l in selected_loads), 2)
    rem_cap = round(max(0.0, capacity_kw - total_power), 2)
    final_score = round(sum(l["priority"] for l in selected_loads), 2)

    explanation = (
        f"Dynamic programming solution selected {len(selected_loads)} loads with total power "
        f"{total_power:.2f} kW out of {capacity_kw:.2f} kW (Objective: {final_score:.1f})."
    )

    return ClassicalOptimizationResult(
        selected_loads=selected_loads,
        deferred_loads=deferred_loads,
        total_selected_power=total_power,
        remaining_capacity=rem_cap,
        capacity_limit=capacity_kw,
        objective_score=final_score,
        feasible=True,
        execution_time_ms=round(elapsed_ms, 3),
        candidates_evaluated=(n + 1) * (int_cap + 1),
        method="Classical Exact (Dynamic Programming)",
        explanation=explanation,
        bitstring=bitstring,
        selected_indices=selected_orig_indices
    )
