# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Capacitated Facility Location Problem (CFLP) solver for supply chain optimization.

Formulation:
  Sets:
    W: set of warehouses
    C: set of customers
    A ⊆ W×C: set of arcs (warehouse-customer pairs)

  Decision Variables:
    y_l ∈ {0,1}: 1 if warehouse l ∈ W is opened, 0 otherwise
    x_lj ≥ 0:    flow from warehouse l to customer j

  Parameters:
    γ_j:  demand of customer j
    c_lj: unit transportation cost from l to j
    β_l:  capacity of warehouse l
    f_l:  fixed cost of opening warehouse l

  Objective:
    min Σ_{l∈W} f_l·y_l + Σ_{(l,j)∈A} c_lj·x_lj

  Constraints:
    1. Demand:   Σ_{l∈W} x_lj ≥ γ_j        ∀j ∈ C
    2. Capacity: Σ_{j∈C} x_lj ≤ β_l·y_l    ∀l ∈ W
    3. Domains:  y_l ∈ {0,1}, x_lj ≥ 0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CFLPResult:
    success: bool
    total_cost: float
    fixed_cost: float
    transport_cost: float
    open_warehouses: list[str]
    flows: dict[tuple[str, str], float]
    message: str
    warehouse_ids: list[str] = field(default_factory=list)
    customer_ids: list[str] = field(default_factory=list)


def solve_cflp(config: dict[str, Any]) -> CFLPResult:
    """
    Solve the CFLP using scipy.optimize.milp.

    config dict keys:
      warehouses: list of {id, capacity, fixed_cost, location}
      customers:  list of {id, demand, location}
      arcs:       optional list of {warehouse, customer, transport_cost}
                  if omitted, all W×C pairs are used with transport_costs matrix
      transport_costs: dict keyed by "{warehouse_id}_{customer_id}" -> float
                       used when arcs not provided
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        raise ImportError("scipy >= 1.8 is required for CFLP. Install with: pip install scipy")

    warehouses = config["warehouses"]
    customers = config["customers"]

    W = [w["id"] for w in warehouses]
    C = [c["id"] for c in customers]
    nW = len(W)
    nC = len(C)

    f = np.array([w["fixed_cost"] for w in warehouses], dtype=float)
    beta = np.array([w["capacity"] for w in warehouses], dtype=float)
    gamma = np.array([c["demand"] for c in customers], dtype=float)

    # Build arc set and cost matrix
    arcs_cfg = config.get("arcs")
    transport_costs_cfg = config.get("transport_costs", {})

    # c_lj[l, j] = unit transport cost from warehouse l to customer j
    c_lj = np.zeros((nW, nC), dtype=float)
    arc_mask = np.zeros((nW, nC), dtype=bool)

    if arcs_cfg:
        for arc in arcs_cfg:
            l = W.index(arc["warehouse"])
            j = C.index(arc["customer"])
            c_lj[l, j] = arc["transport_cost"]
            arc_mask[l, j] = True
    else:
        # All W×C pairs are valid arcs
        arc_mask[:] = True
        for l, wid in enumerate(W):
            for j, cid in enumerate(C):
                key = f"{wid}_{cid}"
                c_lj[l, j] = transport_costs_cfg.get(key, 0.0)

    # Variable ordering: [y_0..y_{nW-1}, x_00, x_01, ..., x_{nW-1,nC-1}]
    # Total vars: nW + nW*nC
    n_vars = nW + nW * nC

    # Objective coefficients
    obj = np.zeros(n_vars)
    obj[:nW] = f
    for l in range(nW):
        for j in range(nC):
            if arc_mask[l, j]:
                obj[nW + l * nC + j] = c_lj[l, j]

    # Integrality: y vars are binary (1), x vars are continuous (0)
    integrality = np.zeros(n_vars)
    integrality[:nW] = 1

    # Bounds: y_l ∈ [0,1], x_lj ∈ [0, inf]
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    ub[:nW] = 1.0
    bounds = Bounds(lb=lb, ub=ub)

    # Constraints
    A_rows, b_lb, b_ub = [], [], []

    # 1. Demand: Σ_l x_lj ≥ γ_j  =>  lb=γ_j, ub=inf
    for j in range(nC):
        row = np.zeros(n_vars)
        for l in range(nW):
            if arc_mask[l, j]:
                row[nW + l * nC + j] = 1.0
        A_rows.append(row)
        b_lb.append(gamma[j])
        b_ub.append(np.inf)

    # 2. Capacity: Σ_j x_lj - β_l·y_l ≤ 0  =>  lb=-inf, ub=0
    for l in range(nW):
        row = np.zeros(n_vars)
        row[l] = -beta[l]
        for j in range(nC):
            if arc_mask[l, j]:
                row[nW + l * nC + j] = 1.0
        A_rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0.0)

    A_matrix = np.vstack(A_rows)
    constraints = LinearConstraint(A_matrix, b_lb, b_ub)

    result = milp(obj, constraints=constraints, integrality=integrality, bounds=bounds)

    if not result.success:
        return CFLPResult(
            success=False,
            total_cost=0.0,
            fixed_cost=0.0,
            transport_cost=0.0,
            open_warehouses=[],
            flows={},
            message=result.message,
            warehouse_ids=W,
            customer_ids=C,
        )

    y_vals = result.x[:nW]
    x_vals = result.x[nW:].reshape(nW, nC)

    open_warehouses = [W[l] for l in range(nW) if y_vals[l] > 0.5]
    flows = {
        (W[l], C[j]): x_vals[l, j]
        for l in range(nW)
        for j in range(nC)
        if arc_mask[l, j] and x_vals[l, j] > 1e-6
    }

    fixed_cost = float(np.dot(f, y_vals))
    transport_cost = float(sum(c_lj[l, j] * x_vals[l, j] for l in range(nW) for j in range(nC) if arc_mask[l, j]))
    total_cost = fixed_cost + transport_cost

    return CFLPResult(
        success=True,
        total_cost=total_cost,
        fixed_cost=fixed_cost,
        transport_cost=transport_cost,
        open_warehouses=open_warehouses,
        flows=flows,
        message="Optimal solution found",
        warehouse_ids=W,
        customer_ids=C,
    )


def print_cflp_results(result: CFLPResult, warehouses: list, customers: list) -> None:
    """Print CFLP results in supply chain format."""
    print(f"\n{'='*60}")
    print("SUPPLY CHAIN OPTIMIZATION RESULTS")
    print(f"{'='*60}")

    if not result.success:
        print(f"Optimization failed: {result.message}")
        return

    print(f"\nStatus: {result.message}")
    print(f"\nCost Summary:")
    print(f"  Fixed cost (warehouse opening): ${result.fixed_cost:,.2f}")
    print(f"  Transportation cost:            ${result.transport_cost:,.2f}")
    print(f"  Total cost:                     ${result.total_cost:,.2f}")

    wh_map = {w["id"]: w for w in warehouses}
    cu_map = {c["id"]: c for c in customers}

    print(f"\nOpen Warehouses ({len(result.open_warehouses)}/{len(result.warehouse_ids)}):")
    for wid in result.open_warehouses:
        w = wh_map.get(wid, {})
        loc = w.get("location", "")
        cap = w.get("capacity", "?")
        print(f"  {wid} — {loc} (capacity: {cap})")

    print(f"\nFlow Allocation (warehouse → customer):")
    for (wid, cid), flow in sorted(result.flows.items()):
        w = wh_map.get(wid, {})
        c = cu_map.get(cid, {})
        w_loc = w.get("location", wid)
        c_loc = c.get("location", cid)
        print(f"  {w_loc} → {c_loc}: {flow:.1f} units")

    print(f"\n{'='*60}\n")
