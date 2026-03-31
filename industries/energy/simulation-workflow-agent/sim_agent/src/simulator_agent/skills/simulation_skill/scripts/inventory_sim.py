# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TraceLink Supply Chain Inventory Simulation Engine

Discrete-time (s, Q) inventory simulation for pharma supply chain scenarios.
Simulates orders, shipments, and demand fulfillment across a multi-echelon network.

Tracks 7 KPIs per time step:
  - order_fulfillment_rate   : units fulfilled / units demanded (network-wide)
  - inventory_doh            : days-on-hand per node (inventory / avg daily demand)
  - shipment_delay           : mean delay vs planned lead time for arrivals that step
  - compliance_rate          : DSCSA serialization rate across received shipments
  - partner_fill_rate        : per-node fill rate (units fulfilled / units demanded)
  - cost_per_unit            : (holding + transport + shortage) / units fulfilled
  - stockout_events          : count of nodes with zero inventory

YAML scenario schema — see data/example_cases/supply_chain/ for examples.
"""

from __future__ import annotations

import json
import logging
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML schema validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = {"scenario", "products", "nodes", "lanes"}


def validate_config(cfg: dict) -> list[str]:
    """Return list of validation errors; empty list means config is valid."""
    errors: list[str] = []
    missing = _REQUIRED_TOP_KEYS - set(cfg.keys())
    if missing:
        errors.append(f"Missing required top-level keys: {sorted(missing)}")
        return errors

    sc = cfg.get("scenario", {})
    for k in ("name", "horizon_days", "time_step_days"):
        if k not in sc:
            errors.append(f"scenario.{k} is required")
    if sc.get("horizon_days", 0) <= 0:
        errors.append("scenario.horizon_days must be > 0")
    if sc.get("time_step_days", 0) <= 0:
        errors.append("scenario.time_step_days must be > 0")

    products = cfg.get("products") or []
    if not products:
        errors.append("At least one product is required")
    product_ids = set()
    for p in products:
        pid = p.get("id")
        if not pid:
            errors.append("Each product must have an 'id'")
        else:
            product_ids.add(pid)
        if p.get("unit_cost", 0) < 0:
            errors.append(f"product {pid}: unit_cost must be >= 0")

    node_ids = set()
    for n in (cfg.get("nodes") or []):
        nid = n.get("id")
        if not nid:
            errors.append("Each node must have an 'id'")
        else:
            node_ids.add(nid)
        if n.get("type") not in ("manufacturer", "distribution_center", "pharmacy"):
            errors.append(f"node {nid}: type must be manufacturer | distribution_center | pharmacy")
        if n.get("initial_inventory", 0) < 0:
            errors.append(f"node {nid}: initial_inventory must be >= 0")

    for lane in (cfg.get("lanes") or []):
        lid = lane.get("id", "?")
        if lane.get("from") not in node_ids:
            errors.append(f"lane {lid}: 'from' node '{lane.get('from')}' not defined in nodes")
        if lane.get("to") not in node_ids:
            errors.append(f"lane {lid}: 'to' node '{lane.get('to')}' not defined in nodes")
        if lane.get("lead_time_steps", 0) < 1:
            errors.append(f"lane {lid}: lead_time_steps must be >= 1")

    return errors


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _build_lane_map(lanes: list[dict]) -> dict[str, dict]:
    """lane_id -> lane config dict; also index by (from, to)."""
    m: dict[str, dict] = {}
    for lane in lanes:
        lid = lane.get("id") or f"{lane['from']}_to_{lane['to']}"
        lane = dict(lane)
        lane["id"] = lid
        m[lid] = lane
    return m


def run_inventory_simulation(cfg: dict) -> dict[str, Any]:
    """
    Run the discrete-time inventory simulation.

    Args:
        cfg: Parsed YAML scenario config dict.

    Returns:
        Results dict with keys:
            scenario_name, horizon_days, time_step_days, steps,
            kpis (per-step time series),
            node_results (per-node per-step inventory + fill rate),
            summary (scalar aggregates of each KPI),
            errors (list of validation errors, empty if successful).
    """
    errors = validate_config(cfg)
    if errors:
        return {"errors": errors, "success": False}

    sc = cfg["scenario"]
    horizon_days: int = int(sc["horizon_days"])
    step_days: int = int(sc["time_step_days"])
    n_steps: int = math.ceil(horizon_days / step_days)
    rng = np.random.default_rng(int(cfg.get("output", {}).get("random_seed", 42)))

    # --- product lookup (use first product for cost params) ---
    products = cfg["products"]
    prod = products[0]  # primary product; multi-product extension is straightforward
    unit_cost: float = float(prod.get("unit_cost", 0.0))
    holding_cost_day: float = float(prod.get("holding_cost_per_unit_day", 0.0))
    shortage_cost_unit: float = float(prod.get("shortage_cost_per_unit", 0.0))

    # --- node state ---
    nodes = cfg["nodes"]
    node_map: dict[str, dict] = {n["id"]: deepcopy(n) for n in nodes}
    inventory: dict[str, float] = {nid: float(n.get("initial_inventory", 0)) for nid, n in node_map.items()}

    # --- lane lookup ---
    lanes = cfg.get("lanes") or []
    lane_map = _build_lane_map(lanes)
    # downstream lanes per supplier node: supplier -> list of lanes
    supplier_lanes: dict[str, list[dict]] = {}
    for lane in lane_map.values():
        supplier_lanes.setdefault(lane["from"], []).append(lane)

    # --- disruption state ---
    disruptions = cfg.get("disruptions") or []

    # --- in-transit shipments: list of dicts ---
    # {lane_id, quantity, remaining_steps, planned_steps, compliance_rate}
    in_transit: list[dict] = []

    # --- result containers ---
    kpis: dict[str, list] = {
        "step": [],
        "day": [],
        "order_fulfillment_rate": [],
        "inventory_doh_mean": [],
        "shipment_delay_mean": [],
        "compliance_rate": [],
        "cost_per_unit": [],
        "stockout_events": [],
    }
    node_results: dict[str, dict[str, list]] = {
        nid: {"inventory": [], "demand": [], "fulfilled": [], "partner_fill_rate": []}
        for nid in node_map
    }

    for step in range(n_steps):
        day = step * step_days

        # --- apply disruptions ---
        active_disruptions = [
            d for d in disruptions
            if d.get("start_step", 0) <= step <= d.get("end_step", n_steps)
        ]
        lane_delay: dict[str, int] = {}
        capacity_reductions: dict[str, float] = {}
        demand_multipliers: dict[str, float] = {}
        for d in active_disruptions:
            dtype = d.get("type", "")
            if dtype == "port_congestion" and d.get("affected_lane"):
                lane_delay[d["affected_lane"]] = lane_delay.get(d["affected_lane"], 0) + int(d.get("lead_time_delay_steps", 1))
            elif dtype == "supplier_shortage" and d.get("affected_node"):
                capacity_reductions[d["affected_node"]] = float(d.get("capacity_reduction", 0.5))
            elif dtype == "demand_spike" and d.get("affected_node"):
                demand_multipliers[d["affected_node"]] = float(d.get("demand_multiplier", 1.5))

        # --- generate demand at pharmacy/customer nodes ---
        step_demand: dict[str, float] = {}
        step_fulfilled: dict[str, float] = {}
        for nid, n in node_map.items():
            if n.get("type") in ("pharmacy", "customer"):
                mean = float(n.get("demand_mean", 0)) * demand_multipliers.get(nid, 1.0)
                std = float(n.get("demand_std", 0))
                demand = max(0.0, float(rng.normal(mean, std)))
                step_demand[nid] = demand
                fulfilled = min(demand, inventory[nid])
                step_fulfilled[nid] = fulfilled
                inventory[nid] = max(0.0, inventory[nid] - fulfilled)
            else:
                step_demand[nid] = 0.0
                step_fulfilled[nid] = 0.0

        # --- advance in-transit shipments and receive arrivals ---
        new_in_transit: list[dict] = []
        arrivals: list[dict] = []
        for s in in_transit:
            s = dict(s)
            s["remaining_steps"] -= 1
            if s["remaining_steps"] <= 0:
                arrivals.append(s)
            else:
                new_in_transit.append(s)
        in_transit = new_in_transit

        # receive arrivals: add to destination inventory
        arrival_delays: list[float] = []
        arrival_compliance: list[float] = []
        for arrival in arrivals:
            dest = lane_map[arrival["lane_id"]]["to"]
            inventory[dest] = min(
                inventory[dest] + arrival["quantity"],
                float(node_map[dest].get("storage_capacity", 1e9)),
            )
            delay = arrival["planned_steps"] - (arrival["planned_steps"] - (arrival.get("extra_delay", 0)))
            arrival_delays.append(float(arrival.get("extra_delay", 0)) * step_days)
            arrival_compliance.append(float(arrival.get("compliance_rate", 1.0)))

        # --- place orders: reorder-point policy ---
        for nid, n in node_map.items():
            rp = float(n.get("reorder_point", 0))
            oq = float(n.get("order_quantity", 0))
            if oq <= 0:
                continue
            if inventory[nid] < rp:
                # find upstream lane(s)
                upstream_lanes = [
                    lane for lane in lane_map.values() if lane["to"] == nid
                ]
                if not upstream_lanes:
                    continue
                lane = upstream_lanes[0]
                supplier = lane["from"]
                cap_factor = 1.0 - capacity_reductions.get(supplier, 0.0)
                available_supply = inventory.get(supplier, 0.0) * cap_factor
                if available_supply <= 0:
                    continue
                shipped = min(oq, available_supply)
                inventory[supplier] = max(0.0, inventory[supplier] - shipped)
                planned_steps = int(lane.get("lead_time_steps", 1))
                extra_delay = lane_delay.get(lane["id"], 0)
                actual_steps = planned_steps + extra_delay
                compliance = float(node_map[nid].get("dscsa_compliance_rate", 1.0))
                in_transit.append({
                    "lane_id": lane["id"],
                    "quantity": shipped,
                    "remaining_steps": actual_steps,
                    "planned_steps": planned_steps,
                    "extra_delay": extra_delay,
                    "compliance_rate": compliance,
                })

        # --- manufacturer replenishment ---
        for nid, n in node_map.items():
            if n.get("type") == "manufacturer":
                cap = float(n.get("production_capacity", 0))
                cap_factor = 1.0 - capacity_reductions.get(nid, 0.0)
                inventory[nid] = min(
                    inventory[nid] + cap * cap_factor,
                    float(n.get("storage_capacity", 1e9)),
                )

        # --- compute KPIs for this step ---
        total_demand = sum(step_demand.values())
        total_fulfilled = sum(step_fulfilled.values())
        fulfillment_rate = (total_fulfilled / total_demand) if total_demand > 0 else 1.0

        # inventory days-on-hand per node (pharmacy nodes only, where demand > 0)
        doh_values: list[float] = []
        for nid, n in node_map.items():
            mean_demand_day = float(n.get("demand_mean", 0)) / step_days
            if mean_demand_day > 0:
                doh_values.append(inventory[nid] / mean_demand_day)
        inventory_doh_mean = float(np.mean(doh_values)) if doh_values else 0.0

        shipment_delay_mean = float(np.mean(arrival_delays)) if arrival_delays else 0.0
        compliance_rate = float(np.mean(arrival_compliance)) if arrival_compliance else 1.0

        # cost per unit
        holding_costs = sum(inventory[nid] * holding_cost_day * step_days for nid in node_map)
        transport_costs = sum(
            arrival["quantity"] * lane_map[arrival["lane_id"]].get("transport_cost_per_unit", 0.0)
            for arrival in arrivals
        )
        shortage_costs = (total_demand - total_fulfilled) * shortage_cost_unit
        total_cost = holding_costs + transport_costs + shortage_costs
        cost_per_unit = (total_cost / total_fulfilled) if total_fulfilled > 0 else total_cost

        stockout_events = sum(1 for nid in node_map if inventory[nid] <= 0)

        kpis["step"].append(step)
        kpis["day"].append(day)
        kpis["order_fulfillment_rate"].append(round(fulfillment_rate, 4))
        kpis["inventory_doh_mean"].append(round(inventory_doh_mean, 2))
        kpis["shipment_delay_mean"].append(round(shipment_delay_mean, 2))
        kpis["compliance_rate"].append(round(compliance_rate, 4))
        kpis["cost_per_unit"].append(round(cost_per_unit, 4))
        kpis["stockout_events"].append(int(stockout_events))

        for nid in node_map:
            node_results[nid]["inventory"].append(round(inventory[nid], 2))
            d = step_demand.get(nid, 0.0)
            f = step_fulfilled.get(nid, 0.0)
            node_results[nid]["demand"].append(round(d, 2))
            node_results[nid]["fulfilled"].append(round(f, 2))
            node_results[nid]["partner_fill_rate"].append(round(f / d, 4) if d > 0 else 1.0)

    # --- summary scalars ---
    summary = {
        "order_fulfillment_rate_mean": round(float(np.mean(kpis["order_fulfillment_rate"])), 4),
        "order_fulfillment_rate_min": round(float(np.min(kpis["order_fulfillment_rate"])), 4),
        "inventory_doh_mean": round(float(np.mean(kpis["inventory_doh_mean"])), 2),
        "inventory_doh_min": round(float(np.min(kpis["inventory_doh_mean"])), 2),
        "shipment_delay_mean_days": round(float(np.mean(kpis["shipment_delay_mean"])), 2),
        "compliance_rate_mean": round(float(np.mean(kpis["compliance_rate"])), 4),
        "cost_per_unit_mean": round(float(np.mean([c for c in kpis["cost_per_unit"] if c > 0])), 4) if any(c > 0 for c in kpis["cost_per_unit"]) else 0.0,
        "total_stockout_events": int(sum(kpis["stockout_events"])),
        "partner_fill_rates": {
            nid: round(float(np.mean([r for r in node_results[nid]["partner_fill_rate"]])), 4)
            for nid in node_map
        },
    }

    return {
        "success": True,
        "errors": [],
        "scenario_name": sc["name"],
        "horizon_days": horizon_days,
        "time_step_days": step_days,
        "steps": n_steps,
        "kpis": kpis,
        "node_results": node_results,
        "summary": summary,
    }


def run_from_yaml(config_path: str, output_dir: str | None = None) -> dict[str, Any]:
    """
    Load a YAML scenario config, run the simulation, save results JSON.

    Args:
        config_path: Path to the .yaml scenario file.
        output_dir: Directory to save results JSON (default: same directory as config).

    Returns:
        Results dict (same as run_inventory_simulation).
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        return {"success": False, "errors": [f"Config file not found: {config_path}"]}

    with open(config_path_obj) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        return {"success": False, "errors": ["Config file is not a valid YAML mapping"]}

    results = run_inventory_simulation(cfg)

    if results.get("success"):
        out_dir = Path(output_dir) if output_dir else config_path_obj.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = config_path_obj.stem
        results_path = out_dir / f"{stem}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        results["results_path"] = str(results_path)
        LOG.info("Simulation results saved to %s", results_path)

    return results
