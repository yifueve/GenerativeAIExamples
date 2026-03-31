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
Supply Chain Results Analysis Tools — TraceLink

Tools for reading and analyzing supply chain simulation results.
Reads JSON output from inventory_sim.py and YAML scenario configs.

KPIs supported:
  order_fulfillment_rate, inventory_doh, shipment_delay,
  compliance_rate, partner_fill_rate, cost_per_unit, stockout_events
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)

_ALL_KPIS = [
    "order_fulfillment_rate",
    "inventory_doh_mean",
    "shipment_delay_mean",
    "compliance_rate",
    "cost_per_unit",
    "stockout_events",
]

_KPI_LABELS = {
    "order_fulfillment_rate": "Order Fulfillment Rate",
    "inventory_doh_mean": "Inventory Days-on-Hand (mean)",
    "shipment_delay_mean": "Shipment Delay — mean days",
    "compliance_rate": "DSCSA Compliance Rate",
    "cost_per_unit": "Cost per Unit ($)",
    "stockout_events": "Stockout Events (# nodes)",
}


def _find_results_json(case_path: str) -> Optional[Path]:
    """
    Find the results JSON file for a scenario.
    Accepts either:
      - A path to a *_results.json file directly, or
      - A path to a .yaml scenario config (looks for matching *_results.json in same dir).
    """
    p = Path(case_path)
    if p.suffix == ".json" and p.exists():
        return p
    if p.suffix in (".yaml", ".yml"):
        results = p.parent / f"{p.stem}_results.json"
        if results.exists():
            return results
        # fallback: any results JSON in same directory
        candidates = sorted(p.parent.glob("*_results.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    if p.is_dir():
        candidates = sorted(p.glob("*_results.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


def _safe_fmt(val: float, precision: int = 2) -> str:
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{precision}f}"


# ============================================================================
# Input Schemas
# ============================================================================


class ReadSimulationSummaryInput(BaseModel):
    case_path: str = Field(
        ...,
        description=(
            "Path to the scenario .yaml config file or *_results.json output file. "
            "Variables to extract (e.g. ['order_fulfillment_rate', 'inventory_doh_mean', "
            "'shipment_delay_mean', 'compliance_rate', 'cost_per_unit', 'stockout_events'])"
        ),
    )
    variables: List[str] = Field(
        ...,
        description=(
            "KPI names to extract. Options: order_fulfillment_rate, inventory_doh_mean, "
            "shipment_delay_mean, compliance_rate, cost_per_unit, stockout_events, "
            "partner_fill_rate:<node_id>"
        ),
    )
    entities: Optional[List[str]] = Field(
        default=None, description="Optional node IDs to filter partner_fill_rate results"
    )


class ReadGridPropertiesInput(BaseModel):
    case_path: str = Field(
        ...,
        description="Path to the scenario .yaml config file — returns network topology summary",
    )
    properties: List[str] = Field(
        ...,
        description=(
            "Network properties to extract. Options: nodes, lanes, products, "
            "disruptions, inventory_policy, compliance_config"
        ),
    )


# ============================================================================
# Tool Functions
# ============================================================================


@tool("read_simulation_summary", args_schema=ReadSimulationSummaryInput)
def read_simulation_summary(
    case_path: str,
    variables: List[str],
    entities: Optional[List[str]] = None,
) -> str:
    """
    Read KPI time-series data from a supply chain simulation results file.

    Accepts a .yaml scenario config path or a *_results.json output path.
    Returns per-step statistics (min, max, final) for each requested KPI.
    """
    try:
        results_path = _find_results_json(case_path)
        if results_path is None:
            return (
                f"Error: No simulation results found for {case_path}. "
                f"Run the scenario first with run_and_heal."
            )

        with open(results_path) as f:
            results = json.load(f)

        if not results.get("success"):
            errs = results.get("errors", [])
            return "Error: Simulation did not complete successfully:\n" + "\n".join(errs)

        kpis = results.get("kpis", {})
        node_results = results.get("node_results", {})
        steps = results.get("steps", 0)
        step_days = results.get("time_step_days", 7)
        horizon = results.get("horizon_days", 0)
        sc_name = results.get("scenario_name", "unknown")

        output = (
            f"✓ Scenario: {sc_name}\n"
            f"  Results from: {results_path.name}\n"
            f"  Horizon: {horizon} days ({steps} steps × {step_days} days)\n\n"
            f"KPI time-series:\n"
        )

        for var in variables:
            if var.startswith("partner_fill_rate:"):
                node_id = var.split(":", 1)[1]
                nr = node_results.get(node_id)
                if nr is None:
                    output += f"  {var}: node not found\n"
                    continue
                data = np.array(nr.get("partner_fill_rate", []))
                if data.size == 0:
                    output += f"  {var}: no data\n"
                    continue
                output += (
                    f"  Partner fill rate [{node_id}]: "
                    f"min={_safe_fmt(np.min(data))}, "
                    f"max={_safe_fmt(np.max(data))}, "
                    f"mean={_safe_fmt(np.mean(data))}, "
                    f"final={_safe_fmt(float(data[-1]))}\n"
                )
            elif var in kpis:
                data = np.array(kpis[var], dtype=float)
                label = _KPI_LABELS.get(var, var)
                output += (
                    f"  {label}: "
                    f"min={_safe_fmt(np.min(data))}, "
                    f"max={_safe_fmt(np.max(data))}, "
                    f"mean={_safe_fmt(np.mean(data))}, "
                    f"final={_safe_fmt(float(data[-1]))}\n"
                )
            else:
                output += f"  {var}: not found (available: {', '.join(list(kpis.keys()) + ['partner_fill_rate:<node_id>'])})\n"

        # append per-node fill rates if no explicit entities filter
        if entities:
            output += "\nFiltered partner fill rates:\n"
            for nid in entities:
                nr = node_results.get(nid)
                if nr is None:
                    output += f"  {nid}: not found\n"
                    continue
                data = np.array(nr.get("partner_fill_rate", []))
                if data.size > 0:
                    output += f"  {nid}: mean={_safe_fmt(np.mean(data))}, min={_safe_fmt(np.min(data))}\n"

        return output

    except Exception as e:
        logger.error("Error reading simulation results: %s", e)
        return f"Error reading simulation results: {e}"


@tool("read_grid_properties", args_schema=ReadGridPropertiesInput)
def read_grid_properties(case_path: str, properties: List[str]) -> str:
    """
    Read supply chain network topology from a scenario YAML config file.

    Returns a structured summary of the requested network properties:
    nodes (manufacturers, DCs, pharmacies), lanes (transport connections),
    products, disruptions, inventory policies, and compliance configuration.
    """
    try:
        config_path = Path(case_path)
        if not config_path.exists():
            return f"Error: Scenario config not found: {case_path}"
        if config_path.suffix not in (".yaml", ".yml"):
            return f"Error: Expected a .yaml scenario config file. Got: {case_path}"

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        sc = cfg.get("scenario", {})
        output = (
            f"✓ Network topology: {sc.get('name', 'unknown')}\n"
            f"  Horizon: {sc.get('horizon_days', '?')} days, "
            f"time step: {sc.get('time_step_days', '?')} days\n\n"
        )

        prop_set = set(p.lower() for p in properties)

        if "nodes" in prop_set or "all" in prop_set:
            nodes = cfg.get("nodes", [])
            output += f"Nodes ({len(nodes)}):\n"
            for n in nodes:
                ntype = n.get("type", "?")
                nid = n.get("id", "?")
                inv = n.get("initial_inventory", 0)
                if ntype == "manufacturer":
                    cap = n.get("production_capacity", "?")
                    output += f"  [{ntype}] {nid}: production_capacity={cap}/step, initial_inventory={inv}\n"
                elif ntype == "distribution_center":
                    rp = n.get("reorder_point", "?")
                    oq = n.get("order_quantity", "?")
                    output += f"  [{ntype}] {nid}: reorder_point={rp}, order_quantity={oq}, initial_inventory={inv}\n"
                elif ntype in ("pharmacy", "customer"):
                    mean = n.get("demand_mean", "?")
                    std = n.get("demand_std", "?")
                    rp = n.get("reorder_point", "?")
                    cr = n.get("dscsa_compliance_rate", 1.0)
                    output += f"  [{ntype}] {nid}: demand={mean}±{std}/step, reorder_point={rp}, compliance={cr:.0%}\n"
            output += "\n"

        if "lanes" in prop_set or "all" in prop_set:
            lanes = cfg.get("lanes", [])
            output += f"Lanes ({len(lanes)}):\n"
            for lane in lanes:
                lid = lane.get("id", f"{lane.get('from')}→{lane.get('to')}")
                lt = lane.get("lead_time_steps", "?")
                tc = lane.get("transport_cost_per_unit", "?")
                output += f"  {lid}: {lane.get('from')} → {lane.get('to')}, lead_time={lt} steps, transport_cost=${tc}/unit\n"
            output += "\n"

        if "products" in prop_set or "all" in prop_set:
            products = cfg.get("products", [])
            output += f"Products ({len(products)}):\n"
            for p in products:
                output += (
                    f"  {p.get('id', '?')}: unit_cost=${p.get('unit_cost', 0):.2f}, "
                    f"holding=${p.get('holding_cost_per_unit_day', 0):.3f}/unit/day, "
                    f"shortage_penalty=${p.get('shortage_cost_per_unit', 0):.2f}/unit\n"
                )
            output += "\n"

        if "disruptions" in prop_set or "all" in prop_set:
            disruptions = cfg.get("disruptions", [])
            output += f"Disruptions ({len(disruptions)}):\n"
            if not disruptions:
                output += "  None configured\n"
            for d in disruptions:
                output += (
                    f"  {d.get('type', '?')}: steps {d.get('start_step', '?')}–{d.get('end_step', '?')}, "
                    f"affects {d.get('affected_lane') or d.get('affected_node', '?')}\n"
                )
            output += "\n"

        if "compliance_config" in prop_set or "all" in prop_set:
            nodes = cfg.get("nodes", [])
            pharmacy_nodes = [n for n in nodes if n.get("type") in ("pharmacy", "customer")]
            output += "DSCSA Compliance Configuration:\n"
            for n in pharmacy_nodes:
                cr = n.get("dscsa_compliance_rate", 1.0)
                output += f"  {n['id']}: serialization_rate={cr:.1%}\n"
            if not pharmacy_nodes:
                output += "  No pharmacy/customer nodes found\n"

        return output

    except Exception as e:
        logger.error("Error reading network topology: %s", e)
        return f"Error reading network topology: {e}"
