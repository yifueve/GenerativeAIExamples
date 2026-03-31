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
Supply Chain Simulation Tools — TraceLink

Tools for running, monitoring, and controlling pharma supply chain inventory simulations.
The simulation engine is inventory_sim.py (discrete-time, multi-echelon (s,Q) model).
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field

from .inventory_sim import run_from_yaml, validate_config
import yaml

logger = logging.getLogger(__name__)

# Thread-local registry for background simulation jobs
_running_sims: dict[int, dict] = {}
_sim_lock = threading.Lock()
_next_pid = 1


# ============================================================================
# Input Schemas
# ============================================================================


class RunSimulationInput(BaseModel):
    data_file: str = Field(
        ...,
        description="Path to the supply chain scenario YAML config file (e.g. scenario.yaml)",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory for output files (default: same directory as config file)",
    )
    background: bool = Field(
        default=False,
        description=(
            "If True, start simulation and return immediately with a job ID. "
            "If False (default), wait for completion and return full results summary."
        ),
    )


class MonitorSimulationInput(BaseModel):
    output_dir: str = Field(
        ..., description="Directory containing simulation output files (results JSON)"
    )
    job_id: Optional[int] = Field(
        default=None, description="Background job ID returned by run_simulation (optional)"
    )


class StopSimulationInput(BaseModel):
    pid: int = Field(..., description="Job ID to stop (returned by run_simulation in background mode)")


# ============================================================================
# Tool Functions
# ============================================================================


def _format_summary(results: dict) -> str:
    """Format simulation results summary for agent output."""
    summary = results.get("summary", {})
    sc_name = results.get("scenario_name", "unknown")
    horizon = results.get("horizon_days", 0)
    steps = results.get("steps", 0)

    pfr = summary.get("partner_fill_rates", {})
    pfr_lines = "\n".join(
        f"    {nid}: {rate:.1%}" for nid, rate in pfr.items()
    )

    return (
        f"✓ Simulation complete: {sc_name}\n"
        f"  Horizon: {horizon} days ({steps} time steps)\n\n"
        f"  KPI Summary:\n"
        f"    Order fulfillment rate (mean): {summary.get('order_fulfillment_rate_mean', 0):.1%}\n"
        f"    Order fulfillment rate (min):  {summary.get('order_fulfillment_rate_min', 0):.1%}\n"
        f"    Inventory days-on-hand (mean): {summary.get('inventory_doh_mean', 0):.1f} days\n"
        f"    Inventory days-on-hand (min):  {summary.get('inventory_doh_min', 0):.1f} days\n"
        f"    Shipment delay (mean):         {summary.get('shipment_delay_mean_days', 0):.1f} days\n"
        f"    DSCSA compliance rate (mean):  {summary.get('compliance_rate_mean', 0):.1%}\n"
        f"    Cost per unit (mean):          ${summary.get('cost_per_unit_mean', 0):.2f}\n"
        f"    Total stockout events:         {summary.get('total_stockout_events', 0)}\n\n"
        f"  Partner fill rates:\n{pfr_lines}\n"
    )


@tool("run_simulation", args_schema=RunSimulationInput)
def run_simulation(
    data_file: str,
    output_dir: Optional[str] = None,
    background: bool = False,
) -> str:
    """
    Run a supply chain inventory simulation from a YAML scenario config file.

    The simulation models multi-echelon pharma supply chains using a discrete-time
    (s, Q) reorder-point policy. Reports 7 KPIs: order fulfillment rate,
    inventory days-on-hand, shipment delay, DSCSA compliance rate,
    partner fill rate, cost-per-unit, and stockout events.

    Primary config format: YAML scenario file (see example_cases/supply_chain/).
    """
    global _next_pid

    config_path = Path(data_file)
    if not config_path.exists():
        return f"Error: Scenario config not found: {data_file}"
    if not data_file.lower().endswith((".yaml", ".yml")):
        return f"Error: Scenario config must be a .yaml file. Got: {data_file}"

    # Validate config before running
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        errs = validate_config(cfg)
        if errs:
            return "Error: Invalid scenario config:\n" + "\n".join(f"  - {e}" for e in errs)
    except Exception as e:
        return f"Error reading scenario config: {e}"

    resolved_output_dir = output_dir or str(config_path.parent)

    if not background:
        try:
            results = run_from_yaml(data_file, resolved_output_dir)
            if not results.get("success"):
                errs = results.get("errors", ["Unknown error"])
                return "Simulation failed:\n" + "\n".join(f"  - {e}" for e in errs)
            out = _format_summary(results)
            if results.get("results_path"):
                out += f"\n  Results saved to: {results['results_path']}"
            return out
        except Exception as e:
            logger.error("Simulation error: %s", e)
            return f"Error running simulation: {e}"
    else:
        with _sim_lock:
            job_id = _next_pid
            _next_pid += 1
            _running_sims[job_id] = {"status": "running", "data_file": data_file, "output_dir": resolved_output_dir}

        def _run():
            try:
                results = run_from_yaml(data_file, resolved_output_dir)
                with _sim_lock:
                    _running_sims[job_id]["status"] = "done" if results.get("success") else "failed"
                    _running_sims[job_id]["results"] = results
            except Exception as e:
                with _sim_lock:
                    _running_sims[job_id]["status"] = "failed"
                    _running_sims[job_id]["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return (
            f"Simulation started in background (job_id={job_id}).\n"
            f"  Config: {data_file}\n"
            f"  Output dir: {resolved_output_dir}\n"
            f"Use monitor_simulation with output_dir or job_id={job_id} to check status."
        )


@tool("monitor_simulation", args_schema=MonitorSimulationInput)
def monitor_simulation(output_dir: str, job_id: Optional[int] = None) -> str:
    """
    Check the status of a supply chain simulation.

    If job_id is provided, returns live status from the background job registry.
    Otherwise, scans output_dir for results JSON files and reports what is available.
    """
    if job_id is not None:
        with _sim_lock:
            job = _running_sims.get(job_id)
        if job is None:
            return f"No simulation job found with job_id={job_id}."
        status = job.get("status", "unknown")
        if status == "running":
            return f"Job {job_id}: still running. Config: {job.get('data_file', '?')}"
        elif status == "done":
            results = job.get("results", {})
            return f"Job {job_id}: completed.\n\n" + _format_summary(results)
        elif status == "failed":
            err = job.get("error") or str(job.get("results", {}).get("errors", "unknown"))
            return f"Job {job_id}: FAILED. Error: {err}"
        return f"Job {job_id}: status={status}"

    out_path = Path(output_dir)
    if not out_path.exists():
        return f"Output directory not found: {output_dir}"

    result_files = list(out_path.glob("*_results.json"))
    if not result_files:
        return f"No results JSON found in {output_dir}. Simulation may still be running or not yet started."

    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest) as f:
            results = json.load(f)
        return f"Results found: {latest.name}\n\n" + _format_summary(results)
    except Exception as e:
        return f"Error reading results file {latest}: {e}"


@tool("stop_simulation", args_schema=StopSimulationInput)
def stop_simulation(pid: int) -> str:
    """
    Stop a background supply chain simulation job by its job ID.
    """
    with _sim_lock:
        job = _running_sims.get(pid)
        if job is None:
            return f"No simulation job found with job_id={pid}."
        if job.get("status") != "running":
            return f"Job {pid} is not running (status={job.get('status')})."
        job["status"] = "stopped"
    return f"Job {pid} marked for stop. (Note: discrete simulation steps are fast; the current step will complete before stopping.)"
