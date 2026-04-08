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
Simulation Tools for OPM Flow

Tools for running, monitoring, and parsing OPM Flow simulations.
"""

import json
import logging
import os
import re
import smtplib
import subprocess
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Optional

from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field


# ============================================================================
# Input Schemas
# ============================================================================


class RunSimulationInput(BaseModel):
    data_file: str = Field(..., description="Path to the simulator input (DATA) file")
    output_dir: Optional[str] = Field(
        default=None, description="Directory for output files (default: same as DATA file)"
    )
    num_threads: int = Field(
        default=1, description="Number of threads for parallel simulation"
    )
    additional_args: Optional[str] = Field(
        default=None, description="Additional command-line arguments"
    )
    background: bool = Field(
        default=True,
        description=(
            "If True, start flow and return immediately. "
            "If False, wait for completion and return full report: on failure includes return code, stdout, stderr, and parsed PRT errors/tail."
        ),
    )


class MonitorSimulationInput(BaseModel):
    output_dir: str = Field(..., description="Directory containing simulation output files")
    tail_lines: int = Field(
        default=80, description="Number of lines to read from end of PRT file"
    )
    use_llm_summary: bool = Field(
        default=False, description="Summarize the PRT tail with an LLM"
    )
    llm_model: Optional[str] = Field(
        default=None, description="Optional LLM model override for summary"
    )


class StopSimulationInput(BaseModel):
    pid: int = Field(..., description="Process ID to stop")
    output_dir: Optional[str] = Field(
        default=None, description="Optional output directory with run metadata"
    )
    force: bool = Field(default=False, description="Force kill the process")


class NotifyOnCompletionInput(BaseModel):
    output_dir: str = Field(..., description="Directory containing simulation output files")
    to_email: str = Field(..., description="Recipient email address")
    subject: Optional[str] = Field(
        default="Simulation completed",
        description="Email subject",
    )
    body: Optional[str] = Field(
        default=None, description="Optional email body override"
    )
    tail_lines: int = Field(
        default=80, description="Number of lines to read from end of PRT file"
    )


class PlotTransportationAssignmentInput(BaseModel):
    results_file: str = Field(
        ...,
        description=(
            "Path to a CFLP optimization results JSON file containing best_solution.open_warehouses "
            "and best_solution.flows (e.g. CFLP_DrugY_experiment_..._results.json)."
        ),
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Optional path to save the plot image (default: plot_transportation_assignment.png next to results_file).",
    )


class PlotSupplyChainKpisInput(BaseModel):
    results_file: str = Field(
        ...,
        description=(
            "Path to a supply chain disruption simulation results JSON file containing kpis and node_results "
            "(e.g. drugY_NE_disruption_results.json)."
        ),
    )
    metrics: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated KPI names to plot. Available: order_fulfillment_rate, inventory_doh_mean, "
            "shipment_delay_mean, compliance_rate, cost_per_unit, stockout_events. "
            "Defaults to all six."
        ),
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Optional path to save the plot image (default: plot_supply_chain_kpis.png next to results_file).",
    )


# ============================================================================
# Helper Functions
# ============================================================================

# When the agent runs in Docker, user paths (e.g. /home/user/.../sim_agent/data/...)
# don't exist in the container. Set OPM_PROJECT_ROOT in the container to the path to
# the sim_agent directory (e.g. /app/sim_agent); paths containing "sim_agent"
# will be resolved relative to it.
def _resolve_data_file_path(data_file: str) -> Path:
    """Resolve DATA file path, including host path -> container path when in Docker."""
    p = Path(data_file)
    if p.exists():
        return p.resolve()
    # Try relative to cwd (e.g. data/knowledge_base/examples/spe1/SPE1CASE1.DATA)
    cwd_candidate = (Path.cwd() / data_file).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    # In Docker: resolve host path to container path using OPM_PROJECT_ROOT
    project_root = os.environ.get("OPM_PROJECT_ROOT")
    if project_root:
        root = Path(project_root)
        # Strip host path prefix: .../sim_agent/data/... -> data/...
        if "sim_agent" in data_file:
            parts = data_file.split("sim_agent", 1)
            suffix = parts[-1].lstrip("/\\")
        elif "data" in data_file:
            parts = data_file.split("data", 1)
            suffix = "data" + parts[-1]
        elif data_file.startswith("/knowledge_base") or data_file.startswith("knowledge_base"):
            # LLM or user sometimes drops "data/" prefix; resolve as data/knowledge_base/...
            rest = data_file.split("knowledge_base", 1)[-1].lstrip("/\\")
            suffix = "data/knowledge_base/" + rest if rest else "data/knowledge_base"
        else:
            suffix = data_file
        container_candidate = (root / suffix).resolve()
        if container_candidate.exists():
            return container_candidate
    return p


def _parse_prt_file(prt_path: Path) -> Dict[str, any]:
    """
    Parse OPM .PRT file for progress and warnings.
    
    Returns dict with:
    - current_time: Current simulation time
    - total_time: Total simulation time
    - progress_pct: Progress percentage
    - warnings: List of warning messages
    - errors: List of error messages
    """
    if not prt_path.exists():
        return {
            "status": "not_started",
            "message": "PRT file not found"
        }
    
    try:
        with open(prt_path, 'r') as f:
            content = f.read()
        
        # Extract timestep information
        # This is a simplified parser - real implementation would be more robust
        lines = content.split('\n')
        
        current_time = None
        warnings = []
        errors = []
        
        for line in lines:
            # Look for timestep information
            if 'Time step' in line or 'Report step' in line:
                # Extract time information
                pass
            
            # Look for warnings
            if 'Warning' in line or 'WARNING' in line:
                warnings.append(line.strip())
            
            # Look for error messages (skip option names like ContinueOnConvergenceError="0")
            line_stripped = line.strip()
            line_upper = line.upper()
            is_option_key = bool(re.search(r"\w+Error\s*=", line) or re.search(r"\w+Error=\"", line))
            is_error_msg = (
                line_stripped.startswith("Error:")
                or " Error:" in line
                or (line_upper.startswith("ERROR") and ":" in line)
                or " Unrecoverable errors" in line_upper
                or "Error summary:" in line_upper
                or ("FATAL" in line_upper and "=" not in line)
                or ("EXCEPTION" in line_upper and "=" not in line)
                or re.search(r"\bfailed\b", line_upper)
            )
            if is_error_msg and not is_option_key:
                errors.append(line_stripped)
        
        return {
            "status": "running",
            "warnings": warnings[-10:] if warnings else [],
            "errors": errors[-20:] if errors else [],  # Keep more errors for failure report
            "message": f"Found {len(warnings)} warnings, {len(errors)} errors"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error parsing PRT file: {str(e)}"
        }


def _find_case_name(data_file: Path) -> str:
    """Extract case name from DATA file path."""
    # OPM uses the DATA file name (without extension) as the case name
    return data_file.stem


def _read_tail(file_path: Path, lines: int = 80) -> str:
    """Read last N lines from a text file without loading it fully."""
    if lines <= 0:
        return ""
    buffer_size = 4096
    data = bytearray()
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        position = f.tell()
        while position > 0 and data.count(b"\n") <= lines:
            read_size = min(buffer_size, position)
            position -= read_size
            f.seek(position)
            data = f.read(read_size) + data
        text = data.decode(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _write_run_metadata(output_dir: Path, metadata: Dict[str, object]) -> Path:
    """Write run metadata for background process tracking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "run.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return meta_path


def _is_simulation_complete(prt_tail_text: str) -> bool:
    return "End of simulation" in prt_tail_text or "End of Simulation" in prt_tail_text


def _send_email(to_email: str, subject: str, body: str) -> str:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_SENDER", username)
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    if not host or not sender:
        return "SMTP_HOST and SMTP_SENDER (or SMTP_USERNAME) must be set."

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=20) as server:
        if use_tls:
            server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(msg)

    return "sent"


# ============================================================================
# Tool Functions
# ============================================================================


@tool("run_simulation", args_schema=RunSimulationInput)
def run_simulation(
    data_file: str,
    output_dir: Optional[str] = None,
    num_threads: int = 1,
    additional_args: Optional[str] = None,
    background: bool = True,
    ) -> str:
    """
    Run a reservoir simulation.
    Use background=False to wait for completion and get a full error report (return code, stdout, stderr, PRT errors and tail) when the run fails.
    """
    try:
        data_file_path = _resolve_data_file_path(data_file)
        logging.getLogger(__name__).debug(
            "run_simulation: requested data_file=%s -> resolved=%s", data_file, data_file_path
        )
        if not data_file_path.exists():
            return f"Error: DATA file not found: {data_file}. (In Docker, set OPM_PROJECT_ROOT to the sim_agent path in the container.)"

        # Determine output directory
        if output_dir is None:
            output_dir = data_file_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = ["flow", str(data_file_path)]

        if output_dir != data_file_path.parent:
            cmd.append(f"--output-dir={str(output_dir)}")

        if num_threads > 1:
            cmd.extend(["--threads-per-process", str(num_threads)])

        if additional_args:
            cmd.extend(additional_args.split())
        # When running in Docker, Flow 2025.x may report "Saturation Function End-point
        # Consistency" (SOGCR slightly negative) for decks that run fine on host. Set
        # OPM_FLOW_EXTRA_ARGS="--CheckSatfuncConsistency=0" to relax the check.
        extra = os.environ.get("OPM_FLOW_EXTRA_ARGS")
        if extra:
            cmd.extend(extra.split())

        case_name = _find_case_name(data_file_path)

        # Run simulation
        if background:
            stdout_path = Path(output_dir) / f"{case_name}.stdout.log"
            stderr_path = Path(output_dir) / f"{case_name}.stderr.log"
            stdout_f = open(stdout_path, "w")
            stderr_f = open(stderr_path, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=data_file_path.parent,
            )
            # Close in parent to avoid file descriptor leaks
            stdout_f.close()
            stderr_f.close()
            meta_path = _write_run_metadata(
                Path(output_dir),
                {
                    "pid": proc.pid,
                    "case_name": case_name,
                    "data_file": str(data_file_path),
                    "output_dir": str(output_dir),
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "command": cmd,
                    "start_time": time.time(),
                },
            )
            # Check if the process exited immediately (e.g., bad args)
            return_code = proc.poll()
            if return_code is not None:
                stderr_tail = ""
                if stderr_path.exists():
                    with open(stderr_path, "r") as f:
                        stderr_tail = f.read()[-1000:]
                return f"""
✗ Simulation failed to start

Case: {case_name}
Return code: {return_code}
Output directory: {output_dir}
Stdout log: {stdout_path}
Stderr log: {stderr_path}
Run metadata: {meta_path}

Error output (tail):
{stderr_tail or "(empty)"}
"""
            return f"""
✓ Simulation started in background

Case: {case_name}
PID: {proc.pid}
Output directory: {output_dir}
Stdout log: {stdout_path}
Stderr log: {stderr_path}
Run metadata: {meta_path}

Check progress via the .PRT file.
"""

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=data_file_path.parent,
        )
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            return f"""
✓ Simulation completed successfully

Case: {case_name}
Data file run: {data_file_path}
Runtime: {elapsed_time:.1f} seconds
Output directory: {output_dir}

You can ask the assistant to plot results or read summary files.
"""
        # Non-background run failed: build a full report with return code, stdout, stderr, and PRT
        report_lines = [
            "✗ Simulation failed",
            "",
            f"Case: {case_name}",
            f"Data file run: {data_file_path}",
            f"Return code: {result.returncode}",
            f"Runtime: {elapsed_time:.1f} seconds",
            f"Output directory: {output_dir}",
            "",
            "--- Stdout (last 2000 chars) ---",
            (result.stdout or "")[-2000:].strip() or "(empty)",
            "",
            "--- Stderr (last 2000 chars) ---",
            (result.stderr or "")[-2000:].strip() or "(empty)",
        ]
        # Parse PRT file for errors and tail (prefer the one matching this run's case name)
        preferred_prt = Path(output_dir) / f"{case_name}.PRT"
        prt_files = list(Path(output_dir).glob("*.PRT"))
        if preferred_prt.exists():
            prt_path = preferred_prt
        elif prt_files:
            prt_path = prt_files[0]
        else:
            prt_path = None
        if prt_path is not None:
            prt_status = _parse_prt_file(prt_path)
            report_lines.append("")
            report_lines.append("--- PRT file summary ---")
            report_lines.append(f"PRT file: {prt_path.name}")
            if prt_status.get("errors"):
                report_lines.append("Errors found in PRT:")
                for err in prt_status["errors"]:
                    report_lines.append(f"  {err}")
            if prt_status.get("warnings"):
                report_lines.append("Warnings in PRT:")
                for w in prt_status["warnings"][-10:]:
                    report_lines.append(f"  {w}")
            prt_tail = _read_tail(prt_path, lines=50)
            if prt_tail:
                report_lines.append("")
                report_lines.append("--- PRT tail (last 50 lines) ---")
                report_lines.append(prt_tail)
        else:
            report_lines.append("")
            report_lines.append("(No .PRT file found in output directory.)")
        return "\n".join(report_lines)
    except FileNotFoundError:
        return """
Error: OPM Flow not found in PATH

Please ensure OPM Flow is installed and accessible.
- Ubuntu: sudo apt install opm-simulators
- From source: https://opm-project.org
- Check installation: which flow
"""
    except Exception as e:
        return f"Error running simulation: {str(e)}"
    

@tool("monitor_simulation", args_schema=MonitorSimulationInput)
def monitor_simulation(
    output_dir: str,
    tail_lines: int = 80,
    use_llm_summary: bool = False,
    llm_model: Optional[str] = None,
) -> str:
    """
    Monitor the progress of a running OPM Flow simulation.
    """
    try:
        output_path = Path(output_dir)

        if not output_path.exists():
            return f"Error: Output directory not found: {output_dir}"

        # Find .PRT file
        prt_files = list(output_path.glob("*.PRT"))

        if not prt_files:
            return f"No .PRT file found in {output_dir}. Simulation may not have started."

        prt_file = prt_files[0]

        # Parse PRT file
        status = _parse_prt_file(prt_file)

        # Read tail
        prt_tail_text = _read_tail(prt_file, lines=tail_lines)

        # Format report
        report = ["=== Simulation Progress ===\n"]
        report.append(f"Status: {status['status']}")
        report.append(f"PRT file: {prt_file.name}")
        report.append(f"Tail lines: {tail_lines}\n")

        if status.get("message"):
            report.append(status["message"])

        report.append("\n--- PRT Tail ---")
        report.append(prt_tail_text or "(empty)")

        if use_llm_summary:
            if not os.getenv("NVIDIA_API_KEY"):
                report.append("\nLLM summary skipped: NVIDIA_API_KEY not set.")
            else:
                from simulator_agent.config import get_config
                from llm_provider import ChatOpenAI

                model = llm_model or get_config().get_llm_model(use_for="tool")
                llm = ChatOpenAI(model=model, max_tokens=512)
                prompt = (
                    "Summarize the following OPM Flow PRT tail. "
                    "Focus on whether the run is progressing, and mention any errors "
                    "or convergence issues. Keep it concise.\n\n"
                    f"{prt_tail_text}"
                )
                summary = llm.invoke(prompt).content.strip()
                report.append("\n--- LLM Summary ---")
                report.append(summary or "(empty)")

        return "\n".join(report)

    except Exception as e:
        return f"Error monitoring simulation: {str(e)}"
    

@tool("stop_simulation", args_schema=StopSimulationInput)
def stop_simulation(
    pid: int, output_dir: Optional[str] = None, force: bool = False
) -> str:
    """
    Stop a running OPM Flow simulation by PID.
    """
    try:
        signal = "-9" if force else "-15"
        result = subprocess.run(
            ["kill", signal, str(pid)], capture_output=True, text=True
        )
        status = "terminated" if result.returncode == 0 else "not_terminated"

        meta_path = None
        if output_dir:
            meta_path = Path(output_dir) / "run.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
                metadata["stop_time"] = time.time()
                metadata["stop_signal"] = "SIGKILL" if force else "SIGTERM"
                metadata["stop_status"] = status
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)

        if result.returncode != 0:
            return (
                "Failed to stop process.\n"
                f"PID: {pid}\n"
                f"Error: {result.stderr.strip() or '(empty)'}"
            )

        return (
            "Simulation stop requested.\n"
            f"PID: {pid}\n"
            f"Signal: {'SIGKILL' if force else 'SIGTERM'}\n"
            f"Run metadata: {meta_path or '(not provided)'}"
        )

    except Exception as e:
        return f"Error stopping simulation: {str(e)}"


@tool("notify_on_completion", args_schema=NotifyOnCompletionInput)
def notify_on_completion(
    output_dir: str,
    to_email: str,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    tail_lines: int = 80,
) -> str:
    """
    Send an email notification if the simulation has completed.
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return f"Error: Output directory not found: {output_dir}"

        prt_files = list(output_path.glob("*.PRT"))
        if not prt_files:
            return f"No .PRT file found in {output_dir}. Simulation may not have started."

        prt_file = prt_files[0]
        prt_tail_text = _read_tail(prt_file, lines=tail_lines)
        if not _is_simulation_complete(prt_tail_text):
            return "Simulation not completed yet. No email sent."

        email_subject = subject or "Simulation completed"
        email_body = body or (
            "Simulation completed.\n\n"
            f"Output directory: {output_dir}\n"
            f"PRT file: {prt_file.name}\n\n"
            "PRT tail:\n"
            f"{prt_tail_text}"
        )
        result = _send_email(to_email, email_subject, email_body)
        if result != "sent":
            return f"Email not sent: {result}"

        return f"Email sent to {to_email}"

    except Exception as e:
        return f"Error sending notification: {str(e)}"


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
    "inventory_doh_mean": "Inventory DoH (days)",
    "shipment_delay_mean": "Shipment Delay (days)",
    "compliance_rate": "Compliance Rate",
    "cost_per_unit": "Cost per Unit ($)",
    "stockout_events": "Stockout Events",
}

# Node type colors for transportation network
_NODE_COLORS = {
    "MFG":   "#1f77b4",   # blue   – manufacturer
    "DC":    "#ff7f0e",   # orange – distribution centre
    "W":     "#ff7f0e",   # orange – warehouse (CFLP)
    "C":     "#2ca02c",   # green  – customer / pharmacy
    "PHARM": "#2ca02c",   # green
    # Battery critical minerals
    "MINE":  "#8c564b",   # brown  – mining source
    "REF":   "#17becf",   # teal   – refinery / processing hub
    "GIGA":  "#e377c2",   # pink   – battery gigafactory
}


def _open_plot(path: str) -> None:
    """Open a saved plot file with the OS default image viewer."""
    import subprocess, sys
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            subprocess.Popen(["start", path], shell=True)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def _node_color(name: str) -> str:
    for prefix, color in _NODE_COLORS.items():
        if name.upper().startswith(prefix):
            return color
    return "#9467bd"


@tool("plot_transportation_assignment", args_schema=PlotTransportationAssignmentInput)
def plot_transportation_assignment(
    results_file: str,
    save_path: Optional[str] = None,
) -> str:
    """
    Plot the supply chain transportation assignment from a CFLP optimization results JSON.
    Draws a bipartite network with open warehouses on the left and customers on the right,
    with edges labelled by flow volume and a bar chart of flows below.
    """
    try:
        import json
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        results_path = Path(results_file)
        if not results_path.exists():
            return f"Error: Results file not found: {results_file}"

        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        best = data.get("best_solution", {})
        open_warehouses = best.get("open_warehouses", [])
        flows: Dict[str, float] = best.get("flows", {})
        total_cost = best.get("objective") or data.get("best_objective")
        opt_summary = data.get("optimization_summary", {})

        if not flows:
            return "Error: No flow assignments found in results file."

        # Parse flows into (warehouse, customer, volume)
        edges = []
        customers = []
        for key, vol in flows.items():
            if "->" in key:
                src, dst = key.split("->", 1)
                edges.append((src.strip(), dst.strip(), float(vol)))
                if dst.strip() not in customers:
                    customers.append(dst.strip())

        warehouses = open_warehouses or sorted({e[0] for e in edges})
        customers = sorted(customers)

        fig, (ax_net, ax_bar) = plt.subplots(
            1, 2, figsize=(14, max(6, max(len(warehouses), len(customers)) * 1.2))
        )

        # ── Network diagram ────────────────────────────────────────────────
        wh_y = {w: i for i, w in enumerate(warehouses)}
        cu_y = {c: i for i, c in enumerate(customers)}
        max_vol = max(v for _, _, v in edges) if edges else 1.0

        for w, c, vol in edges:
            y0 = wh_y[w]
            y1 = cu_y[c]
            lw = 1.0 + 4.0 * vol / max_vol
            ax_net.plot([0, 1], [y0, y1], lw=lw, alpha=0.6, color="#888888")
            ax_net.text(0.5, (y0 + y1) / 2, f"{vol:,.0f}", ha="center", va="bottom",
                        fontsize=7, color="#444444")

        for w, y in wh_y.items():
            ax_net.scatter(0, y, s=300, color=_node_color(w), zorder=5)
            ax_net.text(-0.05, y, w, ha="right", va="center", fontsize=9)
        for c, y in cu_y.items():
            ax_net.scatter(1, y, s=300, color=_node_color(c), zorder=5)
            ax_net.text(1.05, y, c, ha="left", va="center", fontsize=9)

        ax_net.set_xlim(-0.5, 1.5)
        ax_net.set_ylim(-0.8, max(len(warehouses), len(customers)) - 0.2)
        ax_net.axis("off")
        ax_net.set_title("Transportation Network", fontsize=11, fontweight="bold")

        # Legend
        legend_handles = [
            mpatches.Patch(color=_node_color("W"), label="Warehouse / DC"),
            mpatches.Patch(color=_node_color("C"), label="Customer"),
        ]
        ax_net.legend(handles=legend_handles, loc="lower center", fontsize=8)

        # ── Bar chart of flows ─────────────────────────────────────────────
        edge_labels = [f"{w}\n→{c}" for w, c, _ in edges]
        vols = [v for _, _, v in edges]
        colors = [_node_color(w) for w, _, _ in edges]
        x = np.arange(len(edges))
        bars = ax_bar.bar(x, vols, color=colors, edgecolor="white", linewidth=0.5)
        ax_bar.bar_label(bars, labels=[f"{v:,.0f}" for v in vols], padding=3, fontsize=8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(edge_labels, fontsize=8, rotation=30, ha="right")
        ax_bar.set_ylabel("Flow Volume (units)")
        ax_bar.set_title("Flow Volumes by Assignment", fontsize=11, fontweight="bold")
        ax_bar.grid(axis="y", alpha=0.3)

        # Cost annotation
        cost_lines = []
        if opt_summary.get("fixed_cost") is not None:
            cost_lines.append(f"Fixed cost:     ${opt_summary['fixed_cost']:,.0f}")
        if opt_summary.get("transport_cost") is not None:
            cost_lines.append(f"Transport cost: ${opt_summary['transport_cost']:,.0f}")
        if total_cost is not None:
            cost_lines.append(f"Total cost:     ${total_cost:,.0f}")
        if cost_lines:
            ax_bar.text(
                0.98, 0.97, "\n".join(cost_lines),
                transform=ax_bar.transAxes, ha="right", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.8),
            )

        scenario = results_path.stem
        fig.suptitle(f"Transportation Assignment: {scenario}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out_path = save_path or str(results_path.parent / "plot_transportation_assignment.png")
        out_path = str(Path(out_path).resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        _open_plot(out_path)
        return f"Transportation assignment plot saved to: {out_path}"

    except Exception as e:
        return f"Error plotting transportation assignment: {str(e)}"


@tool("plot_supply_chain_kpis", args_schema=PlotSupplyChainKpisInput)
def plot_supply_chain_kpis(
    results_file: str,
    metrics: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Plot supply chain KPI time series from a disruption simulation results JSON.
    Renders a multi-panel figure with one subplot per requested KPI over the simulation horizon.
    Available KPIs: order_fulfillment_rate, inventory_doh_mean, shipment_delay_mean,
    compliance_rate, cost_per_unit, stockout_events.
    """
    try:
        import json
        import matplotlib.pyplot as plt

        results_path = Path(results_file)
        if not results_path.exists():
            return f"Error: Results file not found: {results_file}"

        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        kpis = data.get("kpis", {})
        if not kpis:
            return "Error: No kpis section found in results file."

        days = kpis.get("day") or kpis.get("step")
        if days is None:
            return "Error: kpis must contain a 'day' or 'step' array for the time axis."

        requested = [m.strip() for m in metrics.split(",")] if metrics else _ALL_KPIS
        available = [m for m in requested if m in kpis and m not in ("day", "step")]
        missing = [m for m in requested if m not in kpis and m not in ("day", "step")]
        if not available:
            return (
                f"Error: None of the requested metrics found in kpis. "
                f"Available: {', '.join(k for k in kpis if k not in ('day', 'step'))}"
            )

        n = len(available)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows), squeeze=False)

        kpi_colors = {
            "order_fulfillment_rate": "#2ca02c",
            "inventory_doh_mean":     "#1f77b4",
            "shipment_delay_mean":    "#d62728",
            "compliance_rate":        "#9467bd",
            "cost_per_unit":          "#8c564b",
            "stockout_events":        "#e377c2",
        }

        for idx, metric in enumerate(available):
            ax = axes[idx // ncols][idx % ncols]
            values = kpis[metric]
            color = kpi_colors.get(metric, "#7f7f7f")
            if metric == "stockout_events":
                ax.bar(days, values, color=color, alpha=0.75, width=4)
            else:
                ax.plot(days, values, marker="o", markersize=4, color=color, linewidth=1.8)
                ax.fill_between(days, values, alpha=0.1, color=color)
            ax.set_xlabel("Day")
            ax.set_ylabel(_KPI_LABELS.get(metric, metric))
            ax.set_title(_KPI_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        scenario = data.get("scenario_name", results_path.stem)
        summary = data.get("summary", {})
        subtitle = (
            f"OFR mean={summary.get('order_fulfillment_rate_mean', 'N/A'):.3f}  "
            f"DoH mean={summary.get('inventory_doh_mean', 'N/A'):.1f}d  "
            f"Cost/unit=${summary.get('cost_per_unit_mean', 'N/A'):.0f}"
        ) if summary else ""
        fig.suptitle(f"Supply Chain KPIs — {scenario}\n{subtitle}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        out_path = save_path or str(results_path.parent / "plot_supply_chain_kpis.png")
        out_path = str(Path(out_path).resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        _open_plot(out_path)

        result_msg = f"Supply chain KPI plot saved to: {out_path}"
        if missing:
            result_msg += f"\n(Skipped unavailable metrics: {', '.join(missing)})"
        return result_msg

    except Exception as e:
        return f"Error plotting supply chain KPIs: {str(e)}"
