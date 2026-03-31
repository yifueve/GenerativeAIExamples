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
Tool argument validators.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from ..state import GlobalState, SkillUsed, PLOT_COMPARE_TOOL, PLOT_SUMMARY_TOOL, SIMULATION_INPUT_TOOLS
from .paths import output_dir_from_context
from .plot import infer_plot_metric_with_llm
from ..skills.rag_skill.scripts.extract_keyword import AmbiguousQueryError


def _extract_yaml_from_query(state: GlobalState) -> Optional[str]:
    """Scan the user's original query for a .yaml/.yml file path that exists on disk."""
    query = (state.get("user_input") or state.get("input") or "").strip()
    for match in re.finditer(r"[\w./\-\\]+\.(?:yaml|yml)", query, re.IGNORECASE):
        candidate = Path(match.group(0).strip())
        if candidate.exists():
            return str(candidate.resolve())
    return None


def valid_data_file_path(path_or_name: str, uploaded_files: list[str]) -> tuple[bool, Optional[str]]:
    if not (path_or_name or "").strip():
        return False, "No file path provided."
    s = path_or_name.strip()
    if not (s.upper().endswith(".DATA") or s.lower().endswith(".yaml") or s.lower().endswith(".yml")):
        return False, (
            "run_and_heal requires a scenario config file (.yaml) or simulator input file (.DATA). "
            "You provided a file that does not end with .yaml/.DATA (e.g. .pdf is not supported). "
            "Please upload a valid scenario config (.yaml) or simulator input file (.DATA)."
        )
    p = Path(s)
    if p.exists():
        return True, None
    for u in (uploaded_files or []):
        if u and Path(u).name == Path(s).name:
            return True, None
        sl = s.lower()
        ul = (u or "").strip().lower()
        if u and (ul.endswith(".data") or ul.endswith(".yaml") or ul.endswith(".yml")) and (s in u or Path(u).name == s):
            return True, None
    return False, (f"Scenario config or simulator input file not found: {s}. Please upload the file or provide a valid path.")


def _validate_run_and_heal(state: GlobalState, tool_input: dict) -> tuple[bool, Optional[str]]:
    uploaded = state.get("uploaded_files") or []
    data_file = (tool_input.get("data_file") or "").strip()
    if not data_file and uploaded:
        for u in uploaded:
            ul = (u or "").strip().lower()
            if ul.endswith(".data") or ul.endswith(".yaml") or ul.endswith(".yml"):
                tool_input["data_file"] = u.strip()
                data_file = u.strip()
                break
    if not data_file:
        return False, (
            "To run the scenario, please upload a supply chain scenario config (.yaml) "
            "or simulator input file (.DATA). No valid input file was provided or identified."
        )
    return valid_data_file_path(data_file, uploaded)


def _validate_rag(state: GlobalState, tool_input: dict) -> tuple[bool, Optional[str]]:
    user_input = (state.get("user_input") or state.get("input") or "").strip()
    tool_input["query"] = (tool_input.get("query") or user_input or "TraceLink supply chain documentation").strip()
    return True, None


def _validate_plot_summary(state: GlobalState, tool_input: dict) -> tuple[bool, Optional[str]]:
    data_file = (tool_input.get("data_file") or tool_input.get("file_path") or "").strip()
    if not data_file and (state.get("base_simulation_file") or "").strip():
        data_file = (state.get("base_simulation_file") or "").strip()
        tool_input["data_file"] = data_file
    out_dir = output_dir_from_context(state, tool_input, data_file=data_file)
    tool_input["output_dir"] = out_dir.strip("'\"") if out_dir else "."
    if not (tool_input.get("metric_request") or "").strip():
        user_input = (state.get("user_input") or state.get("input") or "").strip()
        try:
            inferred = infer_plot_metric_with_llm(user_input)
        except AmbiguousQueryError as e:
            return False, str(e)
        if inferred:
            tool_input["metric_request"] = inferred
        else:
            return False, (
                "Could not determine the requested metric from your query. "
                "Please specify the metric explicitly (e.g. FOPT, FOPR) or describe it clearly. "
            )
    return True, None


def _validate_plot_compare(state: GlobalState, tool_input: dict) -> tuple[bool, Optional[str]]:
    output_dir = (tool_input.get("output_dir") or "").strip()
    case_stems = (tool_input.get("case_stems") or "").strip()
    case_paths = (tool_input.get("case_paths") or "").strip()
    base_simulation_file = (state.get("base_simulation_file") or "").strip()
    uploaded = state.get("uploaded_files") or []

    # Derive output_dir from context (state, base file, or uploaded .DATA/.SMSPEC)
    out_dir = output_dir or output_dir_from_context(state, tool_input, data_file=base_simulation_file)
    if out_dir:
        out_dir = out_dir.strip("'\"")
    has_output_location = bool(out_dir and out_dir != ".")

    if not has_output_location:
        return False, (
            "To compare summary metrics, provide an output directory (output_dir) where .SMSPEC files are located, "
            "or upload .SMSPEC files / .DATA files that have corresponding .SMSPEC in the same directory. "
            "You can also run simulations first to generate results."
        )
        if not case_stems and not case_paths:
            agent_generated = [
                u for u in uploaded
                if (u or "").upper().endswith(".DATA") and "_AGENT_GENERATED" in (u or "")
            ]
            smspec_uploaded = [u for u in uploaded if (u or "").strip().upper().endswith(".SMSPEC")]
            data_uploaded = [u for u in uploaded if (u or "").strip().upper().endswith(".DATA")]
            if not agent_generated and not smspec_uploaded and not data_uploaded:
                return False, (
                    "To compare cases, you need either: (1) case_stems or case_paths, or "
                    "(2) uploaded .SMSPEC files, or (3) uploaded .DATA files (with .SMSPEC in same dir after run), "
                    "or (4) a base simulation and an uploaded scenario file (*_AGENT_GENERATED.DATA)."
                )
    return True, None


def _validate_plot_compare_full(state: GlobalState, tool_input: dict) -> tuple[bool, Optional[str]]:
    valid, err = _validate_plot_compare(state, tool_input)
    if not valid:
        return False, err
    data_file = (state.get("base_simulation_file") or "").strip()
    out_dir = output_dir_from_context(state, tool_input, data_file=data_file)
    tool_input["output_dir"] = out_dir.strip("'\"") if out_dir else "."

    # Populate case_paths from uploaded files when not provided
    if not (tool_input.get("case_paths") or "").strip() and not (tool_input.get("case_stems") or "").strip():
        uploaded = state.get("uploaded_files") or []
        smspec_files = [u.strip() for u in uploaded if (u or "").strip().upper().endswith(".SMSPEC")]
        data_files = [u.strip() for u in uploaded if (u or "").strip().upper().endswith(".DATA")]
        if smspec_files:
            tool_input["case_paths"] = ",".join(smspec_files)
        elif len(data_files) >= 2:
            tool_input["case_paths"] = ",".join(data_files)

    if not (tool_input.get("metric_request") or "").strip():
        user_input = (state.get("user_input") or state.get("input") or "").strip()
        try:
            inferred = infer_plot_metric_with_llm(user_input)
        except AmbiguousQueryError as e:
            return False, str(e)
        if inferred:
            tool_input["metric_request"] = inferred
        else:
            return False, (
                "Could not determine the requested metric from your query. "
                "Please specify the metric explicitly (e.g. FOPT, FOPR) or describe it clearly. "
            )
    return True, None


def _validate_simulation_input_tools(state: GlobalState, tool_input: dict) -> tuple[bool, Optional[str]]:
    data_file = (tool_input.get("data_file") or tool_input.get("file_path") or "").strip()
    if not data_file and state.get("uploaded_files"):
        for u in state.get("uploaded_files", []):
            ul = (u or "").strip().lower()
            if ul.endswith(".data") or ul.endswith(".yaml") or ul.endswith(".yml"):
                tool_input["data_file"] = tool_input.get("data_file") or u.strip()
                tool_input["file_path"] = tool_input.get("file_path") or u.strip()
                data_file = u.strip()
                break
    if not data_file:
        yaml_from_query = _extract_yaml_from_query(state)
        if yaml_from_query:
            tool_input["file_path"] = yaml_from_query
            tool_input["data_file"] = yaml_from_query
            data_file = yaml_from_query
    if not (tool_input.get("data_file") or tool_input.get("file_path")):
        return False, ("Tool requires a simulator input file (.DATA) or optimization config (.yaml). Please upload a file or provide file_path/data_file.")
    path = (tool_input.get("data_file") or tool_input.get("file_path") or "").strip()
    pl = path.lower()
    if path and not path.upper().endswith(".DATA") and not pl.endswith(".yaml") and not pl.endswith(".yml"):
        return False, (
            "Tool requires a simulator input file (.DATA) or optimization config (.yaml/.yml). "
            "The provided file does not have a supported extension."
        )
    return True, None


def _validate_run_flow_diagnostics(state: GlobalState, tool_input: dict) -> tuple[bool, Optional[str]]:
    case_path = (tool_input.get("case_path") or "").strip()
    if not case_path and state.get("uploaded_files"):
        for u in state.get("uploaded_files", []):
            if (u or "").strip().upper().endswith(".DATA"):
                tool_input["case_path"] = u.strip()
                case_path = u.strip()
                break
    if not case_path and (state.get("base_simulation_file") or "").strip():
        tool_input["case_path"] = (state.get("base_simulation_file") or "").strip()
        case_path = tool_input["case_path"]
    if not case_path:
        return False, (
            "Network diagnostics requires a scenario config path (.yaml) or results JSON path. "
            "Please upload a .yaml scenario file or provide case_path."
        )
    return True, None


_ValidatorFn = Callable[[GlobalState, dict], tuple[bool, Optional[str]]]

TOOL_VALIDATORS: dict[str, _ValidatorFn] = {
    "run_and_heal": _validate_run_and_heal,
    "simulator_manual": _validate_rag,
    "simulator_examples": _validate_rag,
    "tracelink_docs": _validate_rag,
    "dscsa_regulations": _validate_rag,
    PLOT_SUMMARY_TOOL: _validate_plot_summary,
    PLOT_COMPARE_TOOL: _validate_plot_compare_full,
    "run_flow_diagnostics": _validate_run_flow_diagnostics,
}
for _t in SIMULATION_INPUT_TOOLS:
    if _t != "run_and_heal":
        TOOL_VALIDATORS[_t] = _validate_simulation_input_tools


def validate_args_and_get_update(state: GlobalState) -> tuple[bool, Optional[str], list[SkillUsed]]:
    routing = list(state.get("routing_to") or [])
    if not routing:
        return False, "No skill/tool was selected.", routing

    r = routing[-1]
    tool_name = (r.get("tool_name") or "").strip()
    tool_input = dict(r.get("tool_input") or {})

    validator = TOOL_VALIDATORS.get(tool_name)
    if validator:
        valid, err = validator(state, tool_input)
        if not valid:
            return False, err, routing
        if tool_name in ("simulator_manual", "simulator_examples", "tracelink_docs", "dscsa_regulations"):
            tool_input["collection_name"] = tool_name
        routing[-1] = {**r, "tool_input": tool_input}
        return True, None, routing
    return True, None, routing
