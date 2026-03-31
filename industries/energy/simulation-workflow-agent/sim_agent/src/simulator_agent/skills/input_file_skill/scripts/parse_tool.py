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
Parse Supply Chain Scenario Config Tool — TraceLink

Parses and validates a YAML supply chain scenario config file.
First step in the scenario test chain:
  parse_scenario_config → tracelink_docs → modify_scenario_config → run_and_heal
"""

from pathlib import Path

import yaml
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field

from simulator_agent.skills.simulation_skill.scripts.inventory_sim import validate_config


class ParseDataFileInput(BaseModel):
    file_path: str = Field(
        ..., description="Path to the supply chain scenario YAML config file"
    )


@tool("parse_simulation_input_file", args_schema=ParseDataFileInput)
def parse_simulation_input_file(file_path: str) -> str:
    """
    Parse a supply chain scenario YAML config file and return its structure.

    Validates required sections (scenario, products, nodes, lanes) and reports
    node types, lane connections, products, and any configuration errors.

    First step in the scenario test chain:
    parse_simulation_input_file → tracelink_docs → modify_scenario_config → run_and_heal
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"
        if path.suffix not in (".yaml", ".yml"):
            return f"Error: Expected a .yaml scenario config. Got: {file_path}"

        with open(path) as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            return "Error: File is not a valid YAML mapping."

        # Validation
        errors = validate_config(cfg)

        sc = cfg.get("scenario", {})
        nodes = cfg.get("nodes", [])
        lanes = cfg.get("lanes", [])
        products = cfg.get("products", [])
        disruptions = cfg.get("disruptions", [])

        node_types: dict[str, list[str]] = {}
        for n in nodes:
            t = n.get("type", "unknown")
            node_types.setdefault(t, []).append(n.get("id", "?"))

        output = (
            f"Scenario config: {path.name}\n"
            f"  Name: {sc.get('name', '?')}\n"
            f"  Horizon: {sc.get('horizon_days', '?')} days, "
            f"time step: {sc.get('time_step_days', '?')} days\n\n"
            f"Sections found: scenario, products, nodes, lanes"
        )
        if disruptions:
            output += ", disruptions"
        output += "\n\n"

        output += f"Products ({len(products)}): " + ", ".join(p.get("id", "?") for p in products) + "\n"
        output += f"Nodes ({len(nodes)}):\n"
        for ntype, ids in node_types.items():
            output += f"  {ntype}: {', '.join(ids)}\n"
        output += f"Lanes ({len(lanes)}): "
        output += ", ".join(
            f"{l.get('from', '?')}→{l.get('to', '?')}" for l in lanes
        ) + "\n"

        if disruptions:
            output += f"Disruptions ({len(disruptions)}): " + ", ".join(
                d.get("type", "?") for d in disruptions
            ) + "\n"

        if errors:
            output += "\nValidation errors:\n" + "\n".join(f"  ✗ {e}" for e in errors)
        else:
            output += "\nValidation: ✓ No errors found"

        return output

    except yaml.YAMLError as e:
        return f"Error: Invalid YAML syntax: {e}"
    except Exception as e:
        return f"Error parsing scenario config: {e}"
