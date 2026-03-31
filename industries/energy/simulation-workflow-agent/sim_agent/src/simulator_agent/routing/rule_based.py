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
Rule-based query classification (fallback when LLM unavailable).
"""

from __future__ import annotations

from typing import Any


def classify_query_rule_based(query: str) -> dict[str, Any]:
    q = (query or "").strip().lower()
    if "run" in q and ("scenario" in q or "simulation" in q or "yaml" in q or "supply chain" in q):
        return {"skill": "simulation_skill", "tool": "run_and_heal", "tool_input": {}}
    if "plot" in q and "compare" in q:
        return {"skill": "plot_skill", "tool": "plot_compare_summary_metric", "tool_input": {}}
    if "plot" in q:
        return {"skill": "plot_skill", "tool": "plot_summary_metric", "tool_input": {}}
    if "parse" in q or "modify" in q or "validate" in q or "patch" in q:
        return {"skill": "input_file_skill", "tool": "parse_simulation_input_file", "tool_input": {}}
    if "docs" in q or "documentation" in q or "dscsa" in q or "compliance" in q or "regulation" in q or "search" in q:
        return {"skill": "rag_skill", "tool": "tracelink_docs", "tool_input": {}}
    if "result" in q or "summary" in q or "kpi" in q or "fulfillment" in q or "inventory" in q:
        return {"skill": "results_skill", "tool": "read_simulation_summary", "tool_input": {}}
    return {"skill": "final_response", "tool": None, "tool_input": {}}
