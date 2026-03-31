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
Supply Chain Simulation Skill — TraceLink

Provides tools for running, monitoring, and controlling pharma supply chain
inventory simulations using the discrete-time inventory_sim engine.

Tools:
- run_and_heal: Run scenario + auto-fix config errors on failure (primary run tool)
- run_simulation: Low-level runner (used internally by run_and_heal)
- monitor_simulation: Check simulation progress and read results
- stop_simulation: Stop a background simulation job by job ID
"""

from .scripts.simulation_tools import (
    run_simulation,
    monitor_simulation,
    stop_simulation,
)
from .scripts.self_heal_chain import run_and_heal

__all__ = [
    "run_and_heal",
    "run_simulation",
    "monitor_simulation",
    "stop_simulation",
]
