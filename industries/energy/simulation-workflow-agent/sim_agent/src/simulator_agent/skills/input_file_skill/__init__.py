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
Supply Chain Scenario Config Skill — TraceLink

Provides tools for parsing, modifying, and patching supply chain scenario YAML config files.

Tools:
- parse_simulation_input_file: Parse and validate a scenario YAML config structure
- modify_simulation_input_file: Modify scenario config with natural language instructions
- patch_simulation_input_keyword: Patch a specific parameter block in a scenario config
"""

from .scripts.parse_tool import parse_simulation_input_file
from .scripts.modify_tool import modify_simulation_input_file
from .scripts.patch_tool import patch_simulation_input_keyword

__all__ = [
    "parse_simulation_input_file",
    "modify_simulation_input_file",
    "patch_simulation_input_keyword",
]
