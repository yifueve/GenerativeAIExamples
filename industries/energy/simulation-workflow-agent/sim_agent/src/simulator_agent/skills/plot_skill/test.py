#!/usr/bin/env python3
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
Test suite for Plot Skill tools.

Tests all tools related to plotting:
- plot_transportation_assignment
- plot_supply_chain_kpis

Usage:
    python -m simulator_agent.skills.plot_skill.test
    python -m simulator_agent.skills.plot_skill.test --results-file path/to/results.json
    python -m simulator_agent.skills.plot_skill.test --tool plot_transportation_assignment --keep-plots
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on path so simulator_agent package is found when run as __main__
_skill_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_skill_root) not in sys.path:
    sys.path.insert(0, str(_skill_root))

from simulator_agent.skills.plot_skill import (
    plot_transportation_assignment,
    plot_supply_chain_kpis,
)

_DEFAULT_CFLP_RESULTS = (
    "/workspace/sim_agent/data/example_cases/supply_chain/results/"
    "CFLP_DrugY_results.json"
)
_DEFAULT_KPI_RESULTS = (
    "/workspace/sim_agent/data/example_cases/supply_chain/results/"
    "drugY_NE_disruption_results.json"
)


def test_plot_transportation_assignment(results_file: str, keep_plots: bool = False) -> bool:
    """Test plot_transportation_assignment tool."""
    print("\n" + "=" * 80)
    print("TEST: plot_transportation_assignment")
    print("=" * 80)

    results_path = Path(results_file)
    if not results_path.exists():
        print(f"⚠️  Results file not found: {results_file} — skipping (not a failure).")
        return True

    save_path = str(results_path.parent / "test_plot_transportation_assignment.png")
    try:
        result = plot_transportation_assignment.invoke({
            "results_file": results_file,
            "save_path": save_path,
        })
        print(f"✅ SUCCESS")
        print(f"Result:\n{result}")

        if Path(save_path).exists():
            print(f"✅ Plot file created: {save_path}")
            if not keep_plots:
                Path(save_path).unlink()
                print("   (Cleaned up test plot file)")
            else:
                print(f"   (Plot file kept at: {save_path})")
        else:
            print(f"⚠️  Plot file not found: {save_path}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plot_supply_chain_kpis(results_file: str, keep_plots: bool = False) -> bool:
    """Test plot_supply_chain_kpis tool."""
    print("\n" + "=" * 80)
    print("TEST: plot_supply_chain_kpis")
    print("=" * 80)

    results_path = Path(results_file)
    if not results_path.exists():
        print(f"⚠️  Results file not found: {results_file} — skipping (not a failure).")
        return True

    save_path = str(results_path.parent / "test_plot_supply_chain_kpis.png")
    try:
        result = plot_supply_chain_kpis.invoke({
            "results_file": results_file,
            "save_path": save_path,
        })
        print(f"✅ SUCCESS")
        print(f"Result:\n{result}")

        if Path(save_path).exists():
            print(f"✅ Plot file created: {save_path}")
            if not keep_plots:
                Path(save_path).unlink()
                print("   (Cleaned up test plot file)")
            else:
                print(f"   (Plot file kept at: {save_path})")
        else:
            print(f"⚠️  Plot file not found: {save_path}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(cflp_file: str, kpi_file: str, keep_plots: bool = False) -> None:
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PLOT SKILL TEST SUITE")
    print("=" * 80)

    results = [
        ("plot_transportation_assignment", test_plot_transportation_assignment(cflp_file, keep_plots)),
        ("plot_supply_chain_kpis", test_plot_supply_chain_kpis(kpi_file, keep_plots)),
    ]

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, r in results:
        print(f"{'✅ PASSED' if r else '❌ FAILED'}: {name}")
    print(f"\nTotal: {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Test Plot Skill tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m simulator_agent.skills.plot_skill.test
  python -m simulator_agent.skills.plot_skill.test --cflp-file /path/to/CFLP_results.json --kpi-file /path/to/disruption_results.json --keep-plots
  python -m simulator_agent.skills.plot_skill.test --tool plot_transportation_assignment --cflp-file /path/to/CFLP_results.json
        """,
    )
    parser.add_argument("--cflp-file", default=_DEFAULT_CFLP_RESULTS,
                        help="Path to CFLP optimization results JSON")
    parser.add_argument("--kpi-file", default=_DEFAULT_KPI_RESULTS,
                        help="Path to disruption simulation results JSON")
    parser.add_argument("--tool", choices=["plot_transportation_assignment", "plot_supply_chain_kpis"],
                        help="Run only a specific tool test")
    parser.add_argument("--keep-plots", action="store_true",
                        help="Keep plot files after test (default: delete them)")
    args = parser.parse_args()

    if args.tool == "plot_transportation_assignment":
        success = test_plot_transportation_assignment(args.cflp_file, args.keep_plots)
        sys.exit(0 if success else 1)
    elif args.tool == "plot_supply_chain_kpis":
        success = test_plot_supply_chain_kpis(args.kpi_file, args.keep_plots)
        sys.exit(0 if success else 1)
    else:
        run_all_tests(args.cflp_file, args.kpi_file, args.keep_plots)


if __name__ == "__main__":
    main()
