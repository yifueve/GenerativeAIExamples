#!/usr/bin/env python
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
Main optimization script with YAML configuration support

Usage:
    python run_optimization.py [config_file]
    
    If no config file is provided, uses config.yaml by default
    
Examples:
    python run_optimization.py                      # Use config.yaml
    python run_optimization.py config_simple.yaml   # Quick test
    python run_optimization.py config_thorough.yaml # Thorough search
"""

import sys
import argparse
from pathlib import Path

# Add src and parent directories to path
_src_dir = Path(__file__).parent
_parent_dir = _src_dir.parent
sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_parent_dir))

# Import from same directory
from config_loader import load_config
from orchestrator import run_optimization


def _is_cflp(config_path: str) -> bool:
    """Return True if the config specifies algorithm.type == CFLP."""
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return (cfg.get("algorithm") or {}).get("type", "").upper() == "CFLP"


def run_cflp(config_path: str) -> None:
    """Load CFLP config, solve, and write results JSON for the workflow agent."""
    import json
    import yaml
    from cflp_solver import solve_cflp, print_cflp_results

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    warehouses = cfg["warehouses"]
    customers = cfg["customers"]

    print(f"\n{'='*60}")
    print("STARTING SUPPLY CHAIN OPTIMIZATION (CFLP)")
    print(f"{'='*60}\n")
    scenario_title = cfg.get("scenario_title", "Supply Chain Optimization")
    print(f"  Scenario   : {scenario_title}")
    print(f"  Warehouses : {[w['id'] for w in warehouses]}")
    print(f"  Customers  : {[c['id'] for c in customers]}\n")

    result = solve_cflp(cfg)
    print_cflp_results(result, warehouses, customers)

    # Write results JSON so the workflow agent executor can parse it
    # Subprocess runs with run_cwd=./workflow, so write relative to cwd
    paths = cfg.get("paths", {})
    sim_dir_base = paths.get("sim_dir_base", "dataset/CFLP")
    case_name = paths.get("case_name", "CFLP_DrugY")
    sim_dir = Path.cwd() / sim_dir_base / f"{case_name}.sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    results_file = sim_dir / f"{case_name}_results.json"
    results_data = {
        "best_solution": {
            "objective": result.total_cost if result.success else None,
            "open_warehouses": result.open_warehouses,
            "flows": {f"{k[0]}->{k[1]}": v for k, v in result.flows.items()},
        },
        "best_objective": result.total_cost if result.success else None,
        "optimization_summary": {
            "total_evaluations": 1,
            "status": result.message,
            "fixed_cost": result.fixed_cost,
            "transport_cost": result.transport_cost,
            "total_cost": result.total_cost,
        },
        "success": result.success,
    }
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {results_file}")


def main():
    """Main entry point for configuration-based optimization"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Reservoir optimization with genetic algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimization.py                        # Use default config.yaml
  python run_optimization.py config_simple.yaml     # Quick test
  python run_optimization.py config_thorough.yaml   # Thorough search
  
Available config files:
  config.yaml          - Production optimization (20 pop, 5 gen)
  config_simple.yaml   - Quick test (4 pop, 2 gen)
  config_thorough.yaml - Thorough search (40 pop, 20 gen)
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running optimization'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n{'='*60}")
    print(f"LOADING CONFIGURATION (forecast + robust optimization)")
    print(f"{'='*60}\n")

    # Route to CFLP solver if configured
    if _is_cflp(args.config):
        run_cflp(args.config)
        return

    try:
        config = load_config(args.config)
        config.print_summary()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Dry run - just print config and exit
    if args.dry_run:
        print("Dry run - configuration loaded successfully!")
        print("Remove --dry-run flag to start optimization.")
        sys.exit(0)

    # Run optimization
    print(f"\n{'='*60}")
    print(f"STARTING OPTIMIZATION")
    print(f"{'='*60}\n")
    
    # Determine run name (deterministic)
    config_dict = config.to_dict()
    case_name = None
    if 'case_name' in config_dict:
        case_name = config_dict['case_name']
    run_name = config.run_name or case_name or Path(args.config).stem
    
    # Create simulation directory
    sim_dir_base = config_dict.pop('sim_dir_base')
    case_name = config_dict.pop('case_name')
    sim_dir = Path(sim_dir_base) / f"{case_name}.sim"
    config_dict['sim_dir'] = str(sim_dir)
    print(f"Simulation directory: {sim_dir}")
    
    # Update run_name and timeout in config_dict
    config_dict['run_name'] = run_name
    config_dict['timeout'] = config.timeout
    print(f"Run name: {run_name}\n")
    
    try:
        result = run_optimization(**config_dict)
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Legacy simple results file (for backwards compatibility)
    result_file = Path(sim_dir) / "optimization_results.txt"
    with open(result_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("OPTIMIZATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration file: {args.config}\n")
        f.write(f"Run name: {run_name}\n\n")
        f.write(f"Objective: Maximize {config.objective_mode}\n")
        f.write(f"Population size: {config.pop_size}\n")
        f.write(f"Generations: {config.n_generations}\n")
        f.write(f"Total evaluations: {config.pop_size * config.n_generations}\n\n")
        f.write("Best Solution:\n")
        for var_name, var_value in zip(result.problem.var_names, result.X):
            f.write(f"  {var_name}: {int(var_value)}\n")
        f.write(f"\nBest {config.objective_mode}: {-result.F[0]:.2f}\n\n")
        f.write("-"*60 + "\n")
        f.write("For comprehensive results, see:\n")
        f.write(f"  - {run_name}_results.json (full results with statistics)\n")
        f.write(f"  - {run_name}_evaluations.csv (table of all evaluations)\n")
        f.write(f"  - {run_name}_config.json (optimization configuration)\n")
        f.write(f"  - {run_name}_summary.txt (human-readable summary)\n")
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Legacy results file: {result_file}")
    print(f"\nComprehensive results (JSON/CSV) saved with prefix: {run_name}")
    
    # Generate plots if requested
    if config.save_plots:
        print(f"\nGenerating visualizations...")
        try:
            from visualize_results import visualize_all
            
            if config.save_log and Path(config.log_file).exists():
                visualize_all(
                    config.log_file,
                    output_dir=config.plots_dir,
                    objective_name=config.objective_mode
                )
                print(f"Plots saved to: {config.plots_dir}/")
            else:
                print("Warning: No log file found for visualization")
        except ImportError:
            print("Warning: matplotlib not installed, skipping plots")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

