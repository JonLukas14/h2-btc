from pathlib import Path
import yaml

from build_network import build_test_network


# -----------------------------------------------------------------------------
# 1. Define important project paths
# -----------------------------------------------------------------------------
# BASE_DIR points to the root of the project.
# CONFIG_FILE is the active scenario configuration.
# RESULTS_DIR is where model outputs are stored.
# OUTPUT_NETWORK is the solved PyPSA network file written after optimization.
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "configs" / "scenario.yaml"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_NETWORK = RESULTS_DIR / "network_solved.nc"


# -----------------------------------------------------------------------------
# 2. Main function: load config, build network, solve model, save result
# -----------------------------------------------------------------------------
def main():
    # Read the active scenario configuration from YAML.
    # This config controls things like snapshots, costs, hydrogen mode, and mining.
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Make sure the results folder exists before trying to write outputs.
    RESULTS_DIR.mkdir(exist_ok=True)

    # Build the PyPSA network object from the scenario config and preprocessed data.
    # The actual network structure is created in build_network.py.
    n = build_test_network(cfg)

    # Run the linear optimization using the HiGHS solver.
    # include_objective_constant=False excludes any constant term from the reported
    # objective value so the result focuses on decision-dependent costs.
    status, condition = n.optimize(
        solver_name="highs",
        include_objective_constant=False,
    )

    # Print solve status information for debugging and logging.
    print("Optimization finished")
    print(f"Status: {status}")
    print(f"Condition: {condition}")

    # Stop the workflow if the optimization did not solve successfully.
    # This prevents downstream analysis from using an incomplete or invalid network.
    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"Optimization did not solve cleanly: status={status}, condition={condition}"
        )

    # Export the solved network to a NetCDF file.
    # This file contains the optimized capacities, dispatch, and other results,
    # and is later read by analyze_results.py.
    n.export_to_netcdf(OUTPUT_NETWORK)
    print(f"Saved network to: {OUTPUT_NETWORK}")


# -----------------------------------------------------------------------------
# 3. Script entry point
# -----------------------------------------------------------------------------
# This ensures main() only runs when the file is executed directly,
# not when it is imported from another Python file.
if __name__ == "__main__":
    main()