from pathlib import Path
import argparse
import yaml

from build_network import build_test_network


# -----------------------------------------------------------------------------
# 1. Parse command-line arguments
# -----------------------------------------------------------------------------
# This makes the script reusable for different scenarios, data folders,
# and output folders.
parser = argparse.ArgumentParser(description="Build and solve a PyPSA scenario.")
parser.add_argument("--config", required=True, help="Path to scenario YAML file")
parser.add_argument("--data-dir", required=True, help="Directory containing processed input CSVs")
parser.add_argument("--output-network", required=True, help="Path to solved network NetCDF output")
args = parser.parse_args()


# -----------------------------------------------------------------------------
# 2. Define important file paths
# -----------------------------------------------------------------------------
# CONFIG_FILE is the scenario configuration passed from Snakemake.
# DATA_DIR contains the scenario-specific processed CSV inputs.
# OUTPUT_NETWORK is where the solved PyPSA network will be written.
CONFIG_FILE = Path(args.config)
DATA_DIR = Path(args.data_dir)
OUTPUT_NETWORK = Path(args.output_network)


# -----------------------------------------------------------------------------
# 3. Main function: load config, build network, solve model, save result
# -----------------------------------------------------------------------------
def main():
    # Read the active scenario configuration from YAML.
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Make sure required paths exist before running the model.
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Make sure the parent folder for the solved network exists.
    OUTPUT_NETWORK.parent.mkdir(parents=True, exist_ok=True)

    # Build the PyPSA network object from the scenario config
    # and scenario-specific preprocessed data.
    n = build_test_network(cfg, data_dir=DATA_DIR)

    # Run the optimization using the HiGHS solver.
    status, condition = n.optimize(
        solver_name="highs",
        include_objective_constant=False,
    )

    # Print solve status information for debugging and logging.
    print("Optimization finished")
    print(f"Scenario file: {CONFIG_FILE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Status: {status}")
    print(f"Condition: {condition}")

    # Stop if the optimization did not solve successfully.
    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"Optimization did not solve cleanly: status={status}, condition={condition}"
        )

    # Export the solved network to a NetCDF file.
    n.export_to_netcdf(OUTPUT_NETWORK)
    print(f"Saved network to: {OUTPUT_NETWORK}")


# -----------------------------------------------------------------------------
# 4. Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()