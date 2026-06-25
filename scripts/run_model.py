from pathlib import Path
import yaml

from build_network import build_test_network


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "configs" / "scenario.yaml"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_NETWORK = RESULTS_DIR / "network_solved.nc"


def main():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    RESULTS_DIR.mkdir(exist_ok=True)

    n = build_test_network(cfg)

    status, condition = n.optimize(
        solver_name="highs",
        include_objective_constant=False
    )

    n.export_to_netcdf(OUTPUT_NETWORK)

    print("Optimization finished")
    print(f"Status: {status}")
    print(f"Condition: {condition}")
    print(f"Saved network to: {OUTPUT_NETWORK}")


if __name__ == "__main__":
    main()