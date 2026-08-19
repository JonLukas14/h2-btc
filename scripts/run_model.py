from pathlib import Path
import argparse

import numpy as np
import xarray as xr
import yaml

from build_network import build_test_network


# =============================================================================
# 1. Parse command-line arguments
# =============================================================================
parser = argparse.ArgumentParser(
    description="Build and solve a PyPSA scenario."
)

parser.add_argument(
    "--config",
    required=True,
    help="Path to scenario YAML file",
)

parser.add_argument(
    "--data-dir",
    required=True,
    help="Directory containing processed input CSVs",
)

parser.add_argument(
    "--output-network",
    required=True,
    help="Path to solved network NetCDF output",
)

args = parser.parse_args()


# =============================================================================
# 2. File paths
# =============================================================================
CONFIG_FILE = Path(args.config)
DATA_DIR = Path(args.data_dir)
OUTPUT_NETWORK = Path(args.output_network)


# =============================================================================
# 3. Hydrogen target conversion
# =============================================================================
def get_hydrogen_target_mwh(cfg):
    """
    Convert the annual hydrogen mass-production target from kt H2/year
    to MWh_H2/year on the configured LHV basis.
    """

    hydrogen_cfg = cfg["hydrogen"]

    target_kt_h2 = hydrogen_cfg.get(
        "target_annual_kt_h2"
    )

    if target_kt_h2 is None:
        raise ValueError(
            "hydrogen.target_annual_kt_h2 is not defined."
        )

    target_kt_h2 = float(target_kt_h2)

    if target_kt_h2 <= 0.0:
        raise ValueError(
            "hydrogen.target_annual_kt_h2 must be greater than zero."
        )

    hydrogen_lhv_kwh_per_kg = float(
        hydrogen_cfg.get(
            "hydrogen_lhv_kwh_per_kg",
            33.33,
        )
    )

    if hydrogen_lhv_kwh_per_kg <= 0.0:
        raise ValueError(
            "hydrogen.hydrogen_lhv_kwh_per_kg "
            "must be greater than zero."
        )

    target_mwh_h2 = (
        target_kt_h2
        * 1_000_000.0
        * hydrogen_lhv_kwh_per_kg
        / 1000.0
    )

    return target_mwh_h2


# =============================================================================
# 4. Custom annual hydrogen-production constraint
# =============================================================================
def add_hydrogen_target_constraint(
    n,
    snapshots,
    cfg,
):
    """
    Require annual hydrogen delivery to equal the configured annual target.

    hydrogen_delivery is modeled as a Generator with sign=-1 on the
    hydrogen bus. Its dispatch variable is therefore positive when hydrogen
    leaves the modeled plant.

    Snapshot weighting is included so that this formulation remains correct
    if temporal aggregation is introduced later.
    """

    if "hydrogen_delivery" not in n.generators.index:
        raise KeyError(
            "Generator 'hydrogen_delivery' is missing from the network."
        )

    target_mwh_h2 = get_hydrogen_target_mwh(
        cfg
    )

    # Generator dispatch variable:
    # dimensions = (snapshot, name)
    generator_dispatch = n.model[
        "Generator-p"
    ]

    hydrogen_delivery = (
        generator_dispatch
        .sel(name="hydrogen_delivery")
    )

    # Convert PyPSA snapshot weighting to an xarray object so that
    # Linopy aligns it explicitly with the snapshot dimension.
    snapshot_weights = xr.DataArray(
        n.snapshot_weightings.generators
        .reindex(snapshots)
        .to_numpy(),
        coords={
            "snapshot": snapshots,
        },
        dims=[
            "snapshot",
        ],
    )

    annual_hydrogen_delivery = (
        hydrogen_delivery
        * snapshot_weights
    ).sum(
        "snapshot"
    )

    n.model.add_constraints(
        annual_hydrogen_delivery
        == target_mwh_h2,
        name="GlobalConstraint-hydrogen_delivery_target",
    )

    print("\n--- Hydrogen production constraint ---")
    print(
        f"Annual H2 target: "
        f"{target_mwh_h2:,.2f} MWh_H2"
    )
    print(
        "Constraint: "
        "sum_t(hydrogen_delivery_t * snapshot_weight_t) "
        "= annual target"
    )
    print("--------------------------------------\n")


# =============================================================================
# 5. Validate optimized off-grid S0
# =============================================================================
def validate_solution(
    n,
    cfg,
):
    """
    Validate the most important physical relationships after optimization.
    """

    target_mwh_h2 = get_hydrogen_target_mwh(
        cfg
    )

    hydrogen_cfg = cfg["hydrogen"]

    hydrogen_lhv_kwh_per_kg = float(
        hydrogen_cfg.get(
            "hydrogen_lhv_kwh_per_kg",
            33.33,
        )
    )

    weights = (
        n.snapshot_weightings.generators
    )

    # -------------------------------------------------------------------------
    # Annual energy quantities
    # -------------------------------------------------------------------------
    solar_generation_mwh = float(
        (
            n.generators_t.p["solar"]
            * weights
        ).sum()
    )

    wind_generation_mwh = float(
        (
            n.generators_t.p["wind"]
            * weights
        ).sum()
    )

    electrolyzer_input_mwh = float(
        (
            n.links_t.p0["electrolyzer"]
            * weights
        ).sum()
    )

    electrolyzer_output_mwh = float(
        (
            -n.links_t.p1["electrolyzer"]
            * weights
        ).sum()
    )

    hydrogen_delivery_mwh = float(
        (
            n.generators_t.p[
                "hydrogen_delivery"
            ]
            * weights
        ).sum()
    )

    # -------------------------------------------------------------------------
    # Hourly energy balances
    # -------------------------------------------------------------------------
    electricity_balance = (
        n.generators_t.p["solar"]
        + n.generators_t.p["wind"]
        - n.links_t.p0["electrolyzer"]
    )

    hydrogen_balance = (
        -n.links_t.p1["electrolyzer"]
        - n.generators_t.p[
            "hydrogen_delivery"
        ]
    )

    max_electricity_balance_error_mw = float(
        electricity_balance.abs().max()
    )

    max_hydrogen_balance_error_mw = float(
        hydrogen_balance.abs().max()
    )

    # -------------------------------------------------------------------------
    # Production and efficiency metrics
    # -------------------------------------------------------------------------
    hydrogen_delivered_kg = (
        hydrogen_delivery_mwh
        * 1000.0
        / hydrogen_lhv_kwh_per_kg
    )

    specific_electricity_kwh_per_kg_h2 = (
        electrolyzer_input_mwh
        * 1000.0
        / hydrogen_delivered_kg
    )

    realized_electrolyzer_efficiency = (
        electrolyzer_output_mwh
        / electrolyzer_input_mwh
    )

    # -------------------------------------------------------------------------
    # Numerical validation
    # -------------------------------------------------------------------------
    target_tolerance_mwh = max(
        1e-3,
        target_mwh_h2 * 1e-8,
    )

    if not np.isclose(
        hydrogen_delivery_mwh,
        target_mwh_h2,
        rtol=0.0,
        atol=target_tolerance_mwh,
    ):
        raise RuntimeError(
            "Annual hydrogen target was not met: "
            f"target={target_mwh_h2:.6f} MWh, "
            f"actual={hydrogen_delivery_mwh:.6f} MWh."
        )

    balance_tolerance_mw = 1e-4

    if (
        max_electricity_balance_error_mw
        > balance_tolerance_mw
    ):
        raise RuntimeError(
            "Electricity balance validation failed: "
            f"max error="
            f"{max_electricity_balance_error_mw:.6e} MW."
        )

    if (
        max_hydrogen_balance_error_mw
        > balance_tolerance_mw
    ):
        raise RuntimeError(
            "Hydrogen balance validation failed: "
            f"max error="
            f"{max_hydrogen_balance_error_mw:.6e} MW."
        )

    # -------------------------------------------------------------------------
    # Print validation report
    # -------------------------------------------------------------------------
    print("\n================================================")
    print("OFF-GRID S0 SOLUTION VALIDATION")
    print("================================================")

    print("\nOptimal capacities:")
    print(
        f"  Solar:        "
        f"{n.generators.at['solar', 'p_nom_opt']:.3f} MW"
    )
    print(
        f"  Wind:         "
        f"{n.generators.at['wind', 'p_nom_opt']:.3f} MW"
    )
    print(
        f"  Electrolyzer: "
        f"{n.links.at['electrolyzer', 'p_nom_opt']:.3f} MW_el"
    )

    print("\nAnnual energy:")
    print(
        f"  Solar generation:        "
        f"{solar_generation_mwh:,.3f} MWh"
    )
    print(
        f"  Wind generation:         "
        f"{wind_generation_mwh:,.3f} MWh"
    )
    print(
        f"  Electrolyzer input:      "
        f"{electrolyzer_input_mwh:,.3f} MWh_el"
    )
    print(
        f"  Electrolyzer H2 output:  "
        f"{electrolyzer_output_mwh:,.3f} MWh_H2"
    )
    print(
        f"  H2 delivered:            "
        f"{hydrogen_delivery_mwh:,.3f} MWh_H2"
    )
    print(
        f"  H2 target:               "
        f"{target_mwh_h2:,.3f} MWh_H2"
    )

    print("\nHydrogen mass:")
    print(
        f"  H2 delivered:            "
        f"{hydrogen_delivered_kg:,.3f} kg"
    )
    print(
        f"  H2 delivered:            "
        f"{hydrogen_delivered_kg / 1_000_000.0:.6f} kt"
    )

    print("\nElectrolyzer:")
    print(
        f"  Realized efficiency:     "
        f"{realized_electrolyzer_efficiency:.6f}"
    )
    print(
        f"  Specific electricity:    "
        f"{specific_electricity_kwh_per_kg_h2:.3f} "
        f"kWh_el/kg_H2"
    )

    print("\nBalance validation:")
    print(
        f"  Max electricity error:   "
        f"{max_electricity_balance_error_mw:.3e} MW"
    )
    print(
        f"  Max hydrogen error:      "
        f"{max_hydrogen_balance_error_mw:.3e} MW"
    )

    print("\nChecks:")
    print("  Annual H2 target:       PASS")
    print("  Electricity balance:    PASS")
    print("  Hydrogen balance:       PASS")

    print("================================================\n")


# =============================================================================
# 6. Main
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # Validate paths before reading files
    # -------------------------------------------------------------------------
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Config file not found: "
            f"{CONFIG_FILE}"
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: "
            f"{DATA_DIR}"
        )

    # -------------------------------------------------------------------------
    # Load scenario configuration
    # -------------------------------------------------------------------------
    with CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    OUTPUT_NETWORK.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Build off-grid network
    # -------------------------------------------------------------------------
    n = build_test_network(
        cfg,
        data_dir=DATA_DIR,
    )

    # -------------------------------------------------------------------------
    # Add custom optimization constraints
    # -------------------------------------------------------------------------
    def extra_functionality(
        network,
        snapshots,
    ):
        add_hydrogen_target_constraint(
            network,
            snapshots,
            cfg,
        )

    # -------------------------------------------------------------------------
    # Optimize
    # -------------------------------------------------------------------------
    status, condition = n.optimize(
        solver_name="highs",
        extra_functionality=extra_functionality,
        include_objective_constant=False,
    )

    print("\nOptimization finished")
    print(
        f"Scenario file: {CONFIG_FILE}"
    )
    print(
        f"Data directory: {DATA_DIR}"
    )
    print(
        f"Status: {status}"
    )
    print(
        f"Condition: {condition}"
    )

    if (
        status != "ok"
        or condition != "optimal"
    ):
        raise RuntimeError(
            "Optimization did not solve cleanly: "
            f"status={status}, "
            f"condition={condition}"
        )

    # -------------------------------------------------------------------------
    # Validate physical solution
    # -------------------------------------------------------------------------
    validate_solution(
        n,
        cfg,
    )

    # -------------------------------------------------------------------------
    # Export solved network
    # -------------------------------------------------------------------------
    n.export_to_netcdf(
        OUTPUT_NETWORK
    )

    print(
        f"Saved network to: "
        f"{OUTPUT_NETWORK}"
    )


# =============================================================================
# 7. Script entry point
# =============================================================================
if __name__ == "__main__":
    main()