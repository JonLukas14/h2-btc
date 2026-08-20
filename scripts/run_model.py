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
# 5. Battery inverter power-capacity coupling constraint
# =============================================================================
def add_battery_power_coupling_constraint(
    n,
    cfg,
):
    """
    Couple the charging and discharging Link capacities of the battery.

    The battery uses one physical bidirectional inverter, represented in
    PyPSA by two directed Links:

        battery_charger:
            electricity_bus -> battery_bus

        battery_discharger:
            battery_bus -> electricity_bus

    Link nominal capacity is defined on bus0.

    Therefore:
        - battery_charger.p_nom is the AC-side charging power rating.
        - battery_discharger.p_nom is the battery-side input rating during
          discharge.
        - AC-side discharge capacity equals
          efficiency_discharge * p_nom_discharger.

    Equal AC-side inverter power therefore requires:

        p_nom_charger
        =
        efficiency_discharge * p_nom_discharger
    """

    battery_cfg = cfg.get(
        "battery",
        {},
    )

    battery_enabled = bool(
        battery_cfg.get(
            "enabled",
            False,
        )
    )

    # S0, S2, etc. without battery require no battery constraint.
    if not battery_enabled:
        return

    required_links = [
        "battery_charger",
        "battery_discharger",
    ]

    missing_links = [
        link
        for link in required_links
        if link not in n.links.index
    ]

    if missing_links:
        raise KeyError(
            "Battery power-coupling constraint cannot be added because "
            f"the following Links are missing: {missing_links}"
        )

    for link in required_links:
        if not bool(
            n.links.at[
                link,
                "p_nom_extendable",
            ]
        ):
            raise ValueError(
                f"Battery Link '{link}' must have "
                "p_nom_extendable=true."
            )

    discharge_efficiency = float(
        n.links.at[
            "battery_discharger",
            "efficiency",
        ]
    )

    if not 0.0 < discharge_efficiency <= 1.0:
        raise ValueError(
            "Battery discharge efficiency must be "
            "greater than 0 and at most 1."
        )

    # Extendable Link nominal-capacity decision variables.
    link_p_nom = n.model[
        "Link-p_nom"
    ]

    charger_p_nom = (
        link_p_nom
        .loc["battery_charger"]
    )

    discharger_p_nom = (
        link_p_nom
        .loc["battery_discharger"]
    )

    battery_power_coupling = (
        charger_p_nom
        - discharge_efficiency
        * discharger_p_nom
    )

    n.model.add_constraints(
        battery_power_coupling == 0.0,
        name="Battery-inverter-power-coupling",
    )

    print("\n--- Battery inverter power constraint ---")
    print(
        "Constraint: "
        "p_nom(battery_charger) = "
        "eta_discharge * p_nom(battery_discharger)"
    )
    print(
        f"Discharge efficiency: "
        f"{discharge_efficiency:.6f}"
    )
    print("-----------------------------------------\n")

# =============================================================================
# 6. Validate optimized off-grid scenario
# =============================================================================
def validate_solution(
    n,
    cfg,
):
    """
    Validate the most important physical and numerical relationships
    after optimization for S0, battery-enabled S1, and Bitcoin-enabled S2.
    """

    target_mwh_h2 = get_hydrogen_target_mwh(
        cfg
    )

    hydrogen_cfg = cfg["hydrogen"]

    battery_cfg = cfg.get(
        "battery",
        {},
    )

    battery_enabled = bool(
        battery_cfg.get(
            "enabled",
            False,
        )
    )

    bitcoin_cfg = cfg.get(
        "bitcoin",
        {},
    )

    bitcoin_enabled = bool(
        bitcoin_cfg.get(
            "enabled",
            False,
        )
    )

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
    # Annual renewable and hydrogen quantities
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

    electrolyzer_input = (
        n.links_t.p0["electrolyzer"]
    )

    electrolyzer_output = (
        -n.links_t.p1["electrolyzer"]
    )

    hydrogen_delivery = (
        n.generators_t.p[
            "hydrogen_delivery"
        ]
    )

    electrolyzer_input_mwh = float(
        (
            electrolyzer_input
            * weights
        ).sum()
    )

    electrolyzer_output_mwh = float(
        (
            electrolyzer_output
            * weights
        ).sum()
    )

    hydrogen_delivery_mwh = float(
        (
            hydrogen_delivery
            * weights
        ).sum()
    )

    # -------------------------------------------------------------------------
    # Bitcoin mining quantities
    # -------------------------------------------------------------------------
    #
    # bitcoin_mining_sink is a Generator with sign=-1.
    #
    # Its dispatch variable itself remains positive. The sign only controls
    # how that dispatch enters the electricity-bus balance. Therefore the
    # positive dispatch series below is interpreted directly as electricity
    # consumption by the mining facility.
    bitcoin_consumption = (
        electrolyzer_input
        * 0.0
    )

    bitcoin_capacity_mw = 0.0
    bitcoin_consumption_mwh = 0.0
    bitcoin_utilization_rate = np.nan
    bitcoin_full_capacity_hours = 0.0
    bitcoin_zero_dispatch_hours = 0.0
    bitcoin_max_dispatch_mw = 0.0
    bitcoin_min_dispatch_mw = 0.0

    if bitcoin_enabled:
        bitcoin_asset = "bitcoin_mining_sink"

        if bitcoin_asset not in n.generators.index:
            raise KeyError(
                "Bitcoin mining is enabled, but Generator "
                "'bitcoin_mining_sink' is missing."
            )

        bitcoin_sign = float(
            n.generators.at[
                bitcoin_asset,
                "sign",
            ]
        )

        if not np.isclose(
            bitcoin_sign,
            -1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Bitcoin mining sink must use Generator sign=-1. "
                f"Actual sign={bitcoin_sign}."
            )

        if bool(
            n.generators.at[
                bitcoin_asset,
                "p_nom_extendable",
            ]
        ):
            raise RuntimeError(
                "The current S2 validation case requires fixed "
                "Bitcoin mining capacity."
            )

        configured_bitcoin_capacity_mw = (
            bitcoin_cfg.get(
                "max_capacity_mw"
            )
        )

        if configured_bitcoin_capacity_mw is None:
            raise ValueError(
                "bitcoin.max_capacity_mw must be defined "
                "when Bitcoin mining is enabled."
            )

        configured_bitcoin_capacity_mw = float(
            configured_bitcoin_capacity_mw
        )

        bitcoin_capacity_mw = float(
            n.generators.at[
                bitcoin_asset,
                "p_nom",
            ]
        )

        capacity_tolerance_mw = max(
            1e-9,
            abs(
                configured_bitcoin_capacity_mw
            )
            * 1e-9,
        )

        if not np.isclose(
            bitcoin_capacity_mw,
            configured_bitcoin_capacity_mw,
            rtol=0.0,
            atol=capacity_tolerance_mw,
        ):
            raise RuntimeError(
                "Bitcoin mining capacity does not match "
                "the scenario configuration: "
                f"configured={configured_bitcoin_capacity_mw:.6f} MW, "
                f"network={bitcoin_capacity_mw:.6f} MW."
            )

        bitcoin_consumption = (
            n.generators_t.p[
                bitcoin_asset
            ]
        )

        bitcoin_min_dispatch_mw = float(
            bitcoin_consumption.min()
        )

        bitcoin_max_dispatch_mw = float(
            bitcoin_consumption.max()
        )

        dispatch_tolerance_mw = max(
            1e-6,
            bitcoin_capacity_mw
            * 1e-8,
        )

        if (
            bitcoin_min_dispatch_mw
            < -dispatch_tolerance_mw
        ):
            raise RuntimeError(
                "Bitcoin mining dispatch became negative: "
                f"minimum={bitcoin_min_dispatch_mw:.6e} MW."
            )

        if (
            bitcoin_max_dispatch_mw
            > bitcoin_capacity_mw
            + dispatch_tolerance_mw
        ):
            raise RuntimeError(
                "Bitcoin mining dispatch exceeded installed capacity: "
                f"maximum={bitcoin_max_dispatch_mw:.6f} MW, "
                f"capacity={bitcoin_capacity_mw:.6f} MW."
            )

        bitcoin_consumption_mwh = float(
            (
                bitcoin_consumption
                * weights
            ).sum()
        )

        modeled_hours = float(
            weights.sum()
        )

        if (
            bitcoin_capacity_mw > 0.0
            and modeled_hours > 0.0
        ):
            bitcoin_utilization_rate = (
                bitcoin_consumption_mwh
                / (
                    bitcoin_capacity_mw
                    * modeled_hours
                )
            )

        bitcoin_full_capacity_hours = float(
            weights[
                bitcoin_consumption
                >= (
                    0.99
                    * bitcoin_capacity_mw
                )
            ].sum()
        )

        bitcoin_zero_dispatch_hours = float(
            weights[
                bitcoin_consumption
                <= dispatch_tolerance_mw
            ].sum()
        )

        print("\n--- Bitcoin mining validation ---")
        print(
            f"Fixed mining capacity: "
            f"{bitcoin_capacity_mw:.3f} MW"
        )
        print(
            f"Annual electricity use: "
            f"{bitcoin_consumption_mwh:,.3f} MWh"
        )
        print(
            f"Utilization: "
            f"{bitcoin_utilization_rate * 100.0:.3f}%"
        )
        print(
            f"Minimum dispatch: "
            f"{bitcoin_min_dispatch_mw:.6f} MW"
        )
        print(
            f"Maximum dispatch: "
            f"{bitcoin_max_dispatch_mw:.6f} MW"
        )
        print(
            f"Equivalent full-capacity hours: "
            f"{bitcoin_consumption_mwh / bitcoin_capacity_mw:.2f} h"
        )
        print(
            f"Hours >=99% capacity: "
            f"{bitcoin_full_capacity_hours:.2f} h"
        )
        print(
            f"Zero-dispatch hours: "
            f"{bitcoin_zero_dispatch_hours:.2f} h"
        )
        print("---------------------------------\n")

    # -------------------------------------------------------------------------
    # Battery quantities
    # -------------------------------------------------------------------------
    battery_charge_input = None
    battery_charge_output = None
    battery_discharge_input = None
    battery_discharge_output = None
    battery_store_power = None
    battery_energy = None

    battery_power_mw = 0.0
    battery_discharge_power_ac_mw = 0.0
    battery_energy_mwh = 0.0
    battery_duration_h = np.nan

    battery_charge_input_mwh = 0.0
    battery_charge_output_mwh = 0.0
    battery_discharge_input_mwh = 0.0
    battery_discharge_output_mwh = 0.0
    battery_loss_mwh = 0.0

    battery_power_coupling_error_mw = 0.0
    max_battery_bus_balance_error_mw = 0.0

    battery_soc_min_mwh = 0.0
    battery_soc_max_mwh = 0.0

    if battery_enabled:
        required_links = [
            "battery_charger",
            "battery_discharger",
        ]

        for link in required_links:
            if link not in n.links.index:
                raise KeyError(
                    f"Required battery Link "
                    f"'{link}' is missing."
                )

        if "battery_store" not in n.stores.index:
            raise KeyError(
                "Required Store "
                "'battery_store' is missing."
            )

        # AC electricity withdrawn for charging.
        battery_charge_input = (
            n.links_t.p0[
                "battery_charger"
            ]
        )

        # Energy arriving on the battery bus
        # after charging losses.
        battery_charge_output = (
            -n.links_t.p1[
                "battery_charger"
            ]
        )

        # Energy withdrawn from battery bus
        # into the discharge converter.
        battery_discharge_input = (
            n.links_t.p0[
                "battery_discharger"
            ]
        )

        # Electricity supplied back to AC bus.
        battery_discharge_output = (
            -n.links_t.p1[
                "battery_discharger"
            ]
        )

        # Positive Store.p means net supply from
        # the Store to battery_bus.
        battery_store_power = (
            n.stores_t.p[
                "battery_store"
            ]
        )

        battery_energy = (
            n.stores_t.e[
                "battery_store"
            ]
        )

        # -------------------------------------------------------------
        # Optimized capacities
        # -------------------------------------------------------------
        battery_power_mw = float(
            n.links.at[
                "battery_charger",
                "p_nom_opt",
            ]
        )

        discharger_p_nom_mw = float(
            n.links.at[
                "battery_discharger",
                "p_nom_opt",
            ]
        )

        discharge_efficiency = float(
            n.links.at[
                "battery_discharger",
                "efficiency",
            ]
        )

        charge_efficiency = float(
            n.links.at[
                "battery_charger",
                "efficiency",
            ]
        )

        battery_discharge_power_ac_mw = (
            discharger_p_nom_mw
            * discharge_efficiency
        )

        battery_energy_mwh = float(
            n.stores.at[
                "battery_store",
                "e_nom_opt",
            ]
        )

        if battery_power_mw > 1e-9:
            battery_duration_h = (
                battery_energy_mwh
                / battery_power_mw
            )

        # -------------------------------------------------------------
        # Annual battery energy flows
        # -------------------------------------------------------------
        battery_charge_input_mwh = float(
            (
                battery_charge_input
                * weights
            ).sum()
        )

        battery_charge_output_mwh = float(
            (
                battery_charge_output
                * weights
            ).sum()
        )

        battery_discharge_input_mwh = float(
            (
                battery_discharge_input
                * weights
            ).sum()
        )

        battery_discharge_output_mwh = float(
            (
                battery_discharge_output
                * weights
            ).sum()
        )

        battery_loss_mwh = (
            battery_charge_input_mwh
            - battery_discharge_output_mwh
        )

        # -------------------------------------------------------------
        # Battery bus balance
        # -------------------------------------------------------------
        battery_bus_balance = (
            battery_charge_output
            + battery_store_power
            - battery_discharge_input
        )

        max_battery_bus_balance_error_mw = float(
            battery_bus_balance
            .abs()
            .max()
        )

        # -------------------------------------------------------------
        # Charger/discharger capacity coupling
        # -------------------------------------------------------------
        battery_power_coupling_error_mw = abs(
            battery_power_mw
            - discharge_efficiency
            * discharger_p_nom_mw
        )

        # -------------------------------------------------------------
        # State of charge
        # -------------------------------------------------------------
        battery_soc_min_mwh = float(
            battery_energy.min()
        )

        battery_soc_max_mwh = float(
            battery_energy.max()
        )

        # -------------------------------------------------------------
        # Numerical battery checks
        # -------------------------------------------------------------
        power_coupling_tolerance_mw = max(
            1e-6,
            max(
                abs(battery_power_mw),
                abs(
                    discharge_efficiency
                    * discharger_p_nom_mw
                ),
            )
            * 1e-8,
        )

        if (
            battery_power_coupling_error_mw
            > power_coupling_tolerance_mw
        ):
            raise RuntimeError(
                "Battery inverter power-capacity "
                "coupling failed: "
                f"error="
                f"{battery_power_coupling_error_mw:.6e} MW."
            )

        battery_balance_tolerance_mw = 1e-4

        if (
            max_battery_bus_balance_error_mw
            > battery_balance_tolerance_mw
        ):
            raise RuntimeError(
                "Battery bus balance validation "
                "failed: "
                f"max error="
                f"{max_battery_bus_balance_error_mw:.6e} MW."
            )

        energy_tolerance_mwh = max(
            1e-6,
            battery_energy_mwh * 1e-8,
        )

        if (
            battery_soc_min_mwh
            < -energy_tolerance_mwh
        ):
            raise RuntimeError(
                "Battery state of charge became "
                "negative: "
                f"minimum="
                f"{battery_soc_min_mwh:.6e} MWh."
            )

        if (
            battery_soc_max_mwh
            > battery_energy_mwh
            + energy_tolerance_mwh
        ):
            raise RuntimeError(
                "Battery state of charge exceeded "
                "optimized energy capacity: "
                f"max SOC="
                f"{battery_soc_max_mwh:.6f} MWh, "
                f"capacity="
                f"{battery_energy_mwh:.6f} MWh."
            )

        # With zero standing loss and cyclic storage,
        # realized conversion efficiency should equal
        # eta_charge * eta_discharge whenever the
        # battery actually cycles.
        standing_loss = float(
            n.stores.at[
                "battery_store",
                "standing_loss",
            ]
        )

        if (
            battery_charge_input_mwh > 1e-6
            and abs(standing_loss) < 1e-12
        ):
            realized_round_trip_efficiency = (
                battery_discharge_output_mwh
                / battery_charge_input_mwh
            )

            expected_round_trip_efficiency = (
                charge_efficiency
                * discharge_efficiency
            )

            if not np.isclose(
                realized_round_trip_efficiency,
                expected_round_trip_efficiency,
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "Battery round-trip efficiency "
                    "validation failed: "
                    f"expected="
                    f"{expected_round_trip_efficiency:.6f}, "
                    f"realized="
                    f"{realized_round_trip_efficiency:.6f}."
                )

    # -------------------------------------------------------------------------
    # Hourly electricity balance
    # -------------------------------------------------------------------------
    electricity_balance = (
        n.generators_t.p["solar"]
        + n.generators_t.p["wind"]
        - electrolyzer_input
    )

    if battery_enabled:
        electricity_balance = (
            electricity_balance
            + battery_discharge_output
            - battery_charge_input
        )

    if bitcoin_enabled:
        electricity_balance = (
            electricity_balance
            - bitcoin_consumption
        )

    hydrogen_balance = (
        electrolyzer_output
        - hydrogen_delivery
    )

    max_electricity_balance_error_mw = float(
        electricity_balance
        .abs()
        .max()
    )

    max_hydrogen_balance_error_mw = float(
        hydrogen_balance
        .abs()
        .max()
    )

    # -------------------------------------------------------------------------
    # Hydrogen production and efficiency metrics
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
    # Core numerical validation
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
    # Validation report
    # -------------------------------------------------------------------------
    print("\n================================================")
    print("OFF-GRID SOLUTION VALIDATION")
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

    if battery_enabled:
        print(
            f"  Battery power:"
            f" {battery_power_mw:.3f} MW"
        )
        print(
            f"  Battery energy:"
            f" {battery_energy_mwh:.3f} MWh"
        )

        if np.isfinite(
            battery_duration_h
        ):
            print(
                f"  Battery duration:"
                f" {battery_duration_h:.3f} h"
            )
        else:
            print(
                "  Battery duration: n/a "
                "(zero optimized power)"
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

    if battery_enabled:
        print("\nBattery operation:")
        print(
            f"  Charge from AC:          "
            f"{battery_charge_input_mwh:,.3f} MWh"
        )
        print(
            f"  Charge after losses:     "
            f"{battery_charge_output_mwh:,.3f} MWh"
        )
        print(
            f"  Discharge before losses: "
            f"{battery_discharge_input_mwh:,.3f} MWh"
        )
        print(
            f"  Discharge to AC:         "
            f"{battery_discharge_output_mwh:,.3f} MWh"
        )
        print(
            f"  Total battery losses:    "
            f"{battery_loss_mwh:,.3f} MWh"
        )
        print(
            f"  Minimum SOC:             "
            f"{battery_soc_min_mwh:,.3f} MWh"
        )
        print(
            f"  Maximum SOC:             "
            f"{battery_soc_max_mwh:,.3f} MWh"
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

    if battery_enabled:
        print(
            f"  Max battery-bus error:   "
            f"{max_battery_bus_balance_error_mw:.3e} MW"
        )
        print(
            f"  Power-coupling error:    "
            f"{battery_power_coupling_error_mw:.3e} MW"
        )

    print("\nChecks:")
    print("  Annual H2 target:        PASS")
    print("  Electricity balance:     PASS")
    print("  Hydrogen balance:        PASS")

    if battery_enabled:
        print("  Battery bus balance:     PASS")
        print("  Battery power coupling:  PASS")
        print("  Battery SOC bounds:      PASS")

    print("================================================\n")

# =============================================================================
# 7. Main
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

        add_battery_power_coupling_constraint(
            network,
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
# 8. Script entry point
# =============================================================================
if __name__ == "__main__":
    main()