from pathlib import Path
import argparse

import numpy as np
import xarray as xr
import yaml

from build_network import build_test_network

# =============================================================================
# Bitcoin reference ASIC
# =============================================================================
#
# Reference machine:
# Bitmain Antminer S21 XP, air-cooled
#
# ASIC electrical input:
#   3.645 kW per miner
#
# Facility-side electricity additionally includes PUE:
#   facility power = ASIC power * PUE
#
# IMPORTANT:
# These values describe the named reference machine used to translate
# modeled BTC electrical capacity into an equivalent physical miner count.
# They are not interpreted as a forecast of 2045 ASIC technology.
#
BTC_MINER_NAME = "Bitmain Antminer S21 XP Air-cooled"
BTC_MINER_POWER_KW = 3.645
BTC_MINER_HASHRATE_TH_S = 270.0

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
# 4. Annual hydrogen-delivery expression and custom constraints
# =============================================================================
def get_annual_hydrogen_delivery_expression(
    n,
    snapshots,
):
    """
    Return the weighted annual hydrogen-delivery Linopy expression.

    hydrogen_delivery is represented as a Generator with sign=-1 on the
    hydrogen bus. Its dispatch variable is positive when hydrogen leaves
    the modeled plant.

    Snapshot weighting is included explicitly so that the expression remains
    correct if temporal aggregation is introduced later.
    """

    if "hydrogen_delivery" not in n.generators.index:
        raise KeyError(
            "Generator 'hydrogen_delivery' is missing from the network."
        )

    generator_dispatch = n.model[
        "Generator-p"
    ]

    hydrogen_delivery = (
        generator_dispatch
        .sel(name="hydrogen_delivery")
    )

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

    return annual_hydrogen_delivery


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

    target_mwh_h2 = get_hydrogen_target_mwh(
        cfg
    )

    annual_hydrogen_delivery = (
        get_annual_hydrogen_delivery_expression(
            n,
            snapshots,
        )
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

def add_hydrogen_minimum_constraint(
    n,
    snapshots,
    minimum_mwh_h2,
):
    """
    Require annual hydrogen delivery to remain at or above the Stage-1
    HMAX solution during Stage-2 cost minimization.
    """

    minimum_mwh_h2 = float(
        minimum_mwh_h2
    )

    if minimum_mwh_h2 < 0.0:
        raise ValueError(
            "minimum_mwh_h2 must be non-negative."
        )

    annual_hydrogen_delivery = (
        get_annual_hydrogen_delivery_expression(
            n,
            snapshots,
        )
    )

    n.model.add_constraints(
        annual_hydrogen_delivery
        >= minimum_mwh_h2,
        name="GlobalConstraint-hydrogen_delivery_minimum",
    )

    print("\n--- HMAX Stage-2 hydrogen constraint ---")
    print(
        f"Minimum annual H2 delivery: "
        f"{minimum_mwh_h2:,.6f} MWh_H2"
    )
    print(
        "Constraint: "
        "sum_t(hydrogen_delivery_t * snapshot_weight_t) "
        ">= Stage-1 HMAX minus numerical tolerance"
    )
    print("------------------------------------------\n")


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

    hydrogen_cfg = cfg["hydrogen"]

    hydrogen_mode = str(
        hydrogen_cfg.get(
            "mode",
            "production_target",
        )
    ).strip().lower()

    if hydrogen_mode == "production_target":
        target_mwh_h2 = get_hydrogen_target_mwh(
            cfg
        )

    elif hydrogen_mode == "maximize_production":
        target_mwh_h2 = None

    else:
        raise ValueError(
            "Unsupported hydrogen.mode during solution validation: "
            f"{hydrogen_mode!r}"
        )

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
    bitcoin_utilization_rate = 0.0
    bitcoin_equivalent_full_capacity_hours = 0.0
    bitcoin_full_capacity_hours = 0.0
    bitcoin_zero_dispatch_hours = 0.0
    bitcoin_max_dispatch_mw = 0.0
    bitcoin_min_dispatch_mw = 0.0

    # Physical Bitcoin-miner interpretation
    bitcoin_pue = np.nan
    bitcoin_facility_power_per_miner_kw = np.nan

    bitcoin_installed_miners_equiv = 0.0
    bitcoin_installed_miners_complete = 0

    bitcoin_max_active_miners_equiv = 0.0
    bitcoin_mean_active_miners_equiv = 0.0

    bitcoin_installed_hashrate_ph_s = 0.0
    bitcoin_max_active_hashrate_ph_s = 0.0

    if bitcoin_enabled:
        bitcoin_asset = "bitcoin_mining_sink"
        bitcoin_pue = bitcoin_cfg.get(
            "pue"
        )

        if bitcoin_pue is None:
            raise ValueError(
                "bitcoin.pue must be defined when "
                "Bitcoin mining is enabled."
            )

        bitcoin_pue = float(
            bitcoin_pue
        )

        if bitcoin_pue < 1.0:
            raise ValueError(
                "bitcoin.pue must be >= 1.0."
            )

        bitcoin_facility_power_per_miner_kw = (
            BTC_MINER_POWER_KW
            * bitcoin_pue
        )

        implied_asic_efficiency_j_per_th = (
            BTC_MINER_POWER_KW
            * 1000.0
            / BTC_MINER_HASHRATE_TH_S
        )

        if not np.isclose(
            implied_asic_efficiency_j_per_th,
            13.5,
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Reference ASIC parameters are internally inconsistent: "
                f"power/hashrate implies "
                f"{implied_asic_efficiency_j_per_th:.6f} J/TH."
            )

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

        # -------------------------------------------------------------
        # Bitcoin mining capacity mode
        # -------------------------------------------------------------
        bitcoin_capacity_mode = str(
            bitcoin_cfg.get(
                "capacity_mode",
                "fixed",
            )
        ).strip().lower()

        if bitcoin_capacity_mode not in {
            "fixed",
            "endogenous",
        }:
            raise ValueError(
                "bitcoin.capacity_mode must be either "
                "'fixed' or 'endogenous'."
            )

        bitcoin_p_nom_extendable = bool(
            n.generators.at[
                bitcoin_asset,
                "p_nom_extendable",
            ]
        )

        if bitcoin_capacity_mode == "fixed":
            if bitcoin_p_nom_extendable:
                raise RuntimeError(
                    "Bitcoin capacity mode is 'fixed', but "
                    "bitcoin_mining_sink.p_nom_extendable is True."
                )

            configured_bitcoin_capacity_mw = (
                bitcoin_cfg.get(
                    "max_capacity_mw"
                )
            )

            if configured_bitcoin_capacity_mw is None:
                raise ValueError(
                    "bitcoin.max_capacity_mw must be defined "
                    "when bitcoin.capacity_mode='fixed'."
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
                    "the fixed scenario configuration: "
                    f"configured="
                    f"{configured_bitcoin_capacity_mw:.6f} MW, "
                    f"network="
                    f"{bitcoin_capacity_mw:.6f} MW."
                )

        else:
            if not bitcoin_p_nom_extendable:
                raise RuntimeError(
                    "Bitcoin capacity mode is 'endogenous', but "
                    "bitcoin_mining_sink.p_nom_extendable is False."
                )

            bitcoin_capacity_mw = float(
                n.generators.at[
                    bitcoin_asset,
                    "p_nom_opt",
                ]
            )

            capacity_tolerance_mw = max(
                1e-9,
                abs(
                    bitcoin_capacity_mw
                )
                * 1e-9,
            )

            if (
                bitcoin_capacity_mw
                < -capacity_tolerance_mw
            ):
                raise RuntimeError(
                    "Optimized Bitcoin mining capacity became "
                    f"negative: {bitcoin_capacity_mw:.6e} MW."
                )

            # Remove harmless negative numerical zero.
            if abs(
                bitcoin_capacity_mw
            ) <= capacity_tolerance_mw:
                bitcoin_capacity_mw = 0.0

        # -------------------------------------------------------------
        # Physical miner-count interpretation
        # -------------------------------------------------------------
        bitcoin_installed_miners_equiv = (
            bitcoin_capacity_mw
            * 1000.0
            / bitcoin_facility_power_per_miner_kw
        )

        # Complete physical miners that can be installed without
        # exceeding the modeled facility-side power capacity.
        bitcoin_installed_miners_complete = int(
            np.floor(
                bitcoin_installed_miners_equiv
            )
        )

        # Installed ASIC hashrate.
        bitcoin_installed_hashrate_ph_s = (
            bitcoin_installed_miners_equiv
            * BTC_MINER_HASHRATE_TH_S
            / 1000.0
        )

        bitcoin_consumption = (
            n.generators_t.p[
                bitcoin_asset
            ]
        )

        bitcoin_active_miners_equiv = (
            bitcoin_consumption
            * 1000.0
            / bitcoin_facility_power_per_miner_kw
        )

        bitcoin_min_dispatch_mw = float(
            bitcoin_consumption.min()
        )

        bitcoin_max_dispatch_mw = float(
            bitcoin_consumption.max()
        )
        bitcoin_max_active_miners_equiv = float(
            bitcoin_active_miners_equiv.max()
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

        if modeled_hours > 0.0:
            bitcoin_mean_active_miners_equiv = float(
                (
                    bitcoin_active_miners_equiv
                    * weights
                ).sum()
                / modeled_hours
            )

        # Maximum active hashrate
        bitcoin_max_active_hashrate_ph_s = (
            bitcoin_max_active_miners_equiv
            * BTC_MINER_HASHRATE_TH_S
            / 1000.0
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

            bitcoin_equivalent_full_capacity_hours = (
                bitcoin_consumption_mwh
                / bitcoin_capacity_mw
            )

        if bitcoin_capacity_mw > 1e-9:
            bitcoin_full_capacity_hours = float(
                weights[
                    bitcoin_consumption
                    >= 0.99 * bitcoin_capacity_mw
                ].sum()
            )
        else:
            bitcoin_full_capacity_hours = 0.0

        bitcoin_zero_dispatch_hours = float(
            weights[
                bitcoin_consumption
                <= dispatch_tolerance_mw
            ].sum()
        )

        print("\n--- Bitcoin mining validation ---")

        print(
            f"Reference ASIC: "
            f"{BTC_MINER_NAME}"
        )

        print(
            f"ASIC power: "
            f"{BTC_MINER_POWER_KW:.3f} kW/miner"
        )

        print(
            f"ASIC hashrate: "
            f"{BTC_MINER_HASHRATE_TH_S:.1f} TH/s/miner"
        )

        print(
            f"PUE: "
            f"{bitcoin_pue:.3f}"
        )

        print(
            f"Facility-side power per miner: "
            f"{bitcoin_facility_power_per_miner_kw:.5f} kW"
        )

        print(
            f"Capacity mode: "
            f"{bitcoin_capacity_mode}"
        )

        if bitcoin_capacity_mode == "fixed":
            print(
                f"Fixed mining capacity: "
                f"{bitcoin_capacity_mw:.3f} MW"
            )
        else:
            print(
                f"Optimized mining capacity: "
                f"{bitcoin_capacity_mw:.3f} MW"
            )

        print(
            f"Equivalent installed miners: "
            f"{bitcoin_installed_miners_equiv:,.2f}"
        )

        print(
            f"Complete miners within capacity: "
            f"{bitcoin_installed_miners_complete:,d}"
        )

        print(
            f"Installed hashrate: "
            f"{bitcoin_installed_hashrate_ph_s:,.3f} PH/s"
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
            f"Average active miners: "
            f"{bitcoin_mean_active_miners_equiv:,.2f}"
        )

        print(
            f"Maximum active miners: "
            f"{bitcoin_max_active_miners_equiv:,.2f}"
        )

        print(
            f"Maximum active hashrate: "
            f"{bitcoin_max_active_hashrate_ph_s:,.3f} PH/s"
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
            f"{bitcoin_equivalent_full_capacity_hours:.2f} h"
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
    hmax_stage1_hydrogen_mwh = np.nan
    hmax_tolerance_mwh = np.nan
    hmax_stage2_minimum_hydrogen_mwh = np.nan

    if hydrogen_mode == "production_target":
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

    elif hydrogen_mode == "maximize_production":
        network_meta = dict(
            getattr(
                n,
                "meta",
                {},
            )
            or {}
        )

        required_hmax_metadata = [
            "hmax_stage1_hydrogen_mwh",
            "hmax_tolerance_mwh",
            "hmax_stage2_minimum_hydrogen_mwh",
        ]

        missing_hmax_metadata = [
            key
            for key in required_hmax_metadata
            if key not in network_meta
        ]

        if missing_hmax_metadata:
            raise RuntimeError(
                "HMAX validation metadata are missing from the "
                f"solved network: {missing_hmax_metadata}"
            )

        hmax_stage1_hydrogen_mwh = float(
            network_meta[
                "hmax_stage1_hydrogen_mwh"
            ]
        )

        hmax_tolerance_mwh = float(
            network_meta[
                "hmax_tolerance_mwh"
            ]
        )

        hmax_stage2_minimum_hydrogen_mwh = float(
            network_meta[
                "hmax_stage2_minimum_hydrogen_mwh"
            ]
        )

        hmax_values = [
            hmax_stage1_hydrogen_mwh,
            hmax_tolerance_mwh,
            hmax_stage2_minimum_hydrogen_mwh,
            hydrogen_delivery_mwh,
        ]

        if not all(
            np.isfinite(value)
            for value in hmax_values
        ):
            raise RuntimeError(
                "HMAX validation encountered a non-finite "
                "hydrogen quantity."
            )

        if hmax_stage1_hydrogen_mwh <= 0.0:
            raise RuntimeError(
                "HMAX Stage-1 hydrogen production must be positive."
            )

        if hmax_tolerance_mwh < 0.0:
            raise RuntimeError(
                "HMAX numerical tolerance must be non-negative."
            )

        if hmax_stage2_minimum_hydrogen_mwh <= 0.0:
            raise RuntimeError(
                "HMAX Stage-2 minimum hydrogen production "
                "must be positive."
            )

        expected_stage2_minimum = (
            hmax_stage1_hydrogen_mwh
            - hmax_tolerance_mwh
        )

        metadata_tolerance_mwh = max(
            1e-6,
            hmax_stage1_hydrogen_mwh
            * 1e-10,
        )

        if not np.isclose(
            hmax_stage2_minimum_hydrogen_mwh,
            expected_stage2_minimum,
            rtol=0.0,
            atol=metadata_tolerance_mwh,
        ):
            raise RuntimeError(
                "HMAX metadata are internally inconsistent: "
                f"stage1={hmax_stage1_hydrogen_mwh:.6f} MWh, "
                f"tolerance={hmax_tolerance_mwh:.6f} MWh, "
                f"stage2_minimum="
                f"{hmax_stage2_minimum_hydrogen_mwh:.6f} MWh."
            )

        # Numerical feasibility guard for checking the solved LP.
        # This is deliberately much tighter than the configurable
        # Stage-2 HMAX production allowance.
        hmax_validation_tolerance_mwh = max(
            1e-6,
            hmax_stage1_hydrogen_mwh
            * 1e-12,
        )

        # Stage 2 may use the numerical allowance deliberately, so its
        # production can be slightly below the Stage-1 maximum.
        if (
            hydrogen_delivery_mwh
            + hmax_validation_tolerance_mwh
            < hmax_stage2_minimum_hydrogen_mwh
        ):
            raise RuntimeError(
                "HMAX Stage-2 hydrogen production fell below "
                "the required Stage-1 production floor: "
                f"minimum="
                f"{hmax_stage2_minimum_hydrogen_mwh:.6f} MWh, "
                f"actual={hydrogen_delivery_mwh:.6f} MWh."
            )

        # Conversely, Stage 2 must not materially exceed the Stage-1
        # physical maximum. Such a result would indicate that Stage 1
        # was not actually solved to the maximum.
        if (
            hydrogen_delivery_mwh
            - hmax_validation_tolerance_mwh
            > hmax_stage1_hydrogen_mwh
        ):
            raise RuntimeError(
                "HMAX Stage-2 hydrogen production materially "
                "exceeded the Stage-1 maximum: "
                f"stage1="
                f"{hmax_stage1_hydrogen_mwh:.6f} MWh, "
                f"stage2={hydrogen_delivery_mwh:.6f} MWh."
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
    if hydrogen_mode == "production_target":
        print(
            f"  H2 target:               "
            f"{target_mwh_h2:,.3f} MWh_H2"
        )

    elif hydrogen_mode == "maximize_production":
        print(
            f"  HMAX Stage-1 maximum:    "
            f"{hmax_stage1_hydrogen_mwh:,.3f} MWh_H2"
        )
        print(
            f"  HMAX Stage-2 floor:      "
            f"{hmax_stage2_minimum_hydrogen_mwh:,.3f} MWh_H2"
        )
        print(
            f"  Gap to Stage-1 maximum:  "
            f"{hmax_stage1_hydrogen_mwh - hydrogen_delivery_mwh:,.6f} "
            f"MWh_H2"
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

    if hydrogen_mode == "production_target":
        print(
            "  Annual H2 target:        PASS"
        )
    elif hydrogen_mode == "maximize_production":
        print(
            "  HMAX production:         PASS"
        )
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
    # Hydrogen optimization mode
    # -------------------------------------------------------------------------
    hydrogen_mode = str(
        cfg["hydrogen"].get(
            "mode",
            "production_target",
        )
    ).strip().lower()

    # -------------------------------------------------------------------------
    # Standard fixed-production-target optimization
    # -------------------------------------------------------------------------
    if hydrogen_mode == "production_target":

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

        status, condition = n.optimize(
            solver_name="highs",
            extra_functionality=extra_functionality,
            include_objective_constant=False,
        )

    # -------------------------------------------------------------------------
    # Two-stage maximum-hydrogen optimization
    # -------------------------------------------------------------------------
    elif hydrogen_mode == "maximize_production":

        print(
            "\n================================================"
        )
        print(
            "HMAX STAGE 1: MAXIMIZE ANNUAL HYDROGEN"
        )
        print(
            "================================================"
        )

        # Build a separate network for Stage 1 so that the normal economic
        # objective of the final network remains untouched.
        n_hmax = build_test_network(
            cfg,
            data_dir=DATA_DIR,
        )

        n_hmax.optimize.create_model(
            include_objective_constant=False,
        )

        add_battery_power_coupling_constraint(
            n_hmax,
            cfg,
        )

        annual_hydrogen_delivery_stage1 = (
            get_annual_hydrogen_delivery_expression(
                n_hmax,
                n_hmax.snapshots,
            )
        )

        # Replace PyPSA's economic objective with pure annual-H2
        # maximization. Stage 1 is therefore a physical/resource-potential
        # optimization, not an economic optimization.
        n_hmax.model.add_objective(
            annual_hydrogen_delivery_stage1,
            overwrite=True,
            sense="max",
        )

        status_stage1, condition_stage1 = (
            n_hmax.optimize.solve_model(
                solver_name="highs",
            )
        )

        print(
            "\nHMAX Stage 1 finished"
        )
        print(
            f"Status: {status_stage1}"
        )
        print(
            f"Condition: {condition_stage1}"
        )

        if (
            status_stage1 != "ok"
            or condition_stage1 != "optimal"
        ):
            raise RuntimeError(
                "HMAX Stage 1 did not solve cleanly: "
                f"status={status_stage1}, "
                f"condition={condition_stage1}"
            )

        stage1_weights = (
            n_hmax.snapshot_weightings.generators
            .reindex(n_hmax.snapshots)
        )

        stage1_hydrogen_delivery_mwh = float(
            (
                n_hmax.generators_t.p[
                    "hydrogen_delivery"
                ]
                * stage1_weights
            ).sum()
        )

        if (
            not np.isfinite(
                stage1_hydrogen_delivery_mwh
            )
            or stage1_hydrogen_delivery_mwh <= 0.0
        ):
            raise RuntimeError(
                "HMAX Stage 1 returned a non-positive or "
                "non-finite annual hydrogen quantity: "
                f"{stage1_hydrogen_delivery_mwh!r}"
            )

        # ---------------------------------------------------------------------
        # Configurable lexicographic HMAX tolerance
        # ---------------------------------------------------------------------
        # Defaults reproduce the previously validated formulation exactly:
        #
        #   max(1e-3 MWh_H2, Stage-1 H2 * 1e-8)
        #
        # The parameters are configurable so that the numerical sensitivity
        # of the Stage-2 solution can be tested explicitly.
        hmax_cfg = cfg.get(
            "hydrogen",
            {},
        )

        hmax_relative_tolerance = float(
            hmax_cfg.get(
                "hmax_relative_tolerance",
                1e-8,
            )
        )

        hmax_absolute_tolerance_mwh = float(
            hmax_cfg.get(
                "hmax_absolute_tolerance_mwh",
                1e-3,
            )
        )

        if (
            not np.isfinite(
                hmax_relative_tolerance
            )
            or hmax_relative_tolerance < 0.0
        ):
            raise ValueError(
                "hydrogen.hmax_relative_tolerance must be "
                "finite and non-negative."
            )

        if (
            not np.isfinite(
                hmax_absolute_tolerance_mwh
            )
            or hmax_absolute_tolerance_mwh < 0.0
        ):
            raise ValueError(
                "hydrogen.hmax_absolute_tolerance_mwh must be "
                "finite and non-negative."
            )

        hmax_tolerance_mwh = max(
            hmax_absolute_tolerance_mwh,
            stage1_hydrogen_delivery_mwh
            * hmax_relative_tolerance,
        )

        stage2_minimum_hydrogen_mwh = (
            stage1_hydrogen_delivery_mwh
            - hmax_tolerance_mwh
        )

        print(
            f"Maximum annual H2 from Stage 1: "
            f"{stage1_hydrogen_delivery_mwh:,.6f} MWh_H2"
        )
        print(
            f"HMAX relative tolerance: "
            f"{hmax_relative_tolerance:.12g}"
        )
        print(
            f"HMAX absolute tolerance: "
            f"{hmax_absolute_tolerance_mwh:.12g} MWh_H2"
        )
        print(
            f"Stage-2 effective tolerance: "
            f"{hmax_tolerance_mwh:.6f} MWh_H2"
        )
        print(
            f"Stage-2 minimum H2: "
            f"{stage2_minimum_hydrogen_mwh:,.6f} MWh_H2"
        )

        print(
            "\n================================================"
        )
        print(
            "HMAX STAGE 2: MINIMIZE SYSTEM COST"
        )
        print(
            "================================================"
        )

        # Rebuild the network from the same deterministic inputs.
        # This restores PyPSA's normal annualized system-cost objective.
        n = build_test_network(
            cfg,
            data_dir=DATA_DIR,
        )

        def extra_functionality(
            network,
            snapshots,
        ):
            add_hydrogen_minimum_constraint(
                network,
                snapshots,
                stage2_minimum_hydrogen_mwh,
            )

            add_battery_power_coupling_constraint(
                network,
                cfg,
            )

        status, condition = n.optimize(
            solver_name="highs",
            extra_functionality=extra_functionality,
            include_objective_constant=False,
        )

        # Store Stage-1 information as network metadata so that the analysis
        # script can report and validate the lexicographic HMAX result later.
        n.meta = dict(
            getattr(
                n,
                "meta",
                {},
            )
            or {}
        )

        n.meta[
            "hydrogen_mode"
        ] = "maximize_production"

        n.meta[
            "hmax_stage1_hydrogen_mwh"
        ] = float(
            stage1_hydrogen_delivery_mwh
        )

        n.meta[
            "hmax_relative_tolerance"
        ] = float(
            hmax_relative_tolerance
        )

        n.meta[
            "hmax_absolute_tolerance_mwh"
        ] = float(
            hmax_absolute_tolerance_mwh
        )

        n.meta[
            "hmax_tolerance_mwh"
        ] = float(
            hmax_tolerance_mwh
        )

        n.meta[
            "hmax_stage2_minimum_hydrogen_mwh"
        ] = float(
            stage2_minimum_hydrogen_mwh
        )

    else:
        raise ValueError(
            "Unsupported hydrogen.mode in run_model.py: "
            f"{hydrogen_mode!r}"
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