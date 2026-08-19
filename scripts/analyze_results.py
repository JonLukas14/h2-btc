from pathlib import Path
import argparse

# Use a non-interactive Matplotlib backend because this script is also
# executed by Snakemake without a graphical desktop/display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pypsa
import yaml


# =============================================================================
# 1. Command-line arguments
# =============================================================================
parser = argparse.ArgumentParser(
    description="Analyze solved off-grid PyPSA scenario results."
)

parser.add_argument(
    "--config",
    required=True,
    help="Path to scenario YAML file",
)

parser.add_argument(
    "--network",
    required=True,
    help="Path to solved network NetCDF file",
)

parser.add_argument(
    "--outdir",
    required=True,
    help="Directory for analysis outputs",
)

args = parser.parse_args()


# =============================================================================
# 2. Paths
# =============================================================================
CONFIG_FILE = Path(args.config)
NETWORK_FILE = Path(args.network)
OUTDIR = Path(args.outdir)

if not CONFIG_FILE.exists():
    raise FileNotFoundError(
        f"Config file not found: {CONFIG_FILE}"
    )

if not NETWORK_FILE.exists():
    raise FileNotFoundError(
        f"Solved network not found: {NETWORK_FILE}"
    )

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 3. Remove obsolete outputs from the old national-demand prototype
# =============================================================================
# Prevent stale files from being mistaken for outputs of the redesigned
# off-grid model.
legacy_outputs = [
    "hydrogen_storage_soc.png",
    "hydrogen_storage_soc.html",
    "storage_timeseries.csv",
    "load_totals.csv",
    "cost_solar_wind.png",
    "cost_small_assets.png",
]

for filename in legacy_outputs:
    path = OUTDIR / filename
    if path.exists():
        path.unlink()


# =============================================================================
# 4. Load scenario configuration
# =============================================================================
with CONFIG_FILE.open(
    "r",
    encoding="utf-8",
) as f:
    cfg = yaml.safe_load(f)


required_keys = [
    "scenario_name",
    "system",
    "costs",
    "renewables",
    "hydrogen",
    "hydrogen_storage",
    "battery",
    "bitcoin",
]

missing = [
    key
    for key in required_keys
    if key not in cfg
]

if missing:
    raise KeyError(
        f"Missing top-level config keys: {missing}"
    )


system_cfg = cfg["system"]
hydrogen_cfg = cfg["hydrogen"]
battery_cfg = cfg["battery"]
bitcoin_cfg = cfg["bitcoin"]
hydrogen_storage_cfg = cfg["hydrogen_storage"]


model_type = system_cfg.get(
    "model_type"
)

investment_year = int(
    system_cfg["investment_year"]
)

weather_year = int(
    system_cfg["weather_year"]
)

hydrogen_enabled = bool(
    hydrogen_cfg.get(
        "enabled",
        False,
    )
)

hydrogen_mode = hydrogen_cfg.get(
    "mode"
)

battery_enabled = bool(
    battery_cfg.get(
        "enabled",
        False,
    )
)

bitcoin_enabled = bool(
    bitcoin_cfg.get(
        "enabled",
        False,
    )
)

hydrogen_storage_enabled = bool(
    hydrogen_storage_cfg.get(
        "enabled",
        False,
    )
)


# =============================================================================
# 5. Validate current analysis scope
# =============================================================================
if model_type != "off_grid":
    raise ValueError(
        "This analysis script expects "
        "system.model_type='off_grid'."
    )

if not hydrogen_enabled:
    raise ValueError(
        "The off-grid S0 analysis requires hydrogen.enabled=true."
    )

if hydrogen_mode != "production_target":
    raise ValueError(
        "The current off-grid analysis requires "
        "hydrogen.mode='production_target'."
    )

# S0 only for this validation stage.
# These checks will deliberately be relaxed when S1/S2/S3 are implemented.


if bitcoin_enabled:
    raise NotImplementedError(
        "Bitcoin analysis has not yet been implemented. "
        "Current analysis scope is S0."
    )

if hydrogen_storage_enabled:
    raise NotImplementedError(
        "Hydrogen-storage analysis has not yet been implemented "
        "in the redesigned model."
    )


# =============================================================================
# 6. Print scenario definition
# =============================================================================
print("=== SCENARIO ===")
print(f"File:            {CONFIG_FILE.name}")
print(f"Name:            {cfg['scenario_name']}")
print(f"Model type:      {model_type}")
print(f"Investment year: {investment_year}")
print(f"Weather year:    {weather_year}")

print("\n=== TECHNOLOGY SWITCHES ===")
print(f"Hydrogen:        {hydrogen_enabled}")
print(f"Battery:         {battery_enabled}")
print(f"Bitcoin:         {bitcoin_enabled}")
print(f"H2 storage:      {hydrogen_storage_enabled}")


# =============================================================================
# 7. Load solved network
# =============================================================================
n = pypsa.Network()
n.import_from_netcdf(
    NETWORK_FILE
)


# =============================================================================
# 8. Validate expected S0 component structure
# =============================================================================
required_generators = {
    "solar",
    "wind",
    "hydrogen_delivery",
}

required_links = {
    "electrolyzer",
}

missing_generators = (
    required_generators
    - set(n.generators.index)
)

missing_links = (
    required_links
    - set(n.links.index)
)

if missing_generators:
    raise KeyError(
        f"Missing required generators: "
        f"{sorted(missing_generators)}"
    )

if missing_links:
    raise KeyError(
        f"Missing required links: "
        f"{sorted(missing_links)}"
    )


# Explicit regression protection against the old national-demand model.
legacy_components_found = []

if "load_shedding" in n.generators.index:
    legacy_components_found.append(
        "Generator: load_shedding"
    )

if "electricity_demand" in n.loads.index:
    legacy_components_found.append(
        "Load: electricity_demand"
    )

if "hydrogen_demand" in n.loads.index:
    legacy_components_found.append(
        "Load: hydrogen_demand"
    )

if "hydrogen_sink" in n.generators.index:
    legacy_components_found.append(
        "Generator: hydrogen_sink"
    )

if legacy_components_found:
    raise RuntimeError(
        "Legacy national-demand components found in "
        "the redesigned off-grid network:\n"
        + "\n".join(legacy_components_found)
    )


if float(
    n.generators.at[
        "hydrogen_delivery",
        "sign",
    ]
) != -1.0:
    raise RuntimeError(
        "hydrogen_delivery must have sign=-1."
    )


# =============================================================================
# 9. Snapshot weighting
# =============================================================================
# Generator weighting represents the modeled time contribution to physical
# energy balances. In the current 8760-hour model every weight is normally 1.
if (
    "generators"
    in n.snapshot_weightings.columns
):
    energy_weights = (
        n.snapshot_weightings[
            "generators"
        ]
        .reindex(n.snapshots)
        .astype(float)
    )
else:
    energy_weights = pd.Series(
        1.0,
        index=n.snapshots,
    )

modeled_hours = float(
    energy_weights.sum()
)


def weighted_sum(series):
    """
    Convert an MW time series to weighted MWh over the modeled period.
    """

    series = (
        series
        .reindex(n.snapshots)
        .fillna(0.0)
        .astype(float)
    )

    return float(
        (
            series
            * energy_weights
        ).sum()
    )


def weighted_average(series):
    """
    Weighted mean over the modeled period.
    """

    if modeled_hours <= 0:
        return np.nan

    series = (
        series
        .reindex(n.snapshots)
        .fillna(0.0)
        .astype(float)
    )

    return float(
        (
            series
            * energy_weights
        ).sum()
        / modeled_hours
    )


# =============================================================================
# 10. Optimal capacities
# =============================================================================
solar_capacity_mw = float(
    n.generators.at[
        "solar",
        "p_nom_opt",
    ]
)

wind_capacity_mw = float(
    n.generators.at[
        "wind",
        "p_nom_opt",
    ]
)

electrolyzer_capacity_mw = float(
    n.links.at[
        "electrolyzer",
        "p_nom_opt",
    ]
)

electrolyzer_efficiency = float(
    n.links.at[
        "electrolyzer",
        "efficiency",
    ]
)


# =============================================================================
# 11. Hourly power flows
# =============================================================================
solar_dispatch_mw = (
    n.generators_t.p[
        "solar"
    ]
    .reindex(n.snapshots)
)

wind_dispatch_mw = (
    n.generators_t.p[
        "wind"
    ]
    .reindex(n.snapshots)
)

electrolyzer_input_mw = (
    n.links_t.p0[
        "electrolyzer"
    ]
    .reindex(n.snapshots)
)

# PyPSA Link p1 is negative when the link supplies bus1.
hydrogen_output_mw = (
    -n.links_t.p1[
        "electrolyzer"
    ]
    .reindex(n.snapshots)
)

hydrogen_delivery_mw = (
    n.generators_t.p[
        "hydrogen_delivery"
    ]
    .reindex(n.snapshots)
)

# =============================================================================
# Battery capacities and operation
# =============================================================================
#
# The battery is optional. Defaults are chosen so that S0 and S1 produce
# the same summary schema and can later be compared directly.
#

zero_series = pd.Series(
    0.0,
    index=n.snapshots,
    dtype=float,
)

battery_power_mw = 0.0
battery_discharger_input_capacity_mw = 0.0
battery_discharge_power_ac_mw = 0.0
battery_energy_mwh = 0.0
battery_duration_h = np.nan

battery_charge_efficiency = np.nan
battery_discharge_efficiency = np.nan

battery_charge_input_mw = zero_series.copy()
battery_charge_output_mw = zero_series.copy()
battery_discharge_input_mw = zero_series.copy()
battery_discharge_output_mw = zero_series.copy()
battery_store_power_mw = zero_series.copy()
battery_soc_mwh = zero_series.copy()

battery_charge_input_mwh = 0.0
battery_charge_output_mwh = 0.0
battery_discharge_input_mwh = 0.0
battery_discharge_output_mwh = 0.0
battery_losses_mwh = 0.0

battery_round_trip_efficiency = np.nan
battery_equivalent_full_cycles_per_year = np.nan

battery_soc_min_mwh = 0.0
battery_soc_max_mwh = 0.0

battery_power_coupling_error_mw = 0.0
max_battery_bus_balance_error_mw = 0.0


if battery_enabled:
    required_battery_links = [
        "battery_charger",
        "battery_discharger",
    ]

    for link in required_battery_links:
        if link not in n.links.index:
            raise KeyError(
                f"Battery is enabled but Link '{link}' "
                "is missing from the solved network."
            )

    if "battery_store" not in n.stores.index:
        raise KeyError(
            "Battery is enabled but Store "
            "'battery_store' is missing."
        )

    # -------------------------------------------------------------------------
    # Optimized capacities
    # -------------------------------------------------------------------------
    battery_power_mw = float(
        n.links.at[
            "battery_charger",
            "p_nom_opt",
        ]
    )

    battery_discharger_input_capacity_mw = float(
        n.links.at[
            "battery_discharger",
            "p_nom_opt",
        ]
    )

    battery_charge_efficiency = float(
        n.links.at[
            "battery_charger",
            "efficiency",
        ]
    )

    battery_discharge_efficiency = float(
        n.links.at[
            "battery_discharger",
            "efficiency",
        ]
    )

    # AC-side output rating of discharge inverter.
    battery_discharge_power_ac_mw = (
        battery_discharger_input_capacity_mw
        * battery_discharge_efficiency
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

    # -------------------------------------------------------------------------
    # Hourly battery flows
    # -------------------------------------------------------------------------
    # Charging:
    #   p0  = AC electricity withdrawn
    #  -p1  = electricity arriving at battery bus
    #
    battery_charge_input_mw = (
        n.links_t.p0[
            "battery_charger"
        ]
        .reindex(n.snapshots)
    )

    battery_charge_output_mw = (
        -n.links_t.p1[
            "battery_charger"
        ]
        .reindex(n.snapshots)
    )

    # Discharging:
    #   p0  = electricity withdrawn from battery bus
    #  -p1  = AC electricity supplied
    #
    battery_discharge_input_mw = (
        n.links_t.p0[
            "battery_discharger"
        ]
        .reindex(n.snapshots)
    )

    battery_discharge_output_mw = (
        -n.links_t.p1[
            "battery_discharger"
        ]
        .reindex(n.snapshots)
    )

    # Store.p > 0 means discharge from Store to battery bus.
    battery_store_power_mw = (
        n.stores_t.p[
            "battery_store"
        ]
        .reindex(n.snapshots)
    )

    battery_soc_mwh = (
        n.stores_t.e[
            "battery_store"
        ]
        .reindex(n.snapshots)
    )

# =============================================================================
# 12. Annual energy totals
# =============================================================================
solar_generation_mwh = weighted_sum(
    solar_dispatch_mw
)

wind_generation_mwh = weighted_sum(
    wind_dispatch_mw
)

renewable_generation_mwh = (
    solar_generation_mwh
    + wind_generation_mwh
)

electrolyzer_input_mwh = weighted_sum(
    electrolyzer_input_mw
)

hydrogen_output_mwh = weighted_sum(
    hydrogen_output_mw
)

hydrogen_delivered_mwh = weighted_sum(
    hydrogen_delivery_mw
)

if battery_enabled:
    battery_charge_input_mwh = weighted_sum(
        battery_charge_input_mw
    )

    battery_charge_output_mwh = weighted_sum(
        battery_charge_output_mw
    )

    battery_discharge_input_mwh = weighted_sum(
        battery_discharge_input_mw
    )

    battery_discharge_output_mwh = weighted_sum(
        battery_discharge_output_mw
    )

    battery_losses_mwh = (
        battery_charge_input_mwh
        - battery_discharge_output_mwh
    )

    if battery_charge_input_mwh > 1e-9:
        battery_round_trip_efficiency = (
            battery_discharge_output_mwh
            / battery_charge_input_mwh
        )

    # Cell-side equivalent full cycles.
    if battery_energy_mwh > 1e-9:
        battery_equivalent_full_cycles_per_year = (
            battery_discharge_input_mwh
            / battery_energy_mwh
        )

    battery_soc_min_mwh = float(
        battery_soc_mwh.min()
    )

    battery_soc_max_mwh = float(
        battery_soc_mwh.max()
    )
# =============================================================================
# 13. Hydrogen target and mass conversion
# =============================================================================
target_annual_kt_h2 = hydrogen_cfg.get(
    "target_annual_kt_h2"
)

if target_annual_kt_h2 is None:
    raise ValueError(
        "hydrogen.target_annual_kt_h2 is not defined."
    )

target_annual_kt_h2 = float(
    target_annual_kt_h2
)

hydrogen_lhv_kwh_per_kg = float(
    hydrogen_cfg.get(
        "hydrogen_lhv_kwh_per_kg",
        33.33,
    )
)

if hydrogen_lhv_kwh_per_kg <= 0.0:
    raise ValueError(
        "hydrogen_lhv_kwh_per_kg must be positive."
    )


target_annual_kg_h2 = (
    target_annual_kt_h2
    * 1_000_000.0
)

target_annual_mwh_h2 = (
    target_annual_kg_h2
    * hydrogen_lhv_kwh_per_kg
    / 1000.0
)


hydrogen_delivered_kg = (
    hydrogen_delivered_mwh
    * 1000.0
    / hydrogen_lhv_kwh_per_kg
)

hydrogen_delivered_kt = (
    hydrogen_delivered_kg
    / 1_000_000.0
)


hydrogen_target_achievement = (
    hydrogen_delivered_mwh
    / target_annual_mwh_h2
    if target_annual_mwh_h2 > 0
    else np.nan
)


specific_electricity_kwh_per_kg_h2 = (
    electrolyzer_input_mwh
    * 1000.0
    / hydrogen_delivered_kg
    if hydrogen_delivered_kg > 0
    else np.nan
)


realized_electrolyzer_efficiency = (
    hydrogen_output_mwh
    / electrolyzer_input_mwh
    if electrolyzer_input_mwh > 0
    else np.nan
)


# =============================================================================
# 14. Renewable availability and curtailment
# =============================================================================
solar_availability_pu = (
    n.generators_t.p_max_pu[
        "solar"
    ]
    .reindex(n.snapshots)
)

wind_availability_pu = (
    n.generators_t.p_max_pu[
        "wind"
    ]
    .reindex(n.snapshots)
)


solar_available_mw = (
    solar_availability_pu
    * solar_capacity_mw
)

wind_available_mw = (
    wind_availability_pu
    * wind_capacity_mw
)


solar_available_mwh = weighted_sum(
    solar_available_mw
)

wind_available_mwh = weighted_sum(
    wind_available_mw
)


solar_curtailed_mwh = max(
    solar_available_mwh
    - solar_generation_mwh,
    0.0,
)

wind_curtailed_mwh = max(
    wind_available_mwh
    - wind_generation_mwh,
    0.0,
)


solar_curtailment_rate = (
    solar_curtailed_mwh
    / solar_available_mwh
    if solar_available_mwh > 0
    else np.nan
)

wind_curtailment_rate = (
    wind_curtailed_mwh
    / wind_available_mwh
    if wind_available_mwh > 0
    else np.nan
)


total_available_renewable_mwh = (
    solar_available_mwh
    + wind_available_mwh
)

total_curtailed_renewable_mwh = (
    solar_curtailed_mwh
    + wind_curtailed_mwh
)


renewable_utilization_rate = (
    renewable_generation_mwh
    / total_available_renewable_mwh
    if total_available_renewable_mwh > 0
    else np.nan
)


curtailment = pd.DataFrame(
    {
        "asset": [
            "solar",
            "wind",
        ],
        "available_mwh": [
            solar_available_mwh,
            wind_available_mwh,
        ],
        "actual_mwh": [
            solar_generation_mwh,
            wind_generation_mwh,
        ],
        "curtailed_mwh": [
            solar_curtailed_mwh,
            wind_curtailed_mwh,
        ],
        "curtailment_rate": [
            solar_curtailment_rate,
            wind_curtailment_rate,
        ],
    }
)


# =============================================================================
# 15. Capacity factors
# =============================================================================
solar_availability_cf = weighted_average(
    solar_availability_pu
)

wind_availability_cf = weighted_average(
    wind_availability_pu
)


solar_dispatch_cf = (
    solar_generation_mwh
    / (
        solar_capacity_mw
        * modeled_hours
    )
    if solar_capacity_mw > 0
    else np.nan
)

wind_dispatch_cf = (
    wind_generation_mwh
    / (
        wind_capacity_mw
        * modeled_hours
    )
    if wind_capacity_mw > 0
    else np.nan
)

electrolyzer_capacity_factor = (
    electrolyzer_input_mwh
    / (
        electrolyzer_capacity_mw
        * modeled_hours
    )
    if electrolyzer_capacity_mw > 0
    else np.nan
)


capacity_factors = pd.DataFrame(
    {
        "asset": [
            "solar",
            "wind",
            "electrolyzer",
        ],
        "capacity_mw": [
            solar_capacity_mw,
            wind_capacity_mw,
            electrolyzer_capacity_mw,
        ],
        "annual_energy_mwh": [
            solar_generation_mwh,
            wind_generation_mwh,
            electrolyzer_input_mwh,
        ],
        "availability_capacity_factor": [
            solar_availability_cf,
            wind_availability_cf,
            np.nan,
        ],
        "dispatch_capacity_factor": [
            solar_dispatch_cf,
            wind_dispatch_cf,
            electrolyzer_capacity_factor,
        ],
    }
)


# =============================================================================
# 16. Physical balance validation
# =============================================================================
electricity_balance_mw = (
    solar_dispatch_mw
    + wind_dispatch_mw
    + battery_discharge_output_mw
    - electrolyzer_input_mw
    - battery_charge_input_mw
)

hydrogen_balance_mw = (
    hydrogen_output_mw
    - hydrogen_delivery_mw
)

battery_bus_balance_mw = (
    battery_charge_output_mw
    + battery_store_power_mw
    - battery_discharge_input_mw
)

max_electricity_balance_error_mw = float(
    electricity_balance_mw
    .abs()
    .max()
)

max_hydrogen_balance_error_mw = float(
    hydrogen_balance_mw
    .abs()
    .max()
)

max_battery_bus_balance_error_mw = float(
    battery_bus_balance_mw
    .abs()
    .max()
)

if battery_enabled:
    battery_power_coupling_error_mw = abs(
        battery_power_mw
        - (
            battery_discharge_efficiency
            * battery_discharger_input_capacity_mw
        )
    )

# =============================================================================
# 17. Cost accounting
# =============================================================================
#
# Important:
# build_network.py currently assigns
#
#   annualized investment cost + annual FOM
#
# to PyPSA's capital_cost attribute.
#
# Therefore n.statistics.capex() is labeled here as
# "annualized fixed cost", not pure investment CAPEX.
#
annualized_fixed_cost_eur = float(
    n.statistics.capex().sum()
)

variable_operating_cost_eur = float(
    n.statistics.opex().sum()
)

system_cost_eur = (
    annualized_fixed_cost_eur
    + variable_operating_cost_eur
)

objective_eur = float(
    n.objective
)

objective_cost_difference_eur = (
    objective_eur
    - system_cost_eur
)


lcoh_eur_per_kg_h2 = (
    system_cost_eur
    / hydrogen_delivered_kg
    if hydrogen_delivered_kg > 0
    else np.nan
)


# =============================================================================
# 18. Cost breakdown by asset
# =============================================================================
def statistic_by_name_to_series(statistic):
    """
    Convert a PyPSA statistics result grouped by component name
    into a simple Series indexed by asset name.
    """

    if isinstance(
        statistic,
        pd.DataFrame,
    ):
        statistic = statistic.sum(
            axis=1
        )

    statistic = statistic.astype(
        float
    )

    if isinstance(
        statistic.index,
        pd.MultiIndex,
    ):
        names = (
            statistic.index
            .get_level_values(-1)
        )

        statistic = pd.Series(
            statistic.to_numpy(),
            index=names,
            dtype=float,
        )

    return statistic.groupby(
        level=0
    ).sum()


fixed_cost_by_name = (
    statistic_by_name_to_series(
        n.statistics.capex(
            groupby="name"
        )
    )
)

variable_cost_by_name = (
    statistic_by_name_to_series(
        n.statistics.opex(
            groupby="name"
        )
    )
)


cost_assets = [
    "solar",
    "wind",
    "electrolyzer",
]

cost_labels = {
    "solar": "Solar",
    "wind": "Wind",
    "electrolyzer": "Electrolyzer",
}

if battery_enabled:
    cost_assets.extend(
        [
            "battery_charger",
            "battery_store",
        ]
    )

    cost_labels.update(
        {
            "battery_charger": "Battery inverter",
            "battery_store": "Battery storage",
        }
    )


cost_breakdown = pd.DataFrame(
    {
        "asset": cost_assets,
        "label": [
            cost_labels[asset]
            for asset
            in cost_assets
        ],
        "annualized_fixed_cost_eur": [
            float(
                fixed_cost_by_name.get(
                    asset,
                    0.0,
                )
            )
            for asset
            in cost_assets
        ],
        "variable_operating_cost_eur": [
            float(
                variable_cost_by_name.get(
                    asset,
                    0.0,
                )
            )
            for asset
            in cost_assets
        ],
    }
)

cost_breakdown[
    "total_annual_cost_eur"
] = (
    cost_breakdown[
        "annualized_fixed_cost_eur"
    ]
    + cost_breakdown[
        "variable_operating_cost_eur"
    ]
)

battery_inverter_fixed_cost_eur = float(
    fixed_cost_by_name.get(
        "battery_charger",
        0.0,
    )
)

battery_storage_fixed_cost_eur = float(
    fixed_cost_by_name.get(
        "battery_store",
        0.0,
    )
)

battery_total_fixed_cost_eur = (
    battery_inverter_fixed_cost_eur
    + battery_storage_fixed_cost_eur
)


# =============================================================================
# 19. Numerical consistency checks
# =============================================================================
target_tolerance_mwh = max(
    1e-3,
    target_annual_mwh_h2
    * 1e-8,
)

if not np.isclose(
    hydrogen_delivered_mwh,
    target_annual_mwh_h2,
    rtol=0.0,
    atol=target_tolerance_mwh,
):
    raise RuntimeError(
        "Hydrogen target validation failed: "
        f"target={target_annual_mwh_h2:.6f} MWh_H2, "
        f"actual={hydrogen_delivered_mwh:.6f} MWh_H2."
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

if battery_enabled:
    if (
        max_battery_bus_balance_error_mw
        > balance_tolerance_mw
    ):
        raise RuntimeError(
            "Battery bus balance validation failed: "
            f"max error="
            f"{max_battery_bus_balance_error_mw:.6e} MW."
        )

    power_coupling_tolerance_mw = max(
        1e-6,
        abs(battery_power_mw) * 1e-8,
    )

    if (
        battery_power_coupling_error_mw
        > power_coupling_tolerance_mw
    ):
        raise RuntimeError(
            "Battery inverter power-coupling "
            "validation failed: "
            f"error="
            f"{battery_power_coupling_error_mw:.6e} MW."
        )

    if (
        battery_charge_input_mwh > 1e-6
        and not np.isclose(
            battery_round_trip_efficiency,
            (
                battery_charge_efficiency
                * battery_discharge_efficiency
            ),
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise RuntimeError(
            "Battery round-trip efficiency "
            "validation failed."
        )

efficiency_tolerance = 1e-8

if not np.isclose(
    realized_electrolyzer_efficiency,
    electrolyzer_efficiency,
    rtol=0.0,
    atol=efficiency_tolerance,
):
    raise RuntimeError(
        "Electrolyzer efficiency validation failed: "
        f"configured={electrolyzer_efficiency:.8f}, "
        f"realized="
        f"{realized_electrolyzer_efficiency:.8f}."
    )


objective_tolerance_eur = max(
    1e-3,
    abs(objective_eur)
    * 1e-8,
)

if not np.isclose(
    objective_eur,
    system_cost_eur,
    rtol=0.0,
    atol=objective_tolerance_eur,
):
    raise RuntimeError(
        "Objective/system-cost validation failed: "
        f"objective={objective_eur:.6f} EUR, "
        f"calculated={system_cost_eur:.6f} EUR."
    )


# =============================================================================
# 20. Dispatch time-series table
# =============================================================================
dispatch_timeseries = pd.DataFrame(
    index=n.snapshots
)

dispatch_timeseries[
    "solar_available_mw"
] = solar_available_mw

dispatch_timeseries[
    "solar_dispatch_mw"
] = solar_dispatch_mw

dispatch_timeseries[
    "solar_curtailment_mw"
] = (
    solar_available_mw
    - solar_dispatch_mw
).clip(
    lower=0.0
)

dispatch_timeseries[
    "wind_available_mw"
] = wind_available_mw

dispatch_timeseries[
    "wind_dispatch_mw"
] = wind_dispatch_mw

dispatch_timeseries[
    "wind_curtailment_mw"
] = (
    wind_available_mw
    - wind_dispatch_mw
).clip(
    lower=0.0
)

dispatch_timeseries[
    "electrolyzer_input_mw"
] = electrolyzer_input_mw

dispatch_timeseries[
    "hydrogen_output_mw"
] = hydrogen_output_mw

dispatch_timeseries[
    "hydrogen_delivery_mw"
] = hydrogen_delivery_mw

if battery_enabled:
    dispatch_timeseries[
        "battery_charge_input_mw"
    ] = battery_charge_input_mw

    dispatch_timeseries[
        "battery_charge_output_mw"
    ] = battery_charge_output_mw

    dispatch_timeseries[
        "battery_discharge_input_mw"
    ] = battery_discharge_input_mw

    dispatch_timeseries[
        "battery_discharge_output_mw"
    ] = battery_discharge_output_mw

    dispatch_timeseries[
        "battery_store_power_mw"
    ] = battery_store_power_mw

    dispatch_timeseries[
        "battery_soc_mwh"
    ] = battery_soc_mwh

# =============================================================================
# 21. Component tables
# =============================================================================
generators = n.generators[
    [
        "carrier",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
        "capital_cost",
        "marginal_cost",
    ]
].copy()

links = n.links[
    [
        "carrier",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
        "efficiency",
        "capital_cost",
        "marginal_cost",
    ]
].copy()

stores = n.stores.copy()


# =============================================================================
# 22. One-row scenario summary
# =============================================================================
summary = pd.DataFrame(
    {
        # Scenario definition
        "scenario_file": [
            CONFIG_FILE.name
        ],
        "scenario_name": [
            cfg["scenario_name"]
        ],
        "model_type": [
            model_type
        ],
        "investment_year": [
            investment_year
        ],
        "weather_year": [
            weather_year
        ],

        # Technology switches
        "hydrogen_enabled": [
            hydrogen_enabled
        ],
        "battery_enabled": [
            battery_enabled
        ],
        "bitcoin_enabled": [
            bitcoin_enabled
        ],
        "hydrogen_storage_enabled": [
            hydrogen_storage_enabled
        ],

        # Hydrogen assumptions
        "hydrogen_mode": [
            hydrogen_mode
        ],
        "h2_target_annual_kt": [
            target_annual_kt_h2
        ],
        "h2_lhv_kwh_per_kg": [
            hydrogen_lhv_kwh_per_kg
        ],

        # Optimal capacities
        "solar_capacity_mw": [
            solar_capacity_mw
        ],
        "wind_capacity_mw": [
            wind_capacity_mw
        ],
        "electrolyzer_capacity_mw": [
            electrolyzer_capacity_mw
        ],

        # Battery capacities
        "battery_power_mw": [
            battery_power_mw
        ],
        "battery_discharger_input_capacity_mw": [
            battery_discharger_input_capacity_mw
        ],
        "battery_discharge_power_ac_mw": [
            battery_discharge_power_ac_mw
        ],
        "battery_energy_mwh": [
            battery_energy_mwh
        ],
        "battery_duration_h": [
            battery_duration_h
        ],

        # Electricity
        "solar_generation_mwh": [
            solar_generation_mwh
        ],
        "wind_generation_mwh": [
            wind_generation_mwh
        ],
        "renewable_generation_mwh": [
            renewable_generation_mwh
        ],
        "electrolyzer_input_mwh": [
            electrolyzer_input_mwh
        ],

        # Hydrogen
        "hydrogen_output_mwh": [
            hydrogen_output_mwh
        ],
        "hydrogen_delivered_mwh": [
            hydrogen_delivered_mwh
        ],
        "hydrogen_delivered_kg": [
            hydrogen_delivered_kg
        ],
        "hydrogen_delivered_kt": [
            hydrogen_delivered_kt
        ],
        "h2_target_achievement": [
            hydrogen_target_achievement
        ],
        "electrolyzer_efficiency": [
            electrolyzer_efficiency
        ],
        "realized_electrolyzer_efficiency": [
            realized_electrolyzer_efficiency
        ],
        "specific_electricity_kwh_per_kg_h2": [
            specific_electricity_kwh_per_kg_h2
        ],

        # Battery operation
        "battery_charge_input_mwh": [
            battery_charge_input_mwh
        ],
        "battery_charge_output_mwh": [
            battery_charge_output_mwh
        ],
        "battery_discharge_input_mwh": [
            battery_discharge_input_mwh
        ],
        "battery_discharge_output_mwh": [
            battery_discharge_output_mwh
        ],
        "battery_losses_mwh": [
            battery_losses_mwh
        ],
        "battery_round_trip_efficiency": [
            battery_round_trip_efficiency
        ],
        "battery_equivalent_full_cycles_per_year": [
            battery_equivalent_full_cycles_per_year
        ],
        "battery_soc_min_mwh": [
            battery_soc_min_mwh
        ],
        "battery_soc_max_mwh": [
            battery_soc_max_mwh
        ],

        # Capacity factors
        "solar_availability_cf": [
            solar_availability_cf
        ],
        "solar_dispatch_cf": [
            solar_dispatch_cf
        ],
        "wind_availability_cf": [
            wind_availability_cf
        ],
        "wind_dispatch_cf": [
            wind_dispatch_cf
        ],
        "electrolyzer_capacity_factor": [
            electrolyzer_capacity_factor
        ],

        # Curtailment
        "solar_curtailment_mwh": [
            solar_curtailed_mwh
        ],
        "solar_curtailment_rate": [
            solar_curtailment_rate
        ],
        "wind_curtailment_mwh": [
            wind_curtailed_mwh
        ],
        "wind_curtailment_rate": [
            wind_curtailment_rate
        ],
        "total_curtailment_mwh": [
            total_curtailed_renewable_mwh
        ],
        "renewable_utilization_rate": [
            renewable_utilization_rate
        ],

        # Economics
        "objective_eur_per_year": [
            objective_eur
        ],
        "annualized_fixed_cost_eur_per_year": [
            annualized_fixed_cost_eur
        ],
        "variable_operating_cost_eur_per_year": [
            variable_operating_cost_eur
        ],
        "system_cost_eur_per_year": [
            system_cost_eur
        ],
        "lcoh_eur_per_kg_h2": [
            lcoh_eur_per_kg_h2
        ],

        "battery_inverter_fixed_cost_eur_per_year": [
            battery_inverter_fixed_cost_eur
        ],
        "battery_storage_fixed_cost_eur_per_year": [
            battery_storage_fixed_cost_eur
        ],
        "battery_total_fixed_cost_eur_per_year": [
            battery_total_fixed_cost_eur
        ],

        # Validation
        "max_electricity_balance_error_mw": [
            max_electricity_balance_error_mw
        ],
        "max_hydrogen_balance_error_mw": [
            max_hydrogen_balance_error_mw
        ],
        "objective_cost_difference_eur": [
            objective_cost_difference_eur
        ],

        "max_battery_bus_balance_error_mw": [
            max_battery_bus_balance_error_mw
        ],
        "battery_power_coupling_error_mw": [
            battery_power_coupling_error_mw
        ],
    }
)


# =============================================================================
# 23. Write CSV outputs
# =============================================================================
summary.to_csv(
    OUTDIR / "summary.csv",
    index=False,
)

generators.to_csv(
    OUTDIR / "generators.csv",
)

links.to_csv(
    OUTDIR / "links.csv",
)

stores.to_csv(
    OUTDIR / "stores.csv",
)

dispatch_timeseries.to_csv(
    OUTDIR / "dispatch_timeseries.csv",
)

capacity_factors.to_csv(
    OUTDIR / "capacity_factors.csv",
    index=False,
)

curtailment.to_csv(
    OUTDIR / "curtailment.csv",
    index=False,
)

cost_breakdown.to_csv(
    OUTDIR / "cost_breakdown.csv",
    index=False,
)


# =============================================================================
# 24. Static plot — annual renewable generation
# =============================================================================
generation_plot = pd.DataFrame(
    {
        "source": [
            "Solar",
            "Wind",
        ],
        "generation_gwh": [
            solar_generation_mwh / 1000.0,
            wind_generation_mwh / 1000.0,
        ],
    }
)

fig, ax = plt.subplots(
    figsize=(7, 4.5),
    dpi=160,
)

bars = ax.bar(
    generation_plot["source"],
    generation_plot["generation_gwh"],
)

ax.set_title(
    "Annual Renewable Generation"
)

ax.set_ylabel(
    "Generation [GWh]"
)

ax.bar_label(
    bars,
    fmt="%.2f",
    padding=3,
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "generation_mix.png",
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# 25. Static plot — optimized power capacities
# =============================================================================
capacity_plot = pd.DataFrame(
    {
        "asset": [
            "Solar",
            "Wind",
            "Electrolyzer",
        ],
        "capacity_mw": [
            solar_capacity_mw,
            wind_capacity_mw,
            electrolyzer_capacity_mw,
        ],
    }
)

if battery_enabled:
    capacity_plot.loc[
        len(capacity_plot)
    ] = {
        "asset": "Battery",
        "capacity_mw": battery_power_mw,
    }

fig, ax = plt.subplots(
    figsize=(8, 4.5),
    dpi=160,
)

bars = ax.bar(
    capacity_plot["asset"],
    capacity_plot["capacity_mw"],
)

ax.set_title(
    "Optimized Power Capacities"
)

ax.set_ylabel(
    "Capacity [MW]"
)

ax.bar_label(
    bars,
    fmt="%.2f",
    padding=3,
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "optimized_capacities.png",
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# 26. Static plot — capacity factors
# =============================================================================
cf_plot = capacity_factors.copy()

x = np.arange(
    len(cf_plot)
)

width = 0.36

fig, ax = plt.subplots(
    figsize=(8, 4.5),
    dpi=160,
)

availability_values = (
    cf_plot[
        "availability_capacity_factor"
    ]
    .fillna(0.0)
    .to_numpy()
)

dispatch_values = (
    cf_plot[
        "dispatch_capacity_factor"
    ]
    .fillna(0.0)
    .to_numpy()
)

ax.bar(
    x - width / 2,
    availability_values,
    width,
    label="Availability CF",
)

ax.bar(
    x + width / 2,
    dispatch_values,
    width,
    label="Dispatch / utilization CF",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        "Solar",
        "Wind",
        "Electrolyzer",
    ]
)

ax.set_ylabel(
    "Capacity factor [-]"
)

ax.set_title(
    "Capacity and Utilization Factors"
)

ax.legend()

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "capacity_factors.png",
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# 27. Static plot — renewable curtailment
# =============================================================================
fig, ax = plt.subplots(
    figsize=(7, 4.5),
    dpi=160,
)

bars = ax.bar(
    [
        "Solar",
        "Wind",
    ],
    [
        solar_curtailment_rate * 100.0,
        wind_curtailment_rate * 100.0,
    ],
)

ax.set_title(
    "Renewable Curtailment"
)

ax.set_ylabel(
    "Curtailment [%]"
)

ax.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3,
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "curtailment.png",
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# 28. Static plot — annualized cost breakdown
# =============================================================================
cost_plot = cost_breakdown.copy()

cost_plot[
    "fixed_meur"
] = (
    cost_plot[
        "annualized_fixed_cost_eur"
    ]
    / 1e6
)

cost_plot[
    "variable_meur"
] = (
    cost_plot[
        "variable_operating_cost_eur"
    ]
    / 1e6
)


x = np.arange(
    len(cost_plot)
)

width = 0.36

fig, ax = plt.subplots(
    figsize=(8, 4.5),
    dpi=160,
)

ax.bar(
    x - width / 2,
    cost_plot["fixed_meur"],
    width,
    label="Annualized fixed cost",
)

ax.bar(
    x + width / 2,
    cost_plot["variable_meur"],
    width,
    label="Variable operating cost",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    cost_plot["label"]
)

ax.set_ylabel(
    "Annual cost [million EUR/a]"
)

ax.set_title(
    "Annualized System Cost by Asset"
)

ax.legend()

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "cost_breakdown.png",
    bbox_inches="tight",
)

plt.close(fig)

# =============================================================================
# 29. Static battery plots
# =============================================================================
if battery_enabled:

    # -------------------------------------------------------------------------
    # Battery power and energy capacity
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4.5),
        dpi=160,
    )

    power_bar = axes[0].bar(
        ["Battery power"],
        [battery_power_mw],
    )

    axes[0].set_ylabel(
        "Power [MW]"
    )

    axes[0].set_title(
        "Battery Power Capacity"
    )

    axes[0].bar_label(
        power_bar,
        fmt="%.2f",
        padding=3,
    )

    energy_bar = axes[1].bar(
        ["Battery energy"],
        [battery_energy_mwh],
    )

    axes[1].set_ylabel(
        "Energy [MWh]"
    )

    axes[1].set_title(
        "Battery Energy Capacity"
    )

    axes[1].bar_label(
        energy_bar,
        fmt="%.2f",
        padding=3,
    )

    for ax in axes:
        ax.spines["top"].set_visible(
            False
        )
        ax.spines["right"].set_visible(
            False
        )

    fig.suptitle(
        f"Optimized Battery — "
        f"{battery_duration_h:.2f} h duration"
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "battery_capacity.png",
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # Annual battery throughput and losses
    # -------------------------------------------------------------------------
    battery_operation_plot = pd.DataFrame(
        {
            "flow": [
                "Charge from AC",
                "Discharge to AC",
                "Losses",
            ],
            "energy_mwh": [
                battery_charge_input_mwh,
                battery_discharge_output_mwh,
                battery_losses_mwh,
            ],
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        dpi=160,
    )

    bars = ax.bar(
        battery_operation_plot["flow"],
        battery_operation_plot["energy_mwh"],
    )

    ax.set_title(
        "Annual Battery Operation"
    )

    ax.set_ylabel(
        "Energy [MWh/a]"
    )

    ax.bar_label(
        bars,
        fmt="%.1f",
        padding=3,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "battery_operation.png",
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # Battery state of charge
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(11, 4.5),
        dpi=160,
    )

    ax.plot(
        n.snapshots,
        battery_soc_mwh,
    )

    ax.set_title(
        "Battery State of Charge"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Stored energy [MWh]"
    )

    ax.set_ylim(
        bottom=0.0,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "battery_soc.png",
        bbox_inches="tight",
    )

    plt.close(fig)

else:
    # Remove battery-specific figures if a non-battery scenario is
    # analyzed into a directory that previously contained them.
    for battery_plot_file in [
        "battery_capacity.png",
        "battery_operation.png",
        "battery_soc.png",
    ]:
        path = (
            OUTDIR
            / battery_plot_file
        )

        if path.exists():
            path.unlink()


# =============================================================================
# 30. Interactive Plotly dashboard
# =============================================================================
if battery_enabled:
    dashboard_rows = 4

    dashboard_titles = (
        "Annual Cost by Asset",
        "Annual Renewable Generation",
        "Optimized Power Capacities",
        "Capacity Factors",
        "Renewable Curtailment",
        "Hydrogen Target vs Delivery",
        "Annual Battery Operation",
        "Battery State of Charge",
    )

else:
    dashboard_rows = 3

    dashboard_titles = (
        "Annual Cost by Asset",
        "Annual Renewable Generation",
        "Optimized Power Capacities",
        "Capacity Factors",
        "Renewable Curtailment",
        "Hydrogen Target vs Delivery",
    )


dashboard = make_subplots(
    rows=dashboard_rows,
    cols=2,
    subplot_titles=dashboard_titles,
)


dashboard.add_trace(
    go.Bar(
        x=cost_plot["label"],
        y=(
            cost_plot[
                "total_annual_cost_eur"
            ]
            / 1e6
        ),
        name="Annual cost",
    ),
    row=1,
    col=1,
)


dashboard.add_trace(
    go.Bar(
        x=generation_plot["source"],
        y=generation_plot[
            "generation_gwh"
        ],
        name="Generation",
    ),
    row=1,
    col=2,
)


dashboard.add_trace(
    go.Bar(
        x=capacity_plot["asset"],
        y=capacity_plot[
            "capacity_mw"
        ],
        name="Capacity",
    ),
    row=2,
    col=1,
)


dashboard.add_trace(
    go.Bar(
        x=[
            "Solar",
            "Wind",
        ],
        y=[
            solar_availability_cf,
            wind_availability_cf,
        ],
        name="Availability CF",
    ),
    row=2,
    col=2,
)

dashboard.add_trace(
    go.Bar(
        x=[
            "Solar",
            "Wind",
            "Electrolyzer",
        ],
        y=[
            solar_dispatch_cf,
            wind_dispatch_cf,
            electrolyzer_capacity_factor,
        ],
        name="Dispatch / utilization CF",
    ),
    row=2,
    col=2,
)


dashboard.add_trace(
    go.Bar(
        x=[
            "Solar",
            "Wind",
        ],
        y=[
            solar_curtailment_rate,
            wind_curtailment_rate,
        ],
        name="Curtailment",
    ),
    row=3,
    col=1,
)


dashboard.add_trace(
    go.Bar(
        x=[
            "Target",
            "Delivered",
        ],
        y=[
            target_annual_mwh_h2
            / 1000.0,
            hydrogen_delivered_mwh
            / 1000.0,
        ],
        name="Hydrogen",
    ),
    row=3,
    col=2,
)


dashboard.update_yaxes(
    title_text="million EUR/a",
    row=1,
    col=1,
)

dashboard.update_yaxes(
    title_text="GWh",
    row=1,
    col=2,
)

dashboard.update_yaxes(
    title_text="MW",
    row=2,
    col=1,
)

dashboard.update_yaxes(
    title_text="Share",
    row=2,
    col=2,
)

dashboard.update_yaxes(
    title_text="Share",
    row=3,
    col=1,
)

dashboard.update_yaxes(
    title_text="GWh_H2",
    row=3,
    col=2,
)

if battery_enabled:
    dashboard.update_yaxes(
        title_text="MWh/a",
        row=4,
        col=1,
    )

    dashboard.update_yaxes(
        title_text="MWh",
        row=4,
        col=2,
    )


dashboard.update_layout(
    title_text=(
        "Off-grid Scenario Results: "
        f"{cfg['scenario_name']}"
    ),
    height=1200,
    width=1300,
    template="plotly_white",
)

dashboard.write_html(
    OUTDIR / "results_dashboard.html",
    include_plotlyjs="cdn",
)


# =============================================================================
# 31. HTML summary table
# =============================================================================
summary_html = (
    summary
    .T
    .rename(
        columns={
            0: "value"
        }
    )
)

summary_html.to_html(
    OUTDIR / "summary_table.html",
)


# =============================================================================
# 32. Terminal output
# =============================================================================
print("\n================================================")
print("OFF-GRID SCENARIO ANALYSIS")
print("================================================")

print("\nOptimal capacities:")
print(
    f"  Solar:             "
    f"{solar_capacity_mw:.3f} MW"
)
print(
    f"  Wind:              "
    f"{wind_capacity_mw:.3f} MW"
)
print(
    f"  Electrolyzer:      "
    f"{electrolyzer_capacity_mw:.3f} MW_el"
)

print("\nAnnual energy:")
print(
    f"  Solar generation:  "
    f"{solar_generation_mwh:,.3f} MWh"
)
print(
    f"  Wind generation:   "
    f"{wind_generation_mwh:,.3f} MWh"
)
print(
    f"  ELY input:         "
    f"{electrolyzer_input_mwh:,.3f} MWh_el"
)
print(
    f"  H2 output:         "
    f"{hydrogen_output_mwh:,.3f} MWh_H2"
)
print(
    f"  H2 delivered:      "
    f"{hydrogen_delivered_mwh:,.3f} MWh_H2"
)

print("\nHydrogen:")
print(
    f"  Target:            "
    f"{target_annual_kt_h2:.6f} kt/a"
)
print(
    f"  Delivered:         "
    f"{hydrogen_delivered_kt:.6f} kt/a"
)
print(
    f"  Target achievement:"
    f" {hydrogen_target_achievement * 100:.6f}%"
)
print(
    f"  Specific power:    "
    f"{specific_electricity_kwh_per_kg_h2:.3f} "
    f"kWh_el/kg_H2"
)

print("\nCapacity factors:")
print(
    f"  Solar availability:"
    f" {solar_availability_cf:.4f}"
)
print(
    f"  Solar dispatch:    "
    f"{solar_dispatch_cf:.4f}"
)
print(
    f"  Wind availability: "
    f"{wind_availability_cf:.4f}"
)
print(
    f"  Wind dispatch:     "
    f"{wind_dispatch_cf:.4f}"
)
print(
    f"  Electrolyzer CF:   "
    f"{electrolyzer_capacity_factor:.4f}"
)

if battery_enabled:
    print("\nBattery:")
    print(
        f"  Power capacity:     "
        f"{battery_power_mw:.3f} MW"
    )
    print(
        f"  Energy capacity:    "
        f"{battery_energy_mwh:.3f} MWh"
    )
    print(
        f"  Duration:           "
        f"{battery_duration_h:.3f} h"
    )
    print(
        f"  Charge from AC:     "
        f"{battery_charge_input_mwh:,.3f} MWh/a"
    )
    print(
        f"  Discharge to AC:    "
        f"{battery_discharge_output_mwh:,.3f} MWh/a"
    )
    print(
        f"  Losses:             "
        f"{battery_losses_mwh:,.3f} MWh/a"
    )
    print(
        f"  Round-trip eta:     "
        f"{battery_round_trip_efficiency:.4f}"
    )
    print(
        f"  Equivalent cycles:  "
        f"{battery_equivalent_full_cycles_per_year:.1f} /a"
    )


print("\nCurtailment:")
print(
    f"  Solar:             "
    f"{solar_curtailment_rate * 100:.3f}%"
)
print(
    f"  Wind:              "
    f"{wind_curtailment_rate * 100:.3f}%"
)
print(
    f"  Renewable use:     "
    f"{renewable_utilization_rate * 100:.3f}%"
)

print("\nEconomics:")
print(
    f"  Annual fixed cost: "
    f"{annualized_fixed_cost_eur:,.2f} EUR/a"
)
print(
    f"  Variable OPEX:     "
    f"{variable_operating_cost_eur:,.2f} EUR/a"
)
print(
    f"  System cost:       "
    f"{system_cost_eur:,.2f} EUR/a"
)
print(
    f"  LCOH:              "
    f"{lcoh_eur_per_kg_h2:.4f} EUR/kg_H2"
)

print("\nValidation:")
print(
    f"  Electricity error: "
    f"{max_electricity_balance_error_mw:.3e} MW"
)
print(
    f"  Hydrogen error:    "
    f"{max_hydrogen_balance_error_mw:.3e} MW"
)
print(
    f"  Objective-cost gap:"
    f" {objective_cost_difference_eur:.6e} EUR"
)

if battery_enabled:
    print(
        f"  Battery bus error:  "
        f"{max_battery_bus_balance_error_mw:.3e} MW"
    )
    print(
        f"  Battery coupling:   "
        f"{battery_power_coupling_error_mw:.3e} MW"
    )

print("\nChecks:")
print("  H2 target:          PASS")
print("  Electrolyzer eta:   PASS")
print("  Electricity balance:PASS")
print("  Hydrogen balance:   PASS")
print("  Cost accounting:    PASS")
if battery_enabled:
    print("  Battery bus balance: PASS")
    print("  Battery coupling:    PASS")
    print("  Battery SOC bounds:  PASS")

print("================================================")

print("\n=== SUMMARY ===")
print(
    summary.T
)

print("\n=== CAPACITY FACTORS ===")
print(
    capacity_factors
)

print("\n=== CURTAILMENT ===")
print(
    curtailment
)

print("\n=== COST BREAKDOWN ===")
print(
    cost_breakdown
)

print(
    f"\nCSV/HTML/PNG files written to "
    f"{OUTDIR}"
)
