from pathlib import Path
import math

import pandas as pd
import pypsa
import xarray as xr
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path):
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return ROOT / path


def annuity(rate, lifetime):
    rate = float(rate)
    lifetime = float(lifetime)

    if lifetime <= 0:
        raise ValueError("Lifetime must be greater than zero.")

    if rate == 0:
        return 1.0 / lifetime

    return rate / (1.0 - (1.0 + rate) ** (-lifetime))


def get_cost(costs, technology, parameter, default=None):
    rows = costs[
        (costs["technology"] == technology)
        & (costs["parameter"] == parameter)
    ]

    if rows.empty:
        if default is not None:
            return float(default)

        raise KeyError(
            f"Missing cost parameter: {technology} / {parameter}"
        )

    return float(rows.iloc[0]["value"])


def get_investment_eur_per_mw(costs, technology):
    rows = costs[
        (costs["technology"] == technology)
        & (costs["parameter"] == "investment")
    ]

    if rows.empty:
        raise KeyError(
            f"Missing investment cost for {technology}"
        )

    row = rows.iloc[0]

    value = float(row["value"])
    unit = str(row["unit"]).strip()

    if unit in {"EUR/kW", "EUR/kW_e", "EUR/kWh"}:
        return value * 1000.0

    if unit in {"EUR/MW", "EUR/MW_e", "EUR/MWh"}:
        return value

    raise ValueError(
        f"Unsupported investment unit {unit!r} "
        f"for technology {technology!r}"
    )


def annualized_capital_cost(costs, cfg, technology):
    investment = get_investment_eur_per_mw(
        costs,
        technology,
    )

    fom = get_cost(
        costs,
        technology,
        "FOM",
        0.0,
    )

    lifetime = get_cost(
        costs,
        technology,
        "lifetime",
    )

    discount_rate = float(
        cfg["costs"].get("discount_rate", 0.07)
    )

    return (
        investment * annuity(discount_rate, lifetime)
        + investment * fom / 100.0
    )


def electrolyzer_capital_cost(costs, cfg):
    hydrogen = cfg["hydrogen"]

    override_keys = (
        "electrolyzer_capex_eur_per_kw",
        "electrolyzer_fom_percent_per_year",
        "electrolyzer_lifetime_years",
    )

    values = [
        hydrogen.get(key)
        for key in override_keys
    ]

    if all(value is None for value in values):
        return annualized_capital_cost(
            costs,
            cfg,
            "electrolysis",
        )

    if any(value is None for value in values):
        raise ValueError(
            "Electrolyzer CAPEX, FOM and lifetime overrides "
            "must be supplied together."
        )

    capex_eur_per_kw = float(values[0])
    fom_percent = float(values[1])
    lifetime_years = float(values[2])

    investment_eur_per_mw = (
        capex_eur_per_kw * 1000.0
    )

    discount_rate = float(
        cfg["costs"].get("discount_rate", 0.07)
    )

    return (
        investment_eur_per_mw
        * annuity(discount_rate, lifetime_years)
        + investment_eur_per_mw
        * fom_percent
        / 100.0
    )


def load_profile(path, preferred_columns, snapshots):
    path = resolve_path(path)
    df = pd.read_csv(path)

    column = None

    for candidate in preferred_columns:
        if candidate in df.columns:
            column = candidate
            break

    if column is None:
        raise ValueError(
            f"No supported capacity-factor column found in {path}. "
            f"Expected one of {preferred_columns}; "
            f"found {list(df.columns)}."
        )

    profile = pd.to_numeric(
        df[column],
        errors="raise",
    ).astype(float)

    if len(profile) != snapshots:
        raise ValueError(
            f"{path} contains {len(profile)} rows; "
            f"expected {snapshots}."
        )

    if profile.isna().any():
        raise ValueError(
            f"{path} contains missing capacity factors."
        )

    if (profile < 0).any() or (profile > 1).any():
        raise ValueError(
            f"{path} contains capacity factors outside [0, 1]."
        )

    return profile


def hydrogen_target_mwh(cfg):
    hydrogen = cfg["hydrogen"]

    target_kt = float(
        hydrogen["target_annual_kt_h2"]
    )

    lhv = float(
        hydrogen["hydrogen_lhv_kwh_per_kg"]
    )

    return (
        target_kt
        * 1_000_000.0
        * lhv
        / 1000.0
    )


def build_network(cfg):
    system = cfg["system"]
    renewables = cfg["renewables"]
    hydrogen = cfg["hydrogen"]

    if system["model_type"] != "off_grid":
        raise ValueError(
            "Clean model currently supports only model_type='off_grid'."
        )

    if hydrogen.get("mode") != "production_target":
        raise NotImplementedError(
            "First clean implementation supports only "
            "hydrogen.mode='production_target'."
        )

    battery_cfg = cfg.get("battery", {})
    battery_enabled = bool(
        battery_cfg.get("enabled", False)
    )

    if cfg.get("bitcoin", {}).get("enabled", False):
        raise NotImplementedError(
            "Bitcoin is intentionally disabled until S0 regression passes."
        )

    if cfg.get("hydrogen_storage", {}).get("enabled", False):
        raise NotImplementedError(
            "Hydrogen storage is not part of the current thesis core model."
        )

    snapshots = int(system["snapshots"])
    weather_year = int(system["weather_year"])

    costs_cfg = cfg["costs"]
    active_dataset = costs_cfg["active_dataset"]
    costs_path = costs_cfg["datasets"][active_dataset]

    costs = pd.read_csv(
        resolve_path(costs_path)
    )

    solar_cf = load_profile(
        renewables["solar"]["profile_file"],
        ("capacity_factor", "solar_cf"),
        snapshots,
    )

    wind_cf = load_profile(
        renewables["wind"]["profile_file"],
        ("capacity_factor", "wind_cf"),
        snapshots,
    )

    snapshot_index = pd.date_range(
        start=f"{weather_year}-01-01 00:00:00",
        end=f"{weather_year}-12-31 23:00:00",
        freq="h",
    )

    if len(snapshot_index) == 8784 and snapshots == 8760:
        leap_day = (
            (snapshot_index.month == 2)
            & (snapshot_index.day == 29)
        )
        snapshot_index = snapshot_index[~leap_day]

    if len(snapshot_index) != snapshots:
        raise ValueError(
            f"Snapshot calendar has {len(snapshot_index)} hours; "
            f"expected {snapshots}."
        )

    n = pypsa.Network()
    n.set_snapshots(snapshot_index)

    carriers = [
        "AC",
        "H2",
        "solar",
        "wind",
        "electrolyzer",
        "hydrogen_delivery",
    ]

    if battery_enabled:
        carriers.append("battery")

    n.add(
        "Carrier",
        carriers,
    )

    n.add(
        "Bus",
        "electricity_bus",
        carrier="AC",
    )

    n.add(
        "Bus",
        "hydrogen_bus",
        carrier="H2",
    )

    if battery_enabled:
        n.add(
            "Bus",
            "battery_bus",
            carrier="battery",
        )

    solar_cfg = renewables["solar"]
    solar_kwargs = {}

    if solar_cfg.get("max_capacity_mw") is not None:
        solar_kwargs["p_nom_max"] = float(
            solar_cfg["max_capacity_mw"]
        )

    n.add(
        "Generator",
        "solar",
        bus="electricity_bus",
        carrier="solar",
        p_nom_extendable=bool(
            solar_cfg.get("p_nom_extendable", True)
        ),
        capital_cost=annualized_capital_cost(
            costs,
            cfg,
            "solar-utility",
        ),
        marginal_cost=get_cost(
            costs,
            "solar-utility",
            "VOM",
            0.0,
        ),
        p_max_pu=solar_cf.to_numpy(),
        **solar_kwargs,
    )

    wind_cfg = renewables["wind"]
    wind_kwargs = {}

    if wind_cfg.get("max_capacity_mw") is not None:
        wind_kwargs["p_nom_max"] = float(
            wind_cfg["max_capacity_mw"]
        )

    n.add(
        "Generator",
        "wind",
        bus="electricity_bus",
        carrier="wind",
        p_nom_extendable=bool(
            wind_cfg.get("p_nom_extendable", True)
        ),
        capital_cost=annualized_capital_cost(
            costs,
            cfg,
            "onwind",
        ),
        marginal_cost=get_cost(
            costs,
            "onwind",
            "VOM",
            0.0,
        ),
        p_max_pu=wind_cf.to_numpy(),
        **wind_kwargs,
    )

    if battery_enabled:
        if not bool(
            battery_cfg.get(
                "p_nom_extendable",
                True,
            )
        ):
            raise ValueError(
                "Clean S1 requires battery.p_nom_extendable=true."
            )

        if not bool(
            battery_cfg.get(
                "e_nom_extendable",
                True,
            )
        ):
            raise ValueError(
                "Clean S1 requires battery.e_nom_extendable=true."
            )

        inverter_technology = str(
            battery_cfg.get(
                "inverter_technology",
                "battery inverter",
            )
        )

        storage_technology = str(
            battery_cfg.get(
                "storage_technology",
                "battery storage",
            )
        )

        inverter_efficiency = get_cost(
            costs,
            inverter_technology,
            "efficiency",
        )

        if not 0.0 < inverter_efficiency <= 1.0:
            raise ValueError(
                "Battery inverter efficiency must lie in (0, 1]."
            )

        battery_charge_efficiency = math.sqrt(
            inverter_efficiency
        )

        battery_discharge_efficiency = math.sqrt(
            inverter_efficiency
        )

        standing_loss = float(
            battery_cfg.get(
                "standing_loss",
                0.0,
            )
        )

        if not 0.0 <= standing_loss < 1.0:
            raise ValueError(
                "battery.standing_loss must lie in [0, 1)."
            )

        inverter_annual_cost = annualized_capital_cost(
            costs,
            cfg,
            inverter_technology,
        )

        storage_annual_cost = annualized_capital_cost(
            costs,
            cfg,
            storage_technology,
        )

        n.add(
            "Link",
            "battery_charger",
            bus0="electricity_bus",
            bus1="battery_bus",
            carrier="battery",
            p_nom_extendable=True,
            p_min_pu=0.0,
            efficiency=battery_charge_efficiency,
            capital_cost=inverter_annual_cost,
            marginal_cost=0.0,
        )

        n.add(
            "Store",
            "battery_store",
            bus="battery_bus",
            carrier="battery",
            e_nom_extendable=True,
            e_cyclic=bool(
                battery_cfg.get(
                    "cyclic_state_of_charge",
                    True,
                )
            ),
            standing_loss=standing_loss,
            capital_cost=storage_annual_cost,
            marginal_cost=0.0,
        )

        n.add(
            "Link",
            "battery_discharger",
            bus0="battery_bus",
            bus1="electricity_bus",
            carrier="battery",
            p_nom_extendable=True,
            p_min_pu=0.0,
            efficiency=battery_discharge_efficiency,
            capital_cost=0.0,
            marginal_cost=0.0,
        )

    electrolyzer_efficiency = float(
        hydrogen.get(
            "electrolyzer_efficiency",
            get_cost(
                costs,
                "electrolysis",
                "efficiency",
            ),
        )
    )

    electrolyzer_vom = float(
        hydrogen.get(
            "electrolyzer_variable_cost_eur_per_mwh",
            get_cost(
                costs,
                "electrolysis",
                "VOM",
                0.0,
            ),
        )
    )

    n.add(
        "Link",
        "electrolyzer",
        bus0="electricity_bus",
        bus1="hydrogen_bus",
        carrier="electrolyzer",
        p_nom_extendable=True,
        efficiency=electrolyzer_efficiency,
        capital_cost=electrolyzer_capital_cost(
            costs,
            cfg,
        ),
        marginal_cost=electrolyzer_vom,
    )

    target_mwh = hydrogen_target_mwh(cfg)

    n.add(
        "Generator",
        "hydrogen_delivery",
        bus="hydrogen_bus",
        carrier="hydrogen_delivery",
        sign=-1.0,
        p_nom=target_mwh,
        p_nom_extendable=False,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=0.0,
    )

    return n


def add_hydrogen_target_constraint(n, cfg, snapshots):
    target_mwh = hydrogen_target_mwh(cfg)

    delivery = (
        n.model.variables["Generator-p"]
        .sel(name="hydrogen_delivery")
    )

    weights = xr.DataArray(
        n.snapshot_weightings.generators
        .loc[snapshots]
        .to_numpy(),
        coords={"snapshot": snapshots},
        dims=["snapshot"],
    )

    annual_delivery = (
        delivery * weights
    ).sum("snapshot")

    n.model.add_constraints(
        annual_delivery == target_mwh,
        name="GlobalConstraint-hydrogen_delivery_target",
    )


def add_battery_power_coupling_constraint(n, cfg):
    battery_cfg = cfg.get("battery", {})

    if not bool(
        battery_cfg.get("enabled", False)
    ):
        return

    charger = (
        n.model.variables["Link-p_nom"]
        .loc["battery_charger"]
    )

    discharger = (
        n.model.variables["Link-p_nom"]
        .loc["battery_discharger"]
    )

    discharge_efficiency = float(
        n.links.at[
            "battery_discharger",
            "efficiency",
        ]
    )

    n.model.add_constraints(
        charger
        == discharge_efficiency * discharger,
        name="GlobalConstraint-battery_power_coupling",
    )


def solve_network(n, cfg):
    def extra_functionality(network, snapshots):
        add_hydrogen_target_constraint(
            network,
            cfg,
            snapshots,
        )

        add_battery_power_coupling_constraint(
            network,
            cfg,
        )

    status, condition = n.optimize(
        solver_name="highs",
        extra_functionality=extra_functionality,
        include_objective_constant=True,
    )

    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"Optimization failed: status={status}, "
            f"condition={condition}"
        )

    return n
