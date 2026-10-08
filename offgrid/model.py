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

    hydrogen_mode = str(
        hydrogen.get(
            "mode",
            "production_target",
        )
    ).strip().lower()

    if hydrogen_mode not in {
        "production_target",
        "maximize_production",
    }:
        raise ValueError(
            "Clean off-grid model currently supports "
            "hydrogen.mode='production_target' or "
            "'maximize_production'."
        )

    battery_cfg = cfg.get("battery", {})
    battery_enabled = bool(
        battery_cfg.get("enabled", False)
    )

    bitcoin_cfg = cfg.get("bitcoin", {})
    bitcoin_enabled = bool(
        bitcoin_cfg.get("enabled", False)
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

    if bitcoin_enabled:
        carriers.append("bitcoin_mining")

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

    if bitcoin_enabled:
        operating_mode = str(
            bitcoin_cfg.get(
                "operating_mode",
                "economic_dispatch",
            )
        ).strip().lower()

        if operating_mode != "economic_dispatch":
            raise ValueError(
                "Bitcoin requires "
                "operating_mode='economic_dispatch'."
            )

        # Thesis default:
        # omitted capacity_mode means fixed installed capacity.
        capacity_mode = str(
            bitcoin_cfg.get(
                "capacity_mode",
                "fixed",
            )
        ).strip().lower()

        if capacity_mode == "fixed":
            bitcoin_capacity_mw = float(
                bitcoin_cfg["max_capacity_mw"]
            )

            if bitcoin_capacity_mw <= 0.0:
                raise ValueError(
                    "bitcoin.max_capacity_mw must be greater than zero "
                    "for fixed BTC capacity."
                )

            bitcoin_capacity_kwargs = {
                "p_nom": bitcoin_capacity_mw,
                "p_nom_extendable": False,
                "capital_cost": 0.0,
            }

        elif capacity_mode == "endogenous":
            annualized_capacity_cost = float(
                bitcoin_cfg[
                    "annualized_capacity_cost_eur_per_mw_year"
                ]
            )

            if annualized_capacity_cost < 0.0:
                raise ValueError(
                    "Bitcoin annualized capacity cost "
                    "must be non-negative."
                )

            bitcoin_capacity_kwargs = {
                "p_nom": 0.0,
                "p_nom_extendable": True,
                "capital_cost": annualized_capacity_cost,
            }

            max_capacity_mw = bitcoin_cfg.get(
                "max_capacity_mw"
            )

            if max_capacity_mw is not None:
                max_capacity_mw = float(
                    max_capacity_mw
                )

                if max_capacity_mw <= 0.0:
                    raise ValueError(
                        "bitcoin.max_capacity_mw must be greater than "
                        "zero when an endogenous ceiling is used."
                    )

                bitcoin_capacity_kwargs[
                    "p_nom_max"
                ] = max_capacity_mw

        else:
            raise ValueError(
                "bitcoin.capacity_mode must be either "
                "'fixed' or 'endogenous'."
            )

        hashprice_eur_per_th_day = float(
            bitcoin_cfg[
                "hashprice_eur_per_th_day"
            ]
        )

        asic_efficiency_j_per_th = float(
            bitcoin_cfg[
                "asic_efficiency_j_per_th"
            ]
        )

        pue = float(
            bitcoin_cfg.get(
                "pue",
                1.0,
            )
        )

        other_opex_eur_per_mwh = float(
            bitcoin_cfg.get(
                "other_opex_eur_per_mwh",
                0.0,
            )
        )

        if hashprice_eur_per_th_day < 0.0:
            raise ValueError(
                "bitcoin.hashprice_eur_per_th_day must be non-negative."
            )

        if asic_efficiency_j_per_th <= 0.0:
            raise ValueError(
                "bitcoin.asic_efficiency_j_per_th "
                "must be greater than zero."
            )

        if pue < 1.0:
            raise ValueError(
                "bitcoin.pue must be at least 1.0."
            )

        if other_opex_eur_per_mwh < 0.0:
            raise ValueError(
                "bitcoin.other_opex_eur_per_mwh "
                "must be non-negative."
            )

        # Facility-side electrical intensity.
        #
        # ASIC efficiency:
        #   J/TH = W/(TH/s)
        #
        # PUE expands miner-wall electricity to the
        # complete facility electricity boundary.
        mw_per_th_per_s = (
            asic_efficiency_j_per_th
            * pue
            / 1e6
        )

        # Hashprice is EUR/(TH/s)/day.
        # 1 MWh = 1 MW operated for 1/24 day.
        th_day_per_mwh = (
            (1.0 / 24.0)
            / mw_per_th_per_s
        )

        bitcoin_gross_revenue_eur_per_mwh = (
            hashprice_eur_per_th_day
            * th_day_per_mwh
        )

        bitcoin_net_value_eur_per_mwh = (
            bitcoin_gross_revenue_eur_per_mwh
            - other_opex_eur_per_mwh
        )

        # Flexible electricity consumer.
        #
        # sign=-1 means positive Generator dispatch
        # withdraws electricity from electricity_bus.
        # Negative marginal cost represents BTC
        # operating value in the optimization objective.
        n.add(
            "Generator",
            "bitcoin_mining_sink",
            bus="electricity_bus",
            carrier="bitcoin_mining",
            sign=-1.0,
            p_min_pu=0.0,
            p_max_pu=1.0,
            marginal_cost=-bitcoin_net_value_eur_per_mwh,
            **bitcoin_capacity_kwargs,
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

    if hydrogen_mode == "production_target":
        # Preserve the already validated S0-S3E formulation.
        target_mwh = hydrogen_target_mwh(cfg)

        hydrogen_delivery_kwargs = {
            "p_nom": target_mwh,
            "p_nom_extendable": False,
        }

    elif hydrogen_mode == "maximize_production":
        # HMAX has no prescribed annual H2 quantity.
        #
        # Hydrogen delivery is therefore an extendable zero-cost
        # product sink. The Stage-1 objective will maximize its
        # annual dispatch subject to the physical system limits.
        hydrogen_delivery_kwargs = {
            "p_nom": 0.0,
            "p_nom_extendable": True,
            "p_nom_min": 0.0,
            "capital_cost": 0.0,
        }

    else:
        raise RuntimeError(
            f"Unexpected hydrogen mode {hydrogen_mode!r}."
        )

    n.add(
        "Generator",
        "hydrogen_delivery",
        bus="hydrogen_bus",
        carrier="hydrogen_delivery",
        sign=-1.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=0.0,
        **hydrogen_delivery_kwargs,
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

def get_annual_hydrogen_delivery_expression(
    n,
    snapshots,
):
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

    return (
        delivery * weights
    ).sum("snapshot")


def add_hydrogen_minimum_constraint(
    n,
    snapshots,
    minimum_hydrogen_mwh,
):
    annual_delivery = (
        get_annual_hydrogen_delivery_expression(
            n,
            snapshots,
        )
    )

    n.model.add_constraints(
        annual_delivery >= minimum_hydrogen_mwh,
        name="GlobalConstraint-hydrogen_delivery_minimum",
    )


def solve_network(n, cfg):
    hydrogen_mode = str(
        cfg["hydrogen"].get(
            "mode",
            "production_target",
        )
    ).strip().lower()

    # ========================================================
    # Existing fixed-H2 pathway
    # ========================================================

    if hydrogen_mode == "production_target":

        def extra_functionality(
            network,
            snapshots,
        ):
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

    # ========================================================
    # HMAX: lexicographic two-stage optimization
    # ========================================================

    elif hydrogen_mode == "maximize_production":

        print()
        print("=" * 60)
        print("HMAX STAGE 1: MAXIMIZE ANNUAL HYDROGEN")
        print("=" * 60)

        # Build a separate Stage-1 network so that the normal
        # economic objective of the final Stage-2 network is
        # left untouched.
        n_hmax = build_network(cfg)

        n_hmax.optimize.create_model(
            include_objective_constant=False,
        )

        add_battery_power_coupling_constraint(
            n_hmax,
            cfg,
        )

        annual_hydrogen_stage1 = (
            get_annual_hydrogen_delivery_expression(
                n_hmax,
                n_hmax.snapshots,
            )
        )

        # Pure physical/resource-potential optimization.
        n_hmax.model.add_objective(
            annual_hydrogen_stage1,
            overwrite=True,
            sense="max",
        )

        status_stage1, condition_stage1 = (
            n_hmax.optimize.solve_model(
                solver_name="highs",
            )
        )

        if (
            status_stage1 != "ok"
            or condition_stage1 != "optimal"
        ):
            raise RuntimeError(
                "HMAX Stage 1 failed: "
                f"status={status_stage1}, "
                f"condition={condition_stage1}"
            )

        stage1_weights = (
            n_hmax.snapshot_weightings.generators
            .reindex(n_hmax.snapshots)
        )

        stage1_hydrogen_mwh = float(
            (
                n_hmax.generators_t.p[
                    "hydrogen_delivery"
                ]
                * stage1_weights
            ).sum()
        )

        if (
            not math.isfinite(stage1_hydrogen_mwh)
            or stage1_hydrogen_mwh <= 0.0
        ):
            raise RuntimeError(
                "HMAX Stage 1 returned invalid annual "
                f"hydrogen production: {stage1_hydrogen_mwh!r}"
            )

        hmax_cfg = cfg["hydrogen"]

        relative_tolerance = float(
            hmax_cfg.get(
                "hmax_relative_tolerance",
                1e-8,
            )
        )

        absolute_tolerance_mwh = float(
            hmax_cfg.get(
                "hmax_absolute_tolerance_mwh",
                1e-3,
            )
        )

        if (
            not math.isfinite(relative_tolerance)
            or relative_tolerance < 0.0
        ):
            raise ValueError(
                "hydrogen.hmax_relative_tolerance must be "
                "finite and non-negative."
            )

        if (
            not math.isfinite(absolute_tolerance_mwh)
            or absolute_tolerance_mwh < 0.0
        ):
            raise ValueError(
                "hydrogen.hmax_absolute_tolerance_mwh must be "
                "finite and non-negative."
            )

        hmax_tolerance_mwh = max(
            absolute_tolerance_mwh,
            stage1_hydrogen_mwh
            * relative_tolerance,
        )

        stage2_minimum_hydrogen_mwh = (
            stage1_hydrogen_mwh
            - hmax_tolerance_mwh
        )

        print(
            "Stage-1 maximum H2: "
            f"{stage1_hydrogen_mwh:,.6f} MWh_H2/a"
        )

        print(
            "Stage-2 tolerance:  "
            f"{hmax_tolerance_mwh:,.6f} MWh_H2/a"
        )

        print(
            "Stage-2 minimum H2: "
            f"{stage2_minimum_hydrogen_mwh:,.6f} MWh_H2/a"
        )

        print()
        print("=" * 60)
        print("HMAX STAGE 2: MINIMIZE SYSTEM COST")
        print("=" * 60)

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

        # Preserve the lexicographic information in the solved
        # network so analysis remains reproducible.
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
            stage1_hydrogen_mwh
        )

        n.meta[
            "hmax_relative_tolerance"
        ] = float(
            relative_tolerance
        )

        n.meta[
            "hmax_absolute_tolerance_mwh"
        ] = float(
            absolute_tolerance_mwh
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
            "Unsupported hydrogen mode: "
            f"{hydrogen_mode!r}"
        )

    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"Optimization failed: status={status}, "
            f"condition={condition}"
        )

    return n
