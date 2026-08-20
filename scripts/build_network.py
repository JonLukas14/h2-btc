import math
from pathlib import Path

import pandas as pd
import pypsa


# =============================================================================
# Build PyPSA network from scenario configuration and processed inputs
# =============================================================================
def build_test_network(cfg, data_dir):
    """
    Build the off-grid renewable hydrogen network.

    Current development stage:
        - Solar PV
        - Onshore wind
        - Electrolyzer
        - Electricity bus
        - Hydrogen bus

    Not implemented at this stage:
        - Annual hydrogen production target constraint
        - Battery storage
        - Bitcoin mining
        - Hydrogen storage

    The network is intentionally built in stages so that each subsystem can
    be validated independently.
    """

    data_dir = Path(data_dir)

    # -------------------------------------------------------------------------
    # 1. Read and validate scenario configuration
    # -------------------------------------------------------------------------
    system_cfg = cfg["system"]
    costs_cfg = cfg["costs"]

    renewable_cfg = cfg.get("renewables", {})
    hydrogen_cfg = cfg.get("hydrogen", {})
    hydrogen_storage_cfg = cfg.get("hydrogen_storage", {})
    battery_cfg = cfg.get("battery", {})
    bitcoin_cfg = cfg.get("bitcoin", {})

    model_type = system_cfg.get("model_type")

    if model_type != "off_grid":
        raise ValueError(
            f"This network builder expects system.model_type='off_grid', "
            f"got {model_type!r}."
        )

    snapshots = int(system_cfg["snapshots"])
    investment_year = int(system_cfg["investment_year"])
    weather_year = int(system_cfg["weather_year"])

    hydrogen_enabled = bool(
        hydrogen_cfg.get("enabled", False)
    )

    hydrogen_mode = hydrogen_cfg.get(
        "mode",
        "production_target",
    )

    battery_enabled = bool(
        battery_cfg.get("enabled", False)
    )

    bitcoin_enabled = bool(
        bitcoin_cfg.get("enabled", False)
    )

    hydrogen_storage_enabled = bool(
        hydrogen_storage_cfg.get("enabled", False)
    )

    # -------------------------------------------------------------------------
    # 2. Guard against technologies not implemented in this development stage
    # -------------------------------------------------------------------------


    if hydrogen_storage_enabled:
        raise NotImplementedError(
            "Hydrogen storage is enabled in the scenario, but H2 storage "
            "has not yet been implemented in the redesigned off-grid model."
        )

    if not hydrogen_enabled:
        raise ValueError(
            "The current off-grid reference architecture requires "
            "hydrogen.enabled=true."
        )

    if hydrogen_mode != "production_target":
        raise ValueError(
            "The redesigned off-grid model currently supports only "
            "hydrogen.mode='production_target'."
        )

    # -------------------------------------------------------------------------
    # 3. Required processed inputs
    # -------------------------------------------------------------------------
    active_cost_dataset = costs_cfg["active_dataset"]

    required_files = [
        data_dir / f"{active_cost_dataset}.csv",
        data_dir / "kz_solar_cf.csv",
        data_dir / "kz_wind_cf.csv",
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing processed input files:\n"
            + "\n".join(str(path) for path in missing)
        )

    # -------------------------------------------------------------------------
    # 4. Load processed inputs
    # -------------------------------------------------------------------------
    costs = pd.read_csv(
        data_dir / f"{active_cost_dataset}.csv"
    )

    solar_cf = pd.read_csv(
        data_dir / "kz_solar_cf.csv"
    )["solar_cf"]

    wind_cf = pd.read_csv(
        data_dir / "kz_wind_cf.csv"
    )["wind_cf"]

    for name, series in {
        "solar_cf": solar_cf,
        "wind_cf": wind_cf,
    }.items():
        if len(series) != snapshots:
            raise ValueError(
                f"{name} contains {len(series)} rows, "
                f"but system.snapshots={snapshots}."
            )

    # -------------------------------------------------------------------------
    # 5. Build model snapshot calendar
    # -------------------------------------------------------------------------
    full_year_index = pd.date_range(
        start=f"{weather_year}-01-01 00:00:00",
        end=f"{weather_year}-12-31 23:00:00",
        freq="h",
    )

    if len(full_year_index) == 8784 and snapshots == 8760:
        leap_day = (
            (full_year_index.month == 2)
            & (full_year_index.day == 29)
        )
        snapshot_index = full_year_index[~leap_day]
    else:
        snapshot_index = full_year_index

    if len(snapshot_index) != snapshots:
        raise ValueError(
            f"Snapshot calendar contains {len(snapshot_index)} hours, "
            f"but system.snapshots={snapshots}."
        )

    # -------------------------------------------------------------------------
    # 6. Create PyPSA network
    # -------------------------------------------------------------------------
    n = pypsa.Network()
    n.set_snapshots(snapshot_index)

    # -------------------------------------------------------------------------
    # 7. Define carriers explicitly
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 8. Add energy buses
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 9. Cost helper functions
    # -------------------------------------------------------------------------
    def get_cost_row(technology, parameter):
        rows = costs[
            (costs["technology"] == technology)
            & (costs["parameter"] == parameter)
        ]

        if rows.empty:
            return None

        return rows.iloc[0]

    def get_cost(
        technology,
        parameter,
        default=None,
    ):
        row = get_cost_row(
            technology,
            parameter,
        )

        if row is None:
            if default is not None:
                return float(default)

            raise KeyError(
                f"Missing cost entry for "
                f"{technology} / {parameter}"
            )

        return float(row["value"])

    def get_investment_cost_for_pypsa(technology):
        """
        Convert technology-data investment-cost units to the MW/MWh basis
        required by PyPSA nominal capacities.
        """

        row = get_cost_row(
            technology,
            "investment",
        )

        if row is None:
            raise KeyError(
                f"Missing investment cost for technology "
                f"'{technology}'."
            )

        value = float(row["value"])
        unit = str(row["unit"]).strip()

        if unit in {
            "EUR/kW",
            "EUR/kW_e",
            "EUR/kWh",
        }:
            return value * 1000.0

        if unit in {
            "EUR/MW",
            "EUR/MW_e",
            "EUR/MWh",
        }:
            return value

        raise ValueError(
            f"Unsupported investment-cost unit "
            f"'{unit}' for technology '{technology}'."
        )

    def annuity(rate, lifetime):
        if lifetime <= 0:
            raise ValueError(
                f"Lifetime must be positive, got {lifetime}."
            )

        if rate == 0:
            return 1.0 / lifetime

        return rate / (
            1.0
            - (1.0 + rate) ** (-lifetime)
        )

    def get_annualized_capital_cost(
        technology,
        default_discount_rate=0.07,
    ):
        investment = get_investment_cost_for_pypsa(
            technology
        )

        fom_percent = get_cost(
            technology,
            "FOM",
            0.0,
        )

        lifetime = get_cost(
            technology,
            "lifetime",
        )

        discount_rate = float(
            costs_cfg.get(
                "discount_rate",
                default_discount_rate,
            )
        )

        annualized_capex = (
            investment
            * annuity(
                discount_rate,
                lifetime,
            )
        )

        annualized_fom = (
            investment
            * fom_percent
            / 100.0
        )

        return (
            annualized_capex
            + annualized_fom
        )

    # -------------------------------------------------------------------------
    # 10. Add renewable generators
    # -------------------------------------------------------------------------
    solar_cfg = renewable_cfg.get(
        "solar",
        {},
    )

    wind_cfg = renewable_cfg.get(
        "wind",
        {},
    )

    if bool(solar_cfg.get("enabled", True)):
        solar_kwargs = {}

        solar_max_capacity = solar_cfg.get(
            "max_capacity_mw"
        )

        if solar_max_capacity is not None:
            solar_kwargs["p_nom_max"] = float(
                solar_max_capacity
            )

        n.add(
            "Generator",
            "solar",
            bus="electricity_bus",
            carrier="solar",
            p_nom_extendable=bool(
                solar_cfg.get(
                    "p_nom_extendable",
                    True,
                )
            ),
            capital_cost=get_annualized_capital_cost(
                "solar-utility"
            ),
            marginal_cost=get_cost(
                "solar-utility",
                "VOM",
                0.0,
            ),
            p_max_pu=solar_cf.values,
            **solar_kwargs,
        )

    if bool(wind_cfg.get("enabled", True)):
        wind_kwargs = {}

        wind_max_capacity = wind_cfg.get(
            "max_capacity_mw"
        )

        if wind_max_capacity is not None:
            wind_kwargs["p_nom_max"] = float(
                wind_max_capacity
            )

        n.add(
            "Generator",
            "wind",
            bus="electricity_bus",
            carrier="wind",
            p_nom_extendable=bool(
                wind_cfg.get(
                    "p_nom_extendable",
                    True,
                )
            ),
            capital_cost=get_annualized_capital_cost(
                "onwind"
            ),
            marginal_cost=get_cost(
                "onwind",
                "VOM",
                0.0,
            ),
            p_max_pu=wind_cf.values,
            **wind_kwargs,
        )

    # -------------------------------------------------------------------------
    # 11. Add electrical battery storage
    # -------------------------------------------------------------------------
    #
    # The battery is represented by:
    #
    #   electricity_bus
    #          |
    #          v
    #   battery_charger
    #          |
    #          v
    #      battery_bus
    #          |
    #     battery_store
    #          |
    #          v
    #   battery_discharger
    #          |
    #          v
    #   electricity_bus
    #
    # Power capacity [MW] and energy capacity [MWh] are therefore
    # independently extendable.
    #
    # IMPORTANT:
    # The charger/discharger power-capacity coupling constraint is NOT
    # created here because the Linopy capacity variables do not exist until
    # PyPSA builds the optimization model. That constraint will be added in
    # run_model.py through extra_functionality.
    # -------------------------------------------------------------------------

    if battery_enabled:
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

        battery_p_nom_extendable = bool(
            battery_cfg.get(
                "p_nom_extendable",
                True,
            )
        )

        battery_e_nom_extendable = bool(
            battery_cfg.get(
                "e_nom_extendable",
                True,
            )
        )

        # At the current development stage S1 is explicitly a capacity-
        # expansion case. Fixed battery capacities are not yet supported.
        if not battery_p_nom_extendable:
            raise ValueError(
                "The current S1 battery implementation requires "
                "battery.p_nom_extendable=true."
            )

        if not battery_e_nom_extendable:
            raise ValueError(
                "The current S1 battery implementation requires "
                "battery.e_nom_extendable=true."
            )

        inverter_efficiency = get_cost(
            inverter_technology,
            "efficiency",
        )

        if not 0.0 < inverter_efficiency <= 1.0:
            raise ValueError(
                "Battery inverter efficiency must be "
                "greater than 0 and at most 1."
            )

        # technology-data / PyPSA-Eur convention:
        # the generic battery-inverter efficiency is split symmetrically
        # between charging and discharging.
        battery_charge_efficiency = math.sqrt(
            inverter_efficiency
        )

        battery_discharge_efficiency = math.sqrt(
            inverter_efficiency
        )

        battery_standing_loss = float(
            battery_cfg.get(
                "standing_loss",
                0.0,
            )
        )

        if not 0.0 <= battery_standing_loss < 1.0:
            raise ValueError(
                "battery.standing_loss must be in the interval [0, 1)."
            )

        battery_cyclic = bool(
            battery_cfg.get(
                "cyclic_state_of_charge",
                True,
            )
        )

        battery_inverter_annual_cost = (
            get_annualized_capital_cost(
                inverter_technology
            )
        )

        battery_storage_annual_cost = (
            get_annualized_capital_cost(
                storage_technology
            )
        )

        # Charging Link.
        #
        # The inverter's annualized power cost is assigned here once.
        # The discharging Link below receives zero capital cost because
        # both directions represent the same physical bidirectional inverter.
        n.add(
            "Link",
            "battery_charger",
            bus0="electricity_bus",
            bus1="battery_bus",
            carrier="battery",
            p_nom_extendable=True,
            p_min_pu=0.0,
            efficiency=battery_charge_efficiency,
            capital_cost=battery_inverter_annual_cost,
            marginal_cost=0.0,
        )

        # Energy reservoir.
        n.add(
            "Store",
            "battery_store",
            bus="battery_bus",
            carrier="battery",
            e_nom_extendable=True,
            e_cyclic=battery_cyclic,
            standing_loss=battery_standing_loss,
            capital_cost=battery_storage_annual_cost,
            marginal_cost=0.0,
        )

        # Discharging direction of the same inverter.
        #
        # Its capacity will be coupled to battery_charger during
        # optimization. Therefore no second inverter CAPEX is assigned.
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




    # -------------------------------------------------------------------------
    # 12. Add flexible Bitcoin mining sink
    # -------------------------------------------------------------------------
    #
    # Bitcoin mining is represented as a controllable electricity consumer.
    #
    # A PyPSA Generator with sign=-1 consumes electricity from the
    # electricity bus. Its dispatch can vary freely between zero and the
    # configured fixed mining capacity.
    #
    # The negative marginal cost represents the net operating value obtained
    # from mining one additional MWh of electricity.
    #
    # For the current S2 architecture-validation case, mining capacity is
    # deliberately fixed rather than optimized because mining CAPEX has not
    # yet been introduced into the redesigned model.
    # -------------------------------------------------------------------------

    if bitcoin_enabled:
        bitcoin_operating_mode = bitcoin_cfg.get(
            "operating_mode",
            "economic_dispatch",
        )

        if bitcoin_operating_mode != "economic_dispatch":
            raise ValueError(
                "The current Bitcoin implementation requires "
                "bitcoin.operating_mode='economic_dispatch'."
            )

        bitcoin_capacity_mw = bitcoin_cfg.get(
            "max_capacity_mw"
        )

        if bitcoin_capacity_mw is None:
            raise ValueError(
                "bitcoin.max_capacity_mw must be defined "
                "when Bitcoin mining is enabled."
            )

        bitcoin_capacity_mw = float(
            bitcoin_capacity_mw
        )

        if bitcoin_capacity_mw <= 0.0:
            raise ValueError(
                "bitcoin.max_capacity_mw must be greater than zero."
            )

        hashprice_eur_per_th_day = float(
            bitcoin_cfg.get(
                "hashprice_eur_per_th_day",
                0.0,
            )
        )

        asic_efficiency_j_per_th = float(
            bitcoin_cfg.get(
                "asic_efficiency_j_per_th",
                0.0,
            )
        )

        bitcoin_other_opex_eur_per_mwh = float(
            bitcoin_cfg.get(
                "other_opex_eur_per_mwh",
                0.0,
            )
        )

        if hashprice_eur_per_th_day < 0.0:
            raise ValueError(
                "bitcoin.hashprice_eur_per_th_day "
                "must be non-negative."
            )

        if asic_efficiency_j_per_th <= 0.0:
            raise ValueError(
                "bitcoin.asic_efficiency_j_per_th "
                "must be greater than zero."
            )

        if bitcoin_other_opex_eur_per_mwh < 0.0:
            raise ValueError(
                "bitcoin.other_opex_eur_per_mwh "
                "must be non-negative."
            )

        # ASIC electrical efficiency:
        #
        # J/TH * TH/s = J/s = W
        #
        # Convert W per TH/s to MW per TH/s.
        mw_per_th_per_s = (
            asic_efficiency_j_per_th
            / 1e6
        )

        # Hashprice is quoted per TH/s per day.
        #
        # 1 MWh corresponds to operating 1 MW for one hour, i.e. 1/24 day.
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
            - bitcoin_other_opex_eur_per_mwh
        )

        # Regression check for the previously validated conversion:
        #
        # 16 J/TH and 0.08 EUR/(TH/s)/day
        # -> 2604.1667 TH-day/MWh
        # -> 208.3333 EUR/MWh gross revenue.
        if (
            abs(
                asic_efficiency_j_per_th
                - 16.0
            )
            < 1e-9
            and abs(
                hashprice_eur_per_th_day
                - 0.08
            )
            < 1e-9
            and abs(
                bitcoin_other_opex_eur_per_mwh
            )
            < 1e-9
        ):
            assert abs(
                th_day_per_mwh
                - 2604.166666666667
            ) < 1e-9

            assert abs(
                bitcoin_gross_revenue_eur_per_mwh
                - 208.33333333333334
            ) < 1e-9

        if "bitcoin_mining" not in n.carriers.index:
            n.add(
                "Carrier",
                "bitcoin_mining",
            )

        n.add(
            "Generator",
            "bitcoin_mining_sink",
            bus="electricity_bus",
            carrier="bitcoin_mining",
            sign=-1.0,
            p_nom=bitcoin_capacity_mw,
            p_nom_extendable=False,
            p_min_pu=0.0,
            p_max_pu=1.0,
            marginal_cost=-bitcoin_net_value_eur_per_mwh,
        )


    # -------------------------------------------------------------------------
    # 13. Add electrolyzer
    # -------------------------------------------------------------------------
    electrolyzer_efficiency = get_cost(
        "electrolysis",
        "efficiency",
    )

    configured_efficiency = hydrogen_cfg.get(
        "electrolyzer_efficiency"
    )

    if configured_efficiency is not None:
        electrolyzer_efficiency = float(
            configured_efficiency
        )

    if not 0.0 < electrolyzer_efficiency <= 1.0:
        raise ValueError(
            "Electrolyzer efficiency must be "
            "greater than 0 and at most 1."
        )

    electrolyzer_vom = float(
        hydrogen_cfg.get(
            "electrolyzer_variable_cost_eur_per_mwh",
            get_cost(
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
        capital_cost=get_annualized_capital_cost(
            "electrolysis"
        ),
        marginal_cost=electrolyzer_vom,
    )

    # -------------------------------------------------------------------------
    # 14. Add flexible hydrogen delivery
    # -------------------------------------------------------------------------
    #
    # Hydrogen leaves the modeled plant through a flexible delivery sink.
    #
    # This is NOT an exogenous hourly hydrogen demand profile.
    # The optimizer is free to choose WHEN hydrogen is produced and
    # delivered. Only the annual delivered quantity is constrained.
    #
    # A Generator with sign=-1 behaves as a controllable consumer on the
    # hydrogen bus. Its positive dispatch therefore represents hydrogen
    # leaving the modeled system.
    # -------------------------------------------------------------------------

    target_annual_kt_h2 = hydrogen_cfg.get(
        "target_annual_kt_h2"
    )

    if target_annual_kt_h2 is None:
        raise ValueError(
            "hydrogen.target_annual_kt_h2 must be defined "
            "for production_target mode."
        )

    target_annual_kt_h2 = float(
        target_annual_kt_h2
    )

    if target_annual_kt_h2 <= 0.0:
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

    # 1 kt = 1,000,000 kg.
    #
    # kg * kWh/kg / 1000 = MWh
    target_annual_h2_mwh = (
        target_annual_kt_h2
        * 1_000_000.0
        * hydrogen_lhv_kwh_per_kg
        / 1000.0
    )

    # hydrogen_delivery is only an accounting boundary through which
    # produced hydrogen leaves the modeled plant.
    #
    # Its nominal capacity is deliberately set high enough that it does
    # not impose an artificial hourly delivery constraint. The annual
    # quantity will instead be imposed with a custom Linopy constraint
    # during optimization.
    minimum_snapshot_weight = float(
        n.snapshot_weightings.generators.min()
    )

    hydrogen_delivery_p_nom = (
        target_annual_h2_mwh
        / minimum_snapshot_weight
    )

    n.add(
        "Generator",
        "hydrogen_delivery",
        bus="hydrogen_bus",
        carrier="hydrogen_delivery",
        sign=-1.0,
        p_nom=hydrogen_delivery_p_nom,
        p_nom_extendable=False,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=0.0,
    )

    # -------------------------------------------------------------------------
    # 15. Diagnostics
    # -------------------------------------------------------------------------
    print("\n================================================")
    print("OFF-GRID NETWORK BUILT")
    print("================================================")

    print(f"Scenario:        {cfg.get('scenario_name')}")
    print(f"Model type:      {model_type}")
    print(f"Investment year: {investment_year}")
    print(f"Weather year:    {weather_year}")
    print(f"Snapshots:       {len(n.snapshots)}")

    print(f"First snapshot:  {n.snapshots[0]}")
    print(f"Last snapshot:   {n.snapshots[-1]}")

    print("\nTechnology switches:")
    print(f"  Hydrogen:       {hydrogen_enabled}")
    print(f"  Battery:        {battery_enabled}")
    print(f"  Bitcoin:        {bitcoin_enabled}")
    print(f"  H2 storage:     {hydrogen_storage_enabled}")

    print("\nHydrogen:")
    print(
        f"  Electrolyzer efficiency: "
        f"{electrolyzer_efficiency:.4f}"
    )
    print(
        f"  Annual target [kt]: "
        f"{target_annual_kt_h2}"
    )
    print(
        f"  H2 LHV: "
        f"{hydrogen_lhv_kwh_per_kg:.2f} kWh/kg"
    )

    print(
        f"  Annual target energy: "
        f"{target_annual_h2_mwh:.2f} MWh_H2"
    )

    print(
        "  Annual target constraint: "
        "added during optimization"
    )

    if battery_enabled:
        print("\nBattery:")
        print(
            f"  Inverter technology: "
            f"{inverter_technology}"
        )
        print(
            f"  Storage technology: "
            f"{storage_technology}"
        )
        print(
            f"  Inverter efficiency: "
            f"{inverter_efficiency:.4f}"
        )
        print(
            f"  Charge efficiency: "
            f"{battery_charge_efficiency:.6f}"
        )
        print(
            f"  Discharge efficiency: "
            f"{battery_discharge_efficiency:.6f}"
        )
        print(
            f"  Round-trip efficiency: "
            f"{battery_charge_efficiency * battery_discharge_efficiency:.4f}"
        )
        print(
            f"  Inverter annual cost: "
            f"{battery_inverter_annual_cost:.2f} EUR/MW/a"
        )
        print(
            f"  Storage annual cost: "
            f"{battery_storage_annual_cost:.2f} EUR/MWh/a"
        )
        print(
            f"  Standing loss: "
            f"{battery_standing_loss:.6f}"
        )
        print(
            "  Power coupling constraint: "
            "required during optimization"
        )





    if bitcoin_enabled:
        print("\nBitcoin mining:")
        print(
            f"  Fixed mining capacity: "
            f"{bitcoin_capacity_mw:.3f} MW"
        )
        print(
            f"  ASIC efficiency: "
            f"{asic_efficiency_j_per_th:.3f} J/TH"
        )
        print(
            f"  Hashprice: "
            f"{hashprice_eur_per_th_day:.4f} "
            f"EUR/(TH/s)/day"
        )
        print(
            f"  TH-day per MWh: "
            f"{th_day_per_mwh:.6f}"
        )
        print(
            f"  Gross revenue: "
            f"{bitcoin_gross_revenue_eur_per_mwh:.6f} EUR/MWh"
        )
        print(
            f"  Variable mining OPEX: "
            f"{bitcoin_other_opex_eur_per_mwh:.6f} EUR/MWh"
        )
        print(
            f"  Net operating value: "
            f"{bitcoin_net_value_eur_per_mwh:.6f} EUR/MWh"
        )

    print("\nComponents:")
    print(
        "  Buses:      ",
        list(n.buses.index),
    )
    print(
        "  Generators: ",
        list(n.generators.index),
    )
    print(
        "  Links:      ",
        list(n.links.index),
    )
    print(
        "  Loads:      ",
        list(n.loads.index),
    )
    print(
        "  Stores:     ",
        list(n.stores.index),
    )

    print("================================================\n")

    return n
