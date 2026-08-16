from pathlib import Path
import pandas as pd
import pypsa


# -----------------------------------------------------------------------------
# 1. Build PyPSA network from config and processed input data
# -----------------------------------------------------------------------------
# cfg contains the scenario settings loaded from YAML.
# data_dir points to the scenario-specific folder with processed CSV inputs.
def build_test_network(cfg, data_dir):
    # Convert data_dir to a Path object for safe path handling.
    data_dir = Path(data_dir)

    # Check that the required processed input files exist.
    required_files = [
        data_dir / "kz_electricity_demand.csv",
        data_dir / "kz_hydrogen_demand.csv",
        data_dir / "kz_solar_cf.csv",
        data_dir / "kz_wind_cf.csv",
        data_dir / f"{cfg['costs']['active_dataset']}.csv",
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed input files:\n" + "\n".join(str(p) for p in missing)
        )

    # -------------------------------------------------------------------------
    # 2. Read scenario settings
    # -------------------------------------------------------------------------
    system_cfg = cfg["system"]

    snapshots = int(system_cfg["snapshots"])
    investment_year = int(system_cfg["investment_year"])
    weather_year = int(system_cfg["weather_year"])
    demand_year = int(system_cfg["demand_year"])

    # Technology switches and capacities from config.
    tech_cfg = cfg.get("technology", {})
    hydrogen_cfg = cfg.get("hydrogen", {})
    mining_cfg = cfg.get("mining", {})

    # -------------------------------------------------------------------------
    # 3. Load processed time series and cost data
    # -------------------------------------------------------------------------
    electricity_demand = pd.read_csv(data_dir / "kz_electricity_demand.csv")
    hydrogen_demand = pd.read_csv(data_dir / "kz_hydrogen_demand.csv")
    solar_cf = pd.read_csv(data_dir / "kz_solar_cf.csv")
    wind_cf = pd.read_csv(data_dir / "kz_wind_cf.csv")
    costs = pd.read_csv(data_dir / f"{cfg['costs']['active_dataset']}.csv")

    # Convert series columns to pandas Series for easier use.
    electricity_demand = electricity_demand["electricity_mw"]
    hydrogen_demand = hydrogen_demand["hydrogen_mw"]
    solar_cf = solar_cf["solar_cf"]
    wind_cf = wind_cf["wind_cf"]

    # Validate that every time series matches the configured optimization horizon.
    time_series = {
        "electricity_demand": electricity_demand,
        "hydrogen_demand": hydrogen_demand,
        "solar_cf": solar_cf,
        "wind_cf": wind_cf,
    }

    for name, series in time_series.items():
        if len(series) != snapshots:
            raise ValueError(
                f"{name} contains {len(series)} rows, "
                f"but system.snapshots is {snapshots}."
            )

    # -------------------------------------------------------------------------
    # 4. Build snapshots
    # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
    # 4. Build snapshots
    # -------------------------------------------------------------------------
    # Build the complete calendar year corresponding to the weather data.
    full_year_index = pd.date_range(
        start=f"{weather_year}-01-01 00:00:00",
        end=f"{weather_year}-12-31 23:00:00",
        freq="h",
    )

    # For leap years, remove February 29 when using a 8760-hour model.
    # This keeps the PyPSA timestamps aligned with the normalized
    # demand, solar, and wind input series.
    if len(full_year_index) == 8784 and snapshots == 8760:
        leap_day = (
            (full_year_index.month == 2)
            & (full_year_index.day == 29)
        )
        snapshot_index = full_year_index[~leap_day]
    else:
        snapshot_index = full_year_index

    # Fail explicitly if the calendar and configured horizon do not match.
    if len(snapshot_index) != snapshots:
        raise ValueError(
            f"Snapshot calendar contains {len(snapshot_index)} hours, "
            f"but system.snapshots={snapshots}."
        )
    # -------------------------------------------------------------------------
    # 5. Create empty PyPSA network
    # -------------------------------------------------------------------------
    n = pypsa.Network()
    n.set_snapshots(snapshot_index)

    # -------------------------------------------------------------------------
    # 6. Add buses
    # -------------------------------------------------------------------------
    n.add("Bus", "electricity_bus", carrier="AC")
    n.add("Bus", "hydrogen_bus", carrier="H2")

    # -------------------------------------------------------------------------
    # 7. Helper function to extract cost values
    # -------------------------------------------------------------------------
    # This reads a single parameter value from the processed costs table.
    def get_cost_row(technology, parameter):
        rows = costs[
            (costs["technology"] == technology) &
            (costs["parameter"] == parameter)
        ]

        if rows.empty:
            return None

        return rows.iloc[0]


    def get_cost(technology, parameter, default=None):
        row = get_cost_row(technology, parameter)

        if row is None:
            if default is not None:
                return float(default)

            raise KeyError(
                f"Missing cost entry for {technology} / {parameter}"
            )

        return float(row["value"])


    def get_investment_cost_for_pypsa(technology):
        """
        Convert investment costs from the technology-data units to the
        MW/MWh units used for PyPSA nominal capacities.
        """
        row = get_cost_row(technology, "investment")

        if row is None:
            raise KeyError(
                f"Missing investment cost for technology '{technology}'"
            )

        value = float(row["value"])
        unit = str(row["unit"]).strip()

        # Power technologies:
        # EUR/kW -> EUR/MW
        #
        # Energy storage:
        # EUR/kWh -> EUR/MWh
        if unit in {"EUR/kW", "EUR/kW_e", "EUR/kWh"}:
            return value * 1000.0

        # Already in PyPSA-compatible MW/MWh units.
        if unit in {"EUR/MW", "EUR/MW_e", "EUR/MWh"}:
            return value

        raise ValueError(
            f"Unsupported investment-cost unit '{unit}' "
            f"for technology '{technology}'."
        )


    def annuity(rate, lifetime):
        if lifetime <= 0:
            raise ValueError(
                f"Lifetime must be positive, got {lifetime}"
            )

        if rate == 0:
            return 1.0 / lifetime

        return rate / (
            1.0 - (1.0 + rate) ** (-lifetime)
        )


    def get_annualized_capital_cost(
        technology,
        default_discount_rate=0.07,
    ):
        investment = get_investment_cost_for_pypsa(technology)
        fom_percent = get_cost(technology, "FOM", 0.0)
        lifetime = get_cost(technology, "lifetime")

        discount_rate = float(
            cfg.get("costs", {}).get(
                "discount_rate",
                default_discount_rate,
            )
        )

        annualized_capex = (
            investment
            * annuity(discount_rate, lifetime)
        )

        # FOM is given as % of original investment per year.
        annualized_fom = (
            investment
            * fom_percent
            / 100.0
        )

        return annualized_capex + annualized_fom

    # -------------------------------------------------------------------------
    # 8. Add generators: solar and wind
    # -------------------------------------------------------------------------
    n.add(
        "Generator",
        "solar",
        bus="electricity_bus",
        carrier="solar",
        p_nom_extendable=True,
        capital_cost=get_annualized_capital_cost("solar-utility"),
        marginal_cost=get_cost("solar-utility", "VOM", 0.0),
        efficiency=1.0,
        p_max_pu=solar_cf.values,
    )

    n.add(
        "Generator",
        "wind",
        bus="electricity_bus",
        carrier="wind",
        p_nom_extendable=True,
        capital_cost=get_annualized_capital_cost("onwind"),
        marginal_cost=get_cost("onwind", "VOM", 0.0),
        efficiency=1.0,
        p_max_pu=wind_cf.values,
    )

    # -------------------------------------------------------------------------
    # 9. Add electricity demand load
    # -------------------------------------------------------------------------
    n.add(
        "Load",
        "electricity_demand",
        bus="electricity_bus",
        p_set=electricity_demand.values,
    )

    # -------------------------------------------------------------------------
    # 10. Add load shedding generator
    # -------------------------------------------------------------------------
    # Very expensive fallback generation used to preserve model feasibility.

    load_shedding_cost = float(
        tech_cfg.get(
            "load_shedding_marginal_cost",
            10000.0,
        )
    )

    n.add(
        "Generator",
        "load_shedding",
        bus="electricity_bus",
        carrier="load_shedding",
        p_nom_extendable=True,
        marginal_cost=load_shedding_cost,
    )

    # -------------------------------------------------------------------------
    # 11. Add hydrogen system if enabled
    # -------------------------------------------------------------------------
      # -------------------------------------------------------------------------
    # 11. Add hydrogen system if enabled
    # -------------------------------------------------------------------------
    hydrogen_enabled = bool(
        hydrogen_cfg.get("enabled", True)
    )

    hydrogen_mode = hydrogen_cfg.get(
        "mode",
        "fixed_demand",
    )

    if hydrogen_enabled:

        # ---------------------------------------------------------------------
        # Electrolyzer configuration
        # ---------------------------------------------------------------------
        electrolyzer_efficiency = float(
            hydrogen_cfg.get(
                "electrolyzer_efficiency",
                get_cost(
                    "electrolysis",
                    "efficiency",
                    0.7,
                ),
            )
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

        if not 0.0 < electrolyzer_efficiency <= 1.0:
            raise ValueError(
                "Electrolyzer efficiency must be "
                "greater than 0 and at most 1."
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

        # ---------------------------------------------------------------------
        # Hydrogen storage configuration
        # ---------------------------------------------------------------------
        cyclic_storage = bool(
            hydrogen_cfg.get(
                "cyclic_storage",
                True,
            )
        )

        storage_standing_loss = float(
            hydrogen_cfg.get(
                "storage_standing_loss",
                0.0,
            )
        )

        if not 0.0 <= storage_standing_loss < 1.0:
            raise ValueError(
                "storage_standing_loss must be between "
                "0 and 1."
            )

        n.add(
            "Store",
            "hydrogen_storage",
            bus="hydrogen_bus",
            carrier="hydrogen_storage",
            e_nom_extendable=True,
            capital_cost=get_annualized_capital_cost(
                "hydrogen storage underground"
            ),
            e_cyclic=cyclic_storage,
            standing_loss=storage_standing_loss,
        )

        # ---------------------------------------------------------------------
        # Fixed hydrogen demand
        # ---------------------------------------------------------------------
        if hydrogen_mode == "fixed_demand":
            n.add(
                "Load",
                "hydrogen_demand",
                bus="hydrogen_bus",
                p_set=hydrogen_demand.values,
            )

        # ---------------------------------------------------------------------
        # Legacy flexible hydrogen sink
        # ---------------------------------------------------------------------
        elif hydrogen_mode == "flexible_sink":
            legacy_demand_cfg = cfg.get("demand", {})

            hydrogen_sink_mw = float(
                hydrogen_cfg.get(
                    "max_sink_mw",
                    legacy_demand_cfg.get(
                        "hydrogen_mw",
                        0.0,
                    ),
                )
            )

            if hydrogen_sink_mw <= 0.0:
                raise ValueError(
                    "hydrogen.mode is 'flexible_sink', "
                    "but no positive H2 sink capacity is configured. "
                    "Use hydrogen.max_sink_mw or switch to "
                    "'production_target'."
                )

            n.add(
                "Generator",
                "hydrogen_sink",
                bus="hydrogen_bus",
                carrier="hydrogen_sink",
                sign=-1.0,
                p_nom=hydrogen_sink_mw,
                p_nom_extendable=False,
                p_min_pu=0.0,
                p_max_pu=1.0,
                marginal_cost=0.0,
            )

        # ---------------------------------------------------------------------
        # Annual hydrogen production target
        # ---------------------------------------------------------------------
        elif hydrogen_mode == "production_target":
            target_kt_h2 = float(
                hydrogen_cfg.get(
                    "target_annual_kt_h2",
                    0.0,
                )
            )

            if target_kt_h2 <= 0.0:
                raise ValueError(
                    "hydrogen.mode is 'production_target', "
                    "but target_annual_kt_h2 is missing or <= 0."
                )

            # We intentionally do NOT implement the annual constraint here yet.
            # This will be added in the next validation step using the
            # optimization model / extra_functionality.
            raise NotImplementedError(
                "Hydrogen production_target configuration is valid, "
                "but the annual PyPSA constraint has not yet been "
                "implemented. This is the next model-validation step."
            )

        else:
            raise ValueError(
                f"Unknown hydrogen mode '{hydrogen_mode}'. "
                "Supported modes are: fixed_demand, "
                "flexible_sink, production_target."
            )
        # -------------------------------------------------------------------------
    # 12. Add bitcoin mining sink if enabled
    # -------------------------------------------------------------------------
    # Mining is modeled as an optional electricity consumer.
    # In PyPSA, this is represented as a Generator with sign = -1.
    # A negative marginal cost means that consuming electricity yields value
    # to the optimizer, so the model will dispatch mining when profitable.
    if bool(mining_cfg.get("enabled", False)):
        mining_max_mw = float(mining_cfg.get("max_capacity_mw", 0.0))

        hashprice_eur_per_th_day = float(
            mining_cfg.get("hashprice_eur_per_th_day", 0.08)
        )
        asic_efficiency_j_per_th = float(
            mining_cfg.get("asic_efficiency_j_per_th", 16.0)
        )
        other_opex_eur_per_mwh = float(
            mining_cfg.get("other_opex_eur_per_mwh", 0.0)
        )

        # Convert ASIC efficiency to MW per TH/s:
        # J/s = W, so J/TH * TH/s = W
        mw_per_th_per_s = asic_efficiency_j_per_th / 1e6

        # Validate ASIC efficiency.
        if mw_per_th_per_s <= 0:
            raise ValueError(
                "ASIC efficiency must be greater than zero."
            )

        # 1 MWh means 1 MW operated for 1 hour.
        # Hashprice is quoted per TH/s per day, therefore one hour is 1/24 day.
        th_day_per_mwh = (1.0 / 24.0) / mw_per_th_per_s

        # Gross mining revenue attributable to 1 MWh of electricity.
        gross_revenue_eur_per_mwh = (
            hashprice_eur_per_th_day
            * th_day_per_mwh
        )

        # Net operating value after variable mining OPEX.
        net_value_eur_per_mwh = (
            gross_revenue_eur_per_mwh
            - other_opex_eur_per_mwh
        )

        if (
            abs(asic_efficiency_j_per_th - 16.0) < 1e-9
            and abs(hashprice_eur_per_th_day - 0.08) < 1e-9
            and abs(other_opex_eur_per_mwh) < 1e-9
        ):
            assert abs(th_day_per_mwh - 2604.1667) < 0.01
            assert abs(gross_revenue_eur_per_mwh - 208.3333) < 0.01



        print("\n--- Bitcoin mining calculation ---")
        print(f"ASIC efficiency:     {asic_efficiency_j_per_th:.2f} J/TH")
        print(f"Hashprice:           {hashprice_eur_per_th_day:.4f} EUR/TH/day")
        print(f"TH-day per MWh:      {th_day_per_mwh:.2f}")
        print(f"Gross revenue:       {gross_revenue_eur_per_mwh:.2f} EUR/MWh")
        print(f"Other OPEX:          {other_opex_eur_per_mwh:.2f} EUR/MWh")
        print(f"Net operating value: {net_value_eur_per_mwh:.2f} EUR/MWh")
        print("----------------------------------\n")

        n.add(
            "Generator",
            "bitcoin_mining_sink",
            bus="electricity_bus",
            carrier="bitcoin_mining",
            sign=-1.0,
            p_nom=mining_max_mw,
            p_nom_extendable=False,
            p_min_pu=0.0,
            p_max_pu=1.0,
            marginal_cost=-net_value_eur_per_mwh,
        )

    # -------------------------------------------------------------------------
    # 13. Print basic network diagnostics
    # -------------------------------------------------------------------------
    print("Built network successfully")
    print(f"Investment year: {investment_year}")
    print(f"Weather year: {weather_year}")
    print(f"Demand year: {demand_year}")
    print(f"Snapshots: {len(n.snapshots)}")
    print(f"First snapshot: {n.snapshots[0]}")
    print(f"Last snapshot: {n.snapshots[-1]}")
    print(f"Data directory: {data_dir}")
    print(f"Hydrogen enabled: {hydrogen_enabled}")
    print(f"Hydrogen mode: {hydrogen_mode}")
    print(f"Mining enabled: {bool(mining_cfg.get('enabled', False))}")

    return n