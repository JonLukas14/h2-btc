import pypsa
import pandas as pd
from pathlib import Path


# -----------------------------------------------------------------------------
# 1. Define important project paths
# -----------------------------------------------------------------------------
# BASE_DIR points to the root of the project.
# DATA_DIR is where processed input CSV files are stored.
# KZ_REPO and TECH_REPO point to external/local data source folders.
# In this file, DATA_DIR is the most relevant one for the actual model inputs.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KZ_REPO = BASE_DIR / "pypsa-kz-data"
TECH_REPO = BASE_DIR / "technology-data"


# -----------------------------------------------------------------------------
# 2. Helper function: load a CSV column as a time series
# -----------------------------------------------------------------------------
# This reads a CSV file and turns one column into a pandas Series
# with the model snapshots as the index.
def load_timeseries(path, snapshots, column_name=None):
    df = pd.read_csv(path)

    # If no column name is specified, use the first column in the file.
    if column_name is None:
        series = df.iloc[:, 0]
    else:
        series = df[column_name]

    # Reindex the values onto the model snapshot index.
    series = pd.Series(series.values, index=snapshots)
    return series.astype(float)


# -----------------------------------------------------------------------------
# 3. Helper function: choose the active cost dataset from the config
# -----------------------------------------------------------------------------
# This reads cfg["costs"]["active_dataset"] and finds the matching file path
# inside cfg["costs"]["datasets"].
def get_active_costs_path(cfg, base_dir):
    """
    Resolve the active cost dataset from config.
    Expected config structure:

    costs:
      active_dataset: "costs_2025"
      datasets:
        costs_2025: "data/costs_2025.csv"
        costs_2030: "data/costs_2030.csv"
      discount_rate: 0.07
    """
    costs_cfg = cfg["costs"]
    active_name = costs_cfg["active_dataset"]
    datasets = costs_cfg["datasets"]

    # Stop early if the requested dataset name is not defined.
    if active_name not in datasets:
        raise KeyError(
            f"Unknown cost dataset '{active_name}'. "
            f"Available options: {list(datasets.keys())}"
        )

    # Return the full file path of the selected cost dataset.
    return base_dir / datasets[active_name]


# -----------------------------------------------------------------------------
# 4. Helper function: get a specific cost parameter from the costs table
# -----------------------------------------------------------------------------
# Example: get investment cost for "solar-utility", or lifetime for "onwind".
def get_cost_value(costs_df, technology, parameter):
    row = costs_df[
        (costs_df["technology"] == technology) &
        (costs_df["parameter"] == parameter)
    ]

    # Raise an error if the requested technology/parameter pair is missing.
    if row.empty:
        raise KeyError(f"Missing {parameter} for technology {technology} in costs file")

    return float(row["value"].iloc[0])


# -----------------------------------------------------------------------------
# 5. Helper function: annualize overnight investment cost
# -----------------------------------------------------------------------------
# Converts EUR/MW investment cost into the equivalent model-period capital cost.
# It uses the Capital Recovery Factor (CRF), then scales by modeled hours.
def annualized_capital_cost(overnight_eur_per_mw, lifetime_years, discount_rate, n_snapshots):
    """
    Convert overnight investment cost (EUR/MW) to model-period capital cost.
    Uses Capital Recovery Factor (CRF) to annualize, then scales
    to the fraction of year represented by n_snapshots.
    """
    if discount_rate > 0:
        crf = (discount_rate * (1 + discount_rate) ** lifetime_years) / \
              ((1 + discount_rate) ** lifetime_years - 1)
    else:
        crf = 1.0 / lifetime_years

    annualized = overnight_eur_per_mw * crf
    return annualized * (n_snapshots / 8760)


# -----------------------------------------------------------------------------
# 6. Helper function: convert FOM into a marginal-cost-like value
# -----------------------------------------------------------------------------
# PyPSA separates CAPEX and OPEX, but here you convert fixed O&M into EUR/MWh
# so it shows up inside the operational cost statistics.
def fom_as_marginal_cost(costs_df, technology):
    """
    Convert fixed O&M (% of investment per year) into an equivalent EUR/MWh
    so it shows up in PyPSA OPEX statistics.
    """
    investment_eur_per_mw = get_cost_value(costs_df, technology, "investment")
    fom_percent_per_year = get_cost_value(costs_df, technology, "FOM")
    return investment_eur_per_mw * (fom_percent_per_year / 100.0) / 8760.0


# -----------------------------------------------------------------------------
# 7. Helper function: convert BTC mining economics into EUR/MWh revenue
# -----------------------------------------------------------------------------
# This converts hashprice and ASIC efficiency into electricity-value terms.
# Later this is used as a negative marginal cost for the mining sink.
def mining_revenue_per_mwh(hashprice_eur_per_th_day, asic_efficiency_j_per_th):
    mw_per_th = asic_efficiency_j_per_th / 1e6
    mwh_per_th_day = mw_per_th * 24.0
    return hashprice_eur_per_th_day / mwh_per_th_day


# -----------------------------------------------------------------------------
# 8. Main function: build the PyPSA test network from config and input data
# -----------------------------------------------------------------------------
def build_test_network(cfg):
    # Create an empty PyPSA network object.
    n = pypsa.Network()

    # Read time settings from the config and create hourly snapshots.
    snapshots = int(cfg["system"]["snapshots"])
    year = int(cfg["system"]["year"])
    n.set_snapshots(pd.date_range(f"{year}-01-01 00:00", periods=snapshots, freq="h"))

    # Resolve project root again locally.
    base_dir = Path(__file__).resolve().parent.parent

    # Define the input file paths used in this network.
    costs_path = get_active_costs_path(cfg, base_dir)
    solar_cf_path = base_dir / "data" / "kz_solar_cf.csv"
    wind_cf_path = base_dir / "data" / "kz_wind_cf.csv"
    electricity_demand_path = base_dir / "data" / "kz_electricity_demand.csv"
    hydrogen_demand_path = base_dir / "data" / "kz_hydrogen_demand.csv"

    # Load the selected technology cost dataset.
    costs_df = pd.read_csv(costs_path)
    print(f"Using cost dataset: {cfg['costs']['active_dataset']} -> {costs_path}")

    # Load solar and wind hourly capacity factors.
    solar_cf = load_timeseries(solar_cf_path, n.snapshots)
    wind_cf = load_timeseries(wind_cf_path, n.snapshots)

    # Print quick diagnostics so you can verify the input shapes look right.
    print(f"  Solar CF shape: {solar_cf.shape}, unique values: {solar_cf.nunique()}")
    print(f"  Wind CF shape:  {wind_cf.shape}, unique values: {wind_cf.nunique()}")

    # Build electricity demand.
    # By default this is a flat demand from the config.
    # If the config says so, replace it with an hourly profile from CSV.
    electricity_demand = pd.Series(float(cfg["demand"]["electricity_mw"]), index=n.snapshots)
    if cfg["demand"].get("use_hourly_electricity_profile", False):
        electricity_demand = load_timeseries(electricity_demand_path, n.snapshots)

    # Build hydrogen demand in the same way.
    hydrogen_demand = pd.Series(float(cfg["demand"]["hydrogen_mw"]), index=n.snapshots)
    if cfg["demand"].get("use_hourly_hydrogen_profile", False):
        hydrogen_demand = load_timeseries(hydrogen_demand_path, n.snapshots)

    # Read the discount rate used for annualizing investment costs.
    discount_rate = float(cfg["costs"].get("discount_rate", 0.07))


    # -------------------------------------------------------------------------
    # 9. Compute technology-specific CAPEX and OPEX inputs
    # -------------------------------------------------------------------------
    # Solar capital and operating cost inputs.
    solar_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "solar-utility", "investment"),
        get_cost_value(costs_df, "solar-utility", "lifetime"),
        discount_rate,
        snapshots,
    )
    solar_marginal_cost = fom_as_marginal_cost(costs_df, "solar-utility")

    # Wind capital and operating cost inputs.
    wind_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "onwind", "investment"),
        get_cost_value(costs_df, "onwind", "lifetime"),
        discount_rate,
        snapshots,
    )
    wind_marginal_cost = fom_as_marginal_cost(costs_df, "onwind")

    # Electrolyzer annualized capital cost.
    electrolyzer_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "electrolysis", "investment"),
        get_cost_value(costs_df, "electrolysis", "lifetime"),
        discount_rate,
        snapshots,
    )

    # Hydrogen storage annualized capital cost.
    hydrogen_storage_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "hydrogen storage underground", "investment"),
        get_cost_value(costs_df, "hydrogen storage underground", "lifetime"),
        discount_rate,
        snapshots,
    )


    # -------------------------------------------------------------------------
    # 10. Read config subsections used later in the network definition
    # -------------------------------------------------------------------------
    hyd = cfg["hydrogen"]
    tech = cfg["technology"]

    hydrogen_enabled = bool(hyd.get("enabled", True))
    hydrogen_mode = hyd.get("mode", "fixed_demand")
    electrolyzer_variable_cost = float(hyd.get("electrolyzer_variable_cost_eur_per_mwh", 0.0))
    storage_standing_loss = float(hyd.get("storage_standing_loss", 0.0))


    # -------------------------------------------------------------------------
    # 11. Add carriers
    # -------------------------------------------------------------------------
    # Carriers are labels/categories for components.
    n.add("Carrier", "electricity")
    n.add("Carrier", "hydrogen")
    n.add("Carrier", "solar")
    n.add("Carrier", "wind")
    n.add("Carrier", "electrolyzer")
    n.add("Carrier", "hydrogen_storage")
    n.add("Carrier", "load_shedding")
    n.add("Carrier", "bitcoin_mining")


    # -------------------------------------------------------------------------
    # 12. Add buses
    # -------------------------------------------------------------------------
    # Buses are the nodes where energy is balanced.
    # This model has one electricity bus and one hydrogen bus.
    n.add("Bus", "electricity", carrier="electricity")
    n.add("Bus", "hydrogen", carrier="hydrogen")


    # -------------------------------------------------------------------------
    # 13. Add fixed electricity demand
    # -------------------------------------------------------------------------
    n.add(
        "Load",
        "electricity_demand",
        bus="electricity",
        p_set=electricity_demand,
    )


    # -------------------------------------------------------------------------
    # 14. Add fixed hydrogen demand only if hydrogen mode requires it
    # -------------------------------------------------------------------------
    # In "fixed_demand" mode the model must serve hydrogen demand.
    # In "flexible_sink" mode there is no fixed hydrogen load.
    if hydrogen_enabled and hydrogen_mode == "fixed_demand":
        n.add(
            "Load",
            "hydrogen_demand",
            bus="hydrogen",
            p_set=hydrogen_demand,
        )


    # -------------------------------------------------------------------------
    # 15. Add renewable generators
    # -------------------------------------------------------------------------
    # Solar generator with extendable capacity and time-varying max output.
    n.add(
        "Generator",
        "solar",
        bus="electricity",
        carrier="solar",
        p_nom_extendable=True,
        p_max_pu=solar_cf,
        capital_cost=solar_capital_cost,
        marginal_cost=solar_marginal_cost,
    )

    # Wind generator with extendable capacity and time-varying max output.
    n.add(
        "Generator",
        "wind",
        bus="electricity",
        carrier="wind",
        p_nom_extendable=True,
        p_max_pu=wind_cf,
        capital_cost=wind_capital_cost,
        marginal_cost=wind_marginal_cost,
    )


    # -------------------------------------------------------------------------
    # 16. Add hydrogen system if hydrogen is enabled
    # -------------------------------------------------------------------------
    if hydrogen_enabled:
        # Electrolyzer converts electricity into hydrogen.
        n.add(
            "Link",
            "electrolyzer",
            bus0="electricity",
            bus1="hydrogen",
            carrier="electrolyzer",
            p_nom_extendable=True,
            efficiency=float(hyd["electrolyzer_efficiency"]),
            capital_cost=electrolyzer_capital_cost,
            marginal_cost=electrolyzer_variable_cost,
        )

        # Hydrogen storage store on the hydrogen bus.
        n.add(
            "Store",
            "hydrogen_storage",
            bus="hydrogen",
            carrier="hydrogen_storage",
            e_nom_extendable=True,
            e_cyclic=bool(hyd["cyclic_storage"]),
            standing_loss=storage_standing_loss,
            capital_cost=hydrogen_storage_capital_cost,
        )


    # -------------------------------------------------------------------------
    # 17. Add load shedding as an emergency backup generator
    # -------------------------------------------------------------------------
    # This prevents infeasibility by allowing unmet demand at a very high cost.
    n.add(
        "Generator",
        "load_shedding",
        bus="electricity",
        carrier="load_shedding",
        p_nom=1_000_000,
        marginal_cost=float(tech["load_shedding_marginal_cost"]),
    )


    # -------------------------------------------------------------------------
    # 18. Add optional Bitcoin mining sink
    # -------------------------------------------------------------------------
    mining_cfg = cfg.get("mining", {})
    mining_enabled = bool(mining_cfg.get("enabled", False))

    if mining_enabled:
        # Read mining settings from the config.
        mining_max_mw = float(mining_cfg.get("max_capacity_mw", 0))
        mining_min_frac = float(mining_cfg.get("min_utilization", 0.0))

        hashprice = float(mining_cfg.get("hashprice_eur_per_th_day", 0.0))
        asic_eff = float(mining_cfg.get("asic_efficiency_j_per_th", 16.0))
        other_opex = float(mining_cfg.get("other_opex_eur_per_mwh", 0.0))

        # Convert BTC income into electricity-value terms.
        mining_revenue = mining_revenue_per_mwh(hashprice, asic_eff)

        # Negative marginal cost means the optimizer sees this as profitable demand.
        mining_mc = -(mining_revenue - other_opex)

        # Mining is modeled as a generator with sign=-1,
        # which means it behaves like electricity consumption.
        n.add(
            "Generator",
            "bitcoin_mining_sink",
            bus="electricity",
            carrier="bitcoin_mining",
            p_nom=mining_max_mw,
            p_nom_extendable=False,
            p_min_pu=mining_min_frac,
            p_max_pu=1.0,
            marginal_cost=mining_mc,
            sign=-1,
        )


    # -------------------------------------------------------------------------
    # 19. Return the completed network
    # -------------------------------------------------------------------------
    return n