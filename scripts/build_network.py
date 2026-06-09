import pypsa
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KZ_REPO = BASE_DIR / "pypsa-kz-data"
TECH_REPO = BASE_DIR / "technology-data"

def load_timeseries(path, snapshots, column_name=None):
    df = pd.read_csv(path)

    if column_name is None:
        series = df.iloc[:, 0]
    else:
        series = df[column_name]

    series = pd.Series(series.values, index=snapshots)
    return series.astype(float)


def get_cost_value(costs_df, technology, parameter):
    row = costs_df[
        (costs_df["technology"] == technology) &
        (costs_df["parameter"] == parameter)
    ]

    if row.empty:
        raise KeyError(f"Missing {parameter} for technology {technology} in costs file")

    return float(row["value"].iloc[0])

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

    annualized = overnight_eur_per_mw * crf          # EUR/MW/year
    return annualized * (n_snapshots / 8760)          # EUR/MW for model period

def build_test_network(cfg):
    n = pypsa.Network()

    snapshots = int(cfg["system"]["snapshots"])
    year = int(cfg["system"]["year"])
    n.set_snapshots(pd.date_range(f"{year}-01-01 00:00", periods=snapshots, freq="h"))

    base_dir = Path(__file__).resolve().parent.parent
    data_files = cfg["data_files"]

    costs_path = base_dir / data_files["costs"]
    solar_cf_path = base_dir / data_files["solar_cf"]
    wind_cf_path = base_dir / data_files["wind_cf"]
    electricity_demand_path = base_dir / data_files["electricity_demand"]

    costs_df = pd.read_csv(costs_path)

    solar_cf = load_timeseries(solar_cf_path, n.snapshots)
    wind_cf = load_timeseries(wind_cf_path, n.snapshots)

    print(f"  Solar CF shape: {solar_cf.shape}, unique values: {solar_cf.nunique()}")
    print(f"  Wind CF shape:  {wind_cf.shape}, unique values: {wind_cf.nunique()}")

    electricity_demand = pd.Series(float(cfg["demand"]["electricity_mw"]), index=n.snapshots)
    if cfg["demand"].get("use_hourly_electricity_profile", False):
        electricity_demand = load_timeseries(electricity_demand_path, n.snapshots)

    hydrogen_demand = pd.Series(float(cfg["demand"]["hydrogen_mw"]), index=n.snapshots)
    if cfg["demand"].get("use_hourly_hydrogen_profile", False):
        hydrogen_demand_path = base_dir / data_files["hydrogen_demand"]
        hydrogen_demand = load_timeseries(hydrogen_demand_path, n.snapshots)

    discount_rate = float(cfg.get("costs", {}).get("discount_rate", 0.07))

    solar_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "solar-utility", "investment"),
        get_cost_value(costs_df, "solar-utility", "lifetime"),
        discount_rate, snapshots
    )
    wind_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "onwind", "investment"),
        get_cost_value(costs_df, "onwind", "lifetime"),
        discount_rate, snapshots
    )
    electrolyzer_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "electrolysis", "investment"),
        get_cost_value(costs_df, "electrolysis", "lifetime"),
        discount_rate, snapshots
    )
    hydrogen_storage_capital_cost = annualized_capital_cost(
        get_cost_value(costs_df, "hydrogen storage underground", "investment"),
        get_cost_value(costs_df, "hydrogen storage underground", "lifetime"),
        discount_rate, snapshots
    )

    hyd = cfg["hydrogen"]
    tech = cfg["technology"]

    n.add("Carrier", "electricity")
    n.add("Carrier", "hydrogen")
    n.add("Carrier", "solar")
    n.add("Carrier", "wind")
    n.add("Carrier", "electrolyzer")
    n.add("Carrier", "hydrogen_storage")
    n.add("Carrier", "load_shedding")
    n.add("Carrier", "bitcoin_mining")

    n.add("Bus", "electricity", carrier="electricity")
    n.add("Bus", "hydrogen", carrier="hydrogen")

    n.add(
        "Load",
        "electricity_demand",
        bus="electricity",
        p_set=electricity_demand,
    )

    n.add(
        "Load",
        "hydrogen_demand",
        bus="hydrogen",
        p_set=hydrogen_demand,
    )

    n.add(
        "Generator",
        "solar",
        bus="electricity",
        carrier="solar",
        p_nom_extendable=True,
        p_max_pu=solar_cf,
        capital_cost=solar_capital_cost,
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "wind",
        bus="electricity",
        carrier="wind",
        p_nom_extendable=True,
        p_max_pu=wind_cf,
        capital_cost=wind_capital_cost,
        marginal_cost=0,
    )

    n.add(
        "Link",
        "electrolyzer",
        bus0="electricity",
        bus1="hydrogen",
        carrier="electrolyzer",
        p_nom_extendable=True,
        efficiency=float(hyd["electrolyzer_efficiency"]),
        capital_cost=electrolyzer_capital_cost,
    )

    n.add(
        "Store",
        "hydrogen_storage",
        bus="hydrogen",
        carrier="hydrogen_storage",
        e_nom_extendable=True,
        e_cyclic=bool(hyd["cyclic_storage"]),
        capital_cost=hydrogen_storage_capital_cost,
    )

        # ── Bitcoin mining (flexible electricity sink) ─────────────────────────
    mining_cfg = cfg.get("mining", {})
    mining_enabled = bool(mining_cfg.get("enabled", False))

    if mining_enabled:
        mining_max_mw     = float(mining_cfg.get("max_capacity_mw", 0))
        mining_min_frac   = float(mining_cfg.get("min_utilization", 0.0))
        mining_mc         = float(mining_cfg.get("marginal_cost", -50))

        # Flexible mining load: modelled as a Generator with negative marginal cost.
        # Negative mc means the optimizer wants to run it — it "earns" from mining.
        # p_nom is the max capacity; optimizer decides hourly utilization between 0 and p_nom.
        n.add(
            "Generator",
            "bitcoin_mining",
            bus="electricity",
            carrier="bitcoin_mining",
            p_nom=mining_max_mw,          # fixed installed capacity
            p_nom_extendable=False,       # capacity is a scenario assumption, not optimized
            p_min_pu=mining_min_frac,     # minimum utilization (0 = fully flexible)
            p_max_pu=1.0,
            marginal_cost=mining_mc,      # negative = revenue to system when running
            sign=-1,                      # sign=-1 means this is a load, not a generator
        )

        # Minimum always-on floor (if min_utilization > 0):
        # This is already enforced by p_min_pu above, no separate Load needed.

        print(f"  ✓ Bitcoin mining added: max={mining_max_mw} MW, "
              f"min_utilization={mining_min_frac*100:.0f}%, "
              f"marginal_cost={mining_mc} EUR/MWh")
    else:
        print("  ℹ Bitcoin mining disabled (mining.enabled: false in config)")

    n.add(
        "Generator",
        "load_shedding",
        bus="electricity",
        carrier="load_shedding",
        p_nom=1_000_000,
        marginal_cost=float(tech["load_shedding_marginal_cost"]),
    )

    return n

