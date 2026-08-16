from pathlib import Path
import argparse
import yaml
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------------------------------------------------------
# 1. Parse command-line arguments
# -----------------------------------------------------------------------------
# This makes the script reusable for different scenarios and output folders.
# Example usage:
# python scripts/analyze_results.py \
#   --config configs/scenarios/scenario_s0_reference.yaml \
#   --network results/s0_reference/network_solved.nc \
#   --outdir results/s0_reference
parser = argparse.ArgumentParser(description="Analyze solved PyPSA scenario results.")
parser.add_argument("--config", required=True, help="Path to scenario YAML file")
parser.add_argument("--network", required=True, help="Path to solved network NetCDF file")
parser.add_argument("--outdir", required=True, help="Directory for analysis outputs")
args = parser.parse_args()


# -----------------------------------------------------------------------------
# 2. Define important file paths
# -----------------------------------------------------------------------------
# BASE_DIR points to the project root.
# CONFIG_FILE is the scenario configuration passed from Snakemake.
# NETWORK_FILE is the solved PyPSA network for this scenario.
# OUTDIR is where all CSV, PNG, and HTML outputs will be written.
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(args.config)
NETWORK_FILE = Path(args.network)
OUTDIR = Path(args.outdir)


# Make sure the output folder exists before writing files.
OUTDIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 3. Load scenario configuration from YAML
# -----------------------------------------------------------------------------
# Read the scenario settings so this script knows which case was run.
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Check that the minimum expected top-level keys exist in the config file.
required = [
    "scenario_name",
    "system",
    "costs",
    "technology",
    "hydrogen",
]
missing = [k for k in required if k not in cfg]
if missing:
    raise KeyError(f"Missing top-level keys: {missing}")

system_cfg = cfg["system"]

investment_year = int(system_cfg["investment_year"])
weather_year = int(system_cfg["weather_year"])
demand_year = int(system_cfg["demand_year"])

# -----------------------------------------------------------------------------
# 4. Print scenario settings for debugging and traceability
# -----------------------------------------------------------------------------
# These prints help confirm which scenario was used and which settings were active.
print("=== SCENARIO FILE ===")
print(CONFIG_FILE.name)

print("=== HYDROGEN SETTINGS ===")
print(yaml.dump(cfg.get("hydrogen", {}), sort_keys=False))

print("=== MINING SETTINGS ===")
print(yaml.dump(cfg.get("mining", {}), sort_keys=False))


# -----------------------------------------------------------------------------
# 5. Load the solved PyPSA network
# -----------------------------------------------------------------------------
# Create an empty PyPSA network object, then import the solved network from NetCDF.
n = pypsa.Network()
n.import_from_netcdf(NETWORK_FILE)


# -----------------------------------------------------------------------------
# 6. Read important scenario switches from the config
# -----------------------------------------------------------------------------
# These flags help the script interpret whether hydrogen and mining are active.
hydrogen_cfg = cfg.get("hydrogen", {})
hydrogen_enabled = bool(hydrogen_cfg.get("enabled", True))
hydrogen_mode = hydrogen_cfg.get("mode", "fixed_demand")

mining_cfg = cfg.get("mining", {})
mining_enabled = bool(mining_cfg.get("enabled", False))

# This is the component name used in build_network.py for the mining sink.
mining_asset_name = "bitcoin_mining_sink"


# -----------------------------------------------------------------------------
# 7. Calculate whole-system cost metrics
# -----------------------------------------------------------------------------
# CAPEX = investment costs, OPEX = operational costs, system_cost = total.
total_capex = float(n.statistics.capex().sum())
total_opex = float(n.statistics.opex().sum())
system_cost = total_capex + total_opex


# -----------------------------------------------------------------------------
# 8. Aggregate key energy flows over the modeled period
# -----------------------------------------------------------------------------

# Objective snapshot weights represent the number of modeled hours represented
# by each snapshot. For the current 8760-hour model these are normally all 1.
if "objective" in n.snapshot_weightings.columns:
    snapshot_weights = n.snapshot_weightings["objective"].reindex(n.snapshots)
else:
    snapshot_weights = pd.Series(1.0, index=n.snapshots)


def weighted_sum(series):
    """Return weighted energy total in MWh from an MW time series."""
    series = series.reindex(n.snapshots).fillna(0.0)
    return float((series * snapshot_weights).sum())


# Generator energy totals
generator_dispatch = pd.Series(
    {
        name: weighted_sum(n.generators_t.p[name])
        for name in n.generators_t.p.columns
    },
    dtype=float,
)

# Load energy totals
load_totals = pd.Series(
    {
        name: weighted_sum(n.loads_t.p[name])
        for name in n.loads_t.p.columns
    },
    dtype=float,
)

# Link input/output energy totals
if not n.links.empty:
    link_p0 = pd.Series(
        {
            name: weighted_sum(n.links_t.p0[name])
            for name in n.links_t.p0.columns
        },
        dtype=float,
    )

    link_p1 = pd.Series(
        {
            name: weighted_sum(n.links_t.p1[name])
            for name in n.links_t.p1.columns
        },
        dtype=float,
    )
else:
    link_p0 = pd.Series(dtype=float)
    link_p1 = pd.Series(dtype=float)


solar_generation = float(generator_dispatch.get("solar", 0.0))
wind_generation = float(generator_dispatch.get("wind", 0.0))
load_shedding_generation = float(
    generator_dispatch.get("load_shedding", 0.0)
)


# -----------------------------------------------------------------------------
# 9. Calculate Bitcoin mining metrics
# -----------------------------------------------------------------------------
mining_consumption_mwh = 0.0
mining_utilization_rate = None
mining_gross_revenue_eur = 0.0
mining_variable_opex_eur = 0.0
mining_net_operating_value_eur = 0.0
mining_gross_revenue_eur_per_mwh = 0.0

mining_max_mw = float(
    mining_cfg.get("max_capacity_mw", 0.0)
)

if mining_enabled and mining_asset_name in n.generators.index:

    mining_timeseries = (
        n.generators_t.p[mining_asset_name]
        .abs()
        .reindex(n.snapshots)
        .fillna(0.0)
    )

    mining_consumption_mwh = weighted_sum(mining_timeseries)

    modeled_hours = float(snapshot_weights.sum())

    if mining_max_mw > 0 and modeled_hours > 0:
        mining_utilization_rate = (
            mining_consumption_mwh
            / (mining_max_mw * modeled_hours)
        )

    hashprice_eur_per_th_day = float(
        mining_cfg.get(
            "hashprice_eur_per_th_day",
            0.08,
        )
    )

    asic_efficiency_j_per_th = float(
        mining_cfg.get(
            "asic_efficiency_j_per_th",
            16.0,
        )
    )

    other_opex_eur_per_mwh = float(
        mining_cfg.get(
            "other_opex_eur_per_mwh",
            0.0,
        )
    )

    if asic_efficiency_j_per_th <= 0:
        raise ValueError(
            "ASIC efficiency must be greater than zero."
        )

    mw_per_th_per_s = asic_efficiency_j_per_th / 1e6

    th_day_per_mwh = (
        (1.0 / 24.0)
        / mw_per_th_per_s
    )

    mining_gross_revenue_eur_per_mwh = (
        hashprice_eur_per_th_day
        * th_day_per_mwh
    )

    mining_gross_revenue_eur = (
        mining_consumption_mwh
        * mining_gross_revenue_eur_per_mwh
    )

    mining_variable_opex_eur = (
        mining_consumption_mwh
        * other_opex_eur_per_mwh
    )

    mining_net_operating_value_eur = (
        mining_gross_revenue_eur
        - mining_variable_opex_eur
    )

    print("\n=== BITCOIN MINING RESULTS ===")
    print(
        f"Electricity consumption : "
        f"{mining_consumption_mwh:,.1f} MWh"
    )
    print(
        f"Utilization rate        : "
        f"{mining_utilization_rate * 100:.2f}%"
        if mining_utilization_rate is not None
        else "Utilization rate        : N/A"
    )
    print(
        f"Gross revenue per MWh   : "
        f"{mining_gross_revenue_eur_per_mwh:,.2f} EUR/MWh"
    )
    print(
        f"Gross annual revenue    : "
        f"{mining_gross_revenue_eur:,.2f} EUR"
    )
    print(
        f"Variable mining OPEX    : "
        f"{mining_variable_opex_eur:,.2f} EUR"
    )
    print(
        f"Net operating value     : "
        f"{mining_net_operating_value_eur:,.2f} EUR"
    )


# -----------------------------------------------------------------------------
# 10. Calculate electricity and hydrogen energy metrics
# -----------------------------------------------------------------------------
electricity_demand_total = float(
    load_totals.get(
        "electricity_demand",
        0.0,
    )
)

hydrogen_demand_total = float(
    load_totals.get(
        "hydrogen_demand",
        0.0,
    )
)

electrolyzer_input = float(
    link_p0.get(
        "electrolyzer",
        0.0,
    )
)

# PyPSA Link p1 is negative when energy leaves the link and enters bus1.
hydrogen_output = float(
    -link_p1.get(
        "electrolyzer",
        0.0,
    )
)

# Hydrogen delivered to the flexible / production-target sink.
hydrogen_delivered_mwh = 0.0

if "hydrogen_sink" in n.generators.index:
    hydrogen_sink_ts = (
        n.generators_t.p["hydrogen_sink"]
        .abs()
    )

    hydrogen_delivered_mwh = weighted_sum(
        hydrogen_sink_ts
    )

elif "hydrogen_demand" in n.loads.index:
    hydrogen_delivered_mwh = hydrogen_demand_total


# LHV convention used for hydrogen-energy conversion.
H2_LHV_KWH_PER_KG = 33.3

hydrogen_delivered_kg = (
    hydrogen_delivered_mwh
    * 1000.0
    / H2_LHV_KWH_PER_KG
)

specific_electricity_kwh_per_kg_h2 = (
    electrolyzer_input
    * 1000.0
    / hydrogen_delivered_kg
    if hydrogen_delivered_kg > 0
    else None
)


target_annual_kt_h2 = (
    float(
        hydrogen_cfg.get(
            "target_annual_kt_h2",
            0.0,
        )
    )
    if hydrogen_mode == "production_target"
    else 0.0
)

target_annual_kg_h2 = (
    target_annual_kt_h2
    * 1_000_000.0
)

hydrogen_target_achievement = (
    hydrogen_delivered_kg
    / target_annual_kg_h2
    if target_annual_kg_h2 > 0
    else None
)

# -----------------------------------------------------------------------------
# 11. Export optimized capacities by component type
# -----------------------------------------------------------------------------
# These tables store final installed capacities after optimization.
generators = n.generators[["carrier", "p_nom_opt"]].copy()
links = n.links[["carrier", "p_nom_opt"]].copy()
stores = n.stores[["carrier", "e_nom_opt"]].copy()


# -----------------------------------------------------------------------------
# 12. Calculate simple headline indicators
# -----------------------------------------------------------------------------
# Total renewable generation is used for rough system indicators.
total_renewable_generation = solar_generation + wind_generation


# -----------------------------------------------------------------------------
# 13. Calculate renewable curtailment directly
# -----------------------------------------------------------------------------
curtailment_rows = []

for asset in ["solar", "wind"]:

    if asset not in n.generators.index:
        continue

    p_nom_opt = float(
        n.generators.loc[
            asset,
            "p_nom_opt",
        ]
    )

    if asset in n.generators_t.p_max_pu.columns:
        availability_ts = (
            n.generators_t.p_max_pu[asset]
            * p_nom_opt
        )
    else:
        availability_ts = pd.Series(
            p_nom_opt,
            index=n.snapshots,
        )

    actual_ts = (
        n.generators_t.p[asset]
        .reindex(n.snapshots)
        .fillna(0.0)
    )

    available_mwh = weighted_sum(
        availability_ts
    )

    actual_mwh = weighted_sum(
        actual_ts
    )

    curtailed_mwh = max(
        available_mwh - actual_mwh,
        0.0,
    )

    curtailment_rate = (
        curtailed_mwh / available_mwh
        if available_mwh > 0
        else 0.0
    )

    curtailment_rows.append(
        {
            "asset": asset,
            "available_mwh": available_mwh,
            "actual_mwh": actual_mwh,
            "curtailed_mwh": curtailed_mwh,
            "curtailment_rate": curtailment_rate,
        }
    )

curtailment = pd.DataFrame(
    curtailment_rows
)

# -----------------------------------------------------------------------------
# 14. Extract hydrogen storage state of charge
# -----------------------------------------------------------------------------
# If hydrogen storage exists, read its stored energy over time.
# Otherwise create a zero series to keep the script robust.
if hasattr(n.stores_t, "e") and "hydrogen_storage" in n.stores_t.e.columns:
    storage_soc = n.stores_t.e["hydrogen_storage"].copy()
else:
    storage_soc = pd.Series(0.0, index=n.snapshots, name="hydrogen_storage_soc_mwh")

storage_timeseries = pd.DataFrame({"hydrogen_storage_soc_mwh": storage_soc}, index=n.snapshots)


# -----------------------------------------------------------------------------
# 15. Build hourly dispatch time series table
# -----------------------------------------------------------------------------
# This table is useful for plotting and for checking hourly system behavior.
dispatch_timeseries = pd.DataFrame(index=n.snapshots)
dispatch_timeseries["solar_mw"] = n.generators_t.p.get("solar", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["wind_mw"] = n.generators_t.p.get("wind", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["load_shedding_mw"] = n.generators_t.p.get("load_shedding", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["electricity_demand_mw"] = n.loads_t.p.get("electricity_demand", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["hydrogen_demand_mw"] = n.loads_t.p.get("hydrogen_demand", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["electrolyzer_input_mw"] = n.links_t.p0.get("electrolyzer", pd.Series(0.0, index=n.snapshots)) if not n.links.empty else 0.0
dispatch_timeseries["hydrogen_output_mw"] = -n.links_t.p1.get("electrolyzer", pd.Series(0.0, index=n.snapshots)) if not n.links.empty else 0.0
dispatch_timeseries["mining_mw"] = (
    n.generators_t.p.get(mining_asset_name, pd.Series(0.0, index=n.snapshots)).abs()
    if mining_enabled else pd.Series(0.0, index=n.snapshots)
)


# -----------------------------------------------------------------------------
# 16. Calculate capacity factors
# -----------------------------------------------------------------------------
# Capacity factor = actual yearly output / maximum possible yearly output.
hours = float(snapshot_weights.sum())

solar_p_nom = float(generators.loc["solar", "p_nom_opt"]) if "solar" in generators.index else 0.0
wind_p_nom = float(generators.loc["wind", "p_nom_opt"]) if "wind" in generators.index else 0.0
electrolyzer_p_nom = float(links.loc["electrolyzer", "p_nom_opt"]) if "electrolyzer" in links.index else 0.0

capacity_factors = pd.DataFrame(
    {
        "asset": ["solar", "wind", "electrolyzer"],
        "capacity_opt": [solar_p_nom, wind_p_nom, electrolyzer_p_nom],
        "total_output_or_input_mwh": [solar_generation, wind_generation, electrolyzer_input],
        "capacity_factor": [
            solar_generation / (solar_p_nom * hours) if solar_p_nom > 0 else None,
            wind_generation / (wind_p_nom * hours) if wind_p_nom > 0 else None,
            electrolyzer_input / (electrolyzer_p_nom * hours) if electrolyzer_p_nom > 0 else None,
        ],
    }
)


# -----------------------------------------------------------------------------
# 17. Additional off-grid KPIs
# -----------------------------------------------------------------------------
# These KPIs help interpret reliability and storage behavior in an off-grid system.

# Share of annual electricity demand that had to be covered by load shedding.
load_shedding_share = (
    load_shedding_generation / electricity_demand_total
    if electricity_demand_total > 0 else None
)

# Fraction of available renewable energy that was actually used.
# potential renewable = actual renewable + curtailed renewable
actual_renewable_generation = solar_generation + wind_generation
curtailed_renewable_generation = float(curtailment["curtailed_mwh"].sum())
potential_renewable_generation = (
    actual_renewable_generation + curtailed_renewable_generation
)

renewable_share_used = (
    actual_renewable_generation / potential_renewable_generation
    if potential_renewable_generation > 0 else None
)

# Optimized hydrogen storage energy capacity, if storage exists.
storage_e_nom_opt = (
    float(stores.loc["hydrogen_storage", "e_nom_opt"])
    if "hydrogen_storage" in stores.index else None
)


# -----------------------------------------------------------------------------
# 18. Build one-row summary table
# -----------------------------------------------------------------------------
summary = pd.DataFrame(
    {
        "scenario_file": [
            CONFIG_FILE.name
        ],
        "scenario_name": [
            cfg["scenario_name"]
        ],

        # Scenario definition
        "investment_year": [
            investment_year
        ],
        "weather_year": [
            weather_year
        ],
        "demand_year": [
            demand_year
        ],
        "hydrogen_enabled": [
            hydrogen_enabled
        ],
        "hydrogen_mode": [
            hydrogen_mode
        ],
        "mining_enabled": [
            mining_enabled
        ],

        # Optimization
        "objective_eur": [
            n.objective
        ],
        "annualized_capital_cost_eur": [
            total_capex
        ],
        "net_operating_cost_eur": [
            total_opex
        ],
        "system_cost_eur": [
            system_cost
        ],

        # Electricity
        "electricity_demand_mwh": [
            electricity_demand_total
        ],
        "solar_generation_mwh": [
            solar_generation
        ],
        "wind_generation_mwh": [
            wind_generation
        ],
        "load_shedding_mwh": [
            load_shedding_generation
        ],
        "load_shedding_share": [
            load_shedding_share
        ],
        "renewable_share_used": [
            renewable_share_used
        ],

        # Hydrogen
        "electrolyzer_input_mwh": [
            electrolyzer_input
        ],
        "hydrogen_output_mwh": [
            hydrogen_output
        ],
        "hydrogen_delivered_mwh": [
            hydrogen_delivered_mwh
        ],
        "hydrogen_delivered_kg": [
            hydrogen_delivered_kg
        ],
        "specific_electricity_kwh_per_kg_h2": [
            specific_electricity_kwh_per_kg_h2
        ],
        "h2_target_annual_kt": [
            target_annual_kt_h2
        ],
        "h2_target_achievement": [
            hydrogen_target_achievement
        ],
        "hydrogen_storage_capacity_mwh": [
            storage_e_nom_opt
        ],

        # Bitcoin
        "mining_max_capacity_mw": [
            mining_max_mw
            if mining_enabled
            else 0.0
        ],
        "mining_consumption_mwh": [
            mining_consumption_mwh
        ],
        "mining_utilization_rate": [
            mining_utilization_rate
        ],
        "mining_gross_revenue_eur_per_mwh": [
            mining_gross_revenue_eur_per_mwh
        ],
        "mining_gross_revenue_eur": [
            mining_gross_revenue_eur
        ],
        "mining_variable_opex_eur": [
            mining_variable_opex_eur
        ],
        "mining_net_operating_value_eur": [
            mining_net_operating_value_eur
        ],
    }
)
# -----------------------------------------------------------------------------
# 19. Write core CSV outputs
# -----------------------------------------------------------------------------
summary.to_csv(OUTDIR / "summary.csv", index=False)
generators.to_csv(OUTDIR / "generators.csv")
links.to_csv(OUTDIR / "links.csv")
stores.to_csv(OUTDIR / "stores.csv")
generator_dispatch.to_frame("total_generation_mwh").to_csv(OUTDIR / "generator_dispatch.csv")
load_totals.to_frame("total_load_mwh").to_csv(OUTDIR / "load_totals.csv")
link_p0.to_frame("p0_total_mwh").to_csv(OUTDIR / "link_p0.csv")
link_p1.to_frame("p1_total_mwh").to_csv(OUTDIR / "link_p1.csv")
dispatch_timeseries.to_csv(OUTDIR / "dispatch_timeseries.csv")
capacity_factors.to_csv(OUTDIR / "capacity_factors.csv", index=False)
curtailment.to_csv(OUTDIR / "curtailment.csv", index=False)
storage_timeseries.to_csv(OUTDIR / "storage_timeseries.csv")


# -----------------------------------------------------------------------------
# 20. Prepare compact data tables for plotting
# -----------------------------------------------------------------------------
generation_data = pd.DataFrame(
    {
        "source": ["Solar", "Wind", "Load shedding"],
        "value": [solar_generation, wind_generation, load_shedding_generation],
    }
)

capacity_data = pd.DataFrame(
    {
        "asset": ["Solar", "Wind", "Electrolyzer", "Hydrogen storage", "Load shedding"],
        "value": [
            float(generators.loc["solar", "p_nom_opt"]) if "solar" in generators.index else 0.0,
            float(generators.loc["wind", "p_nom_opt"]) if "wind" in generators.index else 0.0,
            float(links.loc["electrolyzer", "p_nom_opt"]) if "electrolyzer" in links.index else 0.0,
            float(stores.loc["hydrogen_storage", "e_nom_opt"]) if "hydrogen_storage" in stores.index else 0.0,
            float(generators.loc["load_shedding", "p_nom_opt"]) if "load_shedding" in generators.index else 0.0,
        ],
    }
)

# Separate out load shedding so it does not distort the main capacity chart.
real_capacity = capacity_data[capacity_data["asset"] != "Load shedding"].copy()
load_shedding_value = capacity_data.loc[capacity_data["asset"] == "Load shedding", "value"].iloc[0]


# -----------------------------------------------------------------------------
# 21. Build cost tables by technology
# -----------------------------------------------------------------------------
capex_raw = n.statistics.capex(groupby="name")
opex_raw = n.statistics.opex(groupby="name")

capex_by_tech = capex_raw.droplevel(0) if capex_raw.index.nlevels > 1 else capex_raw
opex_by_tech = opex_raw.droplevel(0) if opex_raw.index.nlevels > 1 else opex_raw

all_assets = ["solar", "wind", "electrolyzer", "hydrogen_storage", mining_asset_name]
_lmap = {
    "solar": "Solar",
    "wind": "Wind",
    "electrolyzer": "Electrolyzer",
    "hydrogen_storage": "H₂ Storage",
    mining_asset_name: "BTC Mining",
}

_dl = [_lmap[a] for a in all_assets]
_cv = [float(capex_by_tech.get(a, 0)) / 1e6 for a in all_assets]
_ov = [float(opex_by_tech.get(a, 0)) / 1e6 for a in all_assets]
_tv = [c + o for c, o in zip(_cv, _ov)]


# -----------------------------------------------------------------------------
# 22. Helper functions for cost charts
# -----------------------------------------------------------------------------
def _fmt(v):
    if v == 0:
        return ""
    return f"−{abs(v):,.0f}" if v < 0 else f"{v:,.0f}"


def _cost_chart(x_labels, capex_v, opex_v, total_v, title_suffix, fname):
    fig = go.Figure()
    for name, vals, color in [
        ("CAPEX", capex_v, "#01696f"),
        ("OPEX",  opex_v,  "#7a39bb"),
        ("Total", total_v, "#f5a623"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=x_labels, y=vals,
            marker_color=color,
            text=[_fmt(v) for v in vals],
            textposition="outside",
            textfont=dict(size=12, color="#333"),
            cliponaxis=False,
        ))
    fig.update_layout(
        barmode="group",
        title=dict(
            text=(
                f"Annualised Cost — {title_suffix}<br>"
                f"<span style='font-size:13px;font-weight:normal;color:#666;'>"
                f"Scenario: {cfg['scenario_name']} · Million EUR · Negative OPEX = revenue"
                f"</span>"
            ),
            font=dict(size=16), x=0.5, xanchor="center",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5, font=dict(size=13),
        ),
        font=dict(size=13, family="Arial"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            title="Million EUR", gridcolor="#ebebeb",
            tickformat=",.0f",
            zeroline=True, zerolinecolor="#999", zerolinewidth=1.5,
        ),
        xaxis=dict(tickfont=dict(size=14)),
        margin=dict(t=130, b=120, l=90, r=50),
        width=750, height=520,
    )
    fig.write_image(str(OUTDIR / fname))


# -----------------------------------------------------------------------------
# 23. Create cost charts
# -----------------------------------------------------------------------------
_big = ["Solar", "Wind"]
_bi = [_dl.index(l) for l in _big]
_cost_chart(_big, [_cv[i] for i in _bi], [_ov[i] for i in _bi], [_tv[i] for i in _bi], "Solar & Wind", "cost_solar_wind.png")

_sm = ["Electrolyzer", "H₂ Storage", "BTC Mining"]
_si = [_dl.index(l) for l in _sm]
_cost_chart(_sm, [_cv[i] for i in _si], [_ov[i] for i in _si], [_tv[i] for i in _si], "Electrolyzer, H₂ Storage & BTC Mining", "cost_small_assets.png")


# -----------------------------------------------------------------------------
# 24. Create static PNG charts with Matplotlib
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
bars = ax.bar(generation_data["source"], generation_data["value"], color=["#01696f", "#006494", "#a13544"])
ax.set_title("Generation Mix", fontsize=14, weight="bold")
ax.set_ylabel("MWh")
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUTDIR / "generation_mix.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
bars = ax.bar(
    real_capacity["asset"], real_capacity["value"],
    color=["#01696f", "#006494", "#7a39bb", "#9a9a9a"],
    width=0.68,
)
ax.set_title("Optimized Capacities", fontsize=14, weight="bold")
ax.set_ylabel("MW / MWh")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18)
ax.set_axisbelow(True)
ax.tick_params(axis="x", rotation=18)
for b, v in zip(bars, real_capacity["value"]):
    ax.annotate(
        f"{v:.1f}",
        (b.get_x() + b.get_width() / 2, v),
        ha="center", va="bottom",
        xytext=(0, 3), textcoords="offset points", fontsize=9,
    )
ax.text(
    0.99, 0.95,
    f"Load shedding: {load_shedding_value:,.0f} MW",
    transform=ax.transAxes, ha="right", va="top",
    fontsize=10, color="#444444",
)
fig.tight_layout()
fig.savefig(OUTDIR / "optimized_capacities.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
ax.plot(
    storage_timeseries.index,
    storage_timeseries["hydrogen_storage_soc_mwh"],
    color="#7a39bb",
    linewidth=2,
)
ax.set_title("Hydrogen Storage State of Charge", fontsize=14, weight="bold")
ax.set_ylabel("MWh")
ax.set_xlabel("Time")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTDIR / "hydrogen_storage_soc.png", bbox_inches="tight")
plt.close(fig)


# -----------------------------------------------------------------------------
# 25. Create interactive Plotly HTML for hydrogen storage
# -----------------------------------------------------------------------------
fig_html = go.Figure()
fig_html.add_trace(
    go.Scatter(
        x=storage_timeseries.index,
        y=storage_timeseries["hydrogen_storage_soc_mwh"],
        mode="lines",
        name="Hydrogen SOC",
        line=dict(color="#7a39bb", width=3),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} MWh<extra></extra>",
    )
)
fig_html.update_layout(
    title="Hydrogen Storage State of Charge",
    xaxis_title="Time",
    yaxis_title="MWh",
    template="plotly_white",
    hovermode="x unified",
)
fig_html.write_html(
    OUTDIR / "hydrogen_storage_soc.html",
    full_html=True,
    include_plotlyjs="cdn",
)


# -----------------------------------------------------------------------------
# 26. Prepare dashboard input data
# -----------------------------------------------------------------------------
cost_data = pd.DataFrame(
    {
        "component": [
            "Annualised capital cost",
            "Net operating cost",
            "System cost",
        ],
        "value": [
            total_capex,
            total_opex,
            system_cost,
        ],
    }
)

# -----------------------------------------------------------------------------
# 27. Create one HTML dashboard with multiple subplots
# -----------------------------------------------------------------------------
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "Cost Breakdown", "Generation Mix",
        "Optimized Capacities", "Hydrogen Storage SOC",
        "Capacity Factors", "Curtailment",
    ),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "scatter"}],
        [{"type": "bar"}, {"type": "bar"}],
    ],
)

fig.add_trace(go.Bar(x=cost_data["component"], y=cost_data["value"], marker_color=["#01696f", "#7a39bb", "#444444"], name="Costs"), row=1, col=1)
fig.add_trace(go.Bar(x=generation_data["source"], y=generation_data["value"], marker_color=["#01696f", "#006494", "#a13544"], name="Generation"), row=1, col=2)
fig.add_trace(go.Bar(x=real_capacity["asset"], y=real_capacity["value"], marker_color=["#01696f", "#006494", "#7a39bb", "#9a9a9a"], name="Capacities"), row=2, col=1)
fig.add_trace(go.Scatter(x=storage_timeseries.index, y=storage_timeseries["hydrogen_storage_soc_mwh"], mode="lines", line=dict(color="#7a39bb", width=2), name="Hydrogen SOC"), row=2, col=2)
fig.add_trace(go.Bar(x=capacity_factors["asset"], y=capacity_factors["capacity_factor"], marker_color=["#01696f", "#006494", "#7a39bb"], name="Capacity factors"), row=3, col=1)
fig.add_trace(go.Bar(x=curtailment["asset"], y=curtailment["curtailment_rate"], marker_color=["#01696f", "#006494"], name="Curtailment"), row=3, col=2)

fig.update_yaxes(title_text="EUR", row=1, col=1)
fig.update_yaxes(title_text="MWh", row=1, col=2)
fig.update_yaxes(title_text="MW / MWh", row=2, col=1)
fig.update_yaxes(title_text="MWh", row=2, col=2)
fig.update_yaxes(title_text="Share", row=3, col=1)
fig.update_yaxes(title_text="Share", row=3, col=2)

fig.update_layout(
    title_text=f"Scenario Results: {cfg['scenario_name']}",
    height=1200, width=1300,
    showlegend=False,
    template="plotly_white",
)

fig.write_html(OUTDIR / "results_dashboard.html", include_plotlyjs="cdn")


# -----------------------------------------------------------------------------
# 28. Export summary table as HTML
# -----------------------------------------------------------------------------
summary.style.format(precision=2).to_html(OUTDIR / "summary_table.html", encoding="utf-8")


# -----------------------------------------------------------------------------
# 29. Print outputs to terminal/log for quick inspection
# -----------------------------------------------------------------------------
print("=== SUMMARY ===")
print(summary.T)
print("\n=== GENERATORS ===")
print(generators)
print("\n=== LINKS ===")
print(links)
print("\n=== STORES ===")
print(stores)
print("\n=== CAPACITY FACTORS ===")
print(capacity_factors)
print("\n=== CURTAILMENT ===")
print(curtailment)
print("\n=== DISPATCH TIMESERIES HEAD ===")
print(dispatch_timeseries.head())
print("\n=== STORAGE SOC HEAD ===")
print(storage_timeseries.head())
print(f"\nCSV/HTML/PNG files written to {OUTDIR}")