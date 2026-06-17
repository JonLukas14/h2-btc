from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from build_network import build_test_network

config_dir = Path(__file__).resolve().parent.parent / "configs"
scenario_file = config_dir / "scenario.yaml"

with open(scenario_file, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

required = ["scenario_name", "system", "demand", "costs", "data_files", "technology", "hydrogen"]
missing = [k for k in required if k not in cfg]
if missing:
    raise KeyError(f"Missing top-level keys: {missing}")

outdir = Path(__file__).resolve().parent.parent / "results"
outdir.mkdir(exist_ok=True)

print("=== SCENARIO FILE ===")
print(scenario_file.name)

print("=== MINING SETTINGS ===")
print(yaml.dump(cfg.get("mining", {}), sort_keys=False))

print(cfg.keys())
print(cfg.get("system", {}).keys())

n = build_test_network(cfg)

status, condition = n.optimize(
    solver_name="highs",
    include_objective_constant=False
)

total_capex = float(n.statistics.capex().sum())
total_opex = float(n.statistics.opex().sum())
system_cost = total_capex + total_opex

generator_dispatch = n.generators_t.p.sum()
load_totals = n.loads_t.p.sum()
link_p0 = n.links_t.p0.sum()
link_p1 = n.links_t.p1.sum()

solar_generation = float(generator_dispatch.get("solar", 0.0))
wind_generation = float(generator_dispatch.get("wind", 0.0))
load_shedding_generation = float(generator_dispatch.get("load_shedding", 0.0))

# ── Mining results ──────────────────────────────────────────────────────────
mining_enabled = cfg.get("mining", {}).get("enabled", False)
mining_consumption_mwh = 0.0
mining_utilization_rate = None
mining_max_mw = float(cfg.get("mining", {}).get("max_capacity_mw", 0))

if mining_enabled and "bitcoin_mining" in n.generators.index:
    mining_timeseries = n.generators_t.p.get(
        "bitcoin_mining", pd.Series(0.0, index=n.snapshots)
    ).abs()

    mining_consumption_mwh = float(mining_timeseries.sum())
    hours = len(n.snapshots)

    if mining_max_mw > 0:
        mining_utilization_rate = mining_consumption_mwh / (mining_max_mw * hours)

    print(f"\n=== MINING RESULTS ===")
    print(f"  Total electricity consumed by mining : {mining_consumption_mwh:,.1f} MWh")
    print(f"  Mining utilization rate              : "
          f"{mining_utilization_rate*100:.1f}%" if mining_utilization_rate else "  N/A")
    print(f"  Hours at full capacity               : "
          f"{(mining_timeseries >= mining_max_mw * 0.99).sum()}")
    print(f"  Hours at zero                        : "
          f"{(mining_timeseries <= 0.01).sum()}")

electricity_demand_total = float(load_totals.get("electricity_demand", 0.0))
hydrogen_demand_total = float(load_totals.get("hydrogen_demand", 0.0))

electrolyzer_input = float(link_p0.get("electrolyzer", 0.0))
hydrogen_output = float(-link_p1.get("electrolyzer", 0.0))

total_renewable_generation = solar_generation + wind_generation
simple_lcoe = system_cost / total_renewable_generation if total_renewable_generation > 0 else None
simple_cost_per_h2 = system_cost / hydrogen_output if hydrogen_output > 0 else None

summary = pd.DataFrame(
    {
        "scenario_file": [scenario_file.name],
        "scenario_name": [cfg["scenario_name"]],
        "status": [status],
        "termination": [condition],
        "objective": [n.objective],
        "total_capex": [total_capex],
        "total_opex": [total_opex],
        "system_cost": [system_cost],
        "solar_generation_mwh": [solar_generation],
        "wind_generation_mwh": [wind_generation],
        "load_shedding_mwh": [load_shedding_generation],
        "electricity_demand_mwh": [electricity_demand_total],
        "hydrogen_demand_mwh": [hydrogen_demand_total],
        "electrolyzer_input_mwh": [electrolyzer_input],
        "hydrogen_output_mwh": [hydrogen_output],
        "simple_lcoe_per_mwh": [simple_lcoe],
        "simple_cost_per_hydrogen_mwh": [simple_cost_per_h2],
        "mining_consumption_mwh":       [mining_consumption_mwh],
        "mining_utilization_rate":      [mining_utilization_rate],
        "mining_max_capacity_mw":       [mining_max_mw if mining_enabled else 0],
    }
)

generators = n.generators[["carrier", "p_nom_opt"]].copy()
links = n.links[["carrier", "p_nom_opt"]].copy()
stores = n.stores[["carrier", "e_nom_opt"]].copy()

dispatch_timeseries = pd.DataFrame(index=n.snapshots)
dispatch_timeseries["solar_mw"] = n.generators_t.p.get("solar", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["wind_mw"] = n.generators_t.p.get("wind", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["load_shedding_mw"] = n.generators_t.p.get("load_shedding", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["electricity_demand_mw"] = n.loads_t.p.get("electricity_demand", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["hydrogen_demand_mw"] = n.loads_t.p.get("hydrogen_demand", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["electrolyzer_input_mw"] = n.links_t.p0.get("electrolyzer", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["hydrogen_output_mw"] = -n.links_t.p1.get("electrolyzer", pd.Series(0.0, index=n.snapshots))
dispatch_timeseries["mining_mw"] = (
    n.generators_t.p.get("bitcoin_mining", pd.Series(0.0, index=n.snapshots)).abs()
    if mining_enabled else pd.Series(0.0, index=n.snapshots)
)

hours = len(n.snapshots)

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

# ---- Curtailment ----
try:
    curtailment_stats = n.statistics.curtailment(
        components=["Generator"],
        aggregate_across_components=False,
        groupby="name",
    )
    curtailment_stats = curtailment_stats.reset_index()
    if "Generator" in curtailment_stats.columns:
        curtailment_stats = curtailment_stats.drop(columns=["Generator"])
    curtailment_stats.columns = ["asset", "curtailed_mwh"]
except Exception:
    curtailment_stats = pd.DataFrame({"asset": ["solar", "wind"], "curtailed_mwh": [0.0, 0.0]})

available_energy = pd.DataFrame(
    {
        "asset": ["solar", "wind"],
        "available_mwh": [
            float((n.generators_t.p_max_pu["solar"] * solar_p_nom).sum()) if "solar" in n.generators_t.p_max_pu.columns else 0.0,
            float((n.generators_t.p_max_pu["wind"] * wind_p_nom).sum()) if "wind" in n.generators_t.p_max_pu.columns else 0.0,
        ],
        "actual_mwh": [solar_generation, wind_generation],
    }
)

curtailment = available_energy.merge(curtailment_stats, on="asset", how="left")
curtailment["curtailed_mwh"] = curtailment["curtailed_mwh"].fillna(
    curtailment["available_mwh"] - curtailment["actual_mwh"]
)
curtailment["curtailment_rate"] = curtailment["curtailed_mwh"] / curtailment["available_mwh"]
curtailment["curtailment_rate"] = curtailment["curtailment_rate"].fillna(0.0)

# ---- Storage state of charge ----
if hasattr(n.stores_t, "e") and "hydrogen_storage" in n.stores_t.e.columns:
    storage_soc = n.stores_t.e["hydrogen_storage"].copy()
else:
    storage_soc = pd.Series(0.0, index=n.snapshots, name="hydrogen_storage_soc_mwh")

storage_timeseries = pd.DataFrame({"hydrogen_storage_soc_mwh": storage_soc}, index=n.snapshots)

# ---- Save CSVs ----
summary.to_csv(outdir / "summary.csv", index=False)
generators.to_csv(outdir / "generators.csv")
links.to_csv(outdir / "links.csv")
stores.to_csv(outdir / "stores.csv")
generator_dispatch.to_frame("total_generation_mwh").to_csv(outdir / "generator_dispatch.csv")
load_totals.to_frame("total_load_mwh").to_csv(outdir / "load_totals.csv")
link_p0.to_frame("p0_total_mwh").to_csv(outdir / "link_p0.csv")
link_p1.to_frame("p1_total_mwh").to_csv(outdir / "link_p1.csv")
dispatch_timeseries.to_csv(outdir / "dispatch_timeseries.csv")
capacity_factors.to_csv(outdir / "capacity_factors.csv", index=False)
curtailment.to_csv(outdir / "curtailment.csv", index=False)
storage_timeseries.to_csv(outdir / "storage_timeseries.csv")

# ---- Helper tables for plotting ----
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
            float(generators.loc["solar", "p_nom_opt"]),
            float(generators.loc["wind", "p_nom_opt"]),
            float(links.loc["electrolyzer", "p_nom_opt"]),
            float(stores.loc["hydrogen_storage", "e_nom_opt"]),
            float(generators.loc["load_shedding", "p_nom_opt"]),
        ],
    }
)

real_capacity = capacity_data[capacity_data["asset"] != "Load shedding"].copy()
load_shedding_value = capacity_data.loc[capacity_data["asset"] == "Load shedding", "value"].iloc[0]

# ---- CAPEX by technology chart ----
capex_raw = n.statistics.capex(groupby="name")
opex_raw  = n.statistics.opex(groupby="name")

capex_by_tech = capex_raw.droplevel(0) if capex_raw.index.nlevels > 1 else capex_raw
opex_by_tech  = opex_raw.droplevel(0)  if opex_raw.index.nlevels > 1 else opex_raw

keep = ["solar", "wind", "electrolyzer", "hydrogen_storage"]
labels_map = {
    "solar": "Solar",
    "wind": "Wind",
    "electrolyzer": "Electrolyzer",
    "hydrogen_storage": "H\u2082 Storage",
}

capex_vals = [float(capex_by_tech.get(a, 0)) / 1e6 for a in keep]
opex_vals  = [float(opex_by_tech.get(a, 0))  / 1e6 for a in keep]
total_vals = [c + o for c, o in zip(capex_vals, opex_vals)]
display_labels = [labels_map[a] for a in keep]
colors_capex = ["#f5a623", "#01696f", "#006494", "#7a39bb"]

fig_cost = go.Figure()

for label, val, col in zip(display_labels, capex_vals, colors_capex):
    fig_cost.add_trace(go.Bar(
        name=f"{label} CAPEX",
        x=[label],
        y=[val],
        marker_color=col,
        text=f"{val:,.0f}m" if val > 10 else "",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=12, color="white"),
    ))

for label, tot in zip(display_labels, total_vals):
    fig_cost.add_annotation(
        x=label, y=tot,
        text=f"<b>{tot:,.0f}m</b>",
        showarrow=False,
        yshift=10,
        font=dict(size=13, color="#28251d"),
    )

fig_cost.update_layout(
    barmode="stack",
    title=dict(
        text=(
            f"Annualised System CAPEX by Technology (Million EUR)<br>"
            f"<span style='font-size:15px;font-weight:normal;color:#666;'>"
            f"Scenario: {cfg['scenario_name']} · 7% discount rate"
            f"</span>"
        ),
        font=dict(size=18),
        x=0.5,
        xanchor="center",
    ),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5, font=dict(size=12)),
    font=dict(size=13, family="Arial"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis=dict(title="Million EUR", gridcolor="#ebebeb",
               tickformat=",.0f", tickfont=dict(size=12)),
    xaxis=dict(tickfont=dict(size=14)),
    margin=dict(t=130, b=120, l=90, r=40),
    width=850,
    height=540,
)

fig_cost.update_traces(cliponaxis=False)
fig_cost.write_image(str(outdir / "cost_by_technology.png"))

# ---- Generation mix chart ----
fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
bars = ax.bar(generation_data["source"], generation_data["value"], color=["#01696f", "#006494", "#a13544"])
ax.set_title("Generation Mix", fontsize=14, weight="bold")
ax.set_ylabel("MWh")
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(outdir / "generation_mix.png", bbox_inches="tight")
plt.close(fig)

# ---- Optimized capacities chart ----
fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
bars = ax.bar(
    real_capacity["asset"],
    real_capacity["value"],
    color=["#01696f", "#006494", "#7a39bb", "#9a9a9a"],
    width=0.68
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
fig.savefig(outdir / "optimized_capacities.png", bbox_inches="tight")
plt.close(fig)

# ---- Hydrogen storage SOC chart ----
fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
ax.plot(
    storage_timeseries.index,
    storage_timeseries["hydrogen_storage_soc_mwh"],
    color="#7a39bb", linewidth=2,
)
ax.set_title("Hydrogen Storage State of Charge", fontsize=14, weight="bold")
ax.set_ylabel("MWh")
ax.set_xlabel("Time")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(outdir / "hydrogen_storage_soc.png", bbox_inches="tight")
plt.close(fig)

# ---- Interactive dashboard ----
cost_data = pd.DataFrame(
    {
        "component": ["CAPEX", "OPEX", "TOTAL"],
        "value": [total_capex, total_opex, system_cost],
    }
)

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

fig.add_trace(go.Bar(x=cost_data["component"], y=cost_data["value"],
    marker_color=["#01696f", "#7a39bb", "#444444"], name="Costs"), row=1, col=1)
fig.add_trace(go.Bar(x=generation_data["source"], y=generation_data["value"],
    marker_color=["#01696f", "#006494", "#a13544"], name="Generation"), row=1, col=2)
fig.add_trace(go.Bar(x=real_capacity["asset"], y=real_capacity["value"],
    marker_color=["#01696f", "#006494", "#7a39bb", "#9a9a9a"], name="Capacities"), row=2, col=1)
fig.add_trace(go.Scatter(x=storage_timeseries.index, y=storage_timeseries["hydrogen_storage_soc_mwh"],
    mode="lines", line=dict(color="#7a39bb", width=2), name="Hydrogen SOC"), row=2, col=2)
fig.add_trace(go.Bar(x=capacity_factors["asset"], y=capacity_factors["capacity_factor"],
    marker_color=["#01696f", "#006494", "#7a39bb"], name="Capacity factors"), row=3, col=1)
fig.add_trace(go.Bar(x=curtailment["asset"], y=curtailment["curtailment_rate"],
    marker_color=["#01696f", "#006494"], name="Curtailment"), row=3, col=2)

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

fig.write_html(outdir / "results_dashboard.html", include_plotlyjs="cdn")
summary.style.format(precision=2).to_html(outdir / "summary_table.html", encoding="utf-8")

# ---- Print to terminal ----
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
print("\nCSV/HTML/PNG files written to ../results")

