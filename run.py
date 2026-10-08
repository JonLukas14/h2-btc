import argparse
from pathlib import Path

from offgrid.analysis import summarize, write_summary
from offgrid.model import (
    build_network,
    load_config,
    solve_network,
)


parser = argparse.ArgumentParser(
    description="Run the clean off-grid PyPSA model."
)

parser.add_argument(
    "config",
    help="Scenario YAML file",
)

parser.add_argument(
    "--outdir",
    default=None,
    help="Output directory",
)

args = parser.parse_args()

config_path = Path(args.config)
cfg = load_config(config_path)

if args.outdir is None:
    outdir = (
        Path("results")
        / "clean_rebuild"
        / cfg["scenario_name"]
    )
else:
    outdir = Path(args.outdir)

outdir.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 60)
print("CLEAN OFF-GRID MODEL")
print("=" * 60)
print(f"Scenario: {cfg['scenario_name']}")
print(f"Config:   {config_path}")
print(f"Output:   {outdir}")
print("=" * 60)

network = build_network(cfg)
network = solve_network(
    network,
    cfg,
)

network_path = (
    outdir
    / "network_solved.nc"
)

network.export_to_netcdf(
    network_path
)

summary = summarize(
    network,
    cfg,
)

summary_path = (
    outdir
    / "summary.csv"
)

write_summary(
    summary,
    summary_path,
)

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(
    f"Solar:        "
    f"{summary['solar_capacity_mw']:.6f} MW"
)

print(
    f"Wind:         "
    f"{summary['wind_capacity_mw']:.6f} MW"
)

print(
    f"Electrolyzer: "
    f"{summary['electrolyzer_capacity_mw']:.6f} MW"
)

print(
    f"H2 delivered: "
    f"{summary['hydrogen_delivered_kt']:.6f} kt/a"
)

print(
    f"Battery power: "
    f"{summary['battery_power_mw']:.6f} MW"
)

print(
    f"Battery energy:"
    f" {summary['battery_energy_mwh']:.6f} MWh"
)

print(
    f"System cost:  "
    f"{summary['objective_eur_per_year']:,.2f} EUR/a"
)

print(
    f"LCOH:         "
    f"{summary['lcoh_eur_per_kg_h2']:.6f} EUR/kg"
)

print(
    f"RES use:      "
    f"{100.0 * summary['renewable_utilization_rate']:.6f}%"
)

print("=" * 60)

print(f"Network: {network_path}")
print(f"Summary: {summary_path}")
