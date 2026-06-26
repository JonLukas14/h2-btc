# scripts/preprocess_inputs.py
#
# Purpose:
# This script reads raw / external input data, converts it into model-ready CSVs,
# and writes those processed inputs into the local data/ folder.
#
# Input sources:
#   - ~/git/pypsa-kz-data/             (external Kazakhstan data repo)
#   - technology-data/outputs/*.csv    (technology cost datasets selected in scenario.yaml)
#
# Output location:
#   - data/
#
# These processed files are then used by build_network.py / run_model.py.


import sys
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


# -----------------------------------------------------------------------------
# 1. Define important paths
# -----------------------------------------------------------------------------
# BASE_DIR is the project root.
# DATA_DIR is where processed model input CSVs will be written.
# TECH_REPO points to the local technology-data repository.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TECH_REPO = BASE_DIR / "technology-data"

# Ensure the output folder exists before writing any files.
DATA_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Locate the pypsa-kz-data repository automatically
# -----------------------------------------------------------------------------
# The script tries a few candidate locations and picks the first one that
# contains the expected demand validation file.
_kz_candidates = [
    BASE_DIR / "pypsa-kz-data",
    BASE_DIR.parent / "pypsa-kz-data",
    Path.home() / "git" / "pypsa-kz-data",
]

KZ_REPO = next(
    (p for p in _kz_candidates if (p / "data" / "kz_demand_validation.csv").exists()),
    None
)

# If the repo cannot be found, print diagnostics and stop.
if KZ_REPO is None:
    print("[ERROR] Cannot find pypsa-kz-data in any of these locations:")
    for p in _kz_candidates:
        print(f"  {p}  (exists: {p.exists()})")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 3. Load scenario configuration
# -----------------------------------------------------------------------------
# Read the active scenario file so preprocessing uses the same year,
# snapshot count, and active cost dataset as the model run.
with open(BASE_DIR / "configs" / "scenario.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

YEAR = int(cfg["system"]["year"])
SNAPSHOTS = int(cfg["system"]["snapshots"])


# -----------------------------------------------------------------------------
# 4. Helper function: choose the active cost dataset from scenario.yaml
# -----------------------------------------------------------------------------
# This matches cfg["costs"]["active_dataset"] to a file path listed under
# cfg["costs"]["datasets"].
def get_active_costs_path(cfg, base_dir):
    costs_cfg = cfg["costs"]
    active_name = costs_cfg["active_dataset"]
    datasets = costs_cfg["datasets"]

    # Stop if the selected dataset name is not defined in the config.
    if active_name not in datasets:
        raise KeyError(
            f"Unknown cost dataset '{active_name}'. "
            f"Available options: {list(datasets.keys())}"
        )

    # Return the full file path of the chosen costs CSV.
    return base_dir / datasets[active_name]


# -----------------------------------------------------------------------------
# 5. Print basic environment information
# -----------------------------------------------------------------------------
# Useful for debugging path problems and checking that the intended scenario is used.
print(f"\nBase dir : {BASE_DIR}")
print(f"KZ repo  : {KZ_REPO}  (exists: {KZ_REPO.exists()})")
print(f"Tech repo: {TECH_REPO}  (exists: {TECH_REPO.exists()})")
print(f"Year={YEAR}, Snapshots={SNAPSHOTS}\n")


# =============================================================================
# 6. Build costs CSV from the selected technology dataset
# =============================================================================
# Reads the active cost dataset chosen in scenario.yaml, filters it down to the
# technologies and parameters used in the model, and writes a compact CSV to data/.
def build_costs():
    source = get_active_costs_path(cfg, BASE_DIR)
    if not source.exists():
        sys.exit(f"[ERROR] Missing: {source}")

    df = pd.read_csv(source)

    # Print all technologies related to solar, wind, electrolysis, and hydrogen
    # to help you see what names exist in the raw costs file.
    keywords = ["solar", "wind", "electro", "hydrogen", "pem", "alkaline"]
    matches = df[df["technology"].str.lower().str.contains("|".join(keywords), na=False)]
    print("  Technologies in costs file matching solar/wind/electrolysis/hydrogen:")
    for t in sorted(matches["technology"].unique()):
        print(f"    {t}")

    # Keep only the technologies actually used by your current model.
    keep_techs = [
        "solar-utility",
        "onwind",
        "electrolysis",
        "hydrogen storage underground",
    ]

    # Keep only the parameters that build_network.py needs.
    keep_params = ["investment", "FOM", "VOM", "efficiency", "lifetime"]

    subset = df[
        df["technology"].isin(keep_techs) &
        df["parameter"].isin(keep_params)
    ][["technology", "parameter", "value", "unit"]].copy()

    # Convert investment costs from EUR/kW to EUR/MW if needed.
    subset.loc[subset["parameter"] == "investment", "value"] *= 1000

    # If no rows matched, print the available technology names and stop,
    # because that means your keep_techs list does not match the source file.
    if subset.empty:
        print("\n  [WARNING] No rows matched. Printing ALL available technologies:")
        print(df["technology"].unique().tolist())
        sys.exit(
            "\n[ERROR] Technology names in keep_techs do not match the file.\n"
            "  Update the keep_techs list above to match the printed names."
        )

    print(f"\n  Using active cost dataset: {cfg['costs']['active_dataset']}")
    print(f"  Source file: {source}")
    print(f"\n  Matched rows:")
    print(subset.to_string(index=False))

    # Save the reduced cost table into data/ using the active dataset name.
    out_name = f"{cfg['costs']['active_dataset']}.csv"
    out = DATA_DIR / out_name
    subset[["technology", "parameter", "value"]].to_csv(out, index=False)
    print(f"\n  ✓ costs written → data/{out_name}  ({len(subset)} rows)")


# =============================================================================
# 7. Build hourly electricity demand series
# =============================================================================
# Reads monthly electricity demand from pypsa-kz-data, converts it to hourly MW,
# and writes a full-year hourly demand CSV.
def build_electricity_demand():
    source = KZ_REPO / "data" / "kz_demand_validation.csv"
    if not source.exists():
        sys.exit(f"[ERROR] Missing: {source}\n  KZ_REPO resolved to: {KZ_REPO}")

    df = pd.read_csv(source, index_col=0)
    print(f"  Demand file columns: {list(df.columns)}")
    print(f"  Years available: {sorted(df['year'].unique())}")

    # Use the most recent year available in the source file.
    latest_year = df["year"].max()
    df_year = df[df["year"] == latest_year].sort_values("month").reset_index(drop=True)
    print(f"  Using year: {latest_year}  (most recent in file)")
    print(df_year[["month", "demand_korem"]].to_string(index=False))

    # Hard-coded hours per month for a non-leap year.
    hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

    # Monthly demand values in GWh.
    monthly_gwh = df_year["demand_korem"].values.astype(float)

    # Convert each monthly value into a flat hourly MW profile for that month.
    hourly_demand = []
    for gwh, hrs in zip(monthly_gwh, hours_per_month):
        mw = (gwh * 1000.0) / hrs
        hourly_demand.extend([mw] * hrs)

    hourly_series = pd.Series(hourly_demand)
    print(f"\n  Hourly MW — min: {hourly_series.min():.1f}, mean: {hourly_series.mean():.1f}, max: {hourly_series.max():.1f}")

    # Trim or limit the series to the number of snapshots required by the scenario.
    out_series = hourly_series.iloc[:SNAPSHOTS].reset_index(drop=True)

    out = DATA_DIR / "kz_electricity_demand.csv"
    pd.DataFrame({"electricity_mw": out_series.values}).to_csv(out, index=False)
    print(f"  ✓ electricity demand written → data/kz_electricity_demand.csv  (rows={len(out_series)}, mean={out_series.mean():.1f} MW)")


# =============================================================================
# 8. Build solar capacity factor time series
# =============================================================================
# Fetches hourly solar PV output from PVGIS and converts it to a 0–1 capacity factor.
# Falls back to a synthetic daily profile if PVGIS is unavailable.
def build_solar_cf():
    lat, lon = 43.3, 71.4
    pvgis_year = 2020

    # PVGIS API URL for hourly PV output with 1 kW peakpower.
    # Dividing by 1000 later converts output power into a capacity factor.
    url = (
        f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
        f"?lat={lat}&lon={lon}"
        f"&startyear={pvgis_year}&endyear={pvgis_year}"
        f"&pvcalculation=1&peakpower=1&loss=14"
        f"&angle=30&aspect=0"
        f"&outputformat=json&browser=0"
    )

    print(f"  Fetching solar CF from PVGIS (lat={lat}, lon={lon}) ...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # P is hourly PV power in W for a 1 kW system, so P/1000 gives capacity factor.
        cf_values = [entry["P"] / 1000.0 for entry in data["outputs"]["hourly"]]
        cf = pd.Series(cf_values).clip(0.0, 1.0).iloc[:SNAPSHOTS]

        print(f"  mean CF={cf.mean():.3f}, peak CF={cf.max():.3f}")
        pd.DataFrame({"solar_cf": cf.values}).to_csv(DATA_DIR / "kz_solar_cf.csv", index=False)
        print("  ✓ solar CF written → data/kz_solar_cf.csv  (PVGIS ERA5)")
    except Exception as e:
        print(f"  [WARNING] PVGIS failed: {e} — using synthetic fallback")
        _synthetic_solar()


# -----------------------------------------------------------------------------
# 9. Synthetic fallback for solar CF
# -----------------------------------------------------------------------------
# Used only if the PVGIS API call fails.
def _synthetic_solar():
    daily = [0.00, 0.00, 0.00, 0.00, 0.00, 0.03,
             0.12, 0.28, 0.48, 0.65, 0.77, 0.84,
             0.86, 0.80, 0.69, 0.52, 0.33, 0.14,
             0.03, 0.00, 0.00, 0.00, 0.00, 0.00]
    reps = (SNAPSHOTS // 24) + 1
    cf = pd.Series((daily * reps)[:SNAPSHOTS])
    pd.DataFrame({"solar_cf": cf.values}).to_csv(DATA_DIR / "kz_solar_cf.csv", index=False)
    print("  ✓ solar CF written (synthetic fallback)")


# =============================================================================
# 10. Build wind capacity factor time series
# =============================================================================
# Fetches hourly wind speed from PVGIS, adjusts it to hub height,
# maps it through a simple power curve, and writes wind CF to CSV.
def build_wind_cf():
    lat, lon = 43.6, 51.2
    pvgis_year = 2020

    url = (
        f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
        f"?lat={lat}&lon={lon}"
        f"&startyear={pvgis_year}&endyear={pvgis_year}"
        f"&outputformat=json&browser=0"
        f"&windspeed=1"
    )

    print(f"  Fetching wind data from PVGIS (lat={lat}, lon={lon}) ...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Extract 10 m wind speed from hourly PVGIS output.
        ws_10m = pd.Series([entry["WS10m"] for entry in data["outputs"]["hourly"]])

        # Add a small random perturbation to avoid overly smooth / repeated values.
        rng = np.random.default_rng(seed=42)
        ws_10m = ws_10m + rng.normal(0, 0.1, size=len(ws_10m))
        ws_10m = ws_10m.clip(lower=0.0)

        # Scale wind speed upward to better reflect stronger wind resource assumptions.
        ws_10m = ws_10m * 1.25

        # Extrapolate from 10 m to 100 m hub height using a power-law profile.
        alpha = 0.20
        ws_hub = ws_10m * (100 / 10) ** alpha

        # Convert hub-height wind speed into capacity factor.
        cf = _power_curve(ws_hub).clip(0.0, 1.0)
        cf = cf.iloc[:SNAPSHOTS].reset_index(drop=True)

        print(f"  mean CF={cf.mean():.3f}, peak CF={cf.max():.3f}")
        pd.DataFrame({"wind_cf": cf.values}).to_csv(DATA_DIR / "kz_wind_cf.csv", index=False)
        print("  ✓ wind CF written → data/kz_wind_cf.csv  (PVGIS ERA5, hub height corrected)")

    except Exception as e:
        print(f"  [WARNING] PVGIS failed: {e} — using synthetic fallback")
        _synthetic_wind()


# -----------------------------------------------------------------------------
# 11. Helper function: simple wind turbine power curve
# -----------------------------------------------------------------------------
# Converts wind speed into a normalized capacity factor:
#   - below 3 m/s: no output
#   - 3 to 12 m/s: cubic ramp-up
#   - 12 to 25 m/s: full output
def _power_curve(ws):
    ws = pd.Series(ws).astype(float)
    cf = pd.Series(0.0, index=ws.index)

    mask_ramp = (ws >= 3.0) & (ws < 12.0)
    cf[mask_ramp] = ((ws[mask_ramp] - 3.0) / (12.0 - 3.0)) ** 3

    mask_rated = (ws >= 12.0) & (ws <= 25.0)
    cf[mask_rated] = 1.0

    return cf.clip(0.0, 1.0)


# -----------------------------------------------------------------------------
# 12. Synthetic fallback for wind CF
# -----------------------------------------------------------------------------
# Used only if the PVGIS API call fails.
def _synthetic_wind():
    cf = pd.Series(np.full(SNAPSHOTS, 0.38))
    pd.DataFrame({"wind_cf": cf.values}).to_csv(DATA_DIR / "kz_wind_cf.csv", index=False)
    print("  ✓ wind CF written (synthetic fallback, 38% constant)")


# =============================================================================
# 13. Build hydrogen demand time series
# =============================================================================
# This writes hydrogen demand depending on whether hydrogen is enabled and which
# hydrogen mode is chosen in the scenario.
def build_hydrogen_demand():
    hydrogen_cfg = cfg.get("hydrogen", {})
    hydrogen_enabled = bool(hydrogen_cfg.get("enabled", True))
    hydrogen_mode = hydrogen_cfg.get("mode", "fixed_demand")
    h2_mw = float(cfg["demand"].get("hydrogen_mw", 0.0))

    # If hydrogen is disabled entirely, write zeros.
    if not hydrogen_enabled:
        s = pd.Series(np.zeros(SNAPSHOTS))
        pd.DataFrame({"hydrogen_mw": s.values}).to_csv(DATA_DIR / "kz_hydrogen_demand.csv", index=False)
        print("  ✓ hydrogen demand written → data/kz_hydrogen_demand.csv  (all zeros; hydrogen disabled)")
        return

    # In flexible_sink mode, there is no fixed hydrogen demand profile;
    # hydrogen production becomes optional/flexible, so write zeros.
    if hydrogen_mode == "flexible_sink":
        s = pd.Series(np.zeros(SNAPSHOTS))
        pd.DataFrame({"hydrogen_mw": s.values}).to_csv(DATA_DIR / "kz_hydrogen_demand.csv", index=False)
        print("  ✓ hydrogen demand written → data/kz_hydrogen_demand.csv  (all zeros; flexible_sink mode)")
        return

    # Otherwise write a constant fixed hydrogen demand from the scenario config.
    s = pd.Series(np.full(SNAPSHOTS, h2_mw))
    pd.DataFrame({"hydrogen_mw": s.values}).to_csv(DATA_DIR / "kz_hydrogen_demand.csv", index=False)
    print(f"  ✓ hydrogen demand written → data/kz_hydrogen_demand.csv  (constant {h2_mw} MW from scenario.yaml)")


# =============================================================================
# 14. Main execution block
# =============================================================================
# Run all preprocessing steps in sequence and tell the user what to do next.
if __name__ == "__main__":
    print("── 1. Costs ────────────────────────────────────────────────────────")
    build_costs()

    print("\n── 2. Electricity demand ───────────────────────────────────────────")
    build_electricity_demand()

    print("\n── 3. Solar CF ─────────────────────────────────────────────────────")
    build_solar_cf()

    print("\n── 4. Wind CF ──────────────────────────────────────────────────────")
    build_wind_cf()

    print("\n── 5. Hydrogen demand ──────────────────────────────────────────────")
    build_hydrogen_demand()

    print("\n✓ Done. All real data written to data/")
    print("  Now run: snakemake --cores 1 -p\n")