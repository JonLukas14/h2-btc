import sys
import argparse
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


# -----------------------------------------------------------------------------
# 1. Parse command-line arguments
# -----------------------------------------------------------------------------
# This allows preprocessing to use a chosen scenario file and write outputs
# into a chosen data folder.
parser = argparse.ArgumentParser(description="Preprocess model inputs for a PyPSA scenario.")
parser.add_argument("--config", required=True, help="Path to scenario YAML file")
parser.add_argument("--data-dir", required=True, help="Directory for processed input CSVs")
args = parser.parse_args()


# -----------------------------------------------------------------------------
# 2. Define important paths
# -----------------------------------------------------------------------------
# BASE_DIR is the project root.
# DATA_DIR is where processed model input CSVs will be written.
# TECH_REPO points to the local technology-data repository.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(args.data_dir)
TECH_REPO = BASE_DIR / "technology-data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 3. Locate the pypsa-kz-data repository automatically
# -----------------------------------------------------------------------------
_kz_candidates = [
    BASE_DIR / "pypsa-kz-data",
    BASE_DIR.parent / "pypsa-kz-data",
    Path.home() / "git" / "pypsa-kz-data",
]

KZ_REPO = next(
    (p for p in _kz_candidates if (p / "data" / "kz_demand_validation.csv").exists()),
    None
)

if KZ_REPO is None:
    print("[ERROR] Cannot find pypsa-kz-data in any of these locations:")
    for p in _kz_candidates:
        print(f"  {p}  (exists: {p.exists()})")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 4. Load scenario configuration
# -----------------------------------------------------------------------------
with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

system_cfg = cfg["system"]

SNAPSHOTS = int(system_cfg["snapshots"])
INVESTMENT_YEAR = int(system_cfg["investment_year"])
WEATHER_YEAR = int(system_cfg["weather_year"])
DEMAND_YEAR = int(system_cfg["demand_year"])


# -----------------------------------------------------------------------------
# 5. Helper function: choose the active cost dataset from scenario config
# -----------------------------------------------------------------------------
def get_active_costs_path(cfg, base_dir):
    costs_cfg = cfg["costs"]
    active_name = costs_cfg["active_dataset"]
    datasets = costs_cfg["datasets"]

    if active_name not in datasets:
        raise KeyError(
            f"Unknown cost dataset '{active_name}'. "
            f"Available options: {list(datasets.keys())}"
        )

    return base_dir / datasets[active_name]


# -----------------------------------------------------------------------------
# 6. Print environment information
# -----------------------------------------------------------------------------
print(f"\nBase dir : {BASE_DIR}")
print(f"KZ repo  : {KZ_REPO}  (exists: {KZ_REPO.exists()})")
print(f"Tech repo: {TECH_REPO}  (exists: {TECH_REPO.exists()})")
print(f"Data dir : {DATA_DIR}")
print(
    f"Investment year={INVESTMENT_YEAR}, "
    f"Weather year={WEATHER_YEAR}, "
    f"Demand year={DEMAND_YEAR}, "
    f"Snapshots={SNAPSHOTS}\n"
)


# =============================================================================
# 7. Build costs CSV from selected cost dataset
# =============================================================================
def build_costs():
    source = get_active_costs_path(cfg, BASE_DIR)
    if not source.exists():
        sys.exit(f"[ERROR] Missing: {source}")

    df = pd.read_csv(source)

    keywords = ["solar", "wind", "electro", "hydrogen", "pem", "alkaline"]
    matches = df[df["technology"].str.lower().str.contains("|".join(keywords), na=False)]
    print("  Technologies in costs file matching solar/wind/electrolysis/hydrogen:")
    for t in sorted(matches["technology"].unique()):
        print(f"    {t}")

    keep_techs = [
        "solar-utility",
        "onwind",
        "electrolysis",
        "hydrogen storage underground",
    ]
    keep_params = ["investment", "FOM", "VOM", "efficiency", "lifetime"]

    subset = df[
        df["technology"].isin(keep_techs) &
        df["parameter"].isin(keep_params)
    ][["technology", "parameter", "value", "unit"]].copy()

    #subset.loc[subset["parameter"] == "investment", "value"] *= 1000

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

    out_name = f"{cfg['costs']['active_dataset']}.csv"
    out = DATA_DIR / out_name
    subset[["technology", "parameter", "value", "unit"]].to_csv(out, index=False)
    print(f"\n  ✓ costs written → {out}  ({len(subset)} rows)")


# =============================================================================
# 8. Build hourly electricity demand series
# =============================================================================
def build_electricity_demand():
    source = KZ_REPO / "data" / "kz_demand_validation.csv"
    if not source.exists():
        sys.exit(f"[ERROR] Missing: {source}\n  KZ_REPO resolved to: {KZ_REPO}")

    df = pd.read_csv(source, index_col=0)
    print(f"  Demand file columns: {list(df.columns)}")
    print(f"  Years available: {sorted(df['year'].unique())}")

    available_years = sorted(df["year"].unique())

    if DEMAND_YEAR not in available_years:
        raise ValueError(
            f"Configured demand year {DEMAND_YEAR} is not available "
            f"in kz_demand_validation.csv. "
            f"Available years: {available_years}"
        )

    df_year = (
        df[df["year"] == DEMAND_YEAR]
        .sort_values("month")
        .reset_index(drop=True)
    )

    print(f"  Using configured demand year: {DEMAND_YEAR}")
    print(df_year[["month", "demand_korem"]].to_string(index=False))

    hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    monthly_gwh = df_year["demand_korem"].values.astype(float)

    hourly_demand = []
    for gwh, hrs in zip(monthly_gwh, hours_per_month):
        mw = (gwh * 1000.0) / hrs
        hourly_demand.extend([mw] * hrs)

    hourly_series = pd.Series(hourly_demand)
    print(f"\n  Hourly MW — min: {hourly_series.min():.1f}, mean: {hourly_series.mean():.1f}, max: {hourly_series.max():.1f}")

    out_series = hourly_series.iloc[:SNAPSHOTS].reset_index(drop=True)

    out = DATA_DIR / "kz_electricity_demand.csv"
    pd.DataFrame({"electricity_mw": out_series.values}).to_csv(out, index=False)
    print(f"  ✓ electricity demand written → {out}  (rows={len(out_series)}, mean={out_series.mean():.1f} MW)")


# =============================================================================
# 9. Build solar capacity factor time series
# =============================================================================
def build_solar_cf():
    lat, lon = 43.3, 71.4
    pvgis_year = WEATHER_YEAR

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
        cf_values = [entry["P"] / 1000.0 for entry in data["outputs"]["hourly"]]
        cf = pd.Series(cf_values).clip(0.0, 1.0).iloc[:SNAPSHOTS]
        print(f"  mean CF={cf.mean():.3f}, peak CF={cf.max():.3f}")
        pd.DataFrame({"solar_cf": cf.values}).to_csv(DATA_DIR / "kz_solar_cf.csv", index=False)
        print(f"  ✓ solar CF written → {DATA_DIR / 'kz_solar_cf.csv'}  (PVGIS ERA5)")
    except Exception as e:
        print(f"  [WARNING] PVGIS failed: {e} — using synthetic fallback")
        _synthetic_solar()


def _synthetic_solar():
    daily = [0.00, 0.00, 0.00, 0.00, 0.00, 0.03,
             0.12, 0.28, 0.48, 0.65, 0.77, 0.84,
             0.86, 0.80, 0.69, 0.52, 0.33, 0.14,
             0.03, 0.00, 0.00, 0.00, 0.00, 0.00]
    reps = (SNAPSHOTS // 24) + 1
    cf = pd.Series((daily * reps)[:SNAPSHOTS])
    pd.DataFrame({"solar_cf": cf.values}).to_csv(DATA_DIR / "kz_solar_cf.csv", index=False)
    print(f"  ✓ solar CF written → {DATA_DIR / 'kz_solar_cf.csv'}  (synthetic fallback)")


# =============================================================================
# 10. Build wind capacity factor time series
# =============================================================================
def build_wind_cf():
    lat, lon = 43.6, 51.2
    pvgis_year = WEATHER_YEAR

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

        ws_10m = pd.Series([entry["WS10m"] for entry in data["outputs"]["hourly"]])

        rng = np.random.default_rng(seed=42)
        ws_10m = ws_10m + rng.normal(0, 0.1, size=len(ws_10m))
        ws_10m = ws_10m.clip(lower=0.0)

        ws_10m = ws_10m * 1.25

        alpha = 0.20
        ws_hub = ws_10m * (100 / 10) ** alpha

        cf = _power_curve(ws_hub).clip(0.0, 1.0)
        cf = cf.iloc[:SNAPSHOTS].reset_index(drop=True)

        print(f"  mean CF={cf.mean():.3f}, peak CF={cf.max():.3f}")
        pd.DataFrame({"wind_cf": cf.values}).to_csv(DATA_DIR / "kz_wind_cf.csv", index=False)
        print(f"  ✓ wind CF written → {DATA_DIR / 'kz_wind_cf.csv'}  (PVGIS ERA5, hub height corrected)")
    except Exception as e:
        print(f"  [WARNING] PVGIS failed: {e} — using synthetic fallback")
        _synthetic_wind()


def _power_curve(ws):
    ws = pd.Series(ws).astype(float)
    cf = pd.Series(0.0, index=ws.index)

    mask_ramp = (ws >= 3.0) & (ws < 12.0)
    cf[mask_ramp] = ((ws[mask_ramp] - 3.0) / (12.0 - 3.0)) ** 3

    mask_rated = (ws >= 12.0) & (ws <= 25.0)
    cf[mask_rated] = 1.0

    return cf.clip(0.0, 1.0)


def _synthetic_wind():
    cf = pd.Series(np.full(SNAPSHOTS, 0.38))
    pd.DataFrame({"wind_cf": cf.values}).to_csv(DATA_DIR / "kz_wind_cf.csv", index=False)
    print(f"  ✓ wind CF written → {DATA_DIR / 'kz_wind_cf.csv'}  (synthetic fallback, 38% constant)")


# =============================================================================
# 11. Build hydrogen demand time series
# =============================================================================
def build_hydrogen_demand():
    hydrogen_cfg = cfg.get("hydrogen", {})

    hydrogen_enabled = bool(
        hydrogen_cfg.get("enabled", True)
    )

    hydrogen_mode = hydrogen_cfg.get(
        "mode",
        "fixed_demand"
    )

    # -------------------------------------------------------------------------
    # Case 1: Hydrogen is disabled
    # -------------------------------------------------------------------------
    if not hydrogen_enabled:
        s = pd.Series(np.zeros(SNAPSHOTS))

        pd.DataFrame(
            {"hydrogen_mw": s.values}
        ).to_csv(
            DATA_DIR / "kz_hydrogen_demand.csv",
            index=False
        )

        print(
            f"  ✓ hydrogen demand written → "
            f"{DATA_DIR / 'kz_hydrogen_demand.csv'} "
            f"(all zeros; hydrogen disabled)"
        )
        return

    # -------------------------------------------------------------------------
    # Case 2: Flexible H2 production
    #
    # No fixed hourly H2 demand is imposed here.
    # production_target will later be constrained inside the PyPSA model.
    # -------------------------------------------------------------------------
    if hydrogen_mode in {"flexible_sink", "production_target"}:
        s = pd.Series(np.zeros(SNAPSHOTS))

        pd.DataFrame(
            {"hydrogen_mw": s.values}
        ).to_csv(
            DATA_DIR / "kz_hydrogen_demand.csv",
            index=False
        )

        print(
            f"  ✓ hydrogen demand written → "
            f"{DATA_DIR / 'kz_hydrogen_demand.csv'} "
            f"(all zeros; mode={hydrogen_mode})"
        )
        return

    # -------------------------------------------------------------------------
    # Case 3: Fixed hourly H2 demand
    # -------------------------------------------------------------------------
    if hydrogen_mode == "fixed_demand":
        demand_cfg = cfg.get("demand", {})

        if "hydrogen_mw" not in demand_cfg:
            raise KeyError(
                "Hydrogen mode is 'fixed_demand', but "
                "'demand.hydrogen_mw' is missing from the scenario YAML."
            )

        h2_mw = float(
            demand_cfg["hydrogen_mw"]
        )

        s = pd.Series(
            np.full(SNAPSHOTS, h2_mw)
        )

        pd.DataFrame(
            {"hydrogen_mw": s.values}
        ).to_csv(
            DATA_DIR / "kz_hydrogen_demand.csv",
            index=False
        )

        print(
            f"  ✓ hydrogen demand written → "
            f"{DATA_DIR / 'kz_hydrogen_demand.csv'} "
            f"(constant {h2_mw} MW; fixed_demand mode)"
        )
        return

    # -------------------------------------------------------------------------
    # Invalid hydrogen mode
    # -------------------------------------------------------------------------
    raise ValueError(
        f"Unknown hydrogen mode '{hydrogen_mode}'. "
        "Supported modes are: "
        "fixed_demand, flexible_sink, production_target."
    )

    # -------------------------------------------------------------------------
    # Flexible hydrogen production
    #
    # No fixed hourly H2 demand is required.
    # The annual production target will later be imposed in build_network.py.
    # -------------------------------------------------------------------------
    if hydrogen_mode in {"flexible_sink", "production_target"}:
        s = pd.Series(np.zeros(SNAPSHOTS))

        pd.DataFrame(
            {"hydrogen_mw": s.values}
        ).to_csv(
            DATA_DIR / "kz_hydrogen_demand.csv",
            index=False
        )

        print(
            f"  ✓ hydrogen demand written → "
            f"{DATA_DIR / 'kz_hydrogen_demand.csv'} "
            f"(all zeros; mode={hydrogen_mode})"
        )
        return

    # -------------------------------------------------------------------------
    # Fixed hourly hydrogen demand
    # -------------------------------------------------------------------------
    if hydrogen_mode == "fixed_demand":
        demand_cfg = cfg.get("demand", {})

        if "hydrogen_mw" not in demand_cfg:
            raise KeyError(
                "Hydrogen mode is 'fixed_demand', but "
                "'demand.hydrogen_mw' is missing from the scenario YAML."
            )

        h2_mw = float(demand_cfg["hydrogen_mw"])

        s = pd.Series(np.full(SNAPSHOTS, h2_mw))

        pd.DataFrame(
            {"hydrogen_mw": s.values}
        ).to_csv(
            DATA_DIR / "kz_hydrogen_demand.csv",
            index=False
        )

        print(
            f"  ✓ hydrogen demand written → "
            f"{DATA_DIR / 'kz_hydrogen_demand.csv'} "
            f"(constant {h2_mw} MW; fixed_demand mode)"
        )
        return

    # -------------------------------------------------------------------------
    # Invalid mode
    # -------------------------------------------------------------------------
    raise ValueError(
        f"Unknown hydrogen mode '{hydrogen_mode}'. "
        f"Supported modes are: fixed_demand, flexible_sink, production_target."
    )


# =============================================================================
# 12. Main execution block
# =============================================================================
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

    print("\n✓ Done. All processed inputs written successfully.")