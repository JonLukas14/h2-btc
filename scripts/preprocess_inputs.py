# scripts/preprocess_inputs.py
#
# Reads REAL data from:
#   - ~/git/pypsa-kz-data/           (sibling repo, not inside project)
#   - technology-data/outputs/*.csv  (selected via scenario.yaml)
#
# Writes model-ready CSVs to:
#   - data/

import sys
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TECH_REPO = BASE_DIR / "technology-data"
DATA_DIR.mkdir(exist_ok=True)

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


# ── Config ────────────────────────────────────────────────────────────────────
with open(BASE_DIR / "configs" / "scenario.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

YEAR = int(cfg["system"]["year"])
SNAPSHOTS = int(cfg["system"]["snapshots"])


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


print(f"\nBase dir : {BASE_DIR}")
print(f"KZ repo  : {KZ_REPO}  (exists: {KZ_REPO.exists()})")
print(f"Tech repo: {TECH_REPO}  (exists: {TECH_REPO.exists()})")
print(f"Year={YEAR}, Snapshots={SNAPSHOTS}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 1. COSTS  ← selected via scenario.yaml
# ═════════════════════════════════════════════════════════════════════════════
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

    subset.loc[subset["parameter"] == "investment", "value"] *= 1000

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
    subset[["technology", "parameter", "value"]].to_csv(out, index=False)
    print(f"\n  ✓ costs written → data/{out_name}  ({len(subset)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# 2. ELECTRICITY DEMAND
# ═════════════════════════════════════════════════════════════════════════════
def build_electricity_demand():
    source = KZ_REPO / "data" / "kz_demand_validation.csv"
    if not source.exists():
        sys.exit(f"[ERROR] Missing: {source}\n  KZ_REPO resolved to: {KZ_REPO}")

    df = pd.read_csv(source, index_col=0)
    print(f"  Demand file columns: {list(df.columns)}")
    print(f"  Years available: {sorted(df['year'].unique())}")

    latest_year = df["year"].max()
    df_year = df[df["year"] == latest_year].sort_values("month").reset_index(drop=True)
    print(f"  Using year: {latest_year}  (most recent in file)")
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
    print(f"  ✓ electricity demand written → data/kz_electricity_demand.csv  (rows={len(out_series)}, mean={out_series.mean():.1f} MW)")


# ═════════════════════════════════════════════════════════════════════════════
# 3. SOLAR CF
# ═════════════════════════════════════════════════════════════════════════════
def build_solar_cf():
    lat, lon = 43.3, 71.4
    pvgis_year = 2020

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
        print("  ✓ solar CF written → data/kz_solar_cf.csv  (PVGIS ERA5)")
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
    print("  ✓ solar CF written (synthetic fallback)")


# ═════════════════════════════════════════════════════════════════════════════
# 4. WIND CF
# ═════════════════════════════════════════════════════════════════════════════
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
        print("  ✓ wind CF written → data/kz_wind_cf.csv  (PVGIS ERA5, hub height corrected)")

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
    print("  ✓ wind CF written (synthetic fallback, 38% constant)")


# ═════════════════════════════════════════════════════════════════════════════
# 5. HYDROGEN DEMAND
# ═════════════════════════════════════════════════════════════════════════════
def build_hydrogen_demand():
    h2_mw = float(cfg["demand"].get("hydrogen_mw", 0.0))
    s = pd.Series(np.full(SNAPSHOTS, h2_mw))
    pd.DataFrame({"hydrogen_mw": s.values}).to_csv(DATA_DIR / "kz_hydrogen_demand.csv", index=False)
    print(f"  ✓ hydrogen demand written → data/kz_hydrogen_demand.csv  (constant {h2_mw} MW from scenario.yaml)")


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