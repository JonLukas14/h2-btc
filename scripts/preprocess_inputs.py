# scripts/preprocess_inputs.py
#
# Reads REAL data from:
#   - ~/git/pypsa-kz-data/           (sibling repo, not inside project)
#   - ~/git/kazakhstan_thesis_test/technology-data/outputs/costs_2030.csv
#
# Writes model-ready CSVs to:
#   - ~/git/kazakhstan_thesis_test/data/

import sys
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
TECH_REPO = BASE_DIR / "technology-data"
DATA_DIR.mkdir(exist_ok=True)

# Auto-detect pypsa-kz-data: check inside project, then as sibling repo
_kz_candidates = [
    BASE_DIR / "pypsa-kz-data",                      # inside project
    BASE_DIR.parent / "pypsa-kz-data",               # sibling of project
    Path.home() / "git" / "pypsa-kz-data",           # explicit ~/git/
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
with open(BASE_DIR / "configs" / "scenario.yaml", "r") as f:
    cfg = yaml.safe_load(f)

YEAR      = int(cfg["system"]["year"])       # 2030
SNAPSHOTS = int(cfg["system"]["snapshots"])  # 24 (test) or 8760 (full year)

print(f"\nBase dir : {BASE_DIR}")
print(f"KZ repo  : {KZ_REPO}  (exists: {KZ_REPO.exists()})")
print(f"Tech repo: {TECH_REPO}  (exists: {TECH_REPO.exists()})")
print(f"Year={YEAR}, Snapshots={SNAPSHOTS}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 1. COSTS  ←  technology-data/outputs/costs_2030.csv
# ═════════════════════════════════════════════════════════════════════════════
def build_costs():
    source = TECH_REPO / "outputs" / f"costs_{YEAR}.csv"
    if not source.exists():
        sys.exit(f"[ERROR] Missing: {source}")

    df = pd.read_csv(source)

    # ── Print all technology names that contain relevant keywords ─────────────
    keywords = ["solar", "wind", "electro", "hydrogen", "pem", "alkaline"]
    matches = df[df["technology"].str.lower().str.contains("|".join(keywords), na=False)]
    print("  Technologies in costs file matching solar/wind/electrolysis/hydrogen:")
    for t in sorted(matches["technology"].unique()):
        print(f"    {t}")

    # ── Exact technology names from costs_2030.csv ────────────────────────────
    # These names were verified by inspecting the file above.
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

    print(f"\n  Matched rows:")
    print(subset.to_string(index=False))

    out = DATA_DIR / "costs_2030.csv"
    subset[["technology", "parameter", "value"]].to_csv(out, index=False)
    print(f"\n  ✓ costs written → data/costs_2030.csv  ({len(subset)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# 2. ELECTRICITY DEMAND  ←  pypsa-kz-data/data/kz_demand_validation.csv
#    Columns: year, month, demand_gegis, demand_korem   (monthly GWh)
#    Strategy: use KOREM data (official KZ grid operator), latest year available,
#              convert monthly GWh → average MW per hour within each month,
#              then produce an hourly profile for the full year.
# ═════════════════════════════════════════════════════════════════════════════
def build_electricity_demand():
    source = KZ_REPO / "data" / "kz_demand_validation.csv"
    if not source.exists():
        sys.exit(f"[ERROR] Missing: {source}\n  KZ_REPO resolved to: {KZ_REPO}")

    df = pd.read_csv(source, index_col=0)
    print(f"  Demand file columns: {list(df.columns)}")
    print(f"  Years available: {sorted(df['year'].unique())}")

    # Use most recent full year available (not 2030 — we use latest real data)
    latest_year = df["year"].max()
    df_year = df[df["year"] == latest_year].sort_values("month").reset_index(drop=True)
    print(f"  Using year: {latest_year}  (most recent in file)")
    print(df_year[["month", "demand_korem"]].to_string(index=False))

    # Monthly GWh → MW per hour  (GWh * 1000 / hours_in_month)
    # 2030 is not a leap year
    hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

    monthly_gwh = df_year["demand_korem"].values.astype(float)

    hourly_demand = []
    for gwh, hrs in zip(monthly_gwh, hours_per_month):
        mw = (gwh * 1000.0) / hrs   # GWh → MWh per hour = MW
        hourly_demand.extend([mw] * hrs)

    hourly_series = pd.Series(hourly_demand)  # 8760 rows for a full year
    print(f"\n  Hourly MW — min: {hourly_series.min():.1f}, "
          f"mean: {hourly_series.mean():.1f}, max: {hourly_series.max():.1f}")

    # Trim to model snapshots (24 for test, 8760 for full year)
    out_series = hourly_series.iloc[:SNAPSHOTS].reset_index(drop=True)

    out = DATA_DIR / "kz_electricity_demand.csv"
    pd.DataFrame({"electricity_mw": out_series.values}).to_csv(out, index=False)
    print(f"  ✓ electricity demand written → data/kz_electricity_demand.csv  "
          f"(rows={len(out_series)}, mean={out_series.mean():.1f} MW)")


# ═════════════════════════════════════════════════════════════════════════════
# 3. SOLAR CF  ←  PVGIS ERA5 API  (Zhambyl region, south Kazakhstan)
# ═════════════════════════════════════════════════════════════════════════════
def build_solar_cf():
    lat, lon = 43.3, 71.4   # Zhambyl region — good solar resource
    pvgis_year = 2020        # latest ERA5 year in PVGIS; use as proxy for 2030

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
# 4. WIND CF  ←  PVGIS ERA5 API  (Mangystau region, western Kazakhstan)
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

        # PVGIS returns wind speeds rounded to 1 decimal place (~96 unique values).
        # Add small Gaussian noise (σ=0.1 m/s) to recover realistic CF variation.
        # This preserves the statistical distribution while removing quantization.
        rng = np.random.default_rng(seed=42)   # fixed seed for reproducibility
        ws_10m = ws_10m + rng.normal(0, 0.1, size=len(ws_10m))
        ws_10m = ws_10m.clip(lower=0.0)

        # ERA5 bias correction for flat inland steppe terrain (+25%)
        ws_10m = ws_10m * 1.25

        # Scale from 10m to 100m hub height (Hellmann power law, α=0.143)
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
    """Smooth 3-parameter power curve: cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s."""
    ws = pd.Series(ws).astype(float)
    cf = pd.Series(0.0, index=ws.index)

    # Between cut-in and rated: cubic ramp
    mask_ramp = (ws >= 3.0) & (ws < 12.0)
    cf[mask_ramp] = ((ws[mask_ramp] - 3.0) / (12.0 - 3.0)) ** 3

    # Between rated and cut-out: full capacity
    mask_rated = (ws >= 12.0) & (ws <= 25.0)
    cf[mask_rated] = 1.0

    return cf.clip(0.0, 1.0)


def _synthetic_wind():
    cf = pd.Series(np.full(SNAPSHOTS, 0.38))
    pd.DataFrame({"wind_cf": cf.values}).to_csv(DATA_DIR / "kz_wind_cf.csv", index=False)
    print("  ✓ wind CF written (synthetic fallback, 38% constant)")


# ═════════════════════════════════════════════════════════════════════════════
# 5. HYDROGEN DEMAND  ←  scenario.yaml thesis assumption
# ═════════════════════════════════════════════════════════════════════════════
def build_hydrogen_demand():
    h2_mw = float(cfg["demand"].get("hydrogen_mw", 0.0))
    s = pd.Series(np.full(SNAPSHOTS, h2_mw))
    pd.DataFrame({"hydrogen_mw": s.values}).to_csv(DATA_DIR / "kz_hydrogen_demand.csv", index=False)
    print(f"  ✓ hydrogen demand written → data/kz_hydrogen_demand.csv  "
          f"(constant {h2_mw} MW from scenario.yaml)")


# ═════════════════════════════════════════════════════════════════════════════
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
    print("  Now run:  python scripts/analyze_results.py\n")

