import sys
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
import yaml


# =============================================================================
# 1. Command-line arguments
# =============================================================================
parser = argparse.ArgumentParser(
    description="Preprocess deterministic model inputs for a PyPSA scenario."
)

parser.add_argument(
    "--config",
    required=True,
    help="Path to scenario YAML file",
)

parser.add_argument(
    "--data-dir",
    required=True,
    help="Directory for processed input CSVs",
)

args = parser.parse_args()


# =============================================================================
# 2. Project paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(args.data_dir)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 3. Load scenario configuration
# =============================================================================
with open(
    args.config,
    "r",
    encoding="utf-8",
) as f:
    cfg = yaml.safe_load(f)


system_cfg = cfg["system"]

SNAPSHOTS = int(
    system_cfg["snapshots"]
)

INVESTMENT_YEAR = int(
    system_cfg["investment_year"]
)

WEATHER_YEAR = int(
    system_cfg["weather_year"]
)


# =============================================================================
# 4. Helper functions
# =============================================================================
def get_active_costs_path():
    """
    Return the configured technology-cost dataset.
    """

    costs_cfg = cfg["costs"]

    active_name = costs_cfg[
        "active_dataset"
    ]

    datasets = costs_cfg[
        "datasets"
    ]

    if active_name not in datasets:
        raise KeyError(
            f"Unknown cost dataset '{active_name}'. "
            f"Available options: {list(datasets.keys())}"
        )

    return (
        BASE_DIR
        / datasets[active_name]
    )


def get_profile_path(asset):
    """
    Return the configured reference renewable-profile file.
    """

    renewables_cfg = cfg.get(
        "renewables",
        {}
    )

    asset_cfg = renewables_cfg.get(
        asset,
        {}
    )

    profile_file = asset_cfg.get(
        "profile_file"
    )

    if not profile_file:
        raise KeyError(
            f"renewables.{asset}.profile_file "
            "is not configured."
        )

    path = (
        BASE_DIR
        / profile_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Reference {asset} profile "
            f"does not exist: {path}"
        )

    return path


def sha256_file(path):
    """
    Return SHA256 fingerprint of a file.
    """

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_capacity_factor(
    series,
    name,
):
    """
    Validate an hourly renewable capacity-factor series.
    """

    series = pd.to_numeric(
        series,
        errors="raise",
    )

    if len(series) != SNAPSHOTS:
        raise ValueError(
            f"{name} contains {len(series)} rows, "
            f"but system.snapshots={SNAPSHOTS}."
        )

    if series.isna().any():
        raise ValueError(
            f"{name} contains NaN values."
        )

    if not series.between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            f"{name} contains values outside [0, 1]."
        )

    # Detect clearly invalid constant resource profiles.
    if series.nunique() <= 1:
        raise ValueError(
            f"{name} is constant over the complete "
            "model horizon."
        )

    if float(series.std()) <= 0.01:
        raise ValueError(
            f"{name} shows insufficient temporal variation "
            "for the current hourly resource model."
        )

    return series.reset_index(
        drop=True
    )


# =============================================================================
# 5. Environment information
# =============================================================================
print(
    f"\nBase dir : {BASE_DIR}"
)

print(
    f"Data dir : {DATA_DIR}"
)

print(
    f"Investment year={INVESTMENT_YEAR}, "
    f"Weather year={WEATHER_YEAR}, "
    f"Snapshots={SNAPSHOTS}\n"
)


# =============================================================================
# 6. Technology costs
# =============================================================================
def build_costs():
    source = get_active_costs_path()

    if not source.exists():
        sys.exit(
            f"[ERROR] Missing cost dataset: "
            f"{source}"
        )

    df = pd.read_csv(
        source
    )

    keep_techs = [
        "solar-utility",
        "onwind",
        "electrolysis",
        "hydrogen storage underground",
    ]

    keep_params = [
        "investment",
        "FOM",
        "VOM",
        "efficiency",
        "lifetime",
    ]

    subset = df[
        df["technology"].isin(
            keep_techs
        )
        & df["parameter"].isin(
            keep_params
        )
    ][
        [
            "technology",
            "parameter",
            "value",
            "unit",
        ]
    ].copy()

    if subset.empty:
        raise RuntimeError(
            "No technology-cost rows matched "
            "the required technologies."
        )

    print(
        f"  Active dataset: "
        f"{cfg['costs']['active_dataset']}"
    )

    print(
        f"  Source: {source}"
    )

    out_name = (
        f"{cfg['costs']['active_dataset']}.csv"
    )

    out = (
        DATA_DIR
        / out_name
    )

    subset.to_csv(
        out,
        index=False,
    )

    print(
        f"  ✓ costs written → "
        f"{out} "
        f"({len(subset)} rows)"
    )


# =============================================================================
# 7. Solar reference profile
# =============================================================================
def build_solar_cf():
    source = get_profile_path(
        "solar"
    )

    df = pd.read_csv(
        source
    )

    if "solar_cf" not in df.columns:
        raise KeyError(
            f"{source} does not contain "
            "'solar_cf'."
        )

    cf = validate_capacity_factor(
        df["solar_cf"],
        "solar_cf",
    )

    out = (
        DATA_DIR
        / "kz_solar_cf.csv"
    )

    shutil.copyfile(
        source,
        out,
    )

    print(
        f"  Source: {source}"
    )

    print(
        f"  SHA256: "
        f"{sha256_file(source)}"
    )

    print(
        f"  rows={len(cf)}, "
        f"mean={cf.mean():.6f}, "
        f"std={cf.std():.6f}, "
        f"min={cf.min():.6f}, "
        f"max={cf.max():.6f}"
    )

    print(
        f"  ✓ solar CF written → "
        f"{out}"
    )


# =============================================================================
# 8. Wind reference profile
# =============================================================================
def build_wind_cf():
    source = get_profile_path(
        "wind"
    )

    df = pd.read_csv(
        source
    )

    if "wind_cf" not in df.columns:
        raise KeyError(
            f"{source} does not contain "
            "'wind_cf'."
        )

    cf = validate_capacity_factor(
        df["wind_cf"],
        "wind_cf",
    )

    out = (
        DATA_DIR
        / "kz_wind_cf.csv"
    )

    shutil.copyfile(
        source,
        out,
    )

    print(
        f"  Source: {source}"
    )

    print(
        f"  SHA256: "
        f"{sha256_file(source)}"
    )

    print(
        f"  rows={len(cf)}, "
        f"mean={cf.mean():.6f}, "
        f"std={cf.std():.6f}, "
        f"min={cf.min():.6f}, "
        f"max={cf.max():.6f}, "
        f"unique={cf.nunique()}"
    )

    print(
        f"  ✓ wind CF written → "
        f"{out}"
    )


# =============================================================================
# 9. Main
# =============================================================================
if __name__ == "__main__":
    print(
        "── 1. Costs "
        "────────────────────────────────────────"
    )

    build_costs()

    print(
        "\n── 2. Solar reference profile "
        "──────────────────────"
    )

    build_solar_cf()

    print(
        "\n── 3. Wind reference profile "
        "───────────────────────"
    )

    build_wind_cf()

    print(
        "\n✓ Deterministic off-grid preprocessing "
        "completed successfully."
    )
