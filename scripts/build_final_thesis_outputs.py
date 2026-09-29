#!/usr/bin/env python3

"""
Build final thesis-ready off-grid result tables and figures.

IMPORTANT
---------
This script is post-processing only.

It DOES NOT:
- preprocess inputs,
- build a PyPSA network,
- optimize any scenario,
- rerun any solved case.

It reads existing summary CSV files only.

Inputs
------
Core:
    S0, S1, S2, S3, S2E, S3E

Maximum-production:
    HMAX
    HMAX-B

Merchant:
    H2 value sweep
    H2 x BTC CAPEX grid
    refined H2 x BTC CAPEX cases

Outputs
-------
results/final_thesis_outputs/

    tables/
        table_core_hmax_summary.csv
        table_core_hmax_summary.tex
        table_merchant_thresholds.csv
        table_merchant_thresholds.tex
        merchant_h2_value_sweep.csv
        merchant_grid_all_cases.csv

    figures/
        fig_merchant_regime_map_joint.png/pdf
        fig_merchant_regime_map_joint_b.png/pdf
        fig_btc_viability_boundary.png/pdf
        fig_h2_btc_tradeoff.png/pdf
        fig_battery_entry_regime.png/pdf

    final_output_manifest.txt
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"

OUTDIR = RESULTS_DIR / "final_thesis_outputs"

TABLE_DIR = OUTDIR / "tables"
FIGURE_DIR = OUTDIR / "figures"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CORE_SCENARIOS = {
    "S0": RESULTS_DIR / "s0_thesis_2045" / "summary.csv",
    "S1": RESULTS_DIR / "s1_thesis_2045" / "summary.csv",
    "S2": RESULTS_DIR / "s2_thesis_2045" / "summary.csv",
    "S3": RESULTS_DIR / "s3_thesis_2045" / "summary.csv",
    "S2E": RESULTS_DIR / "s2e_thesis_2045" / "summary.csv",
    "S3E": RESULTS_DIR / "s3e_thesis_2045" / "summary.csv",
}


H2_SWEEP_PATH = (
    RESULTS_DIR
    / "joint_h2_value_sweep"
    / "joint_h2_value_sweep_summary.csv"
)

GRID_PATH = (
    RESULTS_DIR
    / "joint_h2_btc_grid"
    / "joint_h2_btc_grid_summary.csv"
)

REFINED_PATH = (
    RESULTS_DIR
    / "joint_h2_btc_grid_refined"
    / "refined_all_cases.csv"
)


# =============================================================================
# Numerical thresholds used only for result classification
# =============================================================================

BTC_ACTIVE_TOL_MW = 1e-6
BATTERY_ACTIVE_TOL_MW = 1e-6
H2_ACTIVE_TOL_KT = 1e-9

ZERO_TOL = 1e-5


# =============================================================================
# Helpers
# =============================================================================

def require_file(
    path: Path,
    label: str,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return path


def load_single_summary(
    path: Path,
) -> pd.Series:
    require_file(
        path,
        "scenario summary",
    )

    df = pd.read_csv(
        path
    )

    if len(df) != 1:
        raise RuntimeError(
            f"{path} must contain exactly one row; "
            f"found {len(df)}."
        )

    return df.iloc[0].copy()


def bool_value(
    value,
) -> bool:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def clean_small_numeric_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    numeric = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric] = (
        df[numeric]
        .mask(
            df[numeric].abs()
            < ZERO_TOL,
            0.0,
        )
    )

    return df


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def safe_float(
    value,
) -> float:
    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return np.nan

    return value


def classify_regime_from_row(
    row: pd.Series,
) -> str:
    h2 = safe_float(
        row.get(
            "hydrogen_delivered_kt",
            0.0,
        )
    )

    btc = safe_float(
        row.get(
            "bitcoin_capacity_mw",
            0.0,
        )
    )

    battery_power = safe_float(
        row.get(
            "battery_power_mw",
            0.0,
        )
    )

    battery_energy = safe_float(
        row.get(
            "battery_energy_mwh",
            0.0,
        )
    )

    h2_active = (
        np.isfinite(h2)
        and h2 > H2_ACTIVE_TOL_KT
    )

    btc_active = (
        np.isfinite(btc)
        and btc > BTC_ACTIVE_TOL_MW
    )

    battery_active = (
        (
            np.isfinite(
                battery_power
            )
            and battery_power
            > BATTERY_ACTIVE_TOL_MW
        )
        or (
            np.isfinite(
                battery_energy
            )
            and battery_energy
            > BATTERY_ACTIVE_TOL_MW
        )
    )

    parts = []

    if h2_active:
        parts.append("H2")

    if btc_active:
        parts.append("BTC")

    if battery_active:
        parts.append("battery")

    if not parts:
        return "none"

    return "+".join(parts)


# =============================================================================
# Discover HMAX / HMAX-B from solved summaries
# =============================================================================

def discover_hmax_summaries():
    """
    Discover top-level result summaries whose hydrogen mode is
    maximize_production.

    HMAX and HMAX-B are distinguished by battery_enabled metadata,
    not by optimized battery capacity, because the optimal battery
    capacity may legitimately be zero.
    """

    candidates = []

    for path in sorted(
        RESULTS_DIR.glob(
            "*/summary.csv"
        )
    ):
        try:
            row = load_single_summary(
                path
            )
        except Exception:
            continue

        hydrogen_mode = (
            str(
                row.get(
                    "hydrogen_mode",
                    "",
                )
            )
            .strip()
            .lower()
        )

        if (
            hydrogen_mode
            != "maximize_production"
        ):
            continue

        candidates.append(
            (
                path,
                row,
            )
        )

    no_battery = []
    with_battery = []

    for path, row in candidates:
        if bool_value(
            row.get(
                "battery_enabled",
                False,
            )
        ):
            with_battery.append(
                (
                    path,
                    row,
                )
            )
        else:
            no_battery.append(
                (
                    path,
                    row,
                )
            )

    if len(no_battery) != 1:
        raise RuntimeError(
            "Expected exactly one solved HMAX "
            "summary without battery, found "
            f"{len(no_battery)}:\n"
            + "\n".join(
                str(path)
                for path, _
                in no_battery
            )
        )

    if len(with_battery) != 1:
        raise RuntimeError(
            "Expected exactly one solved HMAX "
            "summary with battery enabled, found "
            f"{len(with_battery)}:\n"
            + "\n".join(
                str(path)
                for path, _
                in with_battery
            )
        )

    return {
        "HMAX": no_battery[0],
        "HMAX-B": with_battery[0],
    }


# =============================================================================
# 1. Load core scenarios
# =============================================================================

scenario_rows = []

print(
    "===== CORE SCENARIOS ====="
)

for label, path in (
    CORE_SCENARIOS.items()
):
    row = load_single_summary(
        path
    )

    row["scenario"] = label
    row["result_path"] = str(
        path.relative_to(
            BASE_DIR
        )
    )

    scenario_rows.append(
        row
    )

    print(
        f"PASS — {label}: "
        f"{path.relative_to(BASE_DIR)}"
    )


# =============================================================================
# 2. Load HMAX / HMAX-B
# =============================================================================

print(
    "\n===== HMAX DISCOVERY ====="
)

hmax_cases = (
    discover_hmax_summaries()
)

for label, (
    path,
    row,
) in hmax_cases.items():

    row = row.copy()

    row["scenario"] = label
    row["result_path"] = str(
        path.relative_to(
            BASE_DIR
        )
    )

    scenario_rows.append(
        row
    )

    print(
        f"PASS — {label}: "
        f"{path.relative_to(BASE_DIR)}"
    )


scenario_summary = (
    pd.DataFrame(
        scenario_rows
    )
    .set_index(
        "scenario"
    )
)

scenario_order = [
    "S0",
    "S1",
    "S2",
    "S3",
    "S2E",
    "S3E",
    "HMAX",
    "HMAX-B",
]

scenario_summary = (
    scenario_summary
    .reindex(
        scenario_order
    )
)


# =============================================================================
# 3. Build compact core + HMAX thesis table
# =============================================================================

scenario_columns = [
    "scenario_name",
    "hydrogen_mode",
    "battery_enabled",
    "bitcoin_enabled",
    "solar_capacity_mw",
    "wind_capacity_mw",
    "electrolyzer_capacity_mw",
    "battery_power_mw",
    "battery_energy_mwh",
    "bitcoin_capacity_mw",
    "hydrogen_delivered_kt",
    "renewable_utilization_rate",
    "solar_curtailment_rate",
    "wind_curtailment_rate",
    "gross_system_expenditure_eur_per_year",
    "bitcoin_gross_revenue_eur_per_year",
    "net_system_cost_eur_per_year",
    "lcoh_eur_per_kg_h2",
    "water_requirement_m3_per_year",
    "result_path",
]

scenario_columns = [
    column
    for column
    in scenario_columns
    if column
    in scenario_summary.columns
]

core_hmax_table = (
    scenario_summary[
        scenario_columns
    ]
    .reset_index()
)

if (
    "solar_capacity_mw"
    in core_hmax_table.columns
    and "wind_capacity_mw"
    in core_hmax_table.columns
):
    core_hmax_table.insert(
        core_hmax_table.columns.get_loc(
            "wind_capacity_mw"
        )
        + 1,
        "renewable_capacity_total_mw",
        (
            core_hmax_table[
                "solar_capacity_mw"
            ]
            + core_hmax_table[
                "wind_capacity_mw"
            ]
        ),
    )

core_hmax_table = (
    clean_small_numeric_values(
        core_hmax_table
    )
)

core_csv = (
    TABLE_DIR
    / "table_core_hmax_summary.csv"
)

core_tex = (
    TABLE_DIR
    / "table_core_hmax_summary.tex"
)

core_hmax_table.to_csv(
    core_csv,
    index=False,
)

core_hmax_table.to_latex(
    core_tex,
    index=False,
    na_rep="--",
    float_format="%.4f",
)


# =============================================================================
# 4. Reconstruct merchant H2-value sweep from solved case summaries
# =============================================================================

print(
    "\n===== MERCHANT H2 SWEEP ====="
)

h2_sweep_root = (
    RESULTS_DIR
    / "joint_h2_value_sweep"
)

if not h2_sweep_root.exists():
    raise FileNotFoundError(
        "Missing merchant H2 sweep directory: "
        f"{h2_sweep_root}"
    )


h2_sweep_rows = []

for summary_path in sorted(
    h2_sweep_root.glob(
        "*/summary.csv"
    )
):
    try:
        result = pd.read_csv(
            summary_path
        )
    except Exception:
        continue

    if len(result) != 1:
        continue

    row = result.iloc[0].copy()

    hydrogen_mode = (
        str(
            row.get(
                "hydrogen_mode",
                "",
            )
        )
        .strip()
        .lower()
    )

    if (
        hydrogen_mode
        != "economic_dispatch"
    ):
        continue

    h2_value = safe_float(
        row.get(
            "hydrogen_sale_value_eur_per_kg_h2",
            np.nan,
        )
    )

    if not np.isfinite(
        h2_value
    ):
        continue

    family = (
        "joint_b"
        if bool_value(
            row.get(
                "battery_enabled",
                False,
            )
        )
        else "joint"
    )

    row[
        "joint_family"
    ] = family

    row[
        "sweep_h2_sale_value_eur_per_kg"
    ] = h2_value

    row[
        "source_summary_path"
    ] = str(
        summary_path.relative_to(
            BASE_DIR
        )
    )

    h2_sweep_rows.append(
        row
    )


if not h2_sweep_rows:
    raise RuntimeError(
        "No solved merchant H2 sweep summaries "
        "were discovered."
    )


h2_sweep = pd.DataFrame(
    h2_sweep_rows
)

h2_sweep = (
    h2_sweep
    .sort_values(
        [
            "joint_family",
            "sweep_h2_sale_value_eur_per_kg",
        ]
    )
    .drop_duplicates(
        subset=[
            "joint_family",
            "sweep_h2_sale_value_eur_per_kg",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)


required_h2_sweep = {
    "joint_family",
    "sweep_h2_sale_value_eur_per_kg",
    "hydrogen_delivered_kt",
    "bitcoin_capacity_mw",
    "battery_power_mw",
    "battery_energy_mwh",
}

missing = (
    required_h2_sweep
    - set(
        h2_sweep.columns
    )
)

if missing:
    raise RuntimeError(
        "Merchant H2 sweep is missing "
        f"required columns: {sorted(missing)}"
    )


# These anchor points must be present because they are needed
# for the final sampled-threshold interpretation.
joint_values = set(
    np.round(
        h2_sweep.loc[
            h2_sweep[
                "joint_family"
            ]
            == "joint",
            "sweep_h2_sale_value_eur_per_kg",
        ].astype(float),
        6,
    )
)

required_anchor_values = {
    2.375,
    2.4,
    2.7,
    2.75,
    4.0,
}

missing_anchor_values = (
    required_anchor_values
    - joint_values
)

if missing_anchor_values:
    raise RuntimeError(
        "The reconstructed merchant H2 sweep "
        "is missing required solved anchor values: "
        f"{sorted(missing_anchor_values)}"
    )


h2_sweep = (
    clean_small_numeric_values(
        h2_sweep
    )
)

h2_sweep.to_csv(
    TABLE_DIR
    / "merchant_h2_value_sweep.csv",
    index=False,
)


print(
    f"PASS — reconstructed H2 sweep rows: "
    f"{len(h2_sweep)}"
)

print(
    "PASS — JOINT H2 values: "
    + ", ".join(
        f"{value:g}"
        for value
        in sorted(
            h2_sweep.loc[
                h2_sweep[
                    "joint_family"
                ]
                == "joint",
                "sweep_h2_sale_value_eur_per_kg",
            ].unique()
        )
    )
)


# =============================================================================
# 5. Load original 84-case merchant grid
# =============================================================================

print(
    "\n===== MERCHANT H2 x BTC GRID ====="
)

require_file(
    GRID_PATH,
    "merchant H2 x BTC grid",
)

grid = pd.read_csv(
    GRID_PATH
)

required_grid = {
    "joint_family",
    "sweep_h2_sale_value_eur_per_kg",
    "sweep_btc_capex_eur_per_mw_year",
    "hydrogen_delivered_kt",
    "bitcoin_capacity_mw",
    "battery_power_mw",
    "battery_energy_mwh",
}

missing = (
    required_grid
    - set(
        grid.columns
    )
)

if missing:
    raise RuntimeError(
        "Merchant grid is missing "
        f"required columns: {sorted(missing)}"
    )

grid = grid.copy()

grid["source_set"] = (
    "main_grid"
)

if (
    "merchant_regime"
    not in grid.columns
):
    grid[
        "merchant_regime"
    ] = grid.apply(
        classify_regime_from_row,
        axis=1,
    )

print(
    f"PASS — main-grid rows: "
    f"{len(grid)}"
)


# =============================================================================
# 6. Load 24 refined cases
# =============================================================================

require_file(
    REFINED_PATH,
    "refined merchant grid",
)

refined = pd.read_csv(
    REFINED_PATH
)

rename_refined = {
    "h2_value_eur_per_kg":
        "sweep_h2_sale_value_eur_per_kg",
    "btc_capex_eur_per_mw_year":
        "sweep_btc_capex_eur_per_mw_year",
}

refined = refined.rename(
    columns=rename_refined
)

required_refined = {
    "joint_family",
    "sweep_h2_sale_value_eur_per_kg",
    "sweep_btc_capex_eur_per_mw_year",
    "hydrogen_delivered_kt",
    "bitcoin_capacity_mw",
    "battery_power_mw",
    "battery_energy_mwh",
}

missing = (
    required_refined
    - set(
        refined.columns
    )
)

if missing:
    raise RuntimeError(
        "Refined grid is missing "
        f"required columns: {sorted(missing)}"
    )

refined = refined.copy()

refined["source_set"] = (
    "refined"
)

if (
    "merchant_regime"
    not in refined.columns
):
    refined[
        "merchant_regime"
    ] = refined.apply(
        classify_regime_from_row,
        axis=1,
    )

print(
    f"PASS — refined rows: "
    f"{len(refined)}"
)


# =============================================================================
# 7. Combine main + refined merchant grids
# =============================================================================

merchant_all = pd.concat(
    [
        grid,
        refined,
    ],
    ignore_index=True,
    sort=False,
)

merchant_all = (
    merchant_all
    .sort_values(
        [
            "joint_family",
            "sweep_h2_sale_value_eur_per_kg",
            "sweep_btc_capex_eur_per_mw_year",
            "source_set",
        ]
    )
    .drop_duplicates(
        subset=[
            "joint_family",
            "sweep_h2_sale_value_eur_per_kg",
            "sweep_btc_capex_eur_per_mw_year",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)

merchant_all = (
    clean_small_numeric_values(
        merchant_all
    )
)

merchant_all.to_csv(
    TABLE_DIR
    / "merchant_grid_all_cases.csv",
    index=False,
)

print(
    "\n===== COMBINED MERCHANT CASES ====="
)

print(
    f"Main grid:        {len(grid)}"
)

print(
    f"Refined cases:    {len(refined)}"
)

print(
    f"Unique combined:  "
    f"{len(merchant_all)}"
)


# =============================================================================
# 8. General bracket helpers
# =============================================================================

def sampled_activity_bracket(
    df: pd.DataFrame,
    family: str,
    h2_value: float,
    activity_column: str,
    tolerance: float,
):
    """
    Return highest sampled active CAPEX and first sampled inactive
    CAPEX above it for a fixed H2 value.

    This reports a sampled bracket only.
    It does not interpolate an exact economic threshold.
    """

    subset = df[
        (
            df[
                "joint_family"
            ]
            == family
        )
        & np.isclose(
            df[
                "sweep_h2_sale_value_eur_per_kg"
            ],
            h2_value,
            rtol=0.0,
            atol=1e-9,
        )
    ].copy()

    if subset.empty:
        return (
            np.nan,
            np.nan,
        )

    subset = subset.sort_values(
        "sweep_btc_capex_eur_per_mw_year"
    )

    active = subset[
        subset[
            activity_column
        ].abs()
        > tolerance
    ]

    if active.empty:
        return (
            np.nan,
            float(
                subset[
                    "sweep_btc_capex_eur_per_mw_year"
                ].min()
            ),
        )

    highest_active = float(
        active[
            "sweep_btc_capex_eur_per_mw_year"
        ].max()
    )

    inactive_above = subset[
        (
            subset[
                "sweep_btc_capex_eur_per_mw_year"
            ]
            > highest_active
        )
        & (
            subset[
                activity_column
            ].abs()
            <= tolerance
        )
    ]

    if inactive_above.empty:
        first_inactive = np.nan
    else:
        first_inactive = float(
            inactive_above[
                "sweep_btc_capex_eur_per_mw_year"
            ].min()
        )

    return (
        highest_active,
        first_inactive,
    )


def battery_activity_series(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        (
            df[
                "battery_power_mw"
            ].abs()
            > BATTERY_ACTIVE_TOL_MW
        )
        | (
            df[
                "battery_energy_mwh"
            ].abs()
            > BATTERY_ACTIVE_TOL_MW
        )
    )


# =============================================================================
# 9. Central merchant H2 entry bracket
# =============================================================================

central_joint = (
    h2_sweep[
        h2_sweep[
            "joint_family"
        ]
        == "joint"
    ]
    .sort_values(
        "sweep_h2_sale_value_eur_per_kg"
    )
)

h2_active = (
    central_joint[
        "hydrogen_delivered_kt"
    ]
    > H2_ACTIVE_TOL_KT
)

if not h2_active.any():
    h2_entry_low = np.nan
    h2_entry_high = np.nan
else:
    first_active_index = (
        central_joint[
            h2_active
        ].index[0]
    )

    h2_entry_high = float(
        central_joint.loc[
            first_active_index,
            "sweep_h2_sale_value_eur_per_kg",
        ]
    )

    lower = central_joint[
        central_joint[
            "sweep_h2_sale_value_eur_per_kg"
        ]
        < h2_entry_high
    ]

    lower = lower[
        lower[
            "hydrogen_delivered_kt"
        ]
        <= H2_ACTIVE_TOL_KT
    ]

    h2_entry_low = (
        float(
            lower[
                "sweep_h2_sale_value_eur_per_kg"
            ].max()
        )
        if not lower.empty
        else np.nan
    )


# =============================================================================
# 10. Renewable-cap sampled transition points
# =============================================================================

def first_true_h2_value(
    df,
    column,
):
    if column not in df.columns:
        return np.nan

    subset = df[
        df["joint_family"]
        == "joint"
    ].copy()

    subset = subset[
        subset[column].map(
            bool_value
        )
    ]

    if subset.empty:
        return np.nan

    return float(
        subset[
            "sweep_h2_sale_value_eur_per_kg"
        ].min()
    )


first_solar_cap_h2 = (
    first_true_h2_value(
        h2_sweep,
        "solar_resource_cap_binding",
    )
)

first_wind_cap_h2 = (
    first_true_h2_value(
        h2_sweep,
        "wind_resource_cap_binding",
    )
)


# =============================================================================
# 11. Merchant threshold table
# =============================================================================

threshold_rows = []


def append_capex_bracket(
    label,
    family,
    h2_value,
    activity_column,
    tolerance,
):
    active_max, inactive_min = (
        sampled_activity_bracket(
            merchant_all,
            family,
            h2_value,
            activity_column,
            tolerance,
        )
    )

    threshold_rows.append(
        {
            "finding": label,
            "family": family,
            "h2_value_eur_per_kg": h2_value,
            "lower_bound": active_max,
            "upper_bound": inactive_min,
            "unit": (
                "EUR2020/MW_BTC/a"
            ),
            "interpretation": (
                "Highest sampled active BTC CAPEX "
                "and first sampled inactive CAPEX above it."
            ),
        }
    )


threshold_rows.append(
    {
        "finding":
            "Merchant H2 entry",
        "family":
            "joint, central BTC economics",
        "h2_value_eur_per_kg":
            np.nan,
        "lower_bound":
            h2_entry_low,
        "upper_bound":
            h2_entry_high,
        "unit":
            "EUR2020/kg_H2",
        "interpretation":
            (
                "Highest sampled H2 value with zero "
                "merchant H2 and first sampled value "
                "with positive merchant H2."
            ),
    }
)


append_capex_bracket(
    "Pure BTC viability",
    "joint",
    0.0,
    "bitcoin_capacity_mw",
    BTC_ACTIVE_TOL_MW,
)

append_capex_bracket(
    "Pure BTC viability with battery allowed",
    "joint_b",
    0.0,
    "bitcoin_capacity_mw",
    BTC_ACTIVE_TOL_MW,
)

append_capex_bracket(
    "BTC viability near H2 entry",
    "joint",
    2.38,
    "bitcoin_capacity_mw",
    BTC_ACTIVE_TOL_MW,
)

append_capex_bracket(
    "BTC viability near H2 entry with battery",
    "joint_b",
    2.38,
    "bitcoin_capacity_mw",
    BTC_ACTIVE_TOL_MW,
)

append_capex_bracket(
    "BTC crowd-out at H2 = 2.75 EUR/kg",
    "joint",
    2.75,
    "bitcoin_capacity_mw",
    BTC_ACTIVE_TOL_MW,
)

append_capex_bracket(
    "BTC crowd-out at H2 = 2.75 EUR/kg with battery",
    "joint_b",
    2.75,
    "bitcoin_capacity_mw",
    BTC_ACTIVE_TOL_MW,
)


# Battery-specific bracket at 2.38 EUR/kg.
joint_b_238 = merchant_all[
    (
        merchant_all[
            "joint_family"
        ]
        == "joint_b"
    )
    & np.isclose(
        merchant_all[
            "sweep_h2_sale_value_eur_per_kg"
        ],
        2.38,
        atol=1e-9,
        rtol=0.0,
    )
].copy()

if not joint_b_238.empty:
    joint_b_238[
        "_battery_active_numeric"
    ] = (
        battery_activity_series(
            joint_b_238
        )
        .astype(float)
    )

    active_max, inactive_min = (
        sampled_activity_bracket(
            joint_b_238,
            "joint_b",
            2.38,
            "_battery_active_numeric",
            0.5,
        )
    )

    threshold_rows.append(
        {
            "finding":
                "Battery exit near H2 entry",
            "family":
                "joint_b",
            "h2_value_eur_per_kg":
                2.38,
            "lower_bound":
                active_max,
            "upper_bound":
                inactive_min,
            "unit":
                "EUR2020/MW_BTC/a",
            "interpretation":
                (
                    "Highest sampled BTC CAPEX with "
                    "positive battery investment and "
                    "first sampled BTC CAPEX above it "
                    "with zero battery."
                ),
        }
    )


threshold_rows.append(
    {
        "finding":
            "First sampled solar-cap binding H2 value",
        "family":
            "joint, central BTC economics",
        "h2_value_eur_per_kg":
            np.nan,
        "lower_bound":
            first_solar_cap_h2,
        "upper_bound":
            first_solar_cap_h2,
        "unit":
            "EUR2020/kg_H2",
        "interpretation":
            (
                "First sampled merchant H2 value at "
                "which the solar resource envelope binds."
            ),
    }
)

threshold_rows.append(
    {
        "finding":
            "First sampled wind-cap binding H2 value",
        "family":
            "joint, central BTC economics",
        "h2_value_eur_per_kg":
            np.nan,
        "lower_bound":
            first_wind_cap_h2,
        "upper_bound":
            first_wind_cap_h2,
        "unit":
            "EUR2020/kg_H2",
        "interpretation":
            (
                "First sampled merchant H2 value at "
                "which the wind resource envelope binds."
            ),
    }
)


threshold_table = pd.DataFrame(
    threshold_rows
)

threshold_table.to_csv(
    TABLE_DIR
    / "table_merchant_thresholds.csv",
    index=False,
)

threshold_table.to_latex(
    TABLE_DIR
    / "table_merchant_thresholds.tex",
    index=False,
    na_rep="--",
    float_format="%.4f",
)


# =============================================================================
# 12. Plot helpers
# =============================================================================

def save_figure(
    fig,
    basename,
):
    png = (
        FIGURE_DIR
        / f"{basename}.png"
    )

    pdf = (
        FIGURE_DIR
        / f"{basename}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


REGIME_ORDER = [
    "none",
    "BTC",
    "BTC+battery",
    "H2",
    "H2+BTC",
    "H2+battery",
    "H2+BTC+battery",
]


REGIME_LABELS = {
    "none": "No investment",
    "BTC": "BTC",
    "BTC+battery": "BTC + battery",
    "H2": r"$\mathrm{H_2}$",
    "H2+BTC": r"$\mathrm{H_2}$ + BTC",
    "H2+battery": r"$\mathrm{H_2}$ + battery",
    "H2+BTC+battery": (
        r"$\mathrm{H_2}$ + BTC + battery"
    ),
}


# Fixed semantic colours are used so that the same regime
# has the same appearance in both panels.
REGIME_COLORS = {
    "none": "#D9D9D9",
    "BTC": "#E69F00",
    "BTC+battery": "#D55E00",
    "H2": "#0072B2",
    "H2+BTC": "#009E73",
    "H2+battery": "#CC79A7",
    "H2+BTC+battery": "#56B4E9",
}


def format_h2_tick(
    value,
):
    return (
        f"{float(value):.3f}"
        .rstrip("0")
        .rstrip(".")
    )


def format_btc_capex_tick(
    value_eur_per_mw_year,
):
    value_k = (
        float(
            value_eur_per_mw_year
        )
        / 1000.0
    )

    if np.isclose(
        value_k,
        round(value_k),
    ):
        return f"{value_k:.0f}"

    return f"{value_k:.1f}"


def plot_merchant_regime_comparison(
    merchant_df,
):
    """
    Compare sampled merchant operating regimes for JOINT and
    JOINT-B using one square per solved optimization case.

    Blank locations represent parameter combinations that were
    not evaluated. No interpolation between cases is implied.
    """

    required_columns = {
        "joint_family",
        "sweep_h2_sale_value_eur_per_kg",
        "sweep_btc_capex_eur_per_mw_year",
        "merchant_regime",
    }

    missing = (
        required_columns
        - set(
            merchant_df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Merchant regime comparison is missing "
            f"required columns: {sorted(missing)}"
        )

    plot_df = (
        merchant_df[
            [
                "joint_family",
                "sweep_h2_sale_value_eur_per_kg",
                "sweep_btc_capex_eur_per_mw_year",
                "merchant_regime",
            ]
        ]
        .dropna(
            subset=[
                "joint_family",
                "sweep_h2_sale_value_eur_per_kg",
                "sweep_btc_capex_eur_per_mw_year",
                "merchant_regime",
            ]
        )
        .drop_duplicates(
            subset=[
                "joint_family",
                "sweep_h2_sale_value_eur_per_kg",
                "sweep_btc_capex_eur_per_mw_year",
            ],
            keep="last",
        )
        .copy()
    )

    if plot_df.empty:
        raise RuntimeError(
            "No merchant cases are available "
            "for the regime comparison."
        )

    # --------------------------------------------------------
    # Treat both axes as sampled parameter categories.
    #
    # This is intentional:
    # the figure shows evaluated optimization cases,
    # not a continuous interpolated parameter surface.
    # --------------------------------------------------------

    x_values = sorted(
        plot_df[
            "sweep_btc_capex_eur_per_mw_year"
        ]
        .astype(float)
        .unique()
    )

    y_values = sorted(
        plot_df[
            "sweep_h2_sale_value_eur_per_kg"
        ]
        .astype(float)
        .unique()
    )

    x_position = {
        value: position
        for position, value
        in enumerate(
            x_values
        )
    }

    y_position = {
        value: position
        for position, value
        in enumerate(
            y_values
        )
    }

    plot_df[
        "_x_position"
    ] = (
        plot_df[
            "sweep_btc_capex_eur_per_mw_year"
        ]
        .astype(float)
        .map(
            x_position
        )
    )

    plot_df[
        "_y_position"
    ] = (
        plot_df[
            "sweep_h2_sale_value_eur_per_kg"
        ]
        .astype(float)
        .map(
            y_position
        )
    )

    occurring_regimes = set(
        plot_df[
            "merchant_regime"
        ].astype(str)
    )

    present_regimes = [
        regime
        for regime
        in REGIME_ORDER
        if regime
        in occurring_regimes
    ]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            13.0,
            6.3,
        ),
        sharex=True,
        sharey=True,
    )

    panels = [
        (
            "joint",
            "JOINT",
        ),
        (
            "joint_b",
            "JOINT-B",
        ),
    ]

    for ax, (
        family,
        panel_title,
    ) in zip(
        axes,
        panels,
    ):

        family_df = (
            plot_df[
                plot_df[
                    "joint_family"
                ]
                == family
            ]
            .copy()
        )

        if family_df.empty:
            raise RuntimeError(
                "No merchant cases found for "
                f"{family!r}."
            )

        for regime in (
            present_regimes
        ):
            subset = family_df[
                family_df[
                    "merchant_regime"
                ]
                == regime
            ]

            if subset.empty:
                continue

            ax.scatter(
                subset[
                    "_x_position"
                ],
                subset[
                    "_y_position"
                ],
                marker="s",
                s=245,
                facecolor=(
                    REGIME_COLORS[
                        regime
                    ]
                ),
                edgecolor="black",
                linewidth=0.65,
                zorder=3,
            )

        ax.set_title(
            panel_title,
            fontsize=13,
        )

        ax.set_xlabel(
            (
                "Annualized BTC capacity cost\n"
                r"[$\mathrm{kEUR}_{2020}/"
                r"(\mathrm{MW}\,\mathrm{a})$]"
            )
        )

        ax.set_xticks(
            range(
                len(
                    x_values
                )
            )
        )

        ax.set_xticklabels(
            [
                format_btc_capex_tick(
                    value
                )
                for value
                in x_values
            ],
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            range(
                len(
                    y_values
                )
            )
        )

        ax.set_yticklabels(
            [
                format_h2_tick(
                    value
                )
                for value
                in y_values
            ]
        )

        # Grid lines identify sampled parameter positions.
        ax.grid(
            True,
            which="major",
            linewidth=0.65,
            alpha=0.25,
        )

        ax.set_axisbelow(
            True
        )

        ax.set_xlim(
            -0.65,
            len(
                x_values
            )
            - 0.35,
        )

        ax.set_ylim(
            -0.55,
            len(
                y_values
            )
            - 0.45,
        )

    axes[0].set_ylabel(
        (
            "Hydrogen sale value "
            r"[$\mathrm{EUR}_{2020}/"
            r"\mathrm{kg}_{H_2}$]"
        )
    )

    # --------------------------------------------------------
    # Shared regime legend
    # --------------------------------------------------------

    legend_handles = []

    for regime in (
        present_regimes
    ):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=9,
                markerfacecolor=(
                    REGIME_COLORS[
                        regime
                    ]
                ),
                markeredgecolor="black",
                markeredgewidth=0.65,
                label=(
                    REGIME_LABELS[
                        regime
                    ]
                ),
            )
        )

    fig.legend(
        handles=legend_handles,
        title="Optimal regime",
        loc="center left",
        bbox_to_anchor=(
            0.815,
            0.52,
        ),
        frameon=False,
    )

    # --------------------------------------------------------
    # Common title and sampling note
    # --------------------------------------------------------

    fig.suptitle(
        (
            "Sampled merchant operating regimes: "
            "effect of battery availability"
        ),
        fontsize=15,
        y=0.965,
    )

    fig.text(
        0.43,
        0.025,
        (
            "Each square represents one solved optimization case. "
            "Blank positions were not evaluated; "
            "no interpolation between sampled cases is implied."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.80,
        bottom=0.20,
        top=0.86,
        wspace=0.08,
    )

    save_figure(
        fig,
        "fig_merchant_regime_comparison",
    )


plot_merchant_regime_comparison(
    merchant_all
)



# =============================================================================
# S0-S3 renewable curtailment and utilization comparison
# =============================================================================

print(
    "\n===== S0-S3 CURTAILMENT / RENEWABLE UTILIZATION ====="
)

scenario_paths = {
    "S0": Path("results/s0_thesis_2045/curtailment.csv"),
    "S1": Path("results/s1_thesis_2045/curtailment.csv"),
    "S2": Path("results/s2_thesis_2045/curtailment.csv"),
    "S3": Path("results/s3_thesis_2045/curtailment.csv"),
}

records = []

for scenario, csv_path in scenario_paths.items():

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing curtailment file: {csv_path}"
        )

    curtailment = pd.read_csv(csv_path)

    required_columns = {
        "asset",
        "available_mwh",
        "actual_mwh",
        "curtailed_mwh",
        "curtailment_rate",
    }

    missing = required_columns.difference(
        curtailment.columns
    )

    if missing:
        raise RuntimeError(
            f"{csv_path} is missing columns: "
            f"{sorted(missing)}"
        )

    solar = curtailment.loc[
        curtailment["asset"]
        .astype(str)
        .str.lower()
        == "solar"
    ]

    wind = curtailment.loc[
        curtailment["asset"]
        .astype(str)
        .str.lower()
        == "wind"
    ]

    if len(solar) != 1 or len(wind) != 1:
        raise RuntimeError(
            f"Expected exactly one solar and one wind row in {csv_path}."
        )

    solar = solar.iloc[0]
    wind = wind.iloc[0]

    total_available_mwh = (
        float(solar["available_mwh"])
        + float(wind["available_mwh"])
    )

    total_actual_mwh = (
        float(solar["actual_mwh"])
        + float(wind["actual_mwh"])
    )

    total_curtailed_mwh = (
        float(solar["curtailed_mwh"])
        + float(wind["curtailed_mwh"])
    )

    renewable_utilization_pct = (
        100.0
        * total_actual_mwh
        / total_available_mwh
        if total_available_mwh > 0
        else np.nan
    )

    records.append(
        {
            "scenario": scenario,
            "solar_curtailment_pct":
                100.0 * float(solar["curtailment_rate"]),
            "wind_curtailment_pct":
                100.0 * float(wind["curtailment_rate"]),
            "total_curtailment_gwh":
                total_curtailed_mwh / 1000.0,
            "renewable_utilization_pct":
                renewable_utilization_pct,
        }
    )


curtailment_comparison = pd.DataFrame(
    records
)

scenario_order = [
    "S0",
    "S1",
    "S2",
    "S3",
]

curtailment_comparison[
    "scenario"
] = pd.Categorical(
    curtailment_comparison["scenario"],
    categories=scenario_order,
    ordered=True,
)

curtailment_comparison = (
    curtailment_comparison
    .sort_values("scenario")
    .reset_index(drop=True)
)

curtailment_comparison.to_csv(
    TABLE_DIR
    / "table_s0_s3_curtailment_utilization.csv",
    index=False,
)

print(
    curtailment_comparison.to_string(
        index=False
    )
)


x = np.arange(
    len(curtailment_comparison)
)

width = 0.34

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 5.6),
)


# -------------------------------------------------------------------------
# Panel A — technology-specific curtailment
# -------------------------------------------------------------------------

ax_left = axes[0]

solar_bars = ax_left.bar(
    x - width / 2,
    curtailment_comparison[
        "solar_curtailment_pct"
    ],
    width,
    label="Solar PV",
)

wind_bars = ax_left.bar(
    x + width / 2,
    curtailment_comparison[
        "wind_curtailment_pct"
    ],
    width,
    label="Wind",
)

ax_left.set_title(
    "Renewable curtailment"
)

ax_left.set_ylabel(
    "Curtailment [% of available generation]"
)

ax_left.set_xticks(
    x
)

ax_left.set_xticklabels(
    scenario_order
)

ax_left.grid(
    axis="y",
    alpha=0.25,
)

ax_left.set_axisbelow(
    True
)

for bars in [
    solar_bars,
    wind_bars,
]:
    for bar in bars:

        value = bar.get_height()

        ax_left.annotate(
            f"{value:.1f}",
            (
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


# -------------------------------------------------------------------------
# Panel B — total renewable utilization
# -------------------------------------------------------------------------

ax_right = axes[1]

util_bars = ax_right.bar(
    x,
    curtailment_comparison[
        "renewable_utilization_pct"
    ],
    width=0.55,
)

ax_right.set_title(
    "Overall renewable utilization"
)

ax_right.set_ylabel(
    "Renewable utilization [%]"
)

ax_right.set_xticks(
    x
)

ax_right.set_xticklabels(
    scenario_order
)

ax_right.set_ylim(
    0,
    100,
)

ax_right.grid(
    axis="y",
    alpha=0.25,
)

ax_right.set_axisbelow(
    True
)

for bar in util_bars:

    value = bar.get_height()

    ax_right.annotate(
        f"{value:.1f}%",
        (
            bar.get_x()
            + bar.get_width() / 2,
            value,
        ),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )


# -------------------------------------------------------------------------
# Shared title and legend
# -------------------------------------------------------------------------

fig.suptitle(
    "Renewable curtailment and utilization across central scenarios",
    fontsize=15,
    y=0.97,
)

legend_handles = [
    solar_bars[0],
    wind_bars[0],
]

legend_labels = [
    "Solar PV",
    "Wind",
]

ax_left.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=2,
    frameon=False,
)


# -------------------------------------------------------------------------
# Explanatory note
# -------------------------------------------------------------------------

fig.text(
    0.5,
    0.03,
    (
        "Curtailment is calculated relative to available solar and wind "
        "generation. Renewable utilization is actual renewable generation "
        "divided by total available renewable generation."
    ),
    ha="center",
    va="bottom",
    fontsize=9.0,
)


# Explicitly reserve space below the plots for the legend and note.
fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.25,
    top=0.84,
    wspace=0.28,
)


save_figure(
    fig,
    "fig_curtailment_utilization_s0_s3",
)


# =============================================================================
# 13. BTC break-even estimate from sampled viability brackets
# =============================================================================

print(
    "\n===== BTC BREAK-EVEN ESTIMATES ====="
)

break_even_records = []

for family in [
    "joint",
    "joint_b",
]:

    h2_values = sorted(
        merchant_all.loc[
            merchant_all[
                "joint_family"
            ]
            == family,
            "sweep_h2_sale_value_eur_per_kg",
        ]
        .dropna()
        .astype(float)
        .unique()
    )

    for h2_value in h2_values:

        highest_active_capex, first_inactive_capex = (
            sampled_activity_bracket(
                merchant_all,
                family,
                float(h2_value),
                "bitcoin_capacity_mw",
                BTC_ACTIVE_TOL_MW,
            )
        )

        if (
            not np.isfinite(highest_active_capex)
            or not np.isfinite(first_inactive_capex)
        ):
            continue

        highest_active_capex = float(
            highest_active_capex
        )

        first_inactive_capex = float(
            first_inactive_capex
        )

        if (
            first_inactive_capex
            <= highest_active_capex
        ):
            continue

        midpoint_capex = (
            0.5
            * (
                highest_active_capex
                + first_inactive_capex
            )
        )

        break_even_records.append(
            {
                "joint_family":
                    family,

                "h2_value_eur_per_kg":
                    float(h2_value),

                "highest_active_btc_capex_eur_per_mw_year":
                    highest_active_capex,

                "first_inactive_btc_capex_eur_per_mw_year":
                    first_inactive_capex,

                "estimated_break_even_midpoint_eur_per_mw_year":
                    midpoint_capex,

                "highest_active_btc_capex_keur_per_mw_year":
                    highest_active_capex / 1000.0,

                "first_inactive_btc_capex_keur_per_mw_year":
                    first_inactive_capex / 1000.0,

                "estimated_break_even_midpoint_keur_per_mw_year":
                    midpoint_capex / 1000.0,
            }
        )


btc_break_even = pd.DataFrame(
    break_even_records
)

if btc_break_even.empty:
    raise RuntimeError(
        "No complete BTC break-even brackets available."
    )

btc_break_even = (
    btc_break_even
    .sort_values(
        [
            "h2_value_eur_per_kg",
            "joint_family",
        ]
    )
    .reset_index(
        drop=True
    )
)

btc_break_even.to_csv(
    TABLE_DIR
    / "btc_viability_boundary_points.csv",
    index=False,
)

print(
    btc_break_even[
        [
            "joint_family",
            "h2_value_eur_per_kg",
            "highest_active_btc_capex_keur_per_mw_year",
            "estimated_break_even_midpoint_keur_per_mw_year",
            "first_inactive_btc_capex_keur_per_mw_year",
        ]
    ].to_string(
        index=False
    )
)


# -------------------------------------------------------------------------
# BTC break-even midpoint sensitivity figure
# -------------------------------------------------------------------------
#
# The sampled lower and upper viability bounds remain preserved in
# btc_viability_boundary_points.csv. For the main thesis figure, only
# their arithmetic midpoint is visualized in order to communicate the
# economic sensitivity directly and avoid an unnecessarily complex
# bracket representation.
#
# Lines between markers are visual guides only and do not represent
# interpolation or additional solved model cases.
# -------------------------------------------------------------------------

h2_plot_values = sorted(
    btc_break_even[
        "h2_value_eur_per_kg"
    ]
    .astype(float)
    .unique()
)

x_positions = np.arange(
    len(h2_plot_values)
)

x_lookup = {
    value: position
    for position, value
    in enumerate(h2_plot_values)
}


fig, ax = plt.subplots(
    figsize=(8.8, 5.4)
)


# -------------------------------------------------------------------------
# JOINT
# -------------------------------------------------------------------------

joint_plot = (
    btc_break_even[
        btc_break_even[
            "joint_family"
        ]
        == "joint"
    ]
    .sort_values(
        "h2_value_eur_per_kg"
    )
)

joint_x = [
    x_lookup[
        float(value)
    ]
    for value
    in joint_plot[
        "h2_value_eur_per_kg"
    ]
]

joint_y = (
    joint_plot[
        "estimated_break_even_midpoint_keur_per_mw_year"
    ]
    .astype(float)
    .to_numpy()
)

ax.plot(
    joint_x,
    joint_y,
    marker="o",
    markersize=8,
    linewidth=1.8,
    linestyle="--",
    color="#0072B2",
    markerfacecolor="#0072B2",
    markeredgecolor="black",
    markeredgewidth=0.8,
    label="JOINT",
    zorder=3,
)


# -------------------------------------------------------------------------
# JOINT-B
# -------------------------------------------------------------------------

joint_b_plot = (
    btc_break_even[
        btc_break_even[
            "joint_family"
        ]
        == "joint_b"
    ]
    .sort_values(
        "h2_value_eur_per_kg"
    )
)

joint_b_x = [
    x_lookup[
        float(value)
    ]
    for value
    in joint_b_plot[
        "h2_value_eur_per_kg"
    ]
]

joint_b_y = (
    joint_b_plot[
        "estimated_break_even_midpoint_keur_per_mw_year"
    ]
    .astype(float)
    .to_numpy()
)

ax.plot(
    joint_b_x,
    joint_b_y,
    marker="s",
    markersize=9,
    linewidth=1.8,
    linestyle=":",
    color="#009E73",
    markerfacecolor="white",
    markeredgecolor="#009E73",
    markeredgewidth=2.0,
    label="JOINT-B",
    zorder=4,
)


# -------------------------------------------------------------------------
# Axes
# -------------------------------------------------------------------------

ax.set_title(
    "Estimated BTC break-even capacity cost"
)

ax.set_xlabel(
    (
        "Hydrogen sale value "
        r"[$\mathrm{EUR}_{2020}/"
        r"\mathrm{kg}_{H_2}$]"
    )
)

ax.set_ylabel(
    (
        "Estimated BTC break-even capacity cost "
        r"[$\mathrm{kEUR}_{2020}/"
        r"(\mathrm{MW}\,\mathrm{a})$]"
    )
)


ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [
        (
            f"{value:.3f}"
            .rstrip("0")
            .rstrip(".")
        )
        for value
        in h2_plot_values
    ]
)


ax.grid(
    axis="y",
    linewidth=0.7,
    alpha=0.25,
)

ax.set_axisbelow(
    True
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


ax.legend(
    frameon=False,
    loc="best",
)


# -------------------------------------------------------------------------
# Interpretation note
# -------------------------------------------------------------------------

fig.text(
    0.5,
    0.015,
    (
        "Markers show arithmetic midpoints of sampled BTC viability "
        "brackets. Lines are visual guides only; exact break-even "
        "thresholds were not solved directly."
    ),
    ha="center",
    va="bottom",
    fontsize=9.2,
)


fig.subplots_adjust(
    left=0.13,
    right=0.97,
    bottom=0.19,
    top=0.90,
)


save_figure(
    fig,
    "fig_btc_viability_boundary",
)


# =============================================================================
# 14. H2-BTC trade-off figure
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.6, 5.7)
)


tradeoff_styles = {
    150000.0: {
        "label": (
            r"BTC CAPEX = "
            r"$150\,\mathrm{kEUR}_{2020}/"
            r"(\mathrm{MW}\,\mathrm{a})$"
        ),
        "color": "#0072B2",
        "marker": "o",
        "annotation_offset": (
            6,
            6,
        ),
    },
    200000.0: {
        "label": (
            r"BTC CAPEX = "
            r"$200\,\mathrm{kEUR}_{2020}/"
            r"(\mathrm{MW}\,\mathrm{a})$"
        ),
        "color": "#D55E00",
        "marker": "s",
        "annotation_offset": (
            6,
            -15,
        ),
    },
}


annotation_values = {
    2.38,
    2.50,
    2.75,
}


for btc_capex in [
    150000.0,
    200000.0,
]:
    subset = merchant_all[
        (
            merchant_all[
                "joint_family"
            ]
            == "joint"
        )
        & np.isclose(
            merchant_all[
                "sweep_btc_capex_eur_per_mw_year"
            ],
            btc_capex,
            rtol=0.0,
            atol=1e-6,
        )
    ].copy()

    subset = subset.sort_values(
        "sweep_h2_sale_value_eur_per_kg"
    )

    if subset.empty:
        continue

    style = (
        tradeoff_styles[
            btc_capex
        ]
    )

    # Thin dashed line is only a visual guide between sampled
    # optimization outcomes.
    ax.plot(
        subset[
            "hydrogen_delivered_kt"
        ],
        subset[
            "bitcoin_capacity_mw"
        ],
        linestyle="--",
        linewidth=1.1,
        alpha=0.55,
        color=style[
            "color"
        ],
        zorder=1,
    )

    ax.scatter(
        subset[
            "hydrogen_delivered_kt"
        ],
        subset[
            "bitcoin_capacity_mw"
        ],
        s=62,
        marker=style[
            "marker"
        ],
        facecolor=style[
            "color"
        ],
        edgecolor="black",
        linewidth=0.6,
        label=style[
            "label"
        ],
        zorder=3,
    )

    for _, row in (
        subset.iterrows()
    ):
        h2_value = float(
            row[
                "sweep_h2_sale_value_eur_per_kg"
            ]
        )

        if not any(
            np.isclose(
                h2_value,
                target,
                rtol=0.0,
                atol=1e-9,
            )
            for target
            in annotation_values
        ):
            continue

        btc_capacity = float(
            row[
                "bitcoin_capacity_mw"
            ]
        )

        # Keep annotations above near-zero BTC points so
        # labels are not clipped below the x-axis.
        if btc_capacity < 100.0:
            annotation_offset = (
                8,
                10,
            )
        else:
            annotation_offset = style[
                "annotation_offset"
            ]

        ax.annotate(
            (
                f"{h2_value:.2f} "
                r"$\mathrm{EUR}_{2020}/"
                r"\mathrm{kg}_{H_2}$"
            ),
            (
                row[
                    "hydrogen_delivered_kt"
                ],
                btc_capacity,
            ),
            xytext=annotation_offset,
            textcoords="offset points",
            fontsize=8.5,
        )


ax.set_xlabel(
    (
        "Annual hydrogen production "
        r"[$\mathrm{kt}_{H_2}/\mathrm{a}$]"
    )
)

ax.set_ylabel(
    "Optimized BTC capacity [MW]"
)

ax.set_title(
    (
        r"Sampled $\mathrm{H_2}$–BTC "
        "capacity trade-off"
    )
)


ax.grid(
    True,
    linewidth=0.7,
    alpha=0.25,
)

ax.set_axisbelow(
    True
)

ax.legend(
    frameon=False,
)


fig.text(
    0.5,
    0.015,
    (
        "Markers are solved optimization cases; "
        "dashed lines are visual guides only "
        "and do not represent interpolation."
    ),
    ha="center",
    va="bottom",
    fontsize=9.5,
)


fig.subplots_adjust(
    left=0.12,
    right=0.97,
    bottom=0.19,
    top=0.90,
)


save_figure(
    fig,
    "fig_h2_btc_tradeoff",
)


# =============================================================================
# 15. Battery-selection map
# =============================================================================

joint_b = merchant_all[
    merchant_all[
        "joint_family"
    ]
    == "joint_b"
].copy()


joint_b[
    "battery_selected"
] = (
    battery_activity_series(
        joint_b
    )
    .astype(bool)
)


x_values = sorted(
    joint_b[
        "sweep_btc_capex_eur_per_mw_year"
    ]
    .astype(float)
    .unique()
)

y_values = sorted(
    joint_b[
        "sweep_h2_sale_value_eur_per_kg"
    ]
    .astype(float)
    .unique()
)


x_position = {
    value: position
    for position, value
    in enumerate(
        x_values
    )
}

y_position = {
    value: position
    for position, value
    in enumerate(
        y_values
    )
}


joint_b[
    "_x_position"
] = (
    joint_b[
        "sweep_btc_capex_eur_per_mw_year"
    ]
    .astype(float)
    .map(
        x_position
    )
)

joint_b[
    "_y_position"
] = (
    joint_b[
        "sweep_h2_sale_value_eur_per_kg"
    ]
    .astype(float)
    .map(
        y_position
    )
)


fig, ax = plt.subplots(
    figsize=(9.5, 5.8)
)


# Solved cases with no battery:
no_battery = joint_b[
    ~joint_b[
        "battery_selected"
    ]
]

ax.scatter(
    no_battery[
        "_x_position"
    ],
    no_battery[
        "_y_position"
    ],
    marker="s",
    s=210,
    facecolor="#D9D9D9",
    edgecolor="black",
    linewidth=1.0,
    label="No battery selected",
    zorder=3,
)


# Solved cases with positive battery investment:
with_battery = joint_b[
    joint_b[
        "battery_selected"
    ]
]

ax.scatter(
    with_battery[
        "_x_position"
    ],
    with_battery[
        "_y_position"
    ],
    marker="s",
    s=210,
    facecolor="#0072B2",
    edgecolor="black",
    linewidth=0.8,
    label="Battery selected",
    zorder=3,
)


ax.set_xticks(
    range(
        len(
            x_values
        )
    )
)

ax.set_xticklabels(
    [
        format_btc_capex_tick(
            value
        )
        for value
        in x_values
    ],
    rotation=45,
    ha="right",
)


ax.set_yticks(
    range(
        len(
            y_values
        )
    )
)

ax.set_yticklabels(
    [
        format_h2_tick(
            value
        )
        for value
        in y_values
    ]
)


ax.set_xlabel(
    (
        "Annualized BTC capacity cost "
        r"[$\mathrm{kEUR}_{2020}/"
        r"(\mathrm{MW}\,\mathrm{a})$]"
    )
)

ax.set_ylabel(
    (
        "Hydrogen sale value "
        r"[$\mathrm{EUR}_{2020}/"
        r"\mathrm{kg}_{H_2}$]"
    )
)

ax.set_title(
    "Sampled battery-investment outcomes — JOINT-B"
)


ax.set_xlim(
    -0.65,
    len(
        x_values
    )
    - 0.35,
)

ax.set_ylim(
    -0.55,
    len(
        y_values
    )
    - 0.45,
)


ax.grid(
    True,
    linewidth=0.65,
    alpha=0.25,
)

ax.set_axisbelow(
    True
)


ax.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(
        1.01,
        1.0,
    ),
    borderaxespad=0.0,
)


fig.text(
    0.5,
    0.015,
    (
        "Squares represent solved optimization cases. "
        "Blank positions were not evaluated."
    ),
    ha="center",
    va="bottom",
    fontsize=9.5,
)


fig.subplots_adjust(
    left=0.12,
    right=0.78,
    bottom=0.21,
    top=0.90,
)


save_figure(
    fig,
    "fig_battery_entry_regime",
)


# =============================================================================
# 16. Manifest
# =============================================================================

manifest = OUTDIR / "final_output_manifest.txt"

manifest_lines = [
    "FINAL OFF-GRID THESIS OUTPUTS",
    "==============================",
    "",
    "No optimization was performed by this script.",
    "",
    "Input result sets:",
    "",
]

for label, path in (
    CORE_SCENARIOS.items()
):
    manifest_lines.append(
        f"- {label}: "
        f"{path.relative_to(BASE_DIR)}"
    )

for label, (
    path,
    _,
) in hmax_cases.items():
    manifest_lines.append(
        f"- {label}: "
        f"{path.relative_to(BASE_DIR)}"
    )

manifest_lines.extend(
    [
        (
            "- Merchant H2 sweep: "
            + str(
                H2_SWEEP_PATH.relative_to(
                    BASE_DIR
                )
            )
        ),
        (
            "- Main H2 x BTC grid: "
            + str(
                GRID_PATH.relative_to(
                    BASE_DIR
                )
            )
        ),
        (
            "- Refined grid: "
            + str(
                REFINED_PATH.relative_to(
                    BASE_DIR
                )
            )
        ),
        "",
        "Generated tables:",
    ]
)

for path in sorted(
    TABLE_DIR.iterdir()
):
    manifest_lines.append(
        f"- {path.relative_to(BASE_DIR)}"
    )

manifest_lines.append(
    ""
)

manifest_lines.append(
    "Generated figures:"
)

for path in sorted(
    FIGURE_DIR.iterdir()
):
    manifest_lines.append(
        f"- {path.relative_to(BASE_DIR)}"
    )

manifest.write_text(
    "\n".join(
        manifest_lines
    )
    + "\n",
    encoding="utf-8",
)


# =============================================================================
# 17. Final terminal summary
# =============================================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL THESIS RESULT CONSOLIDATION COMPLETE"
)

print(
    "============================================================"
)

print(
    f"Core + HMAX scenarios: "
    f"{len(core_hmax_table)}"
)

print(
    f"Merchant H2 sweep rows: "
    f"{len(h2_sweep)}"
)

print(
    f"Main merchant grid rows: "
    f"{len(grid)}"
)

print(
    f"Refined merchant rows: "
    f"{len(refined)}"
)

print(
    f"Unique merchant grid rows: "
    f"{len(merchant_all)}"
)

print(
    "\nMerchant sampled thresholds:"
)

print(
    threshold_table.to_string(
        index=False
    )
)

print(
    "\nCore + HMAX summary:"
)

display_columns = [
    column
    for column
    in [
        "scenario",
        "solar_capacity_mw",
        "wind_capacity_mw",
        "renewable_capacity_total_mw",
        "electrolyzer_capacity_mw",
        "battery_power_mw",
        "battery_energy_mwh",
        "bitcoin_capacity_mw",
        "hydrogen_delivered_kt",
        "net_system_cost_eur_per_year",
        "lcoh_eur_per_kg_h2",
    ]
    if column
    in core_hmax_table.columns
]

print(
    core_hmax_table[
        display_columns
    ].to_string(
        index=False
    )
)

print(
    "\nOutputs:"
)

print(
    f"  {OUTDIR.relative_to(BASE_DIR)}"
)

print(
    "\nNo model optimization was run."
)

print(
    "============================================================"
)


if __name__ == "__main__":
    pass
