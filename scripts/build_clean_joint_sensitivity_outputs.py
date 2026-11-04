#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RESULT_FILES = [
    (
        "main_grid",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity"
        / "clean_joint_sensitivity_summary.csv",
    ),
    (
        "refined_175k",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity_refined_175k"
        / "clean_joint_sensitivity_summary.csv",
    ),
    (
        "refined_212p5k",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity_refined_212p5k"
        / "clean_joint_sensitivity_summary.csv",
    ),
    (
        "refined_218p75k",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity_refined_218p75k"
        / "clean_joint_sensitivity_summary.csv",
    ),
    (
        "refined_221p875k",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity_refined_221p875k"
        / "clean_joint_sensitivity_summary.csv",
    ),
    (
        "refined_battery_162p5k",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity_refined_battery_162p5k"
        / "clean_joint_sensitivity_summary.csv",
    ),
    (
        "refined_162p5k",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity_refined_162p5k"
        / "clean_joint_sensitivity_summary.csv",
    ),
    (
        "refined_187p5k",
        BASE_DIR
        / "results"
        / "clean_joint_sensitivity_refined_187p5k"
        / "clean_joint_sensitivity_summary.csv",
    ),
]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "clean_joint_sensitivity_final"
)

BTC_ACTIVE_TOL_MW = 1e-6
BATTERY_ACTIVE_TOL = 1e-6


def load_all_cases() -> pd.DataFrame:
    frames = []

    for source_set, path in RESULT_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing sensitivity result: {path}"
            )

        df = pd.read_csv(path)
        df = df.copy()

        df["source_set"] = source_set
        df["source_file"] = str(
            path.relative_to(BASE_DIR)
        )

        frames.append(df)

    all_cases = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required = {
        "joint_family",
        "sweep_h2_sale_value_eur_per_kg",
        "sweep_btc_capex_eur_per_mw_year",
        "hydrogen_delivered_kt",
        "bitcoin_capacity_mw",
        "battery_power_mw",
        "battery_energy_mwh",
    }

    missing = required - set(all_cases.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    all_cases = (
        all_cases
        .sort_values(
            [
                "joint_family",
                "sweep_h2_sale_value_eur_per_kg",
                "sweep_btc_capex_eur_per_mw_year",
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
        .reset_index(drop=True)
    )

    all_cases["btc_active"] = (
        all_cases["bitcoin_capacity_mw"]
        .fillna(0.0)
        .astype(float)
        > BTC_ACTIVE_TOL_MW
    )

    all_cases["battery_active"] = (
        (
            all_cases["battery_power_mw"]
            .fillna(0.0)
            .astype(float)
            > BATTERY_ACTIVE_TOL
        )
        |
        (
            all_cases["battery_energy_mwh"]
            .fillna(0.0)
            .astype(float)
            > BATTERY_ACTIVE_TOL
        )
    )

    return all_cases


def subset_at(
    df: pd.DataFrame,
    family: str,
    h2_value: float,
) -> pd.DataFrame:
    subset = df[
        (
            df["joint_family"]
            == family
        )
        &
        np.isclose(
            df[
                "sweep_h2_sale_value_eur_per_kg"
            ].astype(float),
            h2_value,
            rtol=0.0,
            atol=1e-9,
        )
    ].copy()

    return subset.sort_values(
        "sweep_btc_capex_eur_per_mw_year"
    )


def sampled_btc_bracket(
    df: pd.DataFrame,
    family: str,
    h2_value: float,
):
    subset = subset_at(
        df,
        family,
        h2_value,
    )

    active = subset[
        subset["btc_active"]
    ]

    if active.empty:
        return np.nan, np.nan

    active_max = float(
        active[
            "sweep_btc_capex_eur_per_mw_year"
        ].max()
    )

    inactive_above = subset[
        (
            ~subset["btc_active"]
        )
        &
        (
            subset[
                "sweep_btc_capex_eur_per_mw_year"
            ]
            > active_max
        )
    ]

    if inactive_above.empty:
        inactive_min = np.nan
    else:
        inactive_min = float(
            inactive_above[
                "sweep_btc_capex_eur_per_mw_year"
            ].min()
        )

    return active_max, inactive_min


def battery_transition_brackets(
    df: pd.DataFrame,
):
    subset = subset_at(
        df,
        "joint_b",
        2.38,
    )

    values = subset[
        [
            "sweep_btc_capex_eur_per_mw_year",
            "battery_active",
        ]
    ].reset_index(drop=True)

    entry = (np.nan, np.nan)
    exit_ = (np.nan, np.nan)

    for i in range(1, len(values)):
        previous = values.iloc[i - 1]
        current = values.iloc[i]

        previous_capex = float(
            previous[
                "sweep_btc_capex_eur_per_mw_year"
            ]
        )

        current_capex = float(
            current[
                "sweep_btc_capex_eur_per_mw_year"
            ]
        )

        previous_active = bool(
            previous["battery_active"]
        )

        current_active = bool(
            current["battery_active"]
        )

        if (
            not previous_active
            and current_active
        ):
            entry = (
                previous_capex,
                current_capex,
            )

        if (
            previous_active
            and not current_active
        ):
            exit_ = (
                previous_capex,
                current_capex,
            )

    return entry, exit_


def build_threshold_table(
    all_cases: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    definitions = [
        (
            "Pure BTC viability",
            "joint",
            0.0,
        ),
        (
            "Pure BTC viability with battery allowed",
            "joint_b",
            0.0,
        ),
        (
            "BTC viability near H2 entry",
            "joint",
            2.38,
        ),
        (
            "BTC viability near H2 entry with battery",
            "joint_b",
            2.38,
        ),
        (
            "BTC crowd-out at H2 = 2.75 EUR/kg",
            "joint",
            2.75,
        ),
        (
            "BTC crowd-out at H2 = 2.75 EUR/kg with battery",
            "joint_b",
            2.75,
        ),
    ]

    for finding, family, h2_value in definitions:
        lower, upper = sampled_btc_bracket(
            all_cases,
            family,
            h2_value,
        )

        rows.append(
            {
                "finding": finding,
                "family": family,
                "h2_value_eur_per_kg": h2_value,
                "lower_bound_eur_per_mw_year": lower,
                "upper_bound_eur_per_mw_year": upper,
                "bracket_width_eur_per_mw_year": (
                    upper - lower
                    if (
                        np.isfinite(lower)
                        and np.isfinite(upper)
                    )
                    else np.nan
                ),
                "interpretation": (
                    "Highest sampled BTC-active CAPEX "
                    "and first sampled BTC-inactive "
                    "CAPEX above it."
                ),
            }
        )

    battery_entry, battery_exit = (
        battery_transition_brackets(
            all_cases
        )
    )

    rows.append(
        {
            "finding":
                "Battery entry near H2 = 2.38 EUR/kg",
            "family":
                "joint_b",
            "h2_value_eur_per_kg":
                2.38,
            "lower_bound_eur_per_mw_year":
                battery_entry[0],
            "upper_bound_eur_per_mw_year":
                battery_entry[1],
            "bracket_width_eur_per_mw_year":
                battery_entry[1] - battery_entry[0],
            "interpretation":
                "Last sampled zero-battery CAPEX "
                "below the active battery region and "
                "first sampled battery-active CAPEX.",
        }
    )

    rows.append(
        {
            "finding":
                "Battery exit near H2 = 2.38 EUR/kg",
            "family":
                "joint_b",
            "h2_value_eur_per_kg":
                2.38,
            "lower_bound_eur_per_mw_year":
                battery_exit[0],
            "upper_bound_eur_per_mw_year":
                battery_exit[1],
            "bracket_width_eur_per_mw_year":
                battery_exit[1] - battery_exit[0],
            "interpretation":
                "Highest sampled battery-active CAPEX "
                "and first sampled zero-battery CAPEX "
                "above it.",
        }
    )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_cases = load_all_cases()

    thresholds = build_threshold_table(
        all_cases
    )

    all_cases_path = (
        OUTPUT_DIR
        / "merchant_all_cases.csv"
    )

    threshold_path = (
        OUTPUT_DIR
        / "merchant_thresholds.csv"
    )

    all_cases.to_csv(
        all_cases_path,
        index=False,
    )

    thresholds.to_csv(
        threshold_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("CLEAN MERCHANT SENSITIVITY CONSOLIDATION")
    print("=" * 72)
    print(
        f"Unique solved cases: {len(all_cases)}"
    )
    print(
        f"All cases: "
        f"{all_cases_path.relative_to(BASE_DIR)}"
    )
    print(
        f"Thresholds: "
        f"{threshold_path.relative_to(BASE_DIR)}"
    )

    print()
    print("===== FINAL SAMPLED THRESHOLDS =====")
    print(
        thresholds.to_string(
            index=False
        )
    )

    print()
    print(
        "NOTE: Bounds are sampled brackets, "
        "not interpolated exact break-even values."
    )


if __name__ == "__main__":
    main()
