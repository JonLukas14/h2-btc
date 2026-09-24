#!/usr/bin/env python3

"""
Two-dimensional merchant sensitivity for the off-grid JOINT model.

Dimensions
----------
1. Hydrogen sale value [EUR/kg_H2]
2. Annualized endogenous Bitcoin capacity cost [EUR/MW/a]

Families
--------
joint
    Merchant H2 + endogenous BTC

joint_b
    Merchant H2 + endogenous BTC + endogenous battery

The script deliberately does not define a central hydrogen sale value.
Both sensitivity dimensions must be supplied explicitly.

For every case:
    preprocess_inputs.py
    run_model.py
    analyze_results.py

Outputs
-------
- one summary.csv per case
- combined joint_h2_btc_grid_summary.csv
- one regime matrix per family
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_TEMPLATES = {
    "joint": (
        BASE_DIR
        / "configs"
        / "scenarios"
        / "scenario_joint_template_thesis_2045.yaml"
    ),
    "joint_b": (
        BASE_DIR
        / "configs"
        / "scenarios"
        / "scenario_joint_b_template_thesis_2045.yaml"
    ),
}


# Numerical activity threshold used only for regime classification.
#
# This does NOT affect the optimization itself.
CAPACITY_ACTIVITY_TOL_MW = 1e-6
HYDROGEN_ACTIVITY_TOL_KT = 1e-9


def number_tag(
    value: float,
) -> str:
    """
    Filesystem-safe deterministic numeric tag.
    """
    return (
        f"{value:.5f}"
        .rstrip("0")
        .rstrip(".")
        .replace("-", "m")
        .replace(".", "p")
    )


def run_command(
    command: list[str],
) -> None:
    print(
        "\n$ "
        + " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=BASE_DIR,
        check=True,
    )


def load_template(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing JOINT template: {path}"
        )

    cfg = yaml.safe_load(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        cfg,
        dict,
    ):
        raise ValueError(
            f"Invalid YAML mapping: {path}"
        )

    hydrogen = cfg.get(
        "hydrogen",
        {},
    )

    bitcoin = cfg.get(
        "bitcoin",
        {},
    )

    if (
        hydrogen.get("mode")
        != "economic_dispatch"
    ):
        raise ValueError(
            f"{path}: expected "
            "hydrogen.mode='economic_dispatch'."
        )

    if (
        bitcoin.get("capacity_mode")
        != "endogenous"
    ):
        raise ValueError(
            f"{path}: expected "
            "bitcoin.capacity_mode='endogenous'."
        )

    return cfg


def validate_nonnegative_finite(
    values: list[float],
    label: str,
) -> None:
    for value in values:
        if (
            not np.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                f"{label} values must be finite "
                "and non-negative. "
                f"Invalid value: {value}"
            )


def configure_case(
    template: dict,
    family: str,
    h2_value: float,
    btc_capex: float,
) -> dict:
    cfg = deepcopy(
        template
    )

    cfg[
        "hydrogen"
    ][
        "sale_value_eur_per_kg_h2"
    ] = float(
        h2_value
    )

    cfg[
        "bitcoin"
    ][
        "annualized_capacity_cost_eur_per_mw_year"
    ] = float(
        btc_capex
    )

    cfg[
        "scenario_name"
    ] = (
        f"{family}_merchant_"
        f"h2_{number_tag(h2_value)}eurkg_"
        f"btc_{number_tag(btc_capex)}eurmwyear"
    )

    return cfg


def classify_regime(
    row: pd.Series,
) -> str:
    """
    Descriptive post-processing classification only.

    No optimization constraints depend on this function.
    """

    h2_kt = float(
        row.get(
            "hydrogen_delivered_kt",
            0.0,
        )
    )

    btc_mw = float(
        row.get(
            "bitcoin_capacity_mw",
            0.0,
        )
    )

    battery_power_mw = float(
        row.get(
            "battery_power_mw",
            0.0,
        )
    )

    battery_energy_mwh = float(
        row.get(
            "battery_energy_mwh",
            0.0,
        )
    )

    h2_active = (
        np.isfinite(h2_kt)
        and h2_kt
        > HYDROGEN_ACTIVITY_TOL_KT
    )

    btc_active = (
        np.isfinite(btc_mw)
        and btc_mw
        > CAPACITY_ACTIVITY_TOL_MW
    )

    battery_active = (
        (
            np.isfinite(
                battery_power_mw
            )
            and battery_power_mw
            > CAPACITY_ACTIVITY_TOL_MW
        )
        or (
            np.isfinite(
                battery_energy_mwh
            )
            and battery_energy_mwh
            > CAPACITY_ACTIVITY_TOL_MW
        )
    )

    if (
        not h2_active
        and not btc_active
        and not battery_active
    ):
        return "none"

    parts = []

    if h2_active:
        parts.append("H2")

    if btc_active:
        parts.append("BTC")

    if battery_active:
        parts.append("battery")

    return "+".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 2D JOINT merchant sensitivity "
            "over H2 sale value and endogenous BTC "
            "capacity cost."
        )
    )

    parser.add_argument(
        "--h2-values-eur-per-kg",
        nargs="+",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--btc-capex-eur-per-mw-year",
        nargs="+",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=[
            "joint",
            "joint_b",
        ],
        default=[
            "joint",
            "joint_b",
        ],
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            BASE_DIR
            / "data"
            / "joint_h2_btc_grid"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            BASE_DIR
            / "results"
            / "joint_h2_btc_grid"
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )

    args = parser.parse_args()

    h2_values = [
        float(value)
        for value
        in args.h2_values_eur_per_kg
    ]

    btc_capex_values = [
        float(value)
        for value
        in args.btc_capex_eur_per_mw_year
    ]

    validate_nonnegative_finite(
        h2_values,
        "H2 sale value",
    )

    validate_nonnegative_finite(
        btc_capex_values,
        "BTC annualized capacity cost",
    )

    args.data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_config_dir = (
        args.output_root
        / "generated_configs"
    )

    generated_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    templates = {
        family: load_template(
            DEFAULT_TEMPLATES[
                family
            ]
        )
        for family
        in args.families
    }

    summaries = []

    total_cases = (
        len(args.families)
        * len(h2_values)
        * len(btc_capex_values)
    )

    case_counter = 0

    for family in args.families:
        template = templates[
            family
        ]

        for h2_value in h2_values:
            for btc_capex in btc_capex_values:
                case_counter += 1

                print(
                    "\n"
                    "============================================================"
                )
                print(
                    f"CASE {case_counter}/{total_cases}"
                )
                print(
                    f"Family:    {family}"
                )
                print(
                    f"H2 value:  "
                    f"{h2_value:.6f} EUR/kg_H2"
                )
                print(
                    f"BTC CAPEX: "
                    f"{btc_capex:,.2f} EUR/MW/a"
                )
                print(
                    "============================================================"
                )

                h2_tag = number_tag(
                    h2_value
                )

                btc_tag = number_tag(
                    btc_capex
                )

                case_name = (
                    f"{family}_"
                    f"h2_{h2_tag}_"
                    f"btc_{btc_tag}"
                )

                case_data_dir = (
                    args.data_root
                    / case_name
                )

                case_outdir = (
                    args.output_root
                    / case_name
                )

                case_outdir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                case_config = (
                    generated_config_dir
                    / f"{case_name}.yaml"
                )

                network_path = (
                    case_outdir
                    / "network_solved.nc"
                )

                summary_path = (
                    case_outdir
                    / "summary.csv"
                )

                cfg = configure_case(
                    template,
                    family,
                    h2_value,
                    btc_capex,
                )

                case_config.write_text(
                    yaml.safe_dump(
                        cfg,
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )

                if not (
                    args.skip_existing
                    and summary_path.exists()
                ):
                    run_command(
                        [
                            sys.executable,
                            "scripts/preprocess_inputs.py",
                            "--config",
                            str(case_config),
                            "--data-dir",
                            str(case_data_dir),
                        ]
                    )

                    run_command(
                        [
                            sys.executable,
                            "scripts/run_model.py",
                            "--config",
                            str(case_config),
                            "--data-dir",
                            str(case_data_dir),
                            "--output-network",
                            str(network_path),
                        ]
                    )

                    run_command(
                        [
                            sys.executable,
                            "scripts/analyze_results.py",
                            "--config",
                            str(case_config),
                            "--network",
                            str(network_path),
                            "--outdir",
                            str(case_outdir),
                        ]
                    )

                if not summary_path.exists():
                    raise FileNotFoundError(
                        "Missing analyzer summary: "
                        f"{summary_path}"
                    )

                summary = pd.read_csv(
                    summary_path
                )

                if len(summary) != 1:
                    raise RuntimeError(
                        "Expected one summary row in "
                        f"{summary_path}; "
                        f"found {len(summary)}."
                    )

                summary.insert(
                    0,
                    "joint_family",
                    family,
                )

                summary.insert(
                    1,
                    "sweep_h2_sale_value_eur_per_kg",
                    h2_value,
                )

                summary.insert(
                    2,
                    "sweep_btc_capex_eur_per_mw_year",
                    btc_capex,
                )

                summary.insert(
                    3,
                    "merchant_regime",
                    classify_regime(
                        summary.iloc[0]
                    ),
                )

                summaries.append(
                    summary
                )

    combined = pd.concat(
        summaries,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "joint_family",
            "sweep_h2_sale_value_eur_per_kg",
            "sweep_btc_capex_eur_per_mw_year",
        ]
    ).reset_index(
        drop=True
    )

    combined_path = (
        args.output_root
        / "joint_h2_btc_grid_summary.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    for family in args.families:
        family_df = combined[
            combined[
                "joint_family"
            ]
            == family
        ].copy()

        regime_matrix = (
            family_df
            .pivot(
                index=(
                    "sweep_h2_sale_value_eur_per_kg"
                ),
                columns=(
                    "sweep_btc_capex_eur_per_mw_year"
                ),
                values="merchant_regime",
            )
            .sort_index()
            .sort_index(
                axis=1,
            )
        )

        regime_path = (
            args.output_root
            / f"regime_matrix_{family}.csv"
        )

        regime_matrix.to_csv(
            regime_path
        )

    display_columns = [
        column
        for column
        in [
            "joint_family",
            "sweep_h2_sale_value_eur_per_kg",
            "sweep_btc_capex_eur_per_mw_year",
            "merchant_regime",
            "solar_capacity_mw",
            "wind_capacity_mw",
            "electrolyzer_capacity_mw",
            "hydrogen_delivered_kt",
            "bitcoin_capacity_mw",
            "bitcoin_consumption_mwh",
            "battery_power_mw",
            "battery_energy_mwh",
            "net_system_cost_eur_per_year",
            "objective_cost_difference_eur",
        ]
        if column
        in combined.columns
    ]

    print(
        "\n"
        "============================================================"
    )
    print(
        "JOINT H2 × BTC GRID COMPLETE"
    )
    print(
        "============================================================"
    )
    print(
        f"Cases:   {len(combined)}"
    )
    print(
        f"Summary: {combined_path}"
    )

    print(
        "\n"
        + combined[
            display_columns
        ].to_string(
            index=False
        )
    )

    for family in args.families:
        regime_path = (
            args.output_root
            / f"regime_matrix_{family}.csv"
        )

        print(
            f"\n=== REGIME MATRIX: {family} ==="
        )

        print(
            pd.read_csv(
                regime_path,
                index_col=0,
            ).to_string()
        )


if __name__ == "__main__":
    main()
