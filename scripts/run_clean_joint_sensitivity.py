#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]

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

DEFAULT_H2_VALUES = [
    0.0,
    2.38,
    2.75,
]

DEFAULT_BTC_CAPEX_VALUES = [
    150000.0,
    200000.0,
    225000.0,
    250000.0,
    300000.0,
    380525.87,
]

BTC_ACTIVITY_TOL_MW = 1e-6
H2_ACTIVITY_TOL_KT = 1e-9
BATTERY_ACTIVITY_TOL_MW = 1e-6
BATTERY_ACTIVITY_TOL_MWH = 1e-6


def number_tag(value: float) -> str:
    return (
        f"{value:.5f}"
        .rstrip("0")
        .rstrip(".")
        .replace("-", "m")
        .replace(".", "p")
    )


def validate_values(
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
                f"and non-negative. Invalid value: {value}"
            )


def load_template(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing template: {path}"
        )

    cfg = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Invalid YAML mapping: {path}"
        )

    if (
        cfg.get("hydrogen", {}).get("mode")
        != "economic_dispatch"
    ):
        raise ValueError(
            f"{path}: expected "
            "hydrogen.mode='economic_dispatch'."
        )

    if (
        cfg.get("bitcoin", {}).get("capacity_mode")
        != "endogenous"
    ):
        raise ValueError(
            f"{path}: expected "
            "bitcoin.capacity_mode='endogenous'."
        )

    return cfg


def configure_case(
    template: dict,
    family: str,
    h2_value: float,
    btc_capex: float,
) -> dict:
    cfg = deepcopy(template)

    cfg["hydrogen"][
        "sale_value_eur_per_kg_h2"
    ] = float(h2_value)

    cfg["bitcoin"][
        "annualized_capacity_cost_eur_per_mw_year"
    ] = float(btc_capex)

    cfg["scenario_name"] = (
        f"{family}_merchant_"
        f"h2_{number_tag(h2_value)}eurkg_"
        f"btc_{number_tag(btc_capex)}eurmwyear"
    )

    return cfg


def classify_regime(row: pd.Series) -> str:
    h2_active = (
        float(
            row.get(
                "hydrogen_delivered_kt",
                0.0,
            )
        )
        > H2_ACTIVITY_TOL_KT
    )

    btc_active = (
        float(
            row.get(
                "bitcoin_capacity_mw",
                0.0,
            )
        )
        > BTC_ACTIVITY_TOL_MW
    )

    battery_power = float(
        row.get(
            "battery_power_mw",
            0.0,
        )
    )

    battery_energy = float(
        row.get(
            "battery_energy_mwh",
            0.0,
        )
    )

    battery_active = (
        battery_power
        > BATTERY_ACTIVITY_TOL_MW
        or battery_energy
        > BATTERY_ACTIVITY_TOL_MWH
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


def read_case_summary(
    summary_path: Path,
    family: str,
    h2_value: float,
    btc_capex: float,
    config_path: Path,
    result_dir: Path,
) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing case summary: {summary_path}"
        )

    df = pd.read_csv(summary_path)

    if len(df) != 1:
        raise RuntimeError(
            f"Expected exactly one row in "
            f"{summary_path}, found {len(df)}."
        )

    df = df.copy()

    df[
        "joint_family"
    ] = family

    df[
        "sweep_h2_sale_value_eur_per_kg"
    ] = float(h2_value)

    df[
        "sweep_btc_capex_eur_per_mw_year"
    ] = float(btc_capex)

    df[
        "generated_config"
    ] = str(
        config_path.relative_to(BASE_DIR)
    )

    df[
        "result_directory"
    ] = str(
        result_dir.relative_to(BASE_DIR)
    )

    df[
        "merchant_regime"
    ] = classify_regime(
        df.iloc[0]
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean JOINT merchant sensitivity runner "
            "for the validated off-grid rebuild."
        )
    )

    parser.add_argument(
        "--h2-values-eur-per-kg",
        nargs="+",
        type=float,
        default=DEFAULT_H2_VALUES,
    )

    parser.add_argument(
        "--btc-capex-eur-per-mw-year",
        nargs="+",
        type=float,
        default=DEFAULT_BTC_CAPEX_VALUES,
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
        "--output-root",
        type=Path,
        default=(
            BASE_DIR
            / "results"
            / "clean_joint_sensitivity"
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

    validate_values(
        h2_values,
        "H2 sale value",
    )

    validate_values(
        btc_capex_values,
        "BTC CAPEX",
    )

    output_root = args.output_root.resolve()

    generated_config_dir = (
        output_root
        / "generated_configs"
    )

    case_root = (
        output_root
        / "cases"
    )

    generated_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    templates = {
        family: load_template(
            DEFAULT_TEMPLATES[family]
        )
        for family
        in args.families
    }

    total_cases = (
        len(args.families)
        * len(h2_values)
        * len(btc_capex_values)
    )

    rows = []
    case_number = 0

    combined_path = (
        output_root
        / "clean_joint_sensitivity_summary.csv"
    )

    for family in args.families:
        template = templates[family]

        for h2_value in h2_values:
            for btc_capex in btc_capex_values:
                case_number += 1

                case_tag = (
                    f"{family}"
                    f"__h2_{number_tag(h2_value)}"
                    f"__btc_{number_tag(btc_capex)}"
                )

                config_path = (
                    generated_config_dir
                    / f"{case_tag}.yaml"
                )

                result_dir = (
                    case_root
                    / case_tag
                )

                summary_path = (
                    result_dir
                    / "summary.csv"
                )

                cfg = configure_case(
                    template,
                    family,
                    h2_value,
                    btc_capex,
                )

                config_path.write_text(
                    yaml.safe_dump(
                        cfg,
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )

                print()
                print("=" * 70)
                print(
                    f"CASE {case_number}/{total_cases}"
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
                print("=" * 70)

                if (
                    args.skip_existing
                    and summary_path.exists()
                ):
                    print(
                        "Using existing solved case."
                    )

                else:
                    result_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    command = [
                        sys.executable,
                        str(
                            BASE_DIR
                            / "run.py"
                        ),
                        str(config_path),
                        "--outdir",
                        str(result_dir),
                    ]

                    print(
                        "$ "
                        + " ".join(command)
                    )

                    subprocess.run(
                        command,
                        cwd=BASE_DIR,
                        check=True,
                    )

                row = read_case_summary(
                    summary_path,
                    family,
                    h2_value,
                    btc_capex,
                    config_path,
                    result_dir,
                )

                rows.append(row)

                combined = pd.concat(
                    rows,
                    ignore_index=True,
                    sort=False,
                )

                combined = (
                    combined
                    .sort_values(
                        [
                            "joint_family",
                            "sweep_h2_sale_value_eur_per_kg",
                            "sweep_btc_capex_eur_per_mw_year",
                        ]
                    )
                    .reset_index(
                        drop=True
                    )
                )

                combined.to_csv(
                    combined_path,
                    index=False,
                )

    combined = pd.read_csv(
        combined_path
    )

    print()
    print("=" * 70)
    print("CLEAN JOINT SENSITIVITY COMPLETE")
    print("=" * 70)
    print(
        f"Cases:   {len(combined)}"
    )
    print(
        f"Summary: "
        f"{combined_path.relative_to(BASE_DIR)}"
    )

    display_columns = [
        "joint_family",
        "sweep_h2_sale_value_eur_per_kg",
        "sweep_btc_capex_eur_per_mw_year",
        "merchant_regime",
        "hydrogen_delivered_kt",
        "bitcoin_capacity_mw",
        "battery_power_mw",
        "battery_energy_mwh",
    ]

    print()
    print(
        combined[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Regime counts:")
    print(
        combined[
            "merchant_regime"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()
