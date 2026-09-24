#!/usr/bin/env python3

"""
Run the resource-constrained merchant H2/BTC sensitivity.

The script does NOT define a central hydrogen sale price.

Hydrogen sale values must be supplied explicitly on the command line.

For each value and scenario family it performs:

    preprocess_inputs.py
    run_model.py
    analyze_results.py

and then writes one combined summary CSV.

Families:
    joint
        Merchant H2 + endogenous BTC

    joint_b
        Merchant H2 + endogenous BTC + endogenous battery
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

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


def value_tag(
    value: float,
) -> str:
    """
    Filesystem-safe deterministic hydrogen-value tag.
    """
    return (
        f"{value:.4f}"
        .replace(
            "-",
            "m",
        )
        .replace(
            ".",
            "p",
        )
    )


def run_command(
    command: list[str],
) -> None:
    print(
        "\n$ "
        + " ".join(
            command
        ),
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

    if (
        cfg.get(
            "hydrogen",
            {},
        ).get(
            "mode"
        )
        != "economic_dispatch"
    ):
        raise ValueError(
            f"{path} is not an economic_dispatch template."
        )

    return cfg


def configure_case(
    template: dict,
    family: str,
    value: float,
) -> dict:
    cfg = deepcopy(
        template
    )

    cfg[
        "hydrogen"
    ][
        "sale_value_eur_per_kg_h2"
    ] = float(
        value
    )

    cfg[
        "scenario_name"
    ] = (
        f"{family}_merchant_h2_"
        f"{value_tag(value)}eur_per_kg"
    )

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run JOINT merchant H2/BTC sensitivity "
            "for explicitly supplied H2 sale values."
        )
    )

    parser.add_argument(
        "--h2-values-eur-per-kg",
        nargs="+",
        required=True,
        type=float,
        help=(
            "Hydrogen sale values in EUR/kg_H2. "
            "No default is intentionally provided."
        ),
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
            / "joint_h2_value_sweep"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            BASE_DIR
            / "results"
            / "joint_h2_value_sweep"
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )

    args = parser.parse_args()

    values = [
        float(
            value
        )
        for value
        in args.h2_values_eur_per_kg
    ]

    for value in values:
        if (
            not pd.notna(
                value
            )
            or value < 0.0
        ):
            raise ValueError(
                "All H2 sale values must be finite "
                "and non-negative."
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

    summaries = []

    templates = {
        family: load_template(
            DEFAULT_TEMPLATES[
                family
            ]
        )
        for family
        in args.families
    }

    for family in args.families:
        template = templates[
            family
        ]

        for value in values:
            tag = value_tag(
                value
            )

            case_name = (
                f"{family}_h2value_{tag}"
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
                value,
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
                        str(
                            case_config
                        ),
                        "--data-dir",
                        str(
                            case_data_dir
                        ),
                    ]
                )

                run_command(
                    [
                        sys.executable,
                        "scripts/run_model.py",
                        "--config",
                        str(
                            case_config
                        ),
                        "--data-dir",
                        str(
                            case_data_dir
                        ),
                        "--output-network",
                        str(
                            network_path
                        ),
                    ]
                )

                run_command(
                    [
                        sys.executable,
                        "scripts/analyze_results.py",
                        "--config",
                        str(
                            case_config
                        ),
                        "--network",
                        str(
                            network_path
                        ),
                        "--outdir",
                        str(
                            case_outdir
                        ),
                    ]
                )

            if not summary_path.exists():
                raise FileNotFoundError(
                    f"Missing analyzer summary: {summary_path}"
                )

            summary = pd.read_csv(
                summary_path
            )

            if len(
                summary
            ) != 1:
                raise RuntimeError(
                    f"Expected one summary row in "
                    f"{summary_path}, got {len(summary)}."
                )

            summary.insert(
                0,
                "joint_family",
                family,
            )

            summary.insert(
                1,
                "sweep_h2_sale_value_eur_per_kg",
                value,
            )

            summaries.append(
                summary
            )

    combined = pd.concat(
        summaries,
        ignore_index=True,
    )

    combined_path = (
        args.output_root
        / "joint_h2_value_sweep_summary.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    print(
        "\n============================================================"
    )
    print(
        "JOINT H2-VALUE SWEEP COMPLETE"
    )
    print(
        "============================================================"
    )
    print(
        f"Cases:    {len(combined)}"
    )
    print(
        f"Summary:  {combined_path}"
    )

    display_columns = [
        column
        for column
        in [
            "joint_family",
            "sweep_h2_sale_value_eur_per_kg",
            "solar_capacity_mw",
            "wind_capacity_mw",
            "electrolyzer_capacity_mw",
            "battery_power_mw",
            "battery_energy_mwh",
            "bitcoin_capacity_mw",
            "hydrogen_delivered_kt",
            "bitcoin_consumption_mwh",
            "hydrogen_gross_revenue_eur_per_year",
            "bitcoin_gross_revenue_eur_per_year",
            "gross_system_expenditure_eur_per_year",
            "net_system_cost_eur_per_year",
            "objective_cost_difference_eur",
        ]
        if column
        in combined.columns
    ]

    print(
        "\n"
        + combined[
            display_columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
