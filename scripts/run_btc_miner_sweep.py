#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Default miner sweep
# =============================================================================
#
# The grid is deliberately dense at low miner counts because we want to
# identify when BTC first begins to alter the renewable-system optimum.
#
# 1567 and 1568 miners are both included because the existing 6 MW S2/S3
# case lies between those two complete-miner counts.
#
DEFAULT_MINERS = [
    0,
    1,
    10,
    25,
    50,
    100,
    250,
    500,
    750,
    1000,
    1250,
    1500,
    1567,
    1568,
    2000,
    2500,
    3000,
]

# Reference machine used consistently in run_model.py:
# Bitmain Antminer S21 XP air-cooled.
BTC_HASHRATE_TH_S = 270.0


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed Bitcoin-miner-count sweeps for "
            "the S2/S3 thesis architectures."
        )
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=[
            "no_battery",
            "battery",
        ],
        default=[
            "no_battery",
            "battery",
        ],
        help=(
            "Sweep without battery, with battery, or both."
        ),
    )

    parser.add_argument(
        "--miners",
        nargs="+",
        type=int,
        default=DEFAULT_MINERS,
        help="Integer miner counts to evaluate.",
    )

    parser.add_argument(
        "--s0-config",
        default=(
            "configs/scenarios/"
            "scenario_s0_thesis_2045.yaml"
        ),
    )

    parser.add_argument(
        "--s1-config",
        default=(
            "configs/scenarios/"
            "scenario_s1_thesis_2045.yaml"
        ),
    )

    parser.add_argument(
        "--s2-template",
        default=(
            "configs/scenarios/"
            "scenario_s2_thesis_2045.yaml"
        ),
    )

    parser.add_argument(
        "--s3-template",
        default=(
            "configs/scenarios/"
            "scenario_s3_thesis_2045.yaml"
        ),
    )

    parser.add_argument(
        "--btc-investment-config",
        default=(
            "configs/scenarios/"
            "scenario_s2e_thesis_2045.yaml"
        ),
        help=(
            "Config from which the central annualized "
            "BTC investment cost is read."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="results/btc_miner_sweep",
    )

    parser.add_argument(
        "--data-root",
        default="data/btc_miner_sweep",
    )

    parser.add_argument(
        "--res-tolerance-mw",
        type=float,
        default=1e-3,
        help=(
            "Minimum renewable-capacity increase counted "
            "as a real investment response."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse a case if its summary.csv already exists."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def load_yaml(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def save_yaml(
    path,
    cfg,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def run(
    cmd,
    cwd,
):
    print(
        "\nRUN:",
        " ".join(
            map(
                str,
                cmd,
            )
        ),
        flush=True,
    )

    subprocess.run(
        [
            str(x)
            for x in cmd
        ],
        cwd=cwd,
        check=True,
    )


def read_summary(
    path,
):
    df = pd.read_csv(
        path
    )

    if len(df) != 1:
        raise RuntimeError(
            f"{path}: expected one summary row, "
            f"found {len(df)}."
        )

    return df.iloc[0]


def run_pipeline(
    root,
    config,
    data_dir,
    result_dir,
    resume,
):
    summary_path = (
        result_dir
        / "summary.csv"
    )

    if (
        resume
        and summary_path.exists()
    ):
        print(
            f"Reusing {summary_path}"
        )

        return read_summary(
            summary_path
        )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    network_path = (
        result_dir
        / "network_solved.nc"
    )

    # -------------------------------------------------------------------------
    # 1. Preprocess
    # -------------------------------------------------------------------------

    run(
        [
            sys.executable,
            "scripts/preprocess_inputs.py",
            "--config",
            config,
            "--data-dir",
            data_dir,
        ],
        root,
    )

    # -------------------------------------------------------------------------
    # 2. Solve
    # -------------------------------------------------------------------------

    run(
        [
            sys.executable,
            "scripts/run_model.py",
            "--config",
            config,
            "--data-dir",
            data_dir,
            "--output-network",
            network_path,
        ],
        root,
    )

    # -------------------------------------------------------------------------
    # 3. Analyze
    # -------------------------------------------------------------------------

    run(
        [
            sys.executable,
            "scripts/analyze_results.py",
            "--config",
            config,
            "--network",
            network_path,
            "--outdir",
            result_dir,
        ],
        root,
    )

    return read_summary(
        summary_path
    )


def miner_capacity_mw(
    template_cfg,
    miners,
):
    """
    Convert a complete integer miner count into
    facility-side electrical capacity.

    ASIC power follows directly from:

        P_ASIC =
            efficiency [J/TH]
            * hashrate [TH/s]

    and facility power additionally includes PUE.
    """

    bitcoin_cfg = (
        template_cfg[
            "bitcoin"
        ]
    )

    efficiency_j_per_th = float(
        bitcoin_cfg[
            "asic_efficiency_j_per_th"
        ]
    )

    pue = float(
        bitcoin_cfg[
            "pue"
        ]
    )

    asic_power_kw = (
        efficiency_j_per_th
        * BTC_HASHRATE_TH_S
        / 1000.0
    )

    facility_power_kw = (
        asic_power_kw
        * pue
    )

    capacity_mw = (
        miners
        * facility_power_kw
        / 1000.0
    )

    return (
        capacity_mw,
        facility_power_kw,
    )


def make_case_config(
    template_path,
    output_path,
    family,
    miners,
):
    """
    Create one temporary fixed-capacity BTC scenario.

    The existing validated S2/S3 operational formulation
    is preserved. BTC investment cost is therefore not
    placed into the optimization of these fixed cases.

    The investment cost is added separately in
    post-processing below.
    """

    cfg = load_yaml(
        template_path
    )

    (
        capacity_mw,
        facility_power_kw,
    ) = miner_capacity_mw(
        cfg,
        miners,
    )

    cfg[
        "scenario_name"
    ] = (
        f"btc_miner_sweep_"
        f"{family}_"
        f"{miners:06d}_miners"
    )

    bitcoin_cfg = (
        cfg[
            "bitcoin"
        ]
    )

    bitcoin_cfg[
        "enabled"
    ] = True

    bitcoin_cfg[
        "operating_mode"
    ] = "economic_dispatch"

    bitcoin_cfg[
        "capacity_mode"
    ] = "fixed"

    bitcoin_cfg[
        "max_capacity_mw"
    ] = float(
        capacity_mw
    )

    # Fixed S2/S3 formulation:
    # BTC CAPEX does not enter the dispatch optimization.
    bitcoin_cfg.pop(
        "annualized_capacity_cost_eur_per_mw_year",
        None,
    )

    save_yaml(
        output_path,
        cfg,
    )

    return (
        capacity_mw,
        facility_power_kw,
    )


def require_columns(
    summary,
    columns,
    source,
):
    missing = [
        column
        for column in columns
        if column
        not in summary.index
    ]

    if missing:
        raise KeyError(
            f"{source}: missing columns "
            f"{missing}"
        )


def first_trigger(
    df,
    column,
    tolerance,
):
    subset = df[
        (
            df[
                "requested_miners"
            ] > 0
        )
        &
        (
            df[
                column
            ] > tolerance
        )
    ]

    if subset.empty:
        return (
            np.nan,
            np.nan,
        )

    row = (
        subset
        .sort_values(
            "requested_miners"
        )
        .iloc[0]
    )

    return (
        int(
            row[
                "requested_miners"
            ]
        ),
        float(
            row[
                "requested_bitcoin_capacity_mw"
            ]
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    output_root = (
        root
        / args.output_root
    )

    data_root = (
        root
        / args.data_root
    )

    generated_config_root = (
        output_root
        / "generated_configs"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    miners = sorted(
        set(
            args.miners
        )
    )

    if (
        not miners
        or miners[0] < 0
    ):
        raise ValueError(
            "Miner counts must be non-negative."
        )

    if 0 not in miners:
        raise ValueError(
            "Include 0 in --miners "
            "so each family has a baseline."
        )

    # -------------------------------------------------------------------------
    # Scenario architecture
    # -------------------------------------------------------------------------

    families = {
        "no_battery": {
            "baseline": (
                root
                / args.s0_config
            ),
            "template": (
                root
                / args.s2_template
            ),
        },

        "battery": {
            "baseline": (
                root
                / args.s1_config
            ),
            "template": (
                root
                / args.s3_template
            ),
        },
    }

    for family in args.families:
        for (
            key,
            path,
        ) in (
            families[
                family
            ].items()
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"{family} {key} "
                    f"not found: {path}"
                )

    # -------------------------------------------------------------------------
    # BTC investment assumption
    # -------------------------------------------------------------------------
    #
    # Read directly from S2E so the sweep cannot silently
    # drift away from the endogenous central assumption.
    #

    investment_config_path = (
        root
        / args.btc_investment_config
    )

    if not investment_config_path.exists():
        raise FileNotFoundError(
            "BTC investment config "
            f"not found: "
            f"{investment_config_path}"
        )

    investment_cfg = load_yaml(
        investment_config_path
    )

    btc_capacity_cost = (
        investment_cfg[
            "bitcoin"
        ].get(
            "annualized_capacity_cost_eur_per_mw_year"
        )
    )

    if (
        btc_capacity_cost is None
        or float(
            btc_capacity_cost
        ) <= 0.0
    ):
        raise ValueError(
            "BTC investment config must contain "
            "a positive annualized capacity cost."
        )

    btc_capacity_cost = float(
        btc_capacity_cost
    )

    # -------------------------------------------------------------------------
    # Exact current analyzer columns used by the sweep
    # -------------------------------------------------------------------------

    core_columns = [
        "solar_capacity_mw",
        "wind_capacity_mw",
        "electrolyzer_capacity_mw",
        "battery_power_mw",
        "battery_energy_mwh",
        "hydrogen_delivered_kg",
        "hydrogen_delivered_kt",
        "total_curtailment_mwh",
        "renewable_utilization_rate",
        "objective_eur_per_year",
        "gross_system_expenditure_eur_per_year",
        "net_system_cost_eur_per_year",
    ]

    bitcoin_columns = [
        "bitcoin_capacity_mw",
        "bitcoin_installed_miners_equiv",
        "bitcoin_consumption_mwh",
        "bitcoin_utilization_rate",
        "bitcoin_gross_revenue_eur_per_year",
        "bitcoin_net_operating_value_eur_per_year",
    ]

    rows = []
    trigger_rows = []

    print(
        "\nBTC MINER SWEEP"
    )

    print(
        "=" * 90
    )

    print(
        "Annualized BTC capacity cost: "
        f"{btc_capacity_cost:,.2f} "
        "EUR/MW/a"
    )

    print(
        "Miner grid:",
        miners,
    )

    print(
        "=" * 90
    )

    # =========================================================================
    # Run sweep families
    # =========================================================================

    for family in args.families:

        baseline_config = (
            families[
                family
            ][
                "baseline"
            ]
        )

        template_config = (
            families[
                family
            ][
                "template"
            ]
        )

        print(
            f"\n\n### {family} ###"
        )

        # ---------------------------------------------------------------------
        # Zero-miner baseline
        # ---------------------------------------------------------------------

        baseline_result_dir = (
            output_root
            / family
            / "000000_miners_baseline"
        )

        baseline_data_dir = (
            data_root
            / family
            / "000000_miners_baseline"
        )

        baseline = run_pipeline(
            root,
            baseline_config,
            baseline_data_dir,
            baseline_result_dir,
            args.resume,
        )

        require_columns(
            baseline,
            core_columns,
            (
                baseline_result_dir
                / "summary.csv"
            ),
        )

        baseline_solar_mw = float(
            baseline[
                "solar_capacity_mw"
            ]
        )

        baseline_wind_mw = float(
            baseline[
                "wind_capacity_mw"
            ]
        )

        baseline_res_mw = (
            baseline_solar_mw
            + baseline_wind_mw
        )

        baseline_net_cost = float(
            baseline[
                "net_system_cost_eur_per_year"
            ]
        )

        rows.append(
            {
                "family":
                    family,

                "battery_available":
                    family
                    == "battery",

                "requested_miners":
                    0,

                "requested_bitcoin_capacity_mw":
                    0.0,

                "bitcoin_capacity_mw":
                    0.0,

                "bitcoin_installed_miners_equiv":
                    0.0,

                "bitcoin_consumption_mwh":
                    0.0,

                "bitcoin_utilization_rate":
                    0.0,

                "bitcoin_gross_revenue_eur_per_year":
                    0.0,

                "bitcoin_net_operating_value_eur_per_year":
                    0.0,

                "solar_capacity_mw":
                    baseline_solar_mw,

                "wind_capacity_mw":
                    baseline_wind_mw,

                "renewable_capacity_mw":
                    baseline_res_mw,

                "delta_solar_capacity_mw":
                    0.0,

                "delta_wind_capacity_mw":
                    0.0,

                "delta_renewable_capacity_mw":
                    0.0,

                "electrolyzer_capacity_mw":
                    float(
                        baseline[
                            "electrolyzer_capacity_mw"
                        ]
                    ),

                "battery_power_mw":
                    float(
                        baseline[
                            "battery_power_mw"
                        ]
                    ),

                "battery_energy_mwh":
                    float(
                        baseline[
                            "battery_energy_mwh"
                        ]
                    ),

                "hydrogen_delivered_kg":
                    float(
                        baseline[
                            "hydrogen_delivered_kg"
                        ]
                    ),

                "hydrogen_delivered_kt":
                    float(
                        baseline[
                            "hydrogen_delivered_kt"
                        ]
                    ),

                "total_curtailment_mwh":
                    float(
                        baseline[
                            "total_curtailment_mwh"
                        ]
                    ),

                "renewable_utilization_rate":
                    float(
                        baseline[
                            "renewable_utilization_rate"
                        ]
                    ),

                "objective_eur_per_year":
                    float(
                        baseline[
                            "objective_eur_per_year"
                        ]
                    ),

                "gross_system_expenditure_eur_per_year":
                    float(
                        baseline[
                            "gross_system_expenditure_eur_per_year"
                        ]
                    ),

                "net_system_cost_excluding_btc_capex_eur_per_year":
                    baseline_net_cost,

                "btc_capacity_cost_eur_per_year":
                    0.0,

                "net_system_cost_including_btc_capex_eur_per_year":
                    baseline_net_cost,

                "delta_corrected_net_cost_vs_baseline_eur_per_year":
                    0.0,

                "economic_benefit_vs_baseline_eur_per_year":
                    0.0,

                "result_dir":
                    str(
                        baseline_result_dir.relative_to(
                            root
                        )
                    ),
            }
        )

        # ---------------------------------------------------------------------
        # Physical miner conversion
        # ---------------------------------------------------------------------

        template = load_yaml(
            template_config
        )

        (
            _,
            facility_power_per_miner_kw,
        ) = miner_capacity_mw(
            template,
            1,
        )

        print(
            f"{family}: "
            f"{facility_power_per_miner_kw:.6f} "
            "kW facility-side per miner"
        )

        # ---------------------------------------------------------------------
        # Positive miner counts
        # ---------------------------------------------------------------------

        for miner_count in miners:

            if miner_count == 0:
                continue

            case_name = (
                f"{miner_count:06d}_miners"
            )

            config_path = (
                generated_config_root
                / family
                / f"{case_name}.yaml"
            )

            (
                requested_capacity_mw,
                _,
            ) = make_case_config(
                template_config,
                config_path,
                family,
                miner_count,
            )

            result_dir = (
                output_root
                / family
                / case_name
            )

            data_dir = (
                data_root
                / family
                / case_name
            )

            summary = run_pipeline(
                root,
                config_path,
                data_dir,
                result_dir,
                args.resume,
            )

            require_columns(
                summary,
                (
                    core_columns
                    + bitcoin_columns
                ),
                (
                    result_dir
                    / "summary.csv"
                ),
            )

            # -----------------------------------------------------------------
            # Validate miner ↔ MW conversion
            # -----------------------------------------------------------------

            model_capacity_mw = float(
                summary[
                    "bitcoin_capacity_mw"
                ]
            )

            capacity_tolerance_mw = max(
                1e-9,
                requested_capacity_mw
                * 1e-9,
            )

            if not np.isclose(
                model_capacity_mw,
                requested_capacity_mw,
                rtol=0.0,
                atol=capacity_tolerance_mw,
            ):
                raise RuntimeError(
                    f"{family}, "
                    f"{miner_count} miners: "
                    "BTC capacity mismatch. "
                    f"requested="
                    f"{requested_capacity_mw:.12f} MW, "
                    f"model="
                    f"{model_capacity_mw:.12f} MW."
                )

            installed_miners_equiv = float(
                summary[
                    "bitcoin_installed_miners_equiv"
                ]
            )

            if not np.isclose(
                installed_miners_equiv,
                float(
                    miner_count
                ),
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"{family}, "
                    f"{miner_count} miners: "
                    "miner conversion mismatch. "
                    f"summary="
                    f"{installed_miners_equiv:.9f}."
                )

            # -----------------------------------------------------------------
            # Renewable response
            # -----------------------------------------------------------------

            solar_mw = float(
                summary[
                    "solar_capacity_mw"
                ]
            )

            wind_mw = float(
                summary[
                    "wind_capacity_mw"
                ]
            )

            renewable_mw = (
                solar_mw
                + wind_mw
            )

            # -----------------------------------------------------------------
            # BTC investment-cost correction
            # -----------------------------------------------------------------
            #
            # IMPORTANT:
            #
            # The fixed S2/S3 formulation treats installed BTC capacity as
            # exogenous. Its CAPEX therefore does not enter the optimization.
            #
            # We add the SAME annualized BTC capacity cost used in S2E/S3E
            # here after solving:
            #
            # total corrected cost
            #   = fixed-case net system cost
            #   + annualized BTC investment cost.
            #
            # This allows the physical sweep and economic interpretation to
            # remain explicitly separated.
            #

            btc_capacity_cost_eur = (
                requested_capacity_mw
                * btc_capacity_cost
            )

            net_cost_excluding_btc_capex = float(
                summary[
                    "net_system_cost_eur_per_year"
                ]
            )

            net_cost_including_btc_capex = (
                net_cost_excluding_btc_capex
                + btc_capacity_cost_eur
            )

            delta_corrected_cost = (
                net_cost_including_btc_capex
                - baseline_net_cost
            )

            economic_benefit = (
                -delta_corrected_cost
            )

            rows.append(
                {
                    "family":
                        family,

                    "battery_available":
                        family
                        == "battery",

                    "requested_miners":
                        miner_count,

                    "requested_bitcoin_capacity_mw":
                        requested_capacity_mw,

                    "bitcoin_capacity_mw":
                        model_capacity_mw,

                    "bitcoin_installed_miners_equiv":
                        installed_miners_equiv,

                    "bitcoin_consumption_mwh":
                        float(
                            summary[
                                "bitcoin_consumption_mwh"
                            ]
                        ),

                    "bitcoin_utilization_rate":
                        float(
                            summary[
                                "bitcoin_utilization_rate"
                            ]
                        ),

                    "bitcoin_gross_revenue_eur_per_year":
                        float(
                            summary[
                                "bitcoin_gross_revenue_eur_per_year"
                            ]
                        ),

                    "bitcoin_net_operating_value_eur_per_year":
                        float(
                            summary[
                                "bitcoin_net_operating_value_eur_per_year"
                            ]
                        ),

                    "solar_capacity_mw":
                        solar_mw,

                    "wind_capacity_mw":
                        wind_mw,

                    "renewable_capacity_mw":
                        renewable_mw,

                    "delta_solar_capacity_mw":
                        (
                            solar_mw
                            - baseline_solar_mw
                        ),

                    "delta_wind_capacity_mw":
                        (
                            wind_mw
                            - baseline_wind_mw
                        ),

                    "delta_renewable_capacity_mw":
                        (
                            renewable_mw
                            - baseline_res_mw
                        ),

                    "electrolyzer_capacity_mw":
                        float(
                            summary[
                                "electrolyzer_capacity_mw"
                            ]
                        ),

                    "battery_power_mw":
                        float(
                            summary[
                                "battery_power_mw"
                            ]
                        ),

                    "battery_energy_mwh":
                        float(
                            summary[
                                "battery_energy_mwh"
                            ]
                        ),

                    "hydrogen_delivered_kg":
                        float(
                            summary[
                                "hydrogen_delivered_kg"
                            ]
                        ),

                    "hydrogen_delivered_kt":
                        float(
                            summary[
                                "hydrogen_delivered_kt"
                            ]
                        ),

                    "total_curtailment_mwh":
                        float(
                            summary[
                                "total_curtailment_mwh"
                            ]
                        ),

                    "renewable_utilization_rate":
                        float(
                            summary[
                                "renewable_utilization_rate"
                            ]
                        ),

                    "objective_eur_per_year":
                        float(
                            summary[
                                "objective_eur_per_year"
                            ]
                        ),

                    "gross_system_expenditure_eur_per_year":
                        float(
                            summary[
                                "gross_system_expenditure_eur_per_year"
                            ]
                        ),

                    "net_system_cost_excluding_btc_capex_eur_per_year":
                        net_cost_excluding_btc_capex,

                    "btc_capacity_cost_eur_per_year":
                        btc_capacity_cost_eur,

                    "net_system_cost_including_btc_capex_eur_per_year":
                        net_cost_including_btc_capex,

                    "delta_corrected_net_cost_vs_baseline_eur_per_year":
                        delta_corrected_cost,

                    "economic_benefit_vs_baseline_eur_per_year":
                        economic_benefit,

                    "result_dir":
                        str(
                            result_dir.relative_to(
                                root
                            )
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Identify first sampled trigger points
        # ---------------------------------------------------------------------

        family_df = (
            pd.DataFrame(
                [
                    row
                    for row in rows
                    if row[
                        "family"
                    ] == family
                ]
            )
            .sort_values(
                "requested_miners"
            )
        )

        (
            res_trigger_miners,
            res_trigger_capacity_mw,
        ) = first_trigger(
            family_df,
            "delta_renewable_capacity_mw",
            args.res_tolerance_mw,
        )

        (
            solar_trigger_miners,
            _,
        ) = first_trigger(
            family_df,
            "delta_solar_capacity_mw",
            args.res_tolerance_mw,
        )

        (
            wind_trigger_miners,
            _,
        ) = first_trigger(
            family_df,
            "delta_wind_capacity_mw",
            args.res_tolerance_mw,
        )

        (
            battery_trigger_miners,
            _,
        ) = first_trigger(
            family_df,
            "battery_power_mw",
            args.res_tolerance_mw,
        )

        (
            btc_dispatch_trigger_miners,
            _,
        ) = first_trigger(
            family_df,
            "bitcoin_consumption_mwh",
            1e-6,
        )

        (
            economic_trigger_miners,
            _,
        ) = first_trigger(
            family_df,
            "economic_benefit_vs_baseline_eur_per_year",
            1.0,
        )

        trigger_rows.append(
            {
                "family":
                    family,

                "baseline_solar_capacity_mw":
                    baseline_solar_mw,

                "baseline_wind_capacity_mw":
                    baseline_wind_mw,

                "baseline_renewable_capacity_mw":
                    baseline_res_mw,

                "first_sampled_btc_dispatch_miners":
                    btc_dispatch_trigger_miners,

                "first_sampled_additional_res_miners":
                    res_trigger_miners,

                "first_sampled_additional_res_btc_capacity_mw":
                    res_trigger_capacity_mw,

                "first_sampled_solar_increase_miners":
                    solar_trigger_miners,

                "first_sampled_wind_increase_miners":
                    wind_trigger_miners,

                "first_sampled_battery_entry_miners":
                    battery_trigger_miners,

                "first_sampled_economically_beneficial_miners":
                    economic_trigger_miners,

                "btc_capacity_cost_eur_per_mw_year":
                    btc_capacity_cost,
            }
        )

    # =========================================================================
    # Save consolidated outputs
    # =========================================================================

    results_df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "family",
                "requested_miners",
            ]
        )
    )

    trigger_df = (
        pd.DataFrame(
            trigger_rows
        )
        .sort_values(
            "family"
        )
    )

    results_path = (
        output_root
        / "btc_miner_sweep_results.csv"
    )

    triggers_path = (
        output_root
        / "btc_miner_sweep_triggers.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    trigger_df.to_csv(
        triggers_path,
        index=False,
    )

    print(
        "\nBTC miner sweep complete."
    )

    print(
        f"Results:  {results_path}"
    )

    print(
        f"Triggers: {triggers_path}"
    )

    print(
        "\nTrigger summary:"
    )

    print(
        trigger_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()