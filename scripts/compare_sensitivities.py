from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("results/thesis_sensitivities")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_summary(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result summary: {path}"
        )

    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"{path} must contain exactly one row."
        )

    return df.iloc[0]


def clean_small_values(df):
    numeric = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric] = df[numeric].mask(
        df[numeric].abs() < 1e-9,
        0.0,
    )

    return df


# =============================================================================
# 1. Hashprice sensitivity
# =============================================================================

hashprice_cases = {
    "S2": {
        20: "results/s2_thesis_2045_hp20/summary.csv",
        32: "results/s2_thesis_2045/summary.csv",
        40: "results/s2_thesis_2045_hp40/summary.csv",
        60: "results/s2_thesis_2045_hp60/summary.csv",
    },

    "S3": {
        20: "results/s3_thesis_2045_hp20/summary.csv",
        24: "results/s3_thesis_2045_hp24/summary.csv",
        28: "results/s3_thesis_2045_hp28/summary.csv",
        30: "results/s3_thesis_2045_hp30/summary.csv",
        32: "results/s3_thesis_2045/summary.csv",
        40: "results/s3_thesis_2045_hp40/summary.csv",
        60: "results/s3_thesis_2045_hp60/summary.csv",
    },
}

hashprice_rows = []

for scenario, cases in hashprice_cases.items():
    for hp, path in cases.items():
        r = load_summary(path)

        hashprice_rows.append({
            "scenario": scenario,
            "hashprice_usd2026_per_ph_day": hp,

            "solar_capacity_mw":
                r["solar_capacity_mw"],

            "wind_capacity_mw":
                r["wind_capacity_mw"],

            "renewable_capacity_mw":
                r["solar_capacity_mw"]
                + r["wind_capacity_mw"],

            "electrolyzer_capacity_mw":
                r["electrolyzer_capacity_mw"],

            "electrolyzer_cf":
                r["electrolyzer_capacity_factor"],

            "battery_power_mw":
                r["battery_power_mw"],

            "battery_energy_mwh":
                r["battery_energy_mwh"],

            "battery_duration_h":
                r["battery_duration_h"],

            "battery_cycles_per_year":
                r[
                    "battery_equivalent_full_cycles_per_year"
                ],

            "bitcoin_consumption_mwh":
                r["bitcoin_consumption_mwh"],

            "bitcoin_utilization":
                r["bitcoin_utilization_rate"],

            "bitcoin_revenue_eur2020_per_year":
                r[
                    "bitcoin_gross_revenue_eur_per_year"
                ],

            "renewable_utilization":
                r["renewable_utilization_rate"],

            "gross_system_expenditure_eur2020_per_year":
                r[
                    "gross_system_expenditure_eur_per_year"
                ],

            "net_system_cost_eur2020_per_year":
                r[
                    "net_system_cost_eur_per_year"
                ],

            "net_system_cost_eur2020_per_kg_h2":
                r[
                    "net_system_cost_eur_per_kg_h2"
                ],
        })

hashprice = clean_small_values(
    pd.DataFrame(hashprice_rows)
)

hashprice.to_csv(
    OUTDIR / "hashprice_sensitivity.csv",
    index=False,
)


# =============================================================================
# 2. Bitcoin-capacity sensitivity
# =============================================================================

capacity_cases = {
    "S2": {
        3: "results/s2_thesis_2045_btc3mw/summary.csv",
        6: "results/s2_thesis_2045/summary.csv",
        12: "results/s2_thesis_2045_btc12mw/summary.csv",
    },

    "S3": {
        3: "results/s3_thesis_2045_btc3mw/summary.csv",
        6: "results/s3_thesis_2045/summary.csv",
        12: "results/s3_thesis_2045_btc12mw/summary.csv",
    },
}

capacity_rows = []

for scenario, cases in capacity_cases.items():
    for btc_mw, path in cases.items():
        r = load_summary(path)

        capacity_rows.append({
            "scenario": scenario,
            "bitcoin_capacity_mw": btc_mw,

            "solar_capacity_mw":
                r["solar_capacity_mw"],

            "wind_capacity_mw":
                r["wind_capacity_mw"],

            "renewable_capacity_mw":
                r["solar_capacity_mw"]
                + r["wind_capacity_mw"],

            "electrolyzer_capacity_mw":
                r["electrolyzer_capacity_mw"],

            "electrolyzer_cf":
                r["electrolyzer_capacity_factor"],

            "battery_power_mw":
                r["battery_power_mw"],

            "battery_energy_mwh":
                r["battery_energy_mwh"],

            "battery_duration_h":
                r["battery_duration_h"],

            "bitcoin_consumption_mwh":
                r["bitcoin_consumption_mwh"],

            "bitcoin_utilization":
                r["bitcoin_utilization_rate"],

            "bitcoin_revenue_eur2020_per_year":
                r[
                    "bitcoin_gross_revenue_eur_per_year"
                ],

            "renewable_utilization":
                r["renewable_utilization_rate"],

            "gross_system_expenditure_eur2020_per_year":
                r[
                    "gross_system_expenditure_eur_per_year"
                ],

            "net_system_cost_eur2020_per_year":
                r[
                    "net_system_cost_eur_per_year"
                ],

            "net_system_cost_eur2020_per_kg_h2":
                r[
                    "net_system_cost_eur_per_kg_h2"
                ],
        })

capacity = clean_small_values(
    pd.DataFrame(capacity_rows)
)

capacity.to_csv(
    OUTDIR / "bitcoin_capacity_sensitivity.csv",
    index=False,
)


# =============================================================================
# 3. Matched S3 - S2 interaction effects
# =============================================================================

interaction_rows = []

# Hashprice comparison where both S2 and S3 exist.
for hp in [20, 32, 40, 60]:
    s2 = hashprice[
        (hashprice["scenario"] == "S2")
        & (
            hashprice[
                "hashprice_usd2026_per_ph_day"
            ] == hp
        )
    ].iloc[0]

    s3 = hashprice[
        (hashprice["scenario"] == "S3")
        & (
            hashprice[
                "hashprice_usd2026_per_ph_day"
            ] == hp
        )
    ].iloc[0]

    interaction_rows.append({
        "sensitivity": "hashprice",
        "value": hp,
        "unit": "USD2026/PH/day",

        "battery_power_mw":
            s3["battery_power_mw"],

        "battery_energy_mwh":
            s3["battery_energy_mwh"],

        "battery_duration_h":
            s3["battery_duration_h"],

        "delta_bitcoin_consumption_mwh":
            s3["bitcoin_consumption_mwh"]
            - s2["bitcoin_consumption_mwh"],

        "delta_net_system_cost_eur2020_per_year":
            s3[
                "net_system_cost_eur2020_per_year"
            ]
            - s2[
                "net_system_cost_eur2020_per_year"
            ],

        "delta_net_cost_eur2020_per_kg_h2":
            s3[
                "net_system_cost_eur2020_per_kg_h2"
            ]
            - s2[
                "net_system_cost_eur2020_per_kg_h2"
            ],
    })


# BTC-capacity comparison.
for btc_mw in [3, 6, 12]:
    s2 = capacity[
        (capacity["scenario"] == "S2")
        & (
            capacity[
                "bitcoin_capacity_mw"
            ] == btc_mw
        )
    ].iloc[0]

    s3 = capacity[
        (capacity["scenario"] == "S3")
        & (
            capacity[
                "bitcoin_capacity_mw"
            ] == btc_mw
        )
    ].iloc[0]

    interaction_rows.append({
        "sensitivity": "bitcoin_capacity",
        "value": btc_mw,
        "unit": "MW",

        "battery_power_mw":
            s3["battery_power_mw"],

        "battery_energy_mwh":
            s3["battery_energy_mwh"],

        "battery_duration_h":
            s3["battery_duration_h"],

        "delta_bitcoin_consumption_mwh":
            s3["bitcoin_consumption_mwh"]
            - s2["bitcoin_consumption_mwh"],

        "delta_net_system_cost_eur2020_per_year":
            s3[
                "net_system_cost_eur2020_per_year"
            ]
            - s2[
                "net_system_cost_eur2020_per_year"
            ],

        "delta_net_cost_eur2020_per_kg_h2":
            s3[
                "net_system_cost_eur2020_per_kg_h2"
            ]
            - s2[
                "net_system_cost_eur2020_per_kg_h2"
            ],
    })

interaction = clean_small_values(
    pd.DataFrame(interaction_rows)
)

interaction.to_csv(
    OUTDIR / "battery_bitcoin_interaction.csv",
    index=False,
)


# =============================================================================
# 4. Identify tested battery-entry interval
# =============================================================================

s3_hp = (
    hashprice[
        hashprice["scenario"] == "S3"
    ]
    .sort_values(
        "hashprice_usd2026_per_ph_day"
    )
    .reset_index(drop=True)
)

entry_threshold = 1e-6

zero_cases = s3_hp[
    s3_hp["battery_power_mw"]
    <= entry_threshold
]

positive_cases = s3_hp[
    s3_hp["battery_power_mw"]
    > entry_threshold
]

if (
    not zero_cases.empty
    and not positive_cases.empty
):
    last_zero = (
        zero_cases[
            "hashprice_usd2026_per_ph_day"
        ]
        .max()
    )

    first_positive = (
        positive_cases[
            "hashprice_usd2026_per_ph_day"
        ]
        .min()
    )

    threshold_text = (
        "Tested battery-entry interval: "
        f"{last_zero:g} < hashprice < "
        f"{first_positive:g} "
        "USD2026/PH/day"
    )
else:
    threshold_text = (
        "Battery-entry interval could not "
        "be bracketed by tested cases."
    )


# =============================================================================
# 5. Print concise results
# =============================================================================

print()
print("=" * 90)
print("HASHPRICE SENSITIVITY")
print("=" * 90)

print(
    hashprice[
        [
            "scenario",
            "hashprice_usd2026_per_ph_day",
            "renewable_capacity_mw",
            "electrolyzer_capacity_mw",
            "battery_power_mw",
            "battery_energy_mwh",
            "bitcoin_utilization",
            "net_system_cost_eur2020_per_kg_h2",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}",
    )
)

print()
print("=" * 90)
print("BITCOIN CAPACITY SENSITIVITY")
print("=" * 90)

print(
    capacity[
        [
            "scenario",
            "bitcoin_capacity_mw",
            "renewable_capacity_mw",
            "electrolyzer_capacity_mw",
            "battery_power_mw",
            "battery_energy_mwh",
            "bitcoin_utilization",
            "net_system_cost_eur2020_per_kg_h2",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}",
    )
)

print()
print("=" * 90)
print("BATTERY-BITCOIN INTERACTION")
print("=" * 90)

print(
    interaction.to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}",
    )
)

print()
print(threshold_text)

print()
print("Outputs:")
print(
    " ",
    OUTDIR / "hashprice_sensitivity.csv",
)
print(
    " ",
    OUTDIR / "bitcoin_capacity_sensitivity.csv",
)
print(
    " ",
    OUTDIR / "battery_bitcoin_interaction.csv",
)
print()
