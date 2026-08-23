from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Paths and plotting defaults
# =============================================================================

OUTDIR = Path("results/thesis_presentation")
FIGDIR = OUTDIR / "figures"
TABLEDIR = OUTDIR / "tables"

FIGDIR.mkdir(parents=True, exist_ok=True)
TABLEDIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
    }
)


# =============================================================================
# Helpers
# =============================================================================

def load_summary(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing summary file: {path}"
        )

    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"{path} must contain exactly one row."
        )

    return df.iloc[0]


def clean_value(value, tol=1e-9):
    if pd.isna(value):
        return np.nan

    value = float(value)

    if abs(value) < tol:
        return 0.0

    return value


def save_figure(fig, stem):
    pdf = FIGDIR / f"{stem}.pdf"
    png = FIGDIR / f"{stem}.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("  ✓", pdf)
    print("  ✓", png)


# =============================================================================
# Central scenarios
# =============================================================================

CENTRAL_CASES = {
    "S0": "results/s0_thesis_2045/summary.csv",
    "S1": "results/s1_thesis_2045/summary.csv",
    "S2": "results/s2_thesis_2045/summary.csv",
    "S3": "results/s3_thesis_2045/summary.csv",
}

central_rows = []

for scenario, path in CENTRAL_CASES.items():
    r = load_summary(path)

    central_rows.append(
        {
            "scenario": scenario,

            "solar_capacity_mw":
                clean_value(
                    r["solar_capacity_mw"]
                ),

            "wind_capacity_mw":
                clean_value(
                    r["wind_capacity_mw"]
                ),

            "renewable_capacity_mw":
                clean_value(
                    r["solar_capacity_mw"]
                    + r["wind_capacity_mw"]
                ),

            "electrolyzer_capacity_mw":
                clean_value(
                    r["electrolyzer_capacity_mw"]
                ),

            "electrolyzer_cf":
                clean_value(
                    r["electrolyzer_capacity_factor"]
                ),

            "battery_power_mw":
                clean_value(
                    r["battery_power_mw"]
                ),

            "battery_energy_mwh":
                clean_value(
                    r["battery_energy_mwh"]
                ),

            "battery_duration_h":
                clean_value(
                    r["battery_duration_h"]
                ),

            "bitcoin_capacity_mw":
                clean_value(
                    r["bitcoin_capacity_mw"]
                ),

            "bitcoin_consumption_mwh":
                clean_value(
                    r["bitcoin_consumption_mwh"]
                ),

            "bitcoin_utilization":
                clean_value(
                    r["bitcoin_utilization_rate"]
                ),

            "renewable_utilization":
                clean_value(
                    r["renewable_utilization_rate"]
                ),

            "gross_expenditure_eur":
                clean_value(
                    r[
                        "gross_system_expenditure_eur_per_year"
                    ]
                ),

            "bitcoin_revenue_eur":
                clean_value(
                    r[
                        "bitcoin_gross_revenue_eur_per_year"
                    ]
                ),

            "net_system_cost_eur":
                clean_value(
                    r[
                        "net_system_cost_eur_per_year"
                    ]
                ),

            "gross_cost_per_kg":
                clean_value(
                    r[
                        "gross_system_expenditure_eur_per_kg_h2"
                    ]
                ),

            "net_cost_per_kg":
                clean_value(
                    r[
                        "net_system_cost_eur_per_kg_h2"
                    ]
                ),
        }
    )

central = (
    pd.DataFrame(central_rows)
    .set_index("scenario")
)


# =============================================================================
# Hashprice sensitivity
# =============================================================================

HASHPRICE_CASES = {
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

hp_rows = []

for scenario, cases in HASHPRICE_CASES.items():

    for hp, path in cases.items():

        r = load_summary(path)

        hp_rows.append(
            {
                "scenario": scenario,
                "hashprice": hp,

                "renewable_capacity_mw":
                    clean_value(
                        r["solar_capacity_mw"]
                        + r["wind_capacity_mw"]
                    ),

                "battery_power_mw":
                    clean_value(
                        r["battery_power_mw"]
                    ),

                "battery_energy_mwh":
                    clean_value(
                        r["battery_energy_mwh"]
                    ),

                "battery_duration_h":
                    clean_value(
                        r["battery_duration_h"]
                    ),

                "bitcoin_utilization":
                    clean_value(
                        r["bitcoin_utilization_rate"]
                    ),

                "net_cost_per_kg":
                    clean_value(
                        r[
                            "net_system_cost_eur_per_kg_h2"
                        ]
                    ),
            }
        )

hashprice = pd.DataFrame(hp_rows)


# =============================================================================
# Bitcoin-capacity sensitivity
# =============================================================================

BTC_CAPACITY_CASES = {
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

btc_rows = []

for scenario, cases in BTC_CAPACITY_CASES.items():

    for btc_mw, path in cases.items():

        r = load_summary(path)

        btc_rows.append(
            {
                "scenario": scenario,
                "bitcoin_capacity_mw": btc_mw,

                "renewable_capacity_mw":
                    clean_value(
                        r["solar_capacity_mw"]
                        + r["wind_capacity_mw"]
                    ),

                "battery_power_mw":
                    clean_value(
                        r["battery_power_mw"]
                    ),

                "battery_energy_mwh":
                    clean_value(
                        r["battery_energy_mwh"]
                    ),

                "battery_duration_h":
                    clean_value(
                        r["battery_duration_h"]
                    ),

                "bitcoin_utilization":
                    clean_value(
                        r["bitcoin_utilization_rate"]
                    ),

                "net_cost_per_kg":
                    clean_value(
                        r[
                            "net_system_cost_eur_per_kg_h2"
                        ]
                    ),
            }
        )

btc_capacity = pd.DataFrame(btc_rows)


# =============================================================================
# Tables
# =============================================================================

central_table = pd.DataFrame(
    {
        scenario: [
            central.loc[
                scenario,
                "solar_capacity_mw"
            ],

            central.loc[
                scenario,
                "wind_capacity_mw"
            ],

            central.loc[
                scenario,
                "renewable_capacity_mw"
            ],

            central.loc[
                scenario,
                "electrolyzer_capacity_mw"
            ],

            central.loc[
                scenario,
                "electrolyzer_cf"
            ] * 100,

            central.loc[
                scenario,
                "battery_power_mw"
            ],

            central.loc[
                scenario,
                "battery_energy_mwh"
            ],

            central.loc[
                scenario,
                "battery_duration_h"
            ],

            central.loc[
                scenario,
                "bitcoin_consumption_mwh"
            ],

            central.loc[
                scenario,
                "bitcoin_utilization"
            ] * 100,

            central.loc[
                scenario,
                "renewable_utilization"
            ] * 100,

            central.loc[
                scenario,
                "gross_cost_per_kg"
            ],

            central.loc[
                scenario,
                "net_cost_per_kg"
            ],
        ]
        for scenario in [
            "S0",
            "S1",
            "S2",
            "S3",
        ]
    },
    index=[
        "Solar PV capacity [MW]",
        "Wind capacity [MW]",
        "Total renewable capacity [MW]",
        "PEM electrolyzer capacity [MW_el]",
        "PEM capacity factor [%]",
        "Battery power [MW]",
        "Battery energy [MWh]",
        "Battery duration [h]",
        "Bitcoin electricity [MWh/a]",
        "Bitcoin utilization [%]",
        "Renewable utilization [%]",
        "Gross system expenditure [EUR2020/kg_H2]",
        "Net system cost [EUR2020/kg_H2]",
    ],
)

central_table.to_csv(
    TABLEDIR / "central_scenarios_table.csv"
)

central_table.to_latex(
    TABLEDIR / "central_scenarios_table.tex",
    float_format="%.3f",
    na_rep="--",
    caption=(
        "Central 2045 results for the "
        "Kanagat off-grid scenarios."
    ),
    label="tab:central_scenarios",
)


hashprice.to_csv(
    TABLEDIR / "hashprice_sensitivity_table.csv",
    index=False,
)

btc_capacity.to_csv(
    TABLEDIR / "bitcoin_capacity_sensitivity_table.csv",
    index=False,
)


# =============================================================================
# Figure 1 — central optimized capacities
# =============================================================================

scenario_order = [
    "S0",
    "S1",
    "S2",
    "S3",
]

capacity_plot = central.loc[
    scenario_order,
    [
        "solar_capacity_mw",
        "wind_capacity_mw",
        "electrolyzer_capacity_mw",
        "battery_power_mw",
    ],
].copy()

capacity_plot.columns = [
    "Solar PV",
    "Wind",
    "PEM electrolyzer",
    "Battery power",
]

fig, ax = plt.subplots(
    figsize=(8.0, 4.8)
)

capacity_plot.plot(
    kind="bar",
    ax=ax,
)

ax.set_xlabel(
    "Scenario"
)

ax.set_ylabel(
    "Optimized capacity [MW]"
)

ax.set_title(
    "Optimized technology capacities by scenario"
)

ax.tick_params(
    axis="x",
    rotation=0,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    title="Technology",
    frameon=False,
)

fig.tight_layout()

save_figure(
    fig,
    "fig01_central_optimized_capacities",
)


# =============================================================================
# Figure 2 — central economics
# =============================================================================
#
# Revenue is plotted below zero because it offsets system expenditure.
# Net system cost is shown as a marker rather than as a third positive bar.
# =============================================================================

x = np.arange(
    len(scenario_order)
)

gross = (
    central.loc[
        scenario_order,
        "gross_expenditure_eur"
    ].to_numpy()
    / 1e6
)

revenue = (
    central.loc[
        scenario_order,
        "bitcoin_revenue_eur"
    ].to_numpy()
    / 1e6
)

net = (
    central.loc[
        scenario_order,
        "net_system_cost_eur"
    ].to_numpy()
    / 1e6
)

fig, ax = plt.subplots(
    figsize=(8.0, 4.8)
)

bar_width = 0.55

ax.bar(
    x,
    gross,
    width=bar_width,
    label="Gross system expenditure",
)

ax.bar(
    x,
    -revenue,
    width=bar_width,
    label="Bitcoin gross revenue",
)

ax.plot(
    x,
    net,
    marker="o",
    linestyle="none",
    markersize=8,
    label="Net system cost",
)

ax.axhline(
    0.0,
    linewidth=1.0,
    linestyle="--",
)

ax.set_xticks(
    x,
    scenario_order,
)

ax.set_xlabel(
    "Scenario"
)

ax.set_ylabel(
    "Annual value [million EUR$_{2020}$/a]"
)

ax.set_title(
    "Central scenario economics"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

save_figure(
    fig,
    "fig02_central_economics",
)


# =============================================================================
# Figure 3 — net system cost vs hashprice
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.8, 4.8)
)

for scenario in [
    "S2",
    "S3",
]:

    subset = (
        hashprice[
            hashprice["scenario"] == scenario
        ]
        .sort_values("hashprice")
    )

    ax.plot(
        subset["hashprice"],
        subset["net_cost_per_kg"],
        marker="o",
        label=scenario,
    )

ax.axhline(
    0.0,
    linewidth=1.0,
    linestyle="--",
)

ax.set_xlabel(
    "Bitcoin hashprice "
    "[USD$_{2026}$/(PH/s)/day]"
)

ax.set_ylabel(
    "Net system cost "
    "[EUR$_{2020}$/kg H$_2$]"
)

ax.set_title(
    "Net system cost sensitivity to Bitcoin hashprice"
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    title="Scenario",
    frameon=False,
)

fig.tight_layout()

save_figure(
    fig,
    "fig03_hashprice_net_system_cost",
)


# =============================================================================
# Figure 4 — S3 battery response to hashprice
# =============================================================================

s3_hp = (
    hashprice[
        hashprice["scenario"] == "S3"
    ]
    .sort_values("hashprice")
)

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(7.8, 8.3),
    sharex=True,
)

for ax in axes:

    ax.axvspan(
        24,
        28,
        alpha=0.10,
    )

    ax.grid(
        alpha=0.25,
    )


axes[0].plot(
    s3_hp["hashprice"],
    s3_hp["battery_power_mw"],
    marker="o",
)

axes[0].set_ylabel(
    "Battery power [MW]"
)

axes[0].set_title(
    "Battery deployment under Bitcoin hashprice sensitivity"
)


axes[1].plot(
    s3_hp["hashprice"],
    s3_hp["battery_energy_mwh"],
    marker="s",
)

axes[1].set_ylabel(
    "Battery energy [MWh]"
)


active = s3_hp[
    s3_hp["battery_power_mw"] > 1e-6
].copy()

axes[2].plot(
    active["hashprice"],
    active["battery_duration_h"],
    marker="o",
)

axes[2].set_ylabel(
    "Battery duration [h]"
)

axes[2].set_xlabel(
    "Bitcoin hashprice "
    "[USD$_{2026}$/(PH/s)/day]"
)

axes[2].set_ylim(
    bottom=0.0,
)

fig.text(
    0.5,
    0.015,
    (
        "Shaded region denotes the tested "
        "battery-entry interval "
        "(24 < hashprice < 28 USD$_{2026}$/(PH/s)/day)."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[
        0.0,
        0.045,
        1.0,
        1.0,
    ]
)

save_figure(
    fig,
    "fig04_hashprice_battery_response",
)


# =============================================================================
# Appendix Figure A1 — BTC-capacity sensitivity
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.8, 4.8)
)

for scenario in [
    "S2",
    "S3",
]:

    subset = (
        btc_capacity[
            btc_capacity["scenario"] == scenario
        ]
        .sort_values(
            "bitcoin_capacity_mw"
        )
    )

    ax.plot(
        subset["bitcoin_capacity_mw"],
        subset["renewable_capacity_mw"],
        marker="o",
        label=scenario,
    )

ax.set_xlabel(
    "Installed Bitcoin mining capacity [MW]"
)

ax.set_ylabel(
    "Optimized renewable capacity [MW]"
)

ax.set_title(
    "Renewable capacity sensitivity to Bitcoin mining scale"
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    title="Scenario",
    frameon=False,
)

fig.tight_layout()

save_figure(
    fig,
    "appendix_figA01_btc_capacity_renewable_response",
)


# =============================================================================
# Console summary
# =============================================================================

print()
print("=" * 80)
print("FINAL THESIS RESULT FIGURES GENERATED")
print("=" * 80)

print()
print("Main-text figures:")
print(
    "  fig01_central_optimized_capacities"
)
print(
    "  fig02_central_economics"
)
print(
    "  fig03_hashprice_net_system_cost"
)
print(
    "  fig04_hashprice_battery_response"
)

print()
print("Appendix figure:")
print(
    "  appendix_figA01_btc_capacity_renewable_response"
)

print()
print("Tables:")
for path in sorted(TABLEDIR.glob("*")):
    print(" ", path)

print()
