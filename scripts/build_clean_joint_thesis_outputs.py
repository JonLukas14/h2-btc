#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "results"
    / "clean_joint_sensitivity_final"
)

CASES_PATH = INPUT_DIR / "merchant_all_cases.csv"
THRESHOLDS_PATH = INPUT_DIR / "merchant_thresholds.csv"

FIGURE_DIR = INPUT_DIR / "figures"
TABLE_DIR = INPUT_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# General plotting style
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


def save_figure(
    fig,
    stem: str,
) -> None:
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close(fig)

    print(f"Saved: {pdf_path.relative_to(BASE_DIR)}")
    print(f"Saved: {png_path.relative_to(BASE_DIR)}")


def clean_small_values(
    df: pd.DataFrame,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    out = df.copy()

    numeric_columns = out.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        values = out[column].astype(float)
        mask = np.isfinite(values) & (
            np.abs(values) < tolerance
        )
        out.loc[mask, column] = 0.0

    return out


require_file(CASES_PATH)
require_file(THRESHOLDS_PATH)

cases = clean_small_values(
    pd.read_csv(CASES_PATH)
)

thresholds = pd.read_csv(
    THRESHOLDS_PATH
)


# =============================================================================
# Validate required columns
# =============================================================================

required_case_columns = {
    "joint_family",
    "sweep_h2_sale_value_eur_per_kg",
    "sweep_btc_capex_eur_per_mw_year",
    "merchant_regime",
    "hydrogen_delivered_kt",
    "bitcoin_capacity_mw",
    "battery_power_mw",
    "battery_energy_mwh",
}

missing = (
    required_case_columns
    - set(cases.columns)
)

if missing:
    raise RuntimeError(
        "merchant_all_cases.csv is missing: "
        f"{sorted(missing)}"
    )


# =============================================================================
# Human-readable labels
# =============================================================================

FAMILY_LABELS = {
    "joint": "JOINT",
    "joint_b": "JOINT-B",
}

REGIME_ORDER = [
    "none",
    "BTC",
    "BTC+battery",
    "H2",
    "H2+BTC",
    "H2+BTC+battery",
]

REGIME_LABELS = {
    "none": "No investment",
    "BTC": "BTC",
    "BTC+battery": "BTC + battery",
    "H2": r"H$_2$",
    "H2+BTC": r"H$_2$ + BTC",
    "H2+BTC+battery":
        r"H$_2$ + BTC + battery",
}

cmap = plt.get_cmap("tab10")

REGIME_COLORS = {
    regime: cmap(i)
    for i, regime
    in enumerate(REGIME_ORDER)
}

REGIME_MARKERS = {
    "none": "o",
    "BTC": "s",
    "BTC+battery": "D",
    "H2": "^",
    "H2+BTC": "v",
    "H2+BTC+battery": "P",
}


# =============================================================================
# FIGURE 1
# Merchant regime map
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.4, 4.4),
    sharey=True,
)

for ax, family in zip(
    axes,
    ["joint", "joint_b"],
):
    subset = cases[
        cases["joint_family"] == family
    ].copy()

    for regime in REGIME_ORDER:
        regime_subset = subset[
            subset["merchant_regime"]
            == regime
        ]

        if regime_subset.empty:
            continue

        ax.scatter(
            regime_subset[
                "sweep_btc_capex_eur_per_mw_year"
            ] / 1000.0,
            regime_subset[
                "sweep_h2_sale_value_eur_per_kg"
            ],
            s=52,
            marker=REGIME_MARKERS[regime],
            color=REGIME_COLORS[regime],
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )

    ax.set_title(
        FAMILY_LABELS[family]
    )

    ax.set_xlabel(
        r"Annualized BTC capacity cost "
        r"[kEUR$_{2020}$/MW$_{BTC}$/a]"
    )

    ax.grid(
        alpha=0.22,
        linewidth=0.6,
    )

    ax.set_xlim(
        140,
        390,
    )

    ax.set_xticks(
        [
            150,
            175,
            200,
            225,
            250,
            300,
            350,
            380.526,
        ]
    )

    ax.set_xticklabels(
        [
            "150",
            "175",
            "200",
            "225",
            "250",
            "300",
            "350",
            "380.5",
        ],
        rotation=35,
        ha="right",
    )

axes[0].set_ylabel(
    r"H$_2$ sale value "
    r"[EUR$_{2020}$/kg$_{H_2}$]"
)

axes[0].set_yticks(
    [0.0, 2.38, 2.75]
)

axes[0].set_yticklabels(
    ["0.00", "2.38", "2.75"]
)

legend_handles = []

for regime in REGIME_ORDER:
    if regime not in set(
        cases["merchant_regime"]
    ):
        continue

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker=REGIME_MARKERS[regime],
            linestyle="none",
            markerfacecolor=REGIME_COLORS[regime],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=7,
            label=REGIME_LABELS[regime],
        )
    )

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=3,
    frameon=False,
)

fig.suptitle(
    "Merchant operating regimes across sampled "
    "H$_2$ values and BTC capacity costs",
    y=1.01,
)

fig.subplots_adjust(
    bottom=0.28,
    wspace=0.12,
)

save_figure(
    fig,
    "fig_merchant_regime_map",
)


# =============================================================================
# FIGURE 2
# H2-BTC trade-off
#
# Only CAPEX values sampled at all three H2 values are used.
# Lines are visual guides between solved points, not interpolation.
# =============================================================================

tradeoff_capex = [
    150000.0,
    175000.0,
    200000.0,
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.4, 4.5),
    sharex=True,
    sharey=True,
)

line_styles = [
    "-",
    "--",
    ":",
]

markers = [
    "o",
    "s",
    "^",
]

for ax, family in zip(
    axes,
    ["joint", "joint_b"],
):
    family_subset = cases[
        cases["joint_family"] == family
    ].copy()

    for index, capex in enumerate(
        tradeoff_capex
    ):
        subset = family_subset[
            np.isclose(
                family_subset[
                    "sweep_btc_capex_eur_per_mw_year"
                ],
                capex,
                rtol=0.0,
                atol=1e-6,
            )
        ].copy()

        subset = subset.sort_values(
            "sweep_h2_sale_value_eur_per_kg"
        )

        if subset.empty:
            continue

        ax.plot(
            subset["hydrogen_delivered_kt"],
            subset["bitcoin_capacity_mw"],
            linestyle=line_styles[index],
            marker=markers[index],
            linewidth=1.5,
            markersize=5.5,
            label=(
                f"{capex / 1000:.0f} "
                "kEUR/MW/a"
            ),
        )

        for _, row in subset.iterrows():
            label = (
                f"{float(row['sweep_h2_sale_value_eur_per_kg']):.2f}"
            )

            ax.annotate(
                label,
                (
                    row["hydrogen_delivered_kt"],
                    row["bitcoin_capacity_mw"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(
        FAMILY_LABELS[family]
    )

    ax.set_xlabel(
        r"H$_2$ delivered [kt/a]"
    )

    ax.grid(
        alpha=0.22,
        linewidth=0.6,
    )

axes[0].set_ylabel(
    r"BTC capacity [MW$_{BTC}$]"
)

axes[1].legend(
    title="BTC CAPEX",
    frameon=False,
    loc="upper right",
)

fig.suptitle(
    "Optimized H$_2$–BTC allocation at selected "
    "BTC capacity costs",
    y=1.01,
)

fig.text(
    0.5,
    -0.02,
    (
        "Point labels show the sampled "
        "H$_2$ sale value in EUR/kg. "
        "Lines only guide the eye between solved cases."
    ),
    ha="center",
    fontsize=8.5,
)

fig.subplots_adjust(
    bottom=0.18,
    wspace=0.10,
)

save_figure(
    fig,
    "fig_merchant_h2_btc_tradeoff",
)


# =============================================================================
# FIGURE 3
# Battery-selection window at H2 = 2.38 EUR/kg
# =============================================================================

battery = cases[
    (
        cases["joint_family"]
        == "joint_b"
    )
    &
    np.isclose(
        cases[
            "sweep_h2_sale_value_eur_per_kg"
        ],
        2.38,
        rtol=0.0,
        atol=1e-9,
    )
].copy()

battery = battery[
    battery[
        "sweep_btc_capex_eur_per_mw_year"
    ]
    <= 225000.0 + 1e-6
].sort_values(
    "sweep_btc_capex_eur_per_mw_year"
)

capex_labels = (
    battery[
        "sweep_btc_capex_eur_per_mw_year"
    ]
    / 1000.0
)

x = np.arange(len(battery))

fig, ax_power = plt.subplots(
    figsize=(8.3, 4.7)
)

ax_energy = ax_power.twinx()

line_power = ax_power.plot(
    x,
    battery["battery_power_mw"],
    marker="o",
    linewidth=1.6,
    label="Battery power",
)

line_energy = ax_energy.plot(
    x,
    battery["battery_energy_mwh"],
    marker="s",
    linestyle="--",
    linewidth=1.5,
    label="Battery energy",
)

ax_power.axvspan(
    0,
    1,
    alpha=0.10,
)

ax_power.axvspan(
    5,
    6,
    alpha=0.10,
)

ax_power.set_xlabel(
    r"Annualized BTC capacity cost "
    r"[kEUR$_{2020}$/MW$_{BTC}$/a]"
)

ax_power.set_ylabel(
    "Battery power [MW]"
)

ax_energy.set_ylabel(
    "Battery energy [MWh]"
)

ax_power.set_title(
    r"Battery investment window at "
    r"H$_2$ = 2.38 EUR/kg"
)

ax_power.grid(
    alpha=0.22,
    linewidth=0.6,
)

ax_power.set_xticks(x)

ax_power.set_xticklabels(
    [
        f"{value:g}"
        for value in capex_labels
    ],
    rotation=30,
    ha="right",
)

lines = line_power + line_energy

ax_power.legend(
    lines,
    [
        line.get_label()
        for line in lines
    ],
    frameon=False,
    loc="upper right",
)

fig.text(
    0.5,
    -0.02,
    (
        "Shaded ranges denote sampled brackets "
        "for battery entry and exit."
    ),
    ha="center",
    fontsize=8.5,
)

fig.subplots_adjust(
    bottom=0.22,
)

save_figure(
    fig,
    "fig_battery_window_h2_238",
)


# =============================================================================
# TABLE 1
# Final sampled thresholds
# =============================================================================

threshold_table = thresholds.copy()

threshold_table[
    "lower_kEUR_per_MW_a"
] = (
    threshold_table[
        "lower_bound_eur_per_mw_year"
    ]
    / 1000.0
)

threshold_table[
    "upper_kEUR_per_MW_a"
] = (
    threshold_table[
        "upper_bound_eur_per_mw_year"
    ]
    / 1000.0
)

threshold_table[
    "width_kEUR_per_MW_a"
] = (
    threshold_table[
        "bracket_width_eur_per_mw_year"
    ]
    / 1000.0
)

threshold_table["family"] = (
    threshold_table["family"]
    .map(FAMILY_LABELS)
    .fillna(
        threshold_table["family"]
    )
)

threshold_table = threshold_table[
    [
        "finding",
        "family",
        "h2_value_eur_per_kg",
        "lower_kEUR_per_MW_a",
        "upper_kEUR_per_MW_a",
        "width_kEUR_per_MW_a",
    ]
]

threshold_csv = (
    TABLE_DIR
    / "table_merchant_thresholds.csv"
)

threshold_table.to_csv(
    threshold_csv,
    index=False,
)

threshold_latex = threshold_table.copy()

threshold_latex.columns = [
    "Finding",
    "Family",
    r"H$_2$ value",
    "Active through",
    "Inactive from",
    "Bracket width",
]

for column in [
    r"H$_2$ value",
    "Active through",
    "Inactive from",
    "Bracket width",
]:
    threshold_latex[column] = (
        threshold_latex[column]
        .map(
            lambda x: (
                ""
                if pd.isna(x)
                else f"{float(x):.3f}".rstrip("0").rstrip(".")
            )
        )
    )

latex_body = threshold_latex.to_latex(
    index=False,
    escape=False,
    column_format=(
        r">{\RaggedRight\arraybackslash}X "
        r"l r r r r"
    ),
)

latex_body = latex_body.replace(
    r"\begin{tabular}{>{\RaggedRight\arraybackslash}X l r r r r}",
    r"\begin{tabularx}{\textwidth}{>{\RaggedRight\arraybackslash}X l r r r r}",
)

latex_body = latex_body.replace(
    r"\end{tabular}",
    r"\end{tabularx}",
)

threshold_tex = (
    TABLE_DIR
    / "table_merchant_thresholds.tex"
)

threshold_tex.write_text(
    (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Sampled economic transition brackets "
        "for the merchant off-grid sensitivity analysis. "
        "Bounds are tested optimization points and are not "
        "interpolated break-even values.}\n"
        "\\label{tab:merchant_thresholds}\n"
        "\\footnotesize\n"
        + latex_body
        + "\\vspace{2pt}\n"
        "\\parbox{0.96\\textwidth}{\\footnotesize "
        "H$_2$ values are in EUR$_{2020}$/kg$_{H_2}$. "
        "Lower, upper and width values are in "
        "kEUR$_{2020}$/MW$_{BTC}$/a.}\n"
        "\\end{table}\n"
    ),
    encoding="utf-8",
)

print(
    f"Saved: {threshold_csv.relative_to(BASE_DIR)}"
)
print(
    f"Saved: {threshold_tex.relative_to(BASE_DIR)}"
)


# =============================================================================
# TABLE 2
# Selected representative Merchant cases
# =============================================================================

selected_cases = [
    ("joint",   0.00, 162500.0),
    ("joint_b", 0.00, 162500.0),

    ("joint",   2.38, 200000.0),
    ("joint_b", 2.38, 200000.0),

    ("joint",   2.75, 162500.0),
    ("joint_b", 2.75, 162500.0),

    ("joint",   2.75, 175000.0),
    ("joint_b", 2.75, 175000.0),
]

selected_rows = []

for family, h2_value, btc_capex in selected_cases:
    subset = cases[
        (
            cases["joint_family"]
            == family
        )
        &
        np.isclose(
            cases[
                "sweep_h2_sale_value_eur_per_kg"
            ],
            h2_value,
            rtol=0.0,
            atol=1e-9,
        )
        &
        np.isclose(
            cases[
                "sweep_btc_capex_eur_per_mw_year"
            ],
            btc_capex,
            rtol=0.0,
            atol=1e-6,
        )
    ]

    if len(subset) != 1:
        raise RuntimeError(
            "Expected one representative case for "
            f"{family}, H2={h2_value}, "
            f"BTC CAPEX={btc_capex}; "
            f"found {len(subset)}."
        )

    selected_rows.append(
        subset.iloc[0]
    )

representative = pd.DataFrame(
    selected_rows
).reset_index(drop=True)

representative_table = pd.DataFrame(
    {
        "Family":
            representative[
                "joint_family"
            ].map(FAMILY_LABELS),

        "H2_value_eur_per_kg":
            representative[
                "sweep_h2_sale_value_eur_per_kg"
            ],

        "BTC_CAPEX_kEUR_per_MW_a":
            representative[
                "sweep_btc_capex_eur_per_mw_year"
            ] / 1000.0,

        "Regime":
            representative[
                "merchant_regime"
            ].map(REGIME_LABELS),

        "H2_delivered_kt":
            representative[
                "hydrogen_delivered_kt"
            ],

        "BTC_capacity_MW":
            representative[
                "bitcoin_capacity_mw"
            ],

        "Battery_power_MW":
            representative[
                "battery_power_mw"
            ],

        "Battery_energy_MWh":
            representative[
                "battery_energy_mwh"
            ],
    }
)

representative_csv = (
    TABLE_DIR
    / "table_merchant_representative_cases.csv"
)

representative_table.to_csv(
    representative_csv,
    index=False,
)

representative_latex = representative_table.copy()

representative_latex.columns = [
    "Family",
    r"H$_2$ value",
    "BTC CAPEX",
    "Regime",
    r"H$_2$ [kt/a]",
    "BTC [MW]",
    "Battery [MW]",
    "Battery [MWh]",
]

for column in [
    r"H$_2$ value",
    "BTC CAPEX",
]:
    representative_latex[column] = (
        representative_latex[column]
        .map(
            lambda x: (
                f"{float(x):.3f}"
                .rstrip("0")
                .rstrip(".")
            )
        )
    )

for column in [
    r"H$_2$ [kt/a]",
    "BTC [MW]",
    "Battery [MW]",
    "Battery [MWh]",
]:
    representative_latex[column] = (
        representative_latex[column]
        .map(
            lambda x: f"{float(x):.2f}"
        )
    )

latex_body = representative_latex.to_latex(
    index=False,
    escape=False,
    column_format="lrrlrrrr",
)

representative_tex = (
    TABLE_DIR
    / "table_merchant_representative_cases.tex"
)

representative_tex.write_text(
    (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Selected representative optimized "
        "merchant cases illustrating BTC-only operation, "
        "H$_2$--BTC coexistence, battery selection, and "
        "BTC crowd-out.}\n"
        "\\label{tab:merchant_representative_cases}\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + latex_body
        + "}\n"
        "\\vspace{2pt}\n"
        "\\parbox{0.96\\textwidth}{\\footnotesize "
        "H$_2$ values are in EUR$_{2020}$/kg$_{H_2}$; "
        "BTC CAPEX is in "
        "kEUR$_{2020}$/MW$_{BTC}$/a.}\n"
        "\\end{table}\n"
    ),
    encoding="utf-8",
)

print(
    f"Saved: {representative_csv.relative_to(BASE_DIR)}"
)
print(
    f"Saved: {representative_tex.relative_to(BASE_DIR)}"
)


# =============================================================================
# Final summary
# =============================================================================

print()
print("=" * 72)
print("FINAL CLEAN MERCHANT THESIS OUTPUTS")
print("=" * 72)

print(
    f"Source cases: {len(cases)}"
)

print()
print("Figures:")
for path in sorted(
    FIGURE_DIR.glob("*")
):
    print(
        f"  {path.relative_to(BASE_DIR)}"
    )

print()
print("Tables:")
for path in sorted(
    TABLE_DIR.glob("*")
):
    print(
        f"  {path.relative_to(BASE_DIR)}"
    )
