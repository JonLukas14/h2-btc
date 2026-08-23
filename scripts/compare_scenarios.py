from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Central thesis scenarios
# =============================================================================

SCENARIOS = {
    "S0": Path("results/s0_thesis_2045/summary.csv"),
    "S1": Path("results/s1_thesis_2045/summary.csv"),
    "S2": Path("results/s2_thesis_2045/summary.csv"),
    "S3": Path("results/s3_thesis_2045/summary.csv"),
}

OUTDIR = Path("results/thesis_comparison")
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load scenario summaries
# =============================================================================

rows = []

for label, path in SCENARIOS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing summary file for {label}: {path}"
        )

    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"{path} must contain exactly one summary row."
        )

    row = df.iloc[0].copy()
    row["scenario"] = label
    rows.append(row)

summary = (
    pd.DataFrame(rows)
    .set_index("scenario")
    .sort_index()
)


# =============================================================================
# Consistency checks
# =============================================================================

common_fields = [
    "investment_year",
    "weather_year",
    "h2_target_annual_kt",
    "h2_lhv_kwh_per_kg",
    "electrolyzer_efficiency",
]

for field in common_fields:
    if summary[field].nunique(dropna=False) != 1:
        raise ValueError(
            f"Central scenarios differ in '{field}':\n"
            f"{summary[field]}"
        )


# =============================================================================
# Derived quantities
# =============================================================================

summary["renewable_capacity_total_mw"] = (
    summary["solar_capacity_mw"]
    + summary["wind_capacity_mw"]
)

# Remove meaningless negative numerical zero.
numeric_columns = summary.select_dtypes(
    include=[np.number]
).columns

summary[numeric_columns] = (
    summary[numeric_columns]
    .mask(
        summary[numeric_columns].abs() < 1e-9,
        0.0,
    )
)


# =============================================================================
# Thesis comparison table
# =============================================================================

metrics = [
    (
        "hydrogen_delivered_kt",
        "Hydrogen production",
        "kt/a",
        1.0,
    ),
    (
        "solar_capacity_mw",
        "Solar PV capacity",
        "MW",
        1.0,
    ),
    (
        "wind_capacity_mw",
        "Wind capacity",
        "MW",
        1.0,
    ),
    (
        "renewable_capacity_total_mw",
        "Total renewable capacity",
        "MW",
        1.0,
    ),
    (
        "electrolyzer_capacity_mw",
        "PEM electrolyzer capacity",
        "MW_el",
        1.0,
    ),
    (
        "electrolyzer_capacity_factor",
        "PEM capacity factor",
        "%",
        100.0,
    ),
    (
        "battery_power_mw",
        "Battery power capacity",
        "MW",
        1.0,
    ),
    (
        "battery_energy_mwh",
        "Battery energy capacity",
        "MWh",
        1.0,
    ),
    (
        "battery_duration_h",
        "Battery duration",
        "h",
        1.0,
    ),
    (
        "battery_equivalent_full_cycles_per_year",
        "Battery equivalent cycles",
        "cycles/a",
        1.0,
    ),
    (
        "solar_curtailment_rate",
        "Solar curtailment",
        "%",
        100.0,
    ),
    (
        "wind_curtailment_rate",
        "Wind curtailment",
        "%",
        100.0,
    ),
    (
        "renewable_utilization_rate",
        "Renewable utilization",
        "%",
        100.0,
    ),
    (
        "bitcoin_capacity_mw",
        "Bitcoin mining capacity",
        "MW",
        1.0,
    ),
    (
        "bitcoin_consumption_mwh",
        "Bitcoin electricity consumption",
        "MWh/a",
        1.0,
    ),
    (
        "bitcoin_utilization_rate",
        "Bitcoin utilization",
        "%",
        100.0,
    ),
    (
        "bitcoin_equivalent_full_load_hours",
        "Bitcoin full-load hours",
        "h/a",
        1.0,
    ),
    (
        "bitcoin_gross_revenue_eur_per_year",
        "Bitcoin gross revenue",
        "EUR2020/a",
        1.0,
    ),
    (
        "annualized_fixed_cost_eur_per_year",
        "Annualized fixed cost",
        "EUR2020/a",
        1.0,
    ),
    (
        "non_bitcoin_variable_operating_cost_eur_per_year",
        "Non-Bitcoin variable OPEX",
        "EUR2020/a",
        1.0,
    ),
    (
        "gross_system_expenditure_eur_per_year",
        "Gross system expenditure",
        "EUR2020/a",
        1.0,
    ),
    (
        "net_system_cost_eur_per_year",
        "Net system cost",
        "EUR2020/a",
        1.0,
    ),
    (
        "gross_system_expenditure_eur_per_kg_h2",
        "Gross system expenditure per kg H2",
        "EUR2020/kg_H2",
        1.0,
    ),
    (
        "net_system_cost_eur_per_kg_h2",
        "Net system cost per kg H2",
        "EUR2020/kg_H2",
        1.0,
    ),
]

records = []

for key, label, unit, scale in metrics:
    record = {
        "Metric": label,
        "Unit": unit,
    }

    for scenario in SCENARIOS:
        value = summary.loc[scenario, key]

        if pd.isna(value):
            record[scenario] = np.nan
        else:
            record[scenario] = float(value) * scale

    records.append(record)

comparison = pd.DataFrame(records)


# =============================================================================
# Save outputs
# =============================================================================

comparison_csv = (
    OUTDIR
    / "central_scenarios_comparison.csv"
)

comparison_html = (
    OUTDIR
    / "central_scenarios_comparison.html"
)

comparison.to_csv(
    comparison_csv,
    index=False,
)

comparison.to_html(
    comparison_html,
    index=False,
    float_format=lambda x: f"{x:,.4f}",
)


# =============================================================================
# Key scenario deltas
# =============================================================================

def pct_change(new, old):
    if abs(old) < 1e-12:
        return np.nan
    return 100.0 * (new - old) / old


print()
print("=" * 78)
print("CENTRAL THESIS SCENARIO COMPARISON")
print("=" * 78)

print()
print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:,.3f}",
    )
)

print()
print("=" * 78)
print("KEY CONTROLLED COMPARISONS")
print("=" * 78)

# S1 - S0
print()
print("S1 - S0: BATTERY EFFECT WITHOUT BITCOIN")
print(
    f"  Battery power: "
    f"{summary.loc['S1', 'battery_power_mw']:.3f} MW"
)
print(
    f"  Battery energy: "
    f"{summary.loc['S1', 'battery_energy_mwh']:.3f} MWh"
)
print(
    f"  Net system-cost change: "
    f"{summary.loc['S1', 'net_system_cost_eur_per_year'] - summary.loc['S0', 'net_system_cost_eur_per_year']:,.2f} EUR/a"
)

# S2 - S0
print()
print("S2 - S0: BITCOIN EFFECT WITHOUT BATTERY")

renew_s0 = summary.loc[
    "S0",
    "renewable_capacity_total_mw",
]
renew_s2 = summary.loc[
    "S2",
    "renewable_capacity_total_mw",
]

print(
    f"  Renewable capacity change: "
    f"{pct_change(renew_s2, renew_s0):.2f}%"
)

print(
    f"  Bitcoin electricity: "
    f"{summary.loc['S2', 'bitcoin_consumption_mwh']:,.2f} MWh/a"
)

print(
    f"  Bitcoin utilization: "
    f"{100 * summary.loc['S2', 'bitcoin_utilization_rate']:.2f}%"
)

print(
    f"  Net system-cost change: "
    f"{pct_change(summary.loc['S2', 'net_system_cost_eur_per_year'], summary.loc['S0', 'net_system_cost_eur_per_year']):.2f}%"
)

# S3 - S2
print()
print("S3 - S2: BATTERY EFFECT WITH BITCOIN")

print(
    f"  Battery power: "
    f"{summary.loc['S3', 'battery_power_mw']:.3f} MW"
)

print(
    f"  Battery energy: "
    f"{summary.loc['S3', 'battery_energy_mwh']:.3f} MWh"
)

print(
    f"  Battery duration: "
    f"{summary.loc['S3', 'battery_duration_h']:.3f} h"
)

print(
    f"  Bitcoin electricity change: "
    f"{summary.loc['S3', 'bitcoin_consumption_mwh'] - summary.loc['S2', 'bitcoin_consumption_mwh']:,.2f} MWh/a"
)

print(
    f"  Net system-cost change: "
    f"{summary.loc['S3', 'net_system_cost_eur_per_year'] - summary.loc['S2', 'net_system_cost_eur_per_year']:,.2f} EUR/a"
)

print(
    f"  Relative net system-cost change: "
    f"{pct_change(summary.loc['S3', 'net_system_cost_eur_per_year'], summary.loc['S2', 'net_system_cost_eur_per_year']):.2f}%"
)

print()
print("Outputs:")
print(f"  {comparison_csv}")
print(f"  {comparison_html}")
print()
