from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

S0_PATH = Path(
    "results/s0_thesis_2045/dispatch_timeseries.csv"
)

S3_PATH = Path(
    "results/s3_thesis_2045/dispatch_timeseries.csv"
)

FIGDIR = Path(
    "results/thesis_presentation/figures"
)

TABLEDIR = Path(
    "results/thesis_presentation/tables"
)

FIGDIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLEDIR.mkdir(
    parents=True,
    exist_ok=True,
)


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# =============================================================================
# Load
# =============================================================================

s0 = pd.read_csv(
    S0_PATH,
    parse_dates=["snapshot"],
)

s3 = pd.read_csv(
    S3_PATH,
    parse_dates=["snapshot"],
)

s0 = (
    s0
    .set_index("snapshot")
    .sort_index()
)

s3 = (
    s3
    .set_index("snapshot")
    .sort_index()
)


# =============================================================================
# Validation
# =============================================================================

if not s0.index.equals(s3.index):
    raise ValueError(
        "S0 and S3 snapshots are not identical."
    )

required_s3 = [
    "battery_charge_input_mw",
    "battery_discharge_output_mw",
    "battery_soc_mwh",
    "bitcoin_consumption_mw",
]

missing = [
    col
    for col in required_s3
    if col not in s3.columns
]

if missing:
    raise KeyError(
        f"Missing S3 columns: {missing}"
    )


# =============================================================================
# Select seven-day high-interaction period
# =============================================================================

s3["battery_ac_throughput_mw"] = (
    s3[
        "battery_charge_input_mw"
    ].clip(lower=0.0)
    +
    s3[
        "battery_discharge_output_mw"
    ].clip(lower=0.0)
)

candidate_starts = (
    s3.index[
        s3.index.hour == 0
    ]
)

best_start = None
best_end = None
best_throughput = None

for start in candidate_starts:

    end = (
        start
        + pd.Timedelta(days=7)
    )

    window = s3.loc[
        (s3.index >= start)
        & (s3.index < end)
    ]

    if len(window) != 168:
        continue

    throughput = (
        window[
            "battery_ac_throughput_mw"
        ].sum()
    )

    if (
        best_throughput is None
        or throughput > best_throughput
    ):

        best_start = start
        best_end = end
        best_throughput = throughput


if best_start is None:
    raise RuntimeError(
        "No complete 168-hour period found."
    )


s0_week = s0.loc[
    (s0.index >= best_start)
    & (s0.index < best_end)
].copy()

s3_week = s3.loc[
    (s3.index >= best_start)
    & (s3.index < best_end)
].copy()


# =============================================================================
# Net battery power
# =============================================================================
#
# Positive = discharge to AC system
# Negative = charging from AC system
# =============================================================================

s3_week["battery_net_power_mw"] = (
    s3_week[
        "battery_discharge_output_mw"
    ]
    -
    s3_week[
        "battery_charge_input_mw"
    ]
)


# =============================================================================
# Clean tiny values
# =============================================================================

for df in [
    s0_week,
    s3_week,
]:

    numeric_cols = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    for col in numeric_cols:

        df[col] = df[col].where(
            df[col].abs() >= 1e-9,
            0.0,
        )


# =============================================================================
# Reproducibility metadata
# =============================================================================

selection = pd.DataFrame(
    [
        {
            "selection_method":
                "Maximum 7-day AC-side battery throughput in S3",

            "window_start":
                best_start,

            "window_end_exclusive":
                best_end,

            "hours":
                len(s3_week),

            "battery_charge_mwh":
                s3_week[
                    "battery_charge_input_mw"
                ].sum(),

            "battery_discharge_mwh":
                s3_week[
                    "battery_discharge_output_mw"
                ].sum(),

            "battery_ac_throughput_mwh":
                best_throughput,

            "btc_consumption_mwh":
                s3_week[
                    "bitcoin_consumption_mw"
                ].sum(),

            "electrolyzer_consumption_mwh":
                s3_week[
                    "electrolyzer_input_mw"
                ].sum(),
        }
    ]
)

selection.to_csv(
    TABLEDIR
    / "dispatch_week_selection.csv",
    index=False,
)


# =============================================================================
# Figure
# =============================================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(11.0, 8.6),
    sharex=True,
)


# -----------------------------------------------------------------------------
# Panel A — S0
# -----------------------------------------------------------------------------

ax = axes[0]

ax.stackplot(
    s0_week.index,
    s0_week["solar_dispatch_mw"],
    s0_week["wind_dispatch_mw"],
    labels=[
        "Solar PV",
        "Wind",
    ],
)

ax.plot(
    s0_week.index,
    -s0_week[
        "electrolyzer_input_mw"
    ],
    linewidth=1.5,
    label="PEM electrolyzer",
)

ax.axhline(
    0.0,
    linewidth=0.8,
    linestyle="--",
)

ax.set_ylabel(
    "Power [MW]"
)

ax.set_title(
    "S0 — hydrogen-only reference"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper right",
)


# -----------------------------------------------------------------------------
# Panel B — S3
# -----------------------------------------------------------------------------

ax = axes[1]

ax.stackplot(
    s3_week.index,
    s3_week["solar_dispatch_mw"],
    s3_week["wind_dispatch_mw"],
    labels=[
        "Solar PV",
        "Wind",
    ],
)

ax.plot(
    s3_week.index,
    -s3_week[
        "electrolyzer_input_mw"
    ],
    linewidth=1.4,
    label="PEM electrolyzer",
)

ax.plot(
    s3_week.index,
    -s3_week[
        "bitcoin_consumption_mw"
    ],
    linewidth=1.3,
    label="Bitcoin mining",
)

ax.plot(
    s3_week.index,
    s3_week[
        "battery_net_power_mw"
    ],
    linewidth=1.2,
    linestyle="--",
    label="Battery net power",
)

ax.axhline(
    0.0,
    linewidth=0.8,
    linestyle="--",
)

ax.set_ylabel(
    "Power [MW]"
)

ax.set_title(
    "S3 — hydrogen + battery + Bitcoin"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper right",
)


# -----------------------------------------------------------------------------
# Panel C — S3 battery SOC
# -----------------------------------------------------------------------------

ax = axes[2]

ax.plot(
    s3_week.index,
    s3_week[
        "battery_soc_mwh"
    ],
    linewidth=1.4,
)

ax.set_ylabel(
    "Battery SOC [MWh]"
)

ax.set_xlabel(
    "Time"
)

ax.set_title(
    "S3 battery state of charge"
)

ax.grid(
    alpha=0.25,
)

ax.set_ylim(
    bottom=0.0,
)


# =============================================================================
# Shared x-axis formatting
# =============================================================================

axes[2].xaxis.set_major_locator(
    mdates.DayLocator()
)

axes[2].xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%d %b"
    )
)


fig.suptitle(
    (
        "Illustrative high-interaction period: "
        f"{best_start:%d %b %Y} – "
        f"{(best_end - pd.Timedelta(hours=1)):%d %b %Y}"
    ),
    y=0.995,
)


fig.text(
    0.5,
    0.008,
    (
        "Positive power denotes electricity supply; "
        "negative power denotes electricity consumption. "
        "Battery net power is positive during discharge "
        "and negative during charging. "
        "The period is selected algorithmically as the "
        "seven-day S3 window with maximum AC-side "
        "battery throughput."
    ),
    ha="center",
    va="bottom",
    fontsize=8.5,
)


fig.tight_layout(
    rect=[
        0.0,
        0.045,
        1.0,
        0.965,
    ]
)


# =============================================================================
# Save
# =============================================================================

pdf_path = (
    FIGDIR
    / "fig05_dispatch_s0_s3_high_interaction_week.pdf"
)

png_path = (
    FIGDIR
    / "fig05_dispatch_s0_s3_high_interaction_week.png"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Console summary
# =============================================================================

print()
print("=" * 80)
print("FINAL S0-S3 DISPATCH FIGURE GENERATED")
print("=" * 80)

print()
print(
    "Selection method: "
    "maximum 7-day AC-side battery throughput in S3"
)

print(
    f"Period:             "
    f"{best_start} to "
    f"{best_end - pd.Timedelta(hours=1)}"
)

print(
    f"Battery charge:     "
    f"{selection.loc[0, 'battery_charge_mwh']:,.2f} MWh"
)

print(
    f"Battery discharge:  "
    f"{selection.loc[0, 'battery_discharge_mwh']:,.2f} MWh"
)

print(
    f"Battery throughput: "
    f"{selection.loc[0, 'battery_ac_throughput_mwh']:,.2f} MWh"
)

print(
    f"BTC consumption:    "
    f"{selection.loc[0, 'btc_consumption_mwh']:,.2f} MWh"
)

print(
    f"PEM consumption:    "
    f"{selection.loc[0, 'electrolyzer_consumption_mwh']:,.2f} MWh"
)

print()
print("Outputs:")
print(" ", pdf_path)
print(" ", png_path)
print(
    " ",
    TABLEDIR
    / "dispatch_week_selection.csv",
)
print()
