# PEM electrolyzer — 2045 parameter provenance

## Purpose

This file documents the techno-economic, efficiency and operating assumptions
for the PEM electrolyzer used in the final 2045 off-grid thesis model.

Unlike utility PV, onshore wind and battery storage, the final electrolyzer
parameters are not copied directly from the generic `electrolysis` row in
PyPSA technology-data v0.13.2.

The generic technology-data row represents an alkaline/AEC reference
technology and is therefore retained only as a common upstream reference.

The thesis base technology is PEM electrolysis (PEMEC), selected because the
off-grid model requires highly flexible operation under variable wind and
solar generation.

No frozen 2030 S0-S3 validation configuration is modified by this provenance
file.

---

## 1. Technology choice

Base electrolyzer technology:

`PEMEC`

PEM electrolysis is selected for the 2045 off-grid base case because the
electrolyzer operates directly with variable renewable electricity.

Literature and technology reviews identify PEM as particularly well suited to:

- rapid load-following;
- intermittent renewable-electricity operation;
- broad part-load operation;
- compact system design;
- pressurized hydrogen production.

This choice is based on operational suitability, not on an assumption that
PEM is universally the cheapest electrolysis technology.

Alkaline electrolysis remains a relevant alternative technology and may be
used later as a technology sensitivity.

---

## 2. Difference from the common technology-data row

The common reference dataset is:

`technology-data/outputs/costs_2045.csv`

PyPSA technology-data:

`v0.13.2`

The generic technology:

`electrolysis`

contains approximately:

- investment: 1100 EUR/kW_e;
- FOM: 4%/year;
- efficiency: 0.6763;
- lifetime: 25 years.

Its source description identifies the underlying DEA technology as:

`86 AEC 100 MW`

and therefore it is not used as the final PEM-specific economic
parameterization.

The legacy technology-data row:

`PEM electrolyzer small size`

is also not used as the final thesis basis because it represents an older
1 MW JRC technology definition with an 8.5-year lifetime that is explicitly
described as likely representing stack rather than complete plant lifetime.

---

## 3. PEM CAPEX

Primary source:

Danish Energy Agency,
Technology Data for Renewable Fuels,
hydrogen production via electrolysis / PEMEC.

The size-specific PEMEC table reports the following CAPEX for a 10 MW plant:

| Cost year | CAPEX |
|---|---:|
| 2040 | 725 EUR/kW_el |
| 2050 | 500 EUR/kW_el |

The catalogue states that these CAPEX values use the 2020 cost level.

The 2045 value is obtained by linear interpolation:

`CAPEX_2045 = 725 + (2045 - 2040)/(2050 - 2040) * (500 - 725)`

Therefore:

`CAPEX_2045 = 612.5 EUR2020/kW_el`

Classification:

DERIVED FROM SOURCE-SUPPORTED 2040 AND 2050 DEA VALUES.

The 10 MW class is used as the reference cost scale.

This does not mean that the optimized electrolyzer capacity is fixed at
10 MW. The optimization remains free to determine installed capacity.

---

## 4. PEM efficiency and hydrogen energy basis

The hydrogen carrier in the custom model uses the lower heating value (LHV).

Configured hydrogen LHV:

`33.33 kWh_H2/kg_H2`

The DEA PEMEC data report LHV hydrogen-output efficiencies of:

- 2040: 61.6%;
- 2050: 66.4%.

The central 2045 efficiency is linearly interpolated:

`eta_2045 = 0.616 + 0.5 * (0.664 - 0.616)`

Therefore:

`eta_LHV_2045 = 0.640`

Classification:

DERIVED FROM SOURCE-SUPPORTED DEA TARGET-YEAR VALUES.

The PyPSA Link therefore represents:

`p_H2,LHV = 0.64 * p_el`

---

## 5. Specific electricity consumption

Specific electricity consumption is derived consistently from the chosen LHV
basis:

`e_el = LHV_H2 / eta_LHV`

Using:

`LHV_H2 = 33.33 kWh/kg`

and:

`eta_LHV = 0.64`

gives:

`e_el = 52.078125 kWh_el/kg_H2`

This is a DERIVED parameter.

It must not be combined with an HHV-based electrolyzer efficiency.

A higher efficiency around 0.67 remains available as a literature-based
sensitivity case, but is not the 2045 central parameter.

---

## 6. System lifetime

Central PEM plant lifetime:

`25 years`

The value represents the electrolyzer plant/system lifetime rather than the
stack replacement interval.

Plant lifetime and stack lifetime are treated separately.

Classification:

SOURCE-SUPPORTED.

---

## 7. Regular fixed O&M

Central regular PEM O&M:

`2% of CAPEX/year`

The DEA technology assessment reports PEMEC yearly OPEX around 2% of CAPEX.

Importantly, this OPEX convention does not include stack replacement.

Therefore the final modelling convention separates:

1. regular annual fixed O&M;
2. stack replacement.

This avoids hiding stack replacement inside a generic FOM percentage.

At the central CAPEX:

`612.5 EUR/kW * 0.02 = 12.25 EUR/kW/year`

or:

`12,250 EUR/MW/year`

Classification:

SOURCE-SUPPORTED RATE + DERIVED MONETARY VALUE.

---

## 8. Stack lifetime

An updated DEA PEMEC technology catalogue reports projected stack replacement
intervals of:

- 2040: 90,000 equivalent operating hours;
- 2050: 105,000 equivalent operating hours.

The 2045 value is linearly interpolated:

`stack_lifetime_2045 = 97,500 equivalent operating hours`

Classification:

DERIVED FROM SOURCE-SUPPORTED DEA VALUES.

This stack lifetime is not treated as the plant lifetime.

---

## 9. Stack replacement cost

DEA industry information reports that the electrolyzer stack represents
approximately:

`30% of total PEMEC CAPEX`

The central stack-replacement cost is therefore approximated as:

`stack_replacement_fraction = 0.30`

At:

`CAPEX = 612.5 EUR2020/kW`

the replacement cost is:

`183.75 EUR2020/kW`

Classification:

SOURCE-SUPPORTED COST FRACTION + DERIVED REPLACEMENT COST.

---

## 10. Throughput-based stack replacement approximation

The linear hourly optimization does not explicitly schedule discrete stack
replacement events.

Stack replacement is therefore represented as a throughput-dependent
electrolyzer operating cost.

For one kW of installed electrolyzer capacity:

`97,500 h * 1 kW = 97.5 MWh_el`

and:

`183.75 EUR / 97.5 MWh_el = 1.884615 EUR/MWh_el`

Central stack-throughput cost:

`1.8846 EUR2020/MWh_el`

This is a MODELLING APPROXIMATION.

It converts the source-supported replacement-cost share and stack lifetime into
a linear throughput cost.

It does not explicitly model:

- discrete replacement timing;
- discounting of individual future stack replacements;
- gradual degradation;
- temperature-dependent degradation;
- start/stop degradation;
- recycling or residual stack value.

The approximation is used to retain a linear optimization formulation while
preventing stack use from being economically free.

---

## 11. Avoidance of O&M double counting

The model uses:

`regular fixed O&M = 2% CAPEX/year`

plus:

`stack throughput cost = 1.8846 EUR/MWh_el`

The fixed O&M convention is selected specifically because the cited DEA
treatment excludes stack replacement.

The stack cost is then represented separately.

A higher O&M percentage that already embeds estimated stack replacement must
not be combined with the explicit throughput stack cost, as this would
double-count replacement expenditure.

---

## 12. Discount rate and annualization

Central thesis real discount rate:

`8%`

Plant lifetime:

`25 years`

Capital recovery factor:

`CRF(8%,25y) = 0.0936787791`

Investment:

`612,500 EUR2020/MW_el`

Annualized CAPEX:

`57,378.25 EUR2020/MW_el/year`

Regular fixed O&M:

`12,250.00 EUR2020/MW_el/year`

Total annualized fixed electrolyzer cost:

`69,628.25 EUR2020/MW_el/year`

These are DERIVED calculations.

The model should calculate the annualized values from the underlying
investment, FOM, lifetime and discount rate rather than hard-coding the final
annualized value.

---

## 13. Operating flexibility

The simplified hourly model allows:

`0 <= p_el,t <= p_nom`

No explicit minimum-load constraint is imposed.

No explicit ramp-rate constraint is imposed.

No startup cost or startup duration is imposed.

No standby electricity consumption is currently represented.

PEM technology has strong literature support for dynamic operation under
variable renewable input.

Nevertheless, unrestricted 0-100% hourly dispatch is a SIMPLIFIED MODELLING
ASSUMPTION rather than a claim that a real PEM plant has no operational
constraints.

The assumption may slightly overstate real operational flexibility.

---

## 14. Hydrogen-production target formulation

The electrolyzer sends hydrogen chemical energy to the hydrogen bus on the
configured LHV basis.

The annual mass target is converted using:

`target_MWh_H2 = target_kt_H2 * 1,000,000 * 33.33 / 1000`

The annual Linopy constraint is:

`sum_t(hydrogen_delivery_t * snapshot_weight_t) = annual_H2_target`

Only the annual hydrogen quantity is constrained.

No exogenous hourly hydrogen-demand profile is imposed.

The hydrogen-delivery component is deliberately oversized so that it does not
constrain the hourly timing of hydrogen production.

The production-target scale itself is a separate scenario assumption and is
not determined by the PEM technology parameterization.

---

## 15. System boundary

The electrolyzer is represented as:

`electricity -> PEM electrolyzer -> H2 chemical energy (LHV)`

The base electrolyzer parameterization covers electrolysis plant conversion and
associated balance-of-plant treatment according to the selected DEA
technology boundary.

The following are not automatically included as downstream hydrogen services:

- hydrogen storage;
- long-distance hydrogen transport;
- hydrogen liquefaction;
- additional downstream compression required by a specific offtake;
- hydrogen sale price or market value.

Downstream compression and storage must be represented separately if they are
introduced later.

Water supply and water-treatment requirements are also tracked separately
rather than silently embedded as an unlimited zero-cost resource.

---

## 16. Comparison with literature

Tremel (2018) supports PEM electrolysis as a highly dynamic technology suited
to fluctuating renewable electricity and supports simplified LHV efficiencies
around the upper part of the approximately 0.64-0.67 range.

Zun and McLellan provide a literature benchmark around 0.64 LHV.

Recent review literature also identifies PEM as especially suitable for
intermittent renewable operation, while emphasizing uncertainty in future
CAPEX, durability and operating costs.

Historical multi-MW PEM CAPEX values should be used as plausibility checks,
not mixed directly with the target-year 2045 DEA CAPEX projection.

---

## 17. Model parameter status

Technology:
FINAL BASE-TECHNOLOGY DECISION — PEMEC.

2045 CAPEX:
CANDIDATE FINAL — 612.5 EUR2020/kW_el, derived from DEA 10 MW 2040/2050
values.

LHV efficiency:
CANDIDATE FINAL — 0.640, derived from DEA 2040/2050 PEMEC efficiency values.

Specific electricity use:
DERIVED — 52.0781 kWh_el/kg_H2.

Plant lifetime:
CANDIDATE FINAL — 25 years.

Regular FOM:
CANDIDATE FINAL — 2% CAPEX/year, excluding stack replacement.

Stack lifetime:
CANDIDATE FINAL — 97,500 equivalent operating hours, interpolated.

Stack replacement fraction:
CANDIDATE FINAL — 30% of system CAPEX.

Stack throughput cost:
CANDIDATE FINAL MODELLING APPROXIMATION — 1.8846 EUR2020/MWh_el.

Discount rate:
PROPOSED FINAL THESIS FINANCIAL ASSUMPTION — 8% real.

0-100% dispatch:
SIMPLIFIED MODELLING ASSUMPTION.

Downstream compression/storage:
EXCLUDED FROM BASE ELECTROLYZER BOUNDARY unless introduced separately.

No frozen validation scenario has been changed by this provenance file.

---

## 18. Sources

Danish Energy Agency,
Technology Data for Renewable Fuels,
Hydrogen production via electrolysis / PEMEC.

Official catalogue landing page:
https://ens.dk/en/analyses-and-statistics/technology-data-renewable-fuels

Catalogue version used for the PEMEC target-year parameter derivation:
https://ens.dk/sites/ens.dk/files/Analyser/version_11_-_technology_data_for_renewable_fuels.pdf

PyPSA technology-data v0.13.2:
https://github.com/PyPSA/technology-data/tree/v0.13.2

Tremel, A. (2018),
Electricity-based Fuels.
Springer.
https://doi.org/10.1007/978-3-319-72459-1

Zun, M. T. and McLellan, B. C. (2023),
Cost Projection of Global Green Hydrogen Production Scenarios.
Hydrogen.
https://doi.org/10.3390/hydrogen4040055

Proost, J. (2019),
State-of-the art CAPEX data for water electrolysers, and their impact on
renewable hydrogen price settings.
International Journal of Hydrogen Energy, 44(9), 4406-4413.
https://doi.org/10.1016/j.ijhydene.2018.07.164

Aminaho, E. N., Aminaho, N. S., and Aminaho, F. (2025),
Techno-economic assessments of electrolyzers for hydrogen production.
Applied Energy, 399, 126515.
https://doi.org/10.1016/j.apenergy.2025.126515

Sources and implementation were checked on 2026-08-21.
