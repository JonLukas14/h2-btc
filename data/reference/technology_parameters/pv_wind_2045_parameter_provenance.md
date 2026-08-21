# Utility-scale PV and onshore wind — 2045 parameter provenance

## Purpose

This file documents the proposed techno-economic parameters for utility-scale
solar PV and onshore wind in the final 2045 off-grid thesis model.

The parameters are based primarily on the common PyPSA technology-data v0.13.2
2045 reference dataset already preserved in this repository:

`technology-data/outputs/costs_2045.csv`

The purpose of this file is to distinguish clearly between:

- upstream source-supported technology parameters;
- upstream currency normalization;
- derived annualized costs;
- model fallback/default values;
- and thesis-specific financial assumptions.

The frozen 2030 S0-S3 validation scenarios are not modified by this parameter
definition.

---

## 1. Common technology-cost dataset

Upstream repository:

`PyPSA/technology-data`

Release:

`v0.13.2`

Cost year:

`2045`

Local source file:

`technology-data/outputs/costs_2045.csv`

SHA256:

`0ce46a0078cfe640e0dcee8f8ac07d95edf60d33cf9bdaa57bddf722e6f4b328`

The identical raw file is used as the technology-data source in the
PyPSA-Earth Kazakhstan workflow.

This provides a common techno-economic reference basis between the custom
off-grid model and the PyPSA-Earth model.

---

## 2. Utility-scale solar PV

Technology-data identifier:

`solar-utility`

Primary upstream source:

Danish Energy Agency (DEA),
technology-data sheet `22 Utility-scale PV`.

### 2045 raw parameters

| Parameter | Model candidate value | Unit | Classification |
|---|---:|---|---|
| Investment cost | 306.7008 | EUR2020/kW_e | SOURCE-SUPPORTED / technology-data output |
| FOM | 2.5269 | % of CAPEX/year | SOURCE-SUPPORTED |
| Lifetime | 40 | years | SOURCE-SUPPORTED |
| VOM | 0.0 | EUR/MWh | MODEL DEFAULT; no explicit solar-utility VOM row |

The utility-scale PV technology is selected instead of the generic PyPSA-Earth
`solar` carrier because the off-grid system represents a large centralized
renewable-energy installation rather than a national mixture of rooftop and
utility-scale PV.

### VOM treatment

The raw technology-data v0.13.2 2045 table does not contain an explicit
`solar-utility / VOM` row.

The custom model currently requests:

`get_cost("solar-utility", "VOM", 0.0)`

and therefore uses:

`VOM = 0 EUR/MWh`

if no explicit value is available.

This must be described as a modelling fallback/default and must not be
attributed to the Danish Energy Agency as an explicit zero-VOM estimate.

---

## 3. Onshore wind

Technology-data identifier:

`onwind`

Primary upstream source:

Danish Energy Agency (DEA),
technology-data sheet `20 Onshore turbines`.

### 2045 raw parameters

| Parameter | Model candidate value | Unit | Classification |
|---|---:|---|---|
| Investment cost | 1026.8091 | EUR2020/kW | SOURCE-SUPPORTED / technology-data output |
| FOM | 1.1817 | % of CAPEX/year | SOURCE-SUPPORTED |
| VOM | 1.3000 | EUR2020/MWh | SOURCE-SUPPORTED |
| Lifetime | 30 | years | SOURCE-SUPPORTED |

The onshore-wind VOM is an explicit technology-data parameter and is therefore
different in evidence status from the utility-PV zero-VOM fallback.

---

## 4. Currency-year treatment

The raw output CSV retains a `currency_year` metadata field.

For the relevant technologies it reports:

- utility-scale PV: source currency year 2020;
- onshore wind: source currency year 2015.

These metadata values describe the original source-money year and must not be
interpreted as the monetary basis that still needs to be converted by the
custom model.

PyPSA technology-data v0.13.2 specifies:

`eur_year: 2020`

The technology-data compilation workflow applies inflation adjustment to
monetary parameters before writing the output cost tables. The inflation
routine applies the adjustment to:

- investment;
- VOM;
- fuel.

Consequently, the numerical monetary values in the generated
`costs_2045.csv` are already harmonized to the technology-data EUR2020
reference basis.

No additional 2015-to-2020 inflation adjustment is therefore applied in the
custom off-grid model.

Applying an additional inflation conversion would double-count the upstream
currency normalization.

---

## 5. Discount-rate convention

### Current validation convention

The frozen validation scenarios use:

`costs.discount_rate = 0.07`

This remains part of the 2030 validation architecture and is not retroactively
changed.

### Proposed final thesis central assumption

Proposed real discount rate:

`8.0 %`

The 8% value is treated as a thesis-level financial modelling assumption for
the 2045 off-grid system.

It is intended to be applied consistently to capital investments across the
off-grid system rather than separately selecting different discount rates for
PV, wind, battery and electrolysis.

The 8% value must not be described as a directly observed Kazakhstan 2045
market WACC.

Supporting context includes Kazakhstan-specific hydrogen techno-economic work
using an 8% discount rate.

### Lower sensitivity benchmark

A lower sensitivity value is:

`6.3 % real`

IRENA's country- and technology-specific financing dataset reports a real
after-tax WACC of 6.3% for Kazakhstan for both:

- utility-scale solar PV;
- onshore wind.

The benchmark relates to financing conditions around 2021 and therefore is not
a direct forecast of financing conditions in 2045.

Its role in the thesis is to provide a Kazakhstan-specific lower financing
benchmark rather than a claim about future financing conditions.

---

## 6. Annualization method

The custom model converts investment costs to the PyPSA MW/MWh nominal
capacity basis before annualization.

For power-capacity technologies:

`EUR/kW * 1000 = EUR/MW`

The capital recovery factor is:

`CRF(r,n) = r / (1 - (1 + r)^(-n))`

Annualized investment cost is:

`annualized CAPEX = investment * CRF(r,n)`

Annual fixed O&M is:

`annualized FOM = investment * FOM / 100`

The value assigned to the PyPSA `capital_cost` attribute is:

`capital_cost = annualized CAPEX + annualized FOM`

Variable operating cost is represented separately through the Generator
`marginal_cost` attribute.

---

## 7. Derived annualized costs at the proposed 8% central rate

These values are derived calculations, not additional external source values.

### Utility-scale PV

Investment:

`306,700.80 EUR2020/MW`

Lifetime:

`40 years`

CRF at 8%:

`0.0838601615`

Annualized CAPEX:

`25,719.98 EUR2020/MW/year`

Annual FOM:

`7,750.02 EUR2020/MW/year`

Total annualized fixed cost:

`33,470.00 EUR2020/MW/year`

### Onshore wind

Investment:

`1,026,809.10 EUR2020/MW`

Lifetime:

`30 years`

CRF at 8%:

`0.0888274334`

Annualized CAPEX:

`91,208.82 EUR2020/MW/year`

Annual FOM:

`12,133.80 EUR2020/MW/year`

Total annualized fixed cost:

`103,342.62 EUR2020/MW/year`

The model should calculate these values from the raw parameters and configured
discount rate rather than hard-code the derived annualized values.

---

## 8. Separation from PyPSA-Earth numerical marginal costs

The solved PyPSA-Earth Kazakhstan network must not be used as the source for
physical renewable VOM assumptions.

In the inspected PyPSA-Earth configuration, renewable marginal costs are
overwritten for dispatch purposes to:

- solar: 0.01 EUR/MWh;
- onwind: 0.015 EUR/MWh.

The solved scenario also uses:

`noisy_costs: true`

which adds a small solve-time perturbation to marginal costs.

Therefore solved renewable marginal costs around approximately
0.019-0.021 EUR/MWh for solar and 0.024-0.026 EUR/MWh for onshore wind are
numerical dispatch/anti-degeneracy quantities, not physical O&M assumptions.

The off-grid model instead uses the source-supported technology-data VOM where
available and explicitly documents fallback values where it is not.

---

## 9. Model parameter status

### Utility-scale PV

Investment cost:
CANDIDATE FINAL — source-supported, 2045 technology-data v0.13.2.

FOM:
CANDIDATE FINAL — source-supported.

Lifetime:
CANDIDATE FINAL — source-supported.

VOM:
CANDIDATE FINAL MODEL DEFAULT — explicit provenance caveat required.

### Onshore wind

Investment cost:
CANDIDATE FINAL — source-supported, 2045 technology-data v0.13.2.

FOM:
CANDIDATE FINAL — source-supported.

VOM:
CANDIDATE FINAL — source-supported.

Lifetime:
CANDIDATE FINAL — source-supported.

### Discount rate

8% central:
PROPOSED THESIS MODELLING ASSUMPTION.

6.3%:
KAZAKHSTAN-SPECIFIC LOWER FINANCING SENSITIVITY BENCHMARK.

No final thesis scenario configuration has been modified by this provenance
file.

---

## 10. Sources

PyPSA technology-data v0.13.2:
https://github.com/PyPSA/technology-data/tree/v0.13.2

2045 output cost file:
https://raw.githubusercontent.com/PyPSA/technology-data/v0.13.2/outputs/costs_2045.csv

technology-data v0.13.2 configuration:
https://raw.githubusercontent.com/PyPSA/technology-data/v0.13.2/config.yaml

technology-data compilation script:
https://raw.githubusercontent.com/PyPSA/technology-data/v0.13.2/scripts/compile_cost_assumptions.py

technology-data inflation helper:
https://raw.githubusercontent.com/PyPSA/technology-data/v0.13.2/scripts/_helpers.py

IRENA (2023), The cost of financing for renewable power:
https://www.irena.org/Publications/2023/May/The-cost-of-financing-for-renewable-power

IRENA financing data appendix:
https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/May/IRENA_Cost_of_financing_renewable_power_Appendix_2023.pdf

H2Diplo (2025), Case Study: Kazakhstan on the Way to Green H2 Ramp-Up:
https://h2diplo.de/wp-content/uploads/2025/10/Study_Kazakhstan_H2_Ramp-Up_EN.pdf

Sources and upstream implementation were checked on 2026-08-21.
