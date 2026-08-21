# Battery storage — 2045 parameter provenance

## Purpose

This file documents the proposed battery-storage techno-economic and technical
parameters for the final 2045 off-grid thesis model.

The battery is represented using separate power and energy components so that
battery power [MW] and battery energy [MWh] can be optimized independently.

The principal upstream techno-economic source is the common PyPSA
technology-data v0.13.2 2045 reference dataset:

`technology-data/outputs/costs_2045.csv`

The frozen 2030 validation scenarios are not modified by this parameter
definition.

---

## 1. Technology representation

The battery is represented in PyPSA by:

- one battery bus;
- one charging Link;
- one energy Store;
- one discharging Link.

Technology-data identifiers:

Power component:

`battery inverter`

Energy component:

`battery storage`

The charging and discharging Links represent the two directions of one
physical bidirectional power-conversion system.

The inverter CAPEX is therefore assigned only once.

Battery power capacity and battery energy capacity are independently
extendable.

No fixed battery duration is imposed in the custom off-grid model.

Battery duration is a derived optimization result:

`duration_h = e_nom_opt_mwh / p_nom_opt_mw`

---

## 2. Common source dataset

Repository:

`PyPSA/technology-data`

Release:

`v0.13.2`

Cost year:

`2045`

Local source file:

`technology-data/outputs/costs_2045.csv`

SHA256:

`0ce46a0078cfe640e0dcee8f8ac07d95edf60d33cf9bdaa57bddf722e6f4b328`

The same raw technology-data dataset is used as the upstream 2045 cost source
in the associated PyPSA-Earth Kazakhstan workflow.

---

## 3. Battery inverter / power component

Technology-data identifier:

`battery inverter`

Primary upstream source:

Danish Energy Agency,
`technology_data_catalogue_for_energy_storage.xlsx`

### 2045 parameters

| Parameter | Value | Unit | Classification |
|---|---:|---|---|
| Investment cost | 85.0720 | EUR2020/kW | SOURCE-SUPPORTED |
| FOM | 0.6750 | % of CAPEX/year | SOURCE-SUPPORTED |
| Round-trip efficiency | 0.9600 | per unit | SOURCE-SUPPORTED |
| Technical lifetime | 10 | years | SOURCE-SUPPORTED |

The technology-data description explicitly labels the efficiency parameter as:

`Round trip efficiency DC`

Therefore 0.96 is interpreted as the round-trip efficiency associated with the
generic battery-inverter/battery power-conversion representation.

---

## 4. Charging and discharging efficiency split

The optimization model requires separate charging-Link and discharging-Link
efficiencies, while the upstream technology-data source provides one
round-trip-efficiency value.

The model therefore assumes a symmetric decomposition:

`eta_charge = sqrt(eta_round_trip)`

`eta_discharge = sqrt(eta_round_trip)`

For:

`eta_round_trip = 0.96`

this gives:

`eta_charge = eta_discharge = 0.9797959`

and therefore:

`eta_charge * eta_discharge = 0.96`

The symmetric split is a MODELLING ASSUMPTION.

The round-trip efficiency itself is source-supported, but the source does not
provide the two one-way efficiencies separately.

---

## 5. Battery energy-storage component

Technology-data identifier:

`battery storage`

Primary upstream source:

Danish Energy Agency,
`technology_data_catalogue_for_energy_storage.xlsx`

### 2045 parameters

| Parameter | Value | Unit | Classification |
|---|---:|---|---|
| Investment cost | 89.8573 | EUR2020/kWh | SOURCE-SUPPORTED |
| Technical lifetime | 30 | years | SOURCE-SUPPORTED |

No explicit FOM, VOM or efficiency row is provided for the battery-storage
energy component in the selected technology-data table.

The custom model therefore does not silently attribute such values to the DEA.

---

## 6. Battery power-capacity coupling

PyPSA Link nominal capacity is defined on `bus0`.

The charging Link is:

`electricity_bus -> battery_bus`

so:

`battery_charger.p_nom`

represents the AC-side charging power capacity.

The discharging Link is:

`battery_bus -> electricity_bus`

so:

`battery_discharger.p_nom`

represents battery-side input power during discharge.

AC-side discharge power capacity is therefore:

`eta_discharge * battery_discharger.p_nom`

Equal AC-side bidirectional inverter power is imposed using:

`battery_charger.p_nom = eta_discharge * battery_discharger.p_nom`

This avoids double-counting inverter power capacity and ensures equal
charge/discharge ratings on the common AC-system basis.

The inverter annualized CAPEX is assigned to the charging Link once.
The discharging Link receives zero additional inverter CAPEX.

---

## 7. Independent optimization of power and energy

The custom off-grid model uses:

`battery_charger.p_nom_extendable = true`

and:

`battery_store.e_nom_extendable = true`

There is no constraint of the form:

`e_nom = max_hours * p_nom`

Therefore:

- inverter power [MW] is optimized endogenously;
- storage energy [MWh] is optimized endogenously;
- resulting storage duration [h] is optimized endogenously.

This treatment is deliberate because battery duration is itself relevant to
the interaction between renewable generation, hydrogen electrolysis and
flexible Bitcoin mining.

---

## 8. Difference from PyPSA-Earth battery duration

The inspected PyPSA-Earth configuration uses a fixed battery storage duration
through:

`max_hours: 6`

for the generic `battery` storage technology.

The custom off-grid model deliberately does not impose this 6-hour duration.

This is an explicit model-boundary difference:

PyPSA-Earth:
fixed duration used for national electricity-system storage representation.

Custom off-grid model:
independent battery MW and MWh optimization so that economically preferred
duration can emerge from the H2/BTC system optimization.

This difference must be considered when comparing battery results between the
two models.

---

## 9. Standing loss

The current off-grid battery configuration uses:

`standing_loss = 0.0`

No standing-loss parameter is supplied by the selected generic battery rows in
technology-data v0.13.2.

The zero standing loss is therefore classified as a SIMPLIFYING MODELLING
ASSUMPTION rather than a source-supported physical claim.

The battery round-trip conversion loss remains represented explicitly through
the 0.96 efficiency.

The omission of self-discharge should be stated as a model limitation,
particularly if long storage residence times occur.

---

## 10. Variable operating costs

No separate battery cycling VOM is currently assigned to:

- battery charger;
- battery store;
- battery discharger.

Their PyPSA marginal costs are therefore zero.

This is a MODELLING ASSUMPTION / fallback and must not be interpreted as proof
that real batteries have zero cycling-related degradation or maintenance cost.

Battery degradation, cycle-dependent replacement and state-of-health dynamics
are not represented explicitly in the linear optimization model.

The technology-component lifetimes and annualized investment costs provide the
principal long-term capital-cost representation.

---

## 11. Discount rate and annualization

The proposed central thesis discount rate is:

`8% real`

The same financial convention is intended to be applied consistently across
PV, wind, battery and electrolysis investments.

Capital recovery factor:

`CRF(r,n) = r / (1 - (1 + r)^(-n))`

Annualized fixed cost:

`capital_cost = investment * CRF + investment * FOM / 100`

where an explicit FOM parameter exists.

### Battery inverter at 8%

Investment:

`85,072 EUR2020/MW`

Lifetime:

`10 years`

CRF:

`0.14902949`

Annualized CAPEX:

`12,678.24 EUR2020/MW/year`

Annual FOM:

`574.24 EUR2020/MW/year`

Total inverter fixed cost:

`13,252.47 EUR2020/MW/year`

### Battery storage at 8%

Investment:

`89,857.30 EUR2020/MWh`

Lifetime:

`30 years`

CRF:

`0.08882743`

Annualized storage fixed cost:

`7,981.79 EUR2020/MWh/year`

These annualized values are DERIVED calculations and should be calculated by
the model rather than hard-coded.

---

## 12. Literature cross-check and boundary interpretation

Recent literature identifies lithium-ion BESS as a leading technology for
renewable integration and off-grid hybrid energy systems.

Contemporary whole-system literature values can differ from the technology-data
2045 parameters because of:

- different reference years;
- different battery chemistries;
- AC versus DC efficiency boundaries;
- whole-system versus component-specific lifetimes;
- fixed-duration packaged BESS versus separately costed MW/MWh components;
- future cost projections versus current observed/project costs.

For this reason, contemporary literature is used primarily as a plausibility
and system-boundary cross-check rather than replacing the 2045
technology-data values without normalization.

In particular, literature reporting whole-BESS lifetimes around 15-20 years
should not be assumed automatically equivalent to the separate technology-data
component lifetimes of 10 years for the inverter and 30 years for the energy
storage component.

---

## 13. Model parameter status

Battery inverter CAPEX:
CANDIDATE FINAL — source-supported, technology-data v0.13.2 / 2045.

Battery inverter FOM:
CANDIDATE FINAL — source-supported.

Battery round-trip efficiency:
CANDIDATE FINAL — source-supported.

Symmetric one-way efficiency split:
CANDIDATE FINAL MODELLING ASSUMPTION.

Battery inverter lifetime:
CANDIDATE FINAL — source-supported.

Battery-storage CAPEX:
CANDIDATE FINAL — source-supported.

Battery-storage lifetime:
CANDIDATE FINAL — source-supported, but system-boundary interpretation must be
retained.

Battery-storage FOM/VOM:
MODEL DEFAULT / ASSUMPTION — not explicit DEA parameters in the selected rows.

Standing loss:
SIMPLIFYING MODELLING ASSUMPTION.

Battery MW/MWh independent optimization:
FINAL ARCHITECTURAL DECISION.

Fixed battery duration:
NOT USED in the custom off-grid model.

No final thesis scenario configuration has been modified by this provenance
file.

---

## 14. Sources

PyPSA technology-data v0.13.2:
https://github.com/PyPSA/technology-data/tree/v0.13.2

2045 output cost file:
https://raw.githubusercontent.com/PyPSA/technology-data/v0.13.2/outputs/costs_2045.csv

Danish Energy Agency:
Technology Data Catalogue for Energy Storage.

Taghizad-Tavana et al. (2025),
Hybrid Renewable Energy Systems for Off-Grid Electrification:
A Comprehensive Review of Storage Technologies, Metaheuristic Optimization
Approaches and Key Challenges.
Eng, 6, 309.
https://doi.org/10.3390/eng6110309

Sources and implementation were checked on 2026-08-21.
