# PyPSA technology-data 2045 cost snapshot provenance

## Purpose

This file documents the provenance of:

`technology-data/outputs/costs_2045.csv`

The CSV is retained as the common upstream techno-economic reference dataset
for the 2045 thesis modelling work.

Individual thesis-model parameters may deliberately deviate from this dataset
where a more technology-specific or Kazakhstan-specific source is judged more
appropriate. Such deviations must be documented separately.

## Upstream source

Repository:

`PyPSA/technology-data`

Release/tag:

`v0.13.2`

Target cost year:

`2045`

Upstream raw file:

`https://raw.githubusercontent.com/PyPSA/technology-data/v0.13.2/outputs/costs_2045.csv`

The local file was downloaded directly from this upstream location without
manual modification.

## File fingerprint

SHA256:

`0ce46a0078cfe640e0dcee8f8ac07d95edf60d33cf9bdaa57bddf722e6f4b328`

## Cross-model consistency

The active PyPSA-Earth configuration used for the Kazakhstan grid-connected
model specifies:

- cost year: 2045
- technology_data_version: v0.13.2
- retrieve_cost_data: true

The PyPSA-Earth raw cost file:

`resources/costs_2045.csv`

has the identical SHA256 fingerprint:

`0ce46a0078cfe640e0dcee8f8ac07d95edf60d33cf9bdaa57bddf722e6f4b328`

Therefore the custom off-grid model and the PyPSA-Earth model can use the same
underlying PyPSA technology-data release as a common techno-economic reference
basis.

## Important distinction

This CSV is the raw upstream technology-data output.

It must not be confused with processed PyPSA-Earth cost tables such as:

- `resources/costs_2045_elec.csv`
- `data/costs.csv`

PyPSA-Earth performs additional processing including annualization, technology
mapping, parameter overwrites and numerical dispatch-cost conventions.

For example, renewable marginal costs in the solved PyPSA-Earth network are
not direct representations of the raw technology-data VOM assumptions.
PyPSA-Earth overwrites selected renewable marginal costs in its configuration,
and the solved scenario additionally used `noisy_costs: true`, which introduces
small numerical perturbations.

The custom off-grid thesis model should therefore use source-supported physical
and economic parameters rather than copying these PyPSA-Earth numerical
dispatch-cost perturbations.

## Evidence classification

technology-data release:
SOURCE-SUPPORTED — PyPSA technology-data v0.13.2.

2045 CSV:
UNMODIFIED UPSTREAM DATA SNAPSHOT.

SHA256 identity between h2-btc and PyPSA-Earth:
REPRODUCIBILITY CHECK.

Use as common techno-economic reference basis:
MODELLING/METHODOLOGICAL DECISION motivated by cross-model consistency.

Technology-specific deviations:
PERMITTED ONLY WITH EXPLICIT SOURCE AND JUSTIFICATION.
