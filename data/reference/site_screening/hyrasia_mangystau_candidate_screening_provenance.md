# Hyrasia One Mangystau candidate-site screening provenance

## Purpose

This screening was performed to identify a representative renewable-resource
location among the five publicly identified Hyrasia One hybrid energy clusters
in Mangystau, Kazakhstan.

The screening is a comparative pre-selection exercise only. It is not the
hourly renewable-resource input used by the final thesis optimization model.

Screening table:

`data/reference/site_screening/hyrasia_mangystau_candidate_screening.csv`

SHA256:

`89e0b0a148cbd8f8ee1936544ab83657decdd50a1c6127c882da97d4e4b003f2`

## Candidate definition

The following Hyrasia One hybrid energy clusters were screened:

- Talap
- Enbek
- Teren oi
- Kanagat
- Rahym

The planned photovoltaic and wind capacities were taken from the official
Hyrasia One project description.

Source:
Hyrasia One, "The Project", SVEVIND Energy Group.

The approximate project-area locations were derived from publicly available
phase locations in the Global Energy Monitor Global Solar Power Tracker and
Global Wind Power Tracker.

Sources:
Global Energy Monitor, "Hyrasia One solar farm".
Global Energy Monitor, "Hyrasia One wind farm".

The coordinates used in the screening are representative project-area
coordinates. They must not be interpreted as surveyed turbine, PV-module,
electrolyzer, or industrial-facility coordinates.

## Screening coordinates and planned capacities

| Site | Latitude | Longitude | Planned PV [MW] | Planned wind [MW] |
|---|---:|---:|---:|---:|
| Talap | 43.7237 | 52.6023 | 1300 | 3000 |
| Enbek | 43.2078 | 52.4944 | 1800 | 3700 |
| Teren oi | 42.4032 | 53.0564 | 3500 | 5200 |
| Kanagat | 42.3152 | 54.8139 | 4900 | 11000 |
| Rahym | 43.3904 | 55.2698 | 1600 | 4000 |

## PV screening method

Solar-resource screening was performed using the European Commission Joint
Research Centre Photovoltaic Geographical Information System (PVGIS), API
version v5_3.

Endpoint:

`PVcalc`

Explicit parameters:

- latitude = candidate-specific latitude
- longitude = candidate-specific longitude
- peakpower = 1.0 kWp
- loss = 0.0 %
- optimalinclination = 1
- outputformat = json

The annual specific PV yield was read from:

`outputs -> totals -> fixed -> E_y`

and stored as:

`pv_yield_kwh_per_kw_a_screening`

The screening capacity factor was derived as:

`pv_cf_screening = E_y / 8760`

The zero-percent loss assumption was deliberately selected for relative
resource screening. It is not a final thesis assumption regarding actual
PV-system losses.

Parameters not explicitly supplied to the API, including the radiation
database, PV technology, mounting-position settings, horizon treatment and
other optional PVGIS inputs, relied on the PVGIS defaults applicable at the
time of execution. Consequently, the PV screening result should be interpreted
as a comparative screening metric rather than a fully specified project-level
PV simulation.

## Wind screening method

Wind-resource screening used the PVGIS v5_3 Typical Meteorological Year (TMY)
endpoint.

Endpoint:

`tmy`

Explicit parameters:

- latitude = candidate-specific latitude
- longitude = candidate-specific longitude
- outputformat = json

The hourly field:

`WS10m`

was extracted from:

`outputs -> tmy_hourly`

For each candidate, the following statistics were calculated:

- arithmetic mean 10 m wind speed:
  `ws10.mean()`
- 50th percentile:
  `numpy.percentile(ws10, 50)`
- 90th percentile:
  `numpy.percentile(ws10, 90)`

The resulting variables are:

- `ws10_mean_m_s`
- `ws10_p50_m_s`
- `ws10_p90_m_s`

These values describe 10 m wind speed from a PVGIS typical meteorological year.
They are not turbine-hub-height wind speeds and they are not turbine capacity
factors.

The TMY data are used only for relative site screening. They are not the final
hourly wind-resource input of the thesis model.

## Candidate comparison

The resulting screening values were:

| Site | PV yield [kWh/kWp/a] | PV CF | Mean WS10 [m/s] | P50 WS10 [m/s] | P90 WS10 [m/s] |
|---|---:|---:|---:|---:|---:|
| Teren oi | 1668.09 | 0.1904 | 5.1745 | 5.10 | 8.00 |
| Kanagat | 1820.88 | 0.2079 | 4.9517 | 4.62 | 8.00 |
| Rahym | 1794.10 | 0.2048 | 4.8588 | 4.62 | 7.79 |
| Enbek | 1648.18 | 0.1881 | 4.7291 | 4.48 | 7.72 |
| Talap | 1621.68 | 0.1851 | 4.5305 | 4.34 | 6.90 |

The output table was sorted first by mean 10 m wind speed and second by annual
PV yield. This ordering was a presentation step and was not an optimization
objective or formal multi-criteria score.

## Representative-site decision

Kanagat was selected as the representative thesis location because it provides
a strong combination of both screened renewable resources:

- highest annual PV yield among the five candidates;
- second-highest mean 10 m wind speed;
- tied highest P90 10 m wind speed;
- largest planned Hyrasia One hybrid-energy cluster in the screened candidate
  set.

Kanagat is therefore described as the best-balanced representative candidate
for the study, not as a mathematically proven globally optimal project site.

## Separation from final renewable profiles

The PVGIS screening values above are not used as the final hourly renewable
dispatch profiles.

After site selection, the final thesis renewable-resource profiles were
generated separately for Kanagat using ERA5 meteorological data for 2013 and
atlite, consistent with the later PyPSA-Earth comparison workflow.

The final profiles and their independent provenance are stored under:

`data/reference/renewable_profiles/`

This distinction prevents the preliminary site-screening method from being
confused with the final thesis renewable-resource representation.

## Evidence classification

Candidate names and planned capacities:
SOURCE-SUPPORTED — official Hyrasia One project information.

Project-area coordinates:
SOURCE-SUPPORTED APPROXIMATE LOCATIONS — Global Energy Monitor trackers.

Screening coordinate selection:
DERIVED REPRESENTATIVE LOCATIONS — not surveyed project coordinates.

PVGIS annual PV yield:
MODELLED SCREENING OUTPUT.

PV screening capacity factor:
DERIVED CALCULATION from annual yield divided by 8760 h.

PVGIS TMY wind statistics:
MODELLED SCREENING OUTPUT / DERIVED STATISTICS.

Kanagat selection:
MODELLING DECISION based on comparative screening results and project context.

Final ERA5/atlite renewable profiles:
SEPARATE THESIS INPUT DATASET; see renewable-profile provenance file.

## Source URLs and retrieval information

Sources were checked/accessed on 2026-08-21.

Hyrasia One project description:
https://hyrasia.one/?page_id=23813270

Global Energy Monitor — Hyrasia One solar farm:
https://www.gem.wiki/Hyrasia_One_solar_farm

Global Energy Monitor — Hyrasia One wind farm:
https://www.gem.wiki/Hyrasia_One_wind_farm

PVGIS API base URL used for the screening:
https://re.jrc.ec.europa.eu/api/v5_3

The PVGIS requests were executed programmatically using the `PVcalc` and `tmy`
endpoints described above. The numerical results in the screening CSV therefore
represent the API responses obtained at the time of execution and the derived
statistics calculated from those responses.
