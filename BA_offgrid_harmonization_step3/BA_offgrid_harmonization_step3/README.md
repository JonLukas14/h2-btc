# Off-grid harmonization – implementation step

This package creates four **new** off-grid scenario YAML files based on the existing, locally checked `offgrid-clean-rebuild` branch. It does not modify legacy scenarios or execute a solver.

## Install/run

```bash
cd /mnt/n/ruj30415/Git/h2-btc
# Confirm that git status -sb is clean.
python /path/to/generate_harmonized_configs.py
```

Generated configurations are stored under `configs/scenarios/harmonized_2026_11_10/`. A source/config SHA256 and Git commit manifest is saved under `results/harmonization_audit_2026_11_10/`.

## Four experiments

- S2E: fixed H2 annual target + optimized BTC, no battery
- S3E: fixed H2 annual target + optimized BTC, battery allowed
- JOINT: H2 sold at EUR2.38/kg and BTC endogenous, no battery
- JOINT-B: same as JOINT, battery allowed

The generator only changes discount rate, PEM economic values/efficiency, BTC annualized investment cost and BTC power-boundary/revenue calibration. It preserves per-scenario weather, renewable capacity bounds, and site parameters.

## Important interpretation

The national ASIC CAPEX assumption uses PUE=1.10 for MW-facility normalization. The harmonized experiment uses PUE=1.10 and **calibrates** hashprice to 0.02238074388 EUR/(TH/s)/day, such that the BTC revenue coefficient remains 62.7967 EUR/MWh, matching the national objective. This is not a 2045 hashprice forecast. Keeping original hashprice with PUE 1.10 instead yields a different BTC revenue sensitivity; consider it as an additional experiment, not an unlabelled reference.

The PEM cost applies national assumptions for the 1 GW DEA technology to the smaller off-grid system. This tests economic consistency, but technological scale-specific cost applicability is not automatically established.

## Running and evaluating

The precise project CLI has not been inspected here. Before launching any run, identify supported commands using the repository's README and help output. Save each solved case's source config, Git revision, source input hashes, solver version, and output summary. Compare harmonized outputs to the existing 2045 baseline by scenario pairing. Do not call an expected economic change a code regression.

## Compare outputs after runs

Use `compare_harmonized_results.py --pair LABEL old-summary.csv new-summary.csv` for each matching scenario pair and `--out results/harmonization_audit_2026_11_10/kpi_deltas.csv`. The parser expects one scenario per summary CSV and checks the 1 kt H2 target for S2E/S3E. It reports changed KPIs and refuses physically impossible summaries. This does not replace solver and energy-balance checks.
