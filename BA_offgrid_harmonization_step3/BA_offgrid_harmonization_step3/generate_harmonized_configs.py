#!/usr/bin/env python3
"""Generate new, non-destructive off-grid scenarios from tracked Clean Rebuild YAML.

Run from the root of the h2-btc repository:
    python /path/to/generate_harmonized_configs.py
Requires PyYAML (already used by offgrid/model.py).
"""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import yaml

ROOT = Path.cwd().resolve()
CONFIGS = ROOT / 'configs' / 'scenarios'
OUTPUT = ROOT / 'configs' / 'scenarios' / 'harmonized_2026_11_10'
METADATA = ROOT / 'results' / 'harmonization_audit_2026_11_10'
NATIONAL_BTC_COST_EUR_PER_MW_YEAR = 324661.55632  # source: E1 output, 2026-11-10
NATIONAL_BTC_REVENUE_EUR_PER_MWH = 62.7967
NATIONAL_STACK_EUR_PER_MWH_EL = 141.92567727354114 / 84769.23076923078 * 1000.0
ASIC_J_PER_TH = 13.5
PUE_FACILITY = 1.10
HASHPRICE_FOR_MATCHED_REVENUE = (NATIONAL_BTC_REVENUE_EUR_PER_MWH * 24 * ASIC_J_PER_TH * PUE_FACILITY / 1e6)

SCENARIOS = [
    ('scenario_s2e_thesis_2045.yaml', 'scenario_s2e_harmonized_2045.yaml', 's2e_harmonized_2045_kanagat'),
    ('scenario_s3e_thesis_2045.yaml', 'scenario_s3e_harmonized_2045.yaml', 's3e_harmonized_2045_kanagat'),
    ('scenario_joint_h2value238.yaml', 'scenario_joint_harmonized_h2value238_2045.yaml', 'joint_harmonized_h2value238_2045_kanagat'),
    ('scenario_joint_b_h2value238.yaml', 'scenario_joint_b_harmonized_h2value238_2045.yaml', 'joint_b_harmonized_h2value238_2045_kanagat'),
]

def git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return p.stdout.strip()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if not (ROOT / 'offgrid' / 'model.py').exists():
        raise SystemExit('ERROR: run from the root of h2-btc (offgrid/model.py not found)')
    if git('branch', '--show-current') != 'offgrid-clean-rebuild':
        raise SystemExit('ERROR: expected offgrid-clean-rebuild branch; no files written')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('ERROR: tracked worktree modifications found; verify or commit/restore first')
    if OUTPUT.exists() or METADATA.exists():
        raise SystemExit('ERROR: output folder already exists; refusing to overwrite')
    staged = []
    for src_name, dest_name, scenario_name in SCENARIOS:
        src = CONFIGS / src_name
        if not src.exists():
            raise SystemExit(f'ERROR: missing source config {src}')
        cfg = yaml.safe_load(src.read_text(encoding='utf-8'))
        if not isinstance(cfg, dict):
            raise SystemExit(f'ERROR: malformed YAML {src}')
        for required in ['costs','hydrogen','bitcoin','system','renewables','battery']:
            if not isinstance(cfg.get(required), dict):
                raise SystemExit(f'ERROR: {src_name} missing {required} section')
        if cfg['bitcoin'].get('capacity_mode') != 'endogenous':
            raise SystemExit(f'ERROR: {src_name} is not an endogenous BTC scenario')
        if cfg['system'].get('investment_year') != 2045:
            raise SystemExit(f'ERROR: {src_name} is not a 2045 scenario')
        if cfg['bitcoin'].get('other_opex_eur_per_mwh', 0) != 0:
            raise SystemExit(f'ERROR: {src_name} uses non-zero BTC OPEX; needs explicit audit')
        if src_name.startswith('scenario_joint') and abs(cfg['hydrogen'].get('sale_value_eur_per_kg_h2',-999) - 2.38) > 1e-8:
            raise SystemExit(f'ERROR: {src_name} has unexpected hydrogen sale price')
        if src_name.startswith('scenario_s') and cfg['hydrogen'].get('mode') != 'production_target':
            raise SystemExit(f'ERROR: {src_name} is not fixed H2 target')
        if src_name.startswith('scenario_joint') and cfg['hydrogen'].get('mode') != 'economic_dispatch':
            raise SystemExit(f'ERROR: {src_name} is not merchant H2')
        # Copies preserve all location, resource, storage, and demand assumptions.
        cfg['scenario_name'] = scenario_name
        cfg['costs']['discount_rate'] = 0.07
        h2=cfg['hydrogen']
        h2.update({
            'electrolyzer_capex_eur_per_kw':530.62,
            'electrolyzer_fom_percent_per_year':2.25,
            'electrolyzer_lifetime_years':32.5,
            'electrolyzer_efficiency':0.6435,
            'electrolyzer_variable_cost_eur_per_mwh':NATIONAL_STACK_EUR_PER_MWH_EL,
        })
        btc=cfg['bitcoin']
        btc.update({
            'annualized_capacity_cost_eur_per_mw_year':NATIONAL_BTC_COST_EUR_PER_MW_YEAR,
            'asic_efficiency_j_per_th':ASIC_J_PER_TH,
            'pue':PUE_FACILITY,
            'hashprice_eur_per_th_day':HASHPRICE_FOR_MATCHED_REVENUE,
            'other_opex_eur_per_mwh':0.0,
        })
        staged.append((src, dest_name, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)))
    # Validation succeeded for all four originals; now create new files only.
    OUTPUT.mkdir(parents=True)
    METADATA.mkdir(parents=True)
    rows=[]
    for src, dest_name, result in staged:
        dest=OUTPUT/dest_name
        dest.write_text('# Generated harmonized sensitivity; original configurations unchanged.\n'
                        '# Adjusted hashprice CALIBRATES the 62.7967 EUR/MWh national BTC reference\n'
                        '# while using PUE 1.10; it is NOT a new market hashprice forecast.\n'
                        + result, encoding='utf-8')
        rows.append({'source_config':str(src.relative_to(ROOT)), 'source_sha256':sha(src),
                     'new_config':str(dest.relative_to(ROOT)), 'generated_sha256':sha(dest),
                     'source_commit':git('rev-parse','HEAD'),
                     'btc_revenue_eur_per_mwh':NATIONAL_BTC_REVENUE_EUR_PER_MWH,
                     'btc_pue':PUE_FACILITY,'btc_hashprice_calibrated_eur_per_th_day':HASHPRICE_FOR_MATCHED_REVENUE,
                     'pem_stack_eur_per_mwh_el':NATIONAL_STACK_EUR_PER_MWH_EL})
    with (METADATA/'scenario_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (METADATA/'README.md').write_text(
        '# Harmonized off-grid scenario experiment\n\n'
        'Base git commit: '+git('rev-parse','HEAD')+'\n\n'
        'Original source YAML is unchanged. Four independent new configurations created.\n'
        'BTC revenue is CALIBRATED at 62.7967 EUR/MWh using PUE 1.10. The required '\
        'hashprice is not independently observed and must not be reported as a market forecast.\n'
        'Economic normalization, scale applicability of 1 GW national PEM costs and '
        'site-specific resource bounds remain thesis limitations.\n'
        'Run solver only after inspecting generated configs and baseline runner syntax.\n',
        encoding='utf-8')
    print(json.dumps({'status':'GENERATED_NOT_SOLVED', 'source_commit':git('rev-parse','HEAD'),
                      'configs':[r['new_config'] for r in rows],
                      'manifest':str((METADATA/'scenario_manifest.csv').relative_to(ROOT)),
                      'matched_revenue_hashprice':HASHPRICE_FOR_MATCHED_REVENUE},indent=2))

if __name__=='__main__':
    main()
