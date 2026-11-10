#!/usr/bin/env python3
"""Compare matched off-grid baseline/harmonized summary CSV pairs.

Usage:
 python compare_harmonized_results.py --pair S2E old_s2e/summary.csv new_s2e/summary.csv [--pair ...] --out audit.csv

A changed assumption is not an automatic regression failure. Report deltas transparently.
"""
import argparse
import csv
import math
from pathlib import Path

FIELDS = [
    'solar_capacity_mw', 'wind_capacity_mw', 'electrolyzer_capacity_mw',
    'battery_power_mw', 'battery_energy_mwh', 'bitcoin_capacity_mw',
    'bitcoin_consumption_mwh', 'bitcoin_gross_revenue_eur_per_mwh',
    'electrolyzer_input_mwh', 'hydrogen_delivered_kt',
    'solar_curtailment_mwh', 'wind_curtailment_mwh',
    'gross_system_expenditure_eur_per_year', 'objective_eur_per_year',
    'net_system_cost_eur_per_year', 'lcoh_eur_per_kg_h2'
]

def read_summary(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f'{path}: expected exactly one summary row, got {len(rows)}')
    return rows[0]

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--pair',nargs=3,action='append',required=True,metavar=('LABEL','BASELINE','HARMONIZED'))
    a.add_argument('--out',required=True)
    args=a.parse_args()
    rows=[]
    for label, bpath, hpath in args.pair:
        b=read_summary(bpath);h=read_summary(hpath)
        for field in FIELDS:
            try:
                bv=float(b[field]);hv=float(h[field])
            except (KeyError,TypeError,ValueError):
                rows.append(dict(case=label,metric=field,baseline=b.get(field,''),harmonized=h.get(field,''),difference='',percent_difference='',status='MISSING_OR_NONNUMERIC'))
                continue
            if not (math.isfinite(bv) and math.isfinite(hv)):
                status='NONFINITE'
                pct=''
            else:
                status='UNCHANGED' if math.isclose(bv,hv,rel_tol=1e-7,abs_tol=1e-6) else 'CHANGED_EXPECTED_IF_PARAMETERS_DIFFER'
                pct=100*(hv-bv)/abs(bv) if abs(bv)>1e-12 else ''
            rows.append(dict(case=label,metric=field,baseline=bv,harmonized=hv,
                             difference=(hv-bv if math.isfinite(hv) and math.isfinite(bv) else ''),percent_difference=pct,status=status))
        for column in ('hydrogen_delivered_kt','solar_capacity_mw','wind_capacity_mw','electrolyzer_capacity_mw','bitcoin_capacity_mw','battery_power_mw'):
            try:
                value=float(h[column])
                if not math.isfinite(value) or value < -1e-5:
                    raise ValueError(f'{label}: implausible {column}={value}')
            except (KeyError,ValueError,TypeError) as exc:
                raise ValueError(f'Invalid harmonized summary {hpath}: {exc}') from exc
        if label.upper() in ('S2E','S3E'):
            got=float(h['hydrogen_delivered_kt'])
            if not math.isclose(got,1.0,rel_tol=2e-5,abs_tol=2e-5):
                raise ValueError(f'{label}: hydrogen fixed target 1 kt/a not met: {got}')
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['case','metric','baseline','harmonized','difference','percent_difference','status']);wr.writeheader();wr.writerows(rows)
    print(f'Wrote {len(rows)} KPI comparisons to {dest}; do not interpret expected sensitivity deltas as code failures.')

if __name__=='__main__': main()
