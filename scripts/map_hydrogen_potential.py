# scripts/map_hydrogen_potential.py
# Green hydrogen production attractiveness map for Kazakhstan
# based on renewable-powered electrolysis suitability.
#
# Method:
# - Sample PVGIS solar and wind resource at a grid of points
# - Compute mean solar and wind capacity factors
# - Build a simple hydrogen attractiveness score from a weighted
#   renewable electricity suitability proxy
# - Restrict best-site selection to points inside Kazakhstan
#
# Outputs:
#   results/hydrogen_potential_map.png
#   results/hydrogen_potential_map.html
#   results/hydrogen_potential_grid.csv

import io
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

LAT_MIN, LAT_MAX = 40.0, 56.0
LON_MIN, LON_MAX = 45.0, 88.0
GRID_STEPS = 6
PVGIS_YEAR = 2020

# Weighting for first-pass electrolysis attractiveness proxy
# Wind gets a slightly higher weight because multi-hour generation profiles
# often help electrolyzer utilization more than solar-only output.
W_WIND = 0.60
W_SOLAR = 0.40

CITY_LABELS = [
    ('Astana', 51.18, 71.45), ('Almaty', 43.25, 76.95), ('Shymkent', 42.30, 69.59),
    ('Aktobe', 50.28, 57.21), ('Atyrau', 47.11, 51.92), ('Mangystau', 43.66, 51.15),
    ('Pavlodar', 52.29, 76.96), ('Karaganda', 49.80, 73.10), ('Kostanay', 53.21, 63.63),
    ('Zhambyl', 42.90, 71.38),
]

def power_curve(ws):
    ws = pd.Series(ws).astype(float)
    cf = pd.Series(0.0, index=ws.index)
    mask_ramp = (ws >= 3.0) & (ws < 12.0)
    mask_rated = (ws >= 12.0) & (ws <= 25.0)
    cf[mask_ramp] = ((ws[mask_ramp] - 3.0) / 9.0) ** 3
    cf[mask_rated] = 1.0
    return cf.clip(0.0, 1.0)

def fetch_solar_cf(lat, lon):
    url = (
        f'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc'
        f'?lat={lat:.2f}&lon={lon:.2f}'
        f'&startyear={PVGIS_YEAR}&endyear={PVGIS_YEAR}'
        f'&pvcalculation=1&peakpower=1&loss=14'
        f'&angle=30&aspect=0&outputformat=json&browser=0'
    )
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        vals = [e['P'] / 1000.0 for e in r.json()['outputs']['hourly']]
        return float(np.mean(vals))
    except Exception as e:
        print(f'Solar error ({lat:.1f},{lon:.1f}): {e}')
        return float('nan')

def fetch_wind_cf(lat, lon):
    url = (
        f'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc'
        f'?lat={lat:.2f}&lon={lon:.2f}'
        f'&startyear={PVGIS_YEAR}&endyear={PVGIS_YEAR}'
        f'&outputformat=json&browser=0&windspeed=1'
    )
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        ws_10m = pd.Series([e['WS10m'] for e in r.json()['outputs']['hourly']])
        rng = np.random.default_rng(seed=42)
        ws_10m = (ws_10m + rng.normal(0, 0.1, len(ws_10m))).clip(lower=0)
        ws_10m = ws_10m * 1.25
        ws_hub = ws_10m * (100 / 10) ** 0.20
        return float(power_curve(ws_hub).mean())
    except Exception as e:
        print(f'Wind error ({lat:.1f},{lon:.1f}): {e}')
        return float('nan')

print(f'Sampling PVGIS at {GRID_STEPS*GRID_STEPS} points ({GRID_STEPS}x{GRID_STEPS} grid) ...')
lats = np.linspace(LAT_MIN + 1.5, LAT_MAX - 1.5, GRID_STEPS)
lons = np.linspace(LON_MIN + 2.0, LON_MAX - 2.0, GRID_STEPS)
results = []
for i, (lat, lon) in enumerate([(a, b) for a in lats for b in lons], 1):
    print(f'[{i:02d}/{GRID_STEPS*GRID_STEPS}] ({lat:.2f}, {lon:.2f}) ...', end=' ', flush=True)
    s = fetch_solar_cf(lat, lon)
    time.sleep(0.4)
    w = fetch_wind_cf(lat, lon)
    time.sleep(0.4)
    results.append({'lat': round(lat,2), 'lon': round(lon,2), 'solar_cf_mean': s, 'wind_cf_mean': w})
    print(f'solar={s:.3f} wind={w:.3f}')

df = pd.DataFrame(results)

# First-pass green hydrogen attractiveness score
# Weighted sum of solar and wind CF, then normalized to 0-100 for readability.
df['h2_score_raw'] = W_WIND * df['wind_cf_mean'] + W_SOLAR * df['solar_cf_mean']
min_raw = df['h2_score_raw'].min()
max_raw = df['h2_score_raw'].max()
df['h2_score'] = 100 * (df['h2_score_raw'] - min_raw) / (max_raw - min_raw + 1e-12)

def classify(score):
    if score >= 80:
        return 'Very high'
    elif score >= 60:
        return 'High'
    elif score >= 40:
        return 'Medium'
    elif score >= 20:
        return 'Low'
    return 'Very low'

df['h2_class'] = df['h2_score'].apply(classify)
df.to_csv(OUT_DIR / 'hydrogen_potential_grid.csv', index=False)
print('Grid data saved -> results/hydrogen_potential_grid.csv')

print('Loading map boundaries ...')
raw = requests.get('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson', timeout=30)
raw.raise_for_status()
countries = gpd.read_file(io.StringIO(raw.text))
neighbours = ['Russia', 'China', 'Uzbekistan', 'Kyrgyzstan', 'Turkmenistan', 'Tajikistan', 'Azerbaijan', 'Kazakhstan']
context_countries = countries[countries['NAME'].isin(neighbours)].copy()
kz_only = countries[countries['ISO_A2'] == 'KZ'].copy()

raw2 = requests.get('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson', timeout=30)
raw2.raise_for_status()
provinces = gpd.read_file(io.StringIO(raw2.text))
oblasts = provinces[provinces['iso_a2'] == 'KZ'].copy()

points_gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
points_in_kz = gpd.sjoin(points_gdf, kz_only[['geometry']], predicate='within', how='inner').drop(columns=['index_right'])

def score_to_size(values, min_size=140, max_size=1700):
    arr = np.array(values, dtype=float)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax == vmin:
        return np.full_like(arr, (min_size + max_size) / 2)
    scaled = (arr - vmin) / (vmax - vmin)
    return min_size + scaled * (max_size - min_size)

def get_best_row():
    valid = points_in_kz.dropna(subset=['h2_score'])
    return valid.loc[valid['h2_score'].idxmax()]

def legend_handles(cmap_name, valid_values, best_value):
    cmap = cm.get_cmap(cmap_name)
    vmin, vmax = float(np.nanmin(valid_values)), float(np.nanmax(valid_values))
    vmid = (vmin + vmax) / 2
    vals = [vmin, vmid, vmax]
    sizes = score_to_size(vals)
    norm = Normalize(vmin=vmin, vmax=vmax)
    handles = []
    for i, (v, s) in enumerate(zip(vals, sizes)):
        handles.append(
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=cmap(norm(v)), markeredgecolor='white',
                   markeredgewidth=1.0, alpha=0.9,
                   markersize=np.sqrt(s) * 0.48,
                   label=('Score ≤ ' if i == 0 else 'Score ≥ ' if i == 2 else 'Score ≈ ') + f'{v:.0f}')
        )
    handles.append(
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='none', markeredgecolor='red', markeredgewidth=2,
               markersize=12, label=f'Best site in Kazakhstan  Score ≈ {best_value:.1f}')
    )
    return handles

def plot_h2_map():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160, facecolor='#d6e8f5')
    non_kz = context_countries[context_countries['ISO_A2'] != 'KZ']
    non_kz.plot(ax=ax, color='#e8e8e8', edgecolor='#aaaaaa', linewidth=0.6, zorder=1)
    kz_only.plot(ax=ax, color='white', edgecolor='#333333', linewidth=1.4, zorder=2)
    if not oblasts.empty:
        oblasts.plot(ax=ax, color='none', edgecolor='#999999', linewidth=0.55, linestyle='--', zorder=3)

    valid = df.dropna(subset=['h2_score']).copy()
    sizes = score_to_size(valid['h2_score'].values)
    norm = Normalize(vmin=valid['h2_score'].min(), vmax=valid['h2_score'].max())
    sc = ax.scatter(valid['lon'], valid['lat'], s=sizes, c=valid['h2_score'], cmap='PuBuGn', norm=norm,
                    alpha=0.84, edgecolors='white', linewidths=0.9, zorder=5)

    for _, row in valid.iterrows():
        ax.text(row['lon'], row['lat'], f"{row['h2_score']:.0f}", ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

    best = get_best_row()
    best_size = score_to_size([best['h2_score']])[0]
    ax.scatter(best['lon'], best['lat'], s=best_size * 1.8, facecolors='none', edgecolors='red', linewidths=2.5, zorder=7)
    ax.scatter(best['lon'], best['lat'], s=120, marker='*', color='red', edgecolors='white', linewidths=0.5, zorder=8)

    for name, clat, clon in CITY_LABELS:
        if LON_MIN <= clon <= LON_MAX and LAT_MIN <= clat <= LAT_MAX:
            ax.plot(clon, clat, 'o', ms=3, color='#444', zorder=9)
            ax.text(clon + 0.3, clat + 0.3, name, fontsize=7.5, color='#333', fontstyle='italic', zorder=9)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.022, pad=0.012, shrink=0.78)
    cbar.set_label('Green hydrogen attractiveness score (0-100, unitless)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    handles = legend_handles('PuBuGn', valid['h2_score'].values, float(best['h2_score']))
    leg = ax.legend(handles=handles, title='Bubble size reflects attractiveness', title_fontsize=10, fontsize=10,
                    loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.97,
                    edgecolor='#cccccc', borderpad=1.15, labelspacing=0.95, handletextpad=1.0,
                    borderaxespad=0.8, handlelength=1.4)
    leg.get_frame().set_facecolor('white')

    ax.set_xlim(LON_MIN - 1, LON_MAX + 1)
    ax.set_ylim(LAT_MIN - 1, LAT_MAX + 1)
    ax.set_xlabel('Longitude (E)', fontsize=11)
    ax.set_ylabel('Latitude (N)', fontsize=11)
    ax.set_title('Kazakhstan - Green Hydrogen Production Attractiveness\n(larger bubble = more attractive for renewable electrolysis)', fontsize=15, weight='bold', pad=14)
    ax.grid(True, alpha=0.12, linestyle=':', color='#555')
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.text(0.99, 0.01,
             f'Source: PVGIS ERA5 {PVGIS_YEAR} | Score = {W_WIND:.0%} wind CF + {W_SOLAR:.0%} solar CF',
             ha='right', va='bottom', fontsize=8, color='#888')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'hydrogen_potential_map.png', bbox_inches='tight', dpi=160)
    plt.close(fig)
    print('PNG saved -> results/hydrogen_potential_map.png')

best = get_best_row()
print(f"Best green H2 site in Kazakhstan: ({best.lat:.2f}N, {best.lon:.2f}E) score={best.h2_score:.1f}, class={best.h2_class}")
plot_h2_map()

# Interactive HTML map
valid = df.dropna(subset=['h2_score']).copy()
smin, smax = valid['h2_score'].min(), valid['h2_score'].max()
sizes_px = 8 + (valid['h2_score'] - smin) / (smax - smin + 1e-9) * 32

trace = go.Scattermapbox(
    lat=valid['lat'], lon=valid['lon'], mode='markers',
    marker=dict(size=sizes_px, color=valid['h2_score'], colorscale='PuBuGn',
                cmin=float(smin), cmax=float(smax), opacity=0.84, sizemode='diameter',
                colorbar=dict(title=dict(text='H2 score', side='right'), thickness=14, len=0.65)),
    text=[f"<b>{r.lat:.2f}N {r.lon:.2f}E</b><br>H2 score: <b>{r.h2_score:.1f}</b><br>Class: {r.h2_class}<br>Wind CF={r.wind_cf_mean:.3f}, Solar CF={r.solar_cf_mean:.3f}" for _, r in valid.iterrows()],
    hovertemplate='%{text}<extra></extra>', name='H2 score'
)

best_marker = go.Scattermapbox(
    lat=[best.lat], lon=[best.lon], mode='markers+text',
    marker=dict(size=18, symbol='star', color='red'),
    text=[f'Best H2 site in Kazakhstan score={best.h2_score:.1f}'],
    textposition='top right', textfont=dict(size=11, color='red'), name='Best site'
)

fig = go.Figure(data=[trace, best_marker])
fig.update_layout(
    title=dict(text='Kazakhstan Green Hydrogen Production Attractiveness', font=dict(size=17), x=0.5),
    mapbox=dict(style='carto-positron', center=dict(lat=48.5, lon=66.5), zoom=3.9),
    height=680, margin=dict(l=0, r=0, t=65, b=0),
    annotations=[dict(
        text='Score is a first-pass, unitless suitability proxy based on 60% wind CF + 40% solar CF. Higher = more attractive for renewable electrolysis.',
        align='left', showarrow=False, xref='paper', yref='paper', x=0.01, y=0.01,
        xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.88)', bordercolor='#cccccc', borderwidth=1, font=dict(size=10)
    )]
)
fig.write_html(OUT_DIR / 'hydrogen_potential_map.html', include_plotlyjs='cdn')
print('Interactive HTML map saved -> results/hydrogen_potential_map.html')
print('Done! Check your results/ folder.')