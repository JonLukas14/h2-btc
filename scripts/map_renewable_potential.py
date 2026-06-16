# scripts/map_renewable_potential.py
# Bubble map of solar and wind capacity factors across Kazakhstan.
# Outputs: solar_potential_map.png, wind_potential_map.png,
#          renewable_potential_map.html, renewable_potential_grid.csv
# Usage:   python scripts/map_renewable_potential.py
# Install: pip install requests numpy pandas geopandas matplotlib plotly

import io
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
OUT_DIR    = BASE_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

LAT_MIN, LAT_MAX = 40.0, 56.0
LON_MIN, LON_MAX = 45.0, 88.0
GRID_STEPS = 6       # 6x6 = 36 points, ~2-3 min. Raise for finer detail.
PVGIS_YEAR = 2020

CITY_LABELS = [
    ('Astana',    51.18, 71.45),
    ('Almaty',    43.25, 76.95),
    ('Shymkent',  42.30, 69.59),
    ('Aktobe',    50.28, 57.21),
    ('Atyrau',    47.11, 51.92),
    ('Mangystau', 43.66, 51.15),
    ('Pavlodar',  52.29, 76.96),
    ('Karaganda', 49.80, 73.10),
    ('Kostanay',  53.21, 63.63),
    ('Zhambyl',   42.90, 71.38),
]

# ---------------------------------------------------------------------------
# WIND POWER CURVE  (same as build_network.py)
# ---------------------------------------------------------------------------
def power_curve(ws):
    ws = pd.Series(ws).astype(float)
    cf = pd.Series(0.0, index=ws.index)
    mask_ramp  = (ws >= 3.0) & (ws < 12.0)
    mask_rated = (ws >= 12.0) & (ws <= 25.0)
    cf[mask_ramp]  = ((ws[mask_ramp] - 3.0) / 9.0) ** 3
    cf[mask_rated] = 1.0
    return cf.clip(0.0, 1.0)

# ---------------------------------------------------------------------------
# PVGIS API HELPERS
# ---------------------------------------------------------------------------
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
        print(f'    Solar error ({lat:.1f},{lon:.1f}): {e}')
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
        rng    = np.random.default_rng(seed=42)
        ws_10m = (ws_10m + rng.normal(0, 0.1, len(ws_10m))).clip(lower=0)
        ws_10m = ws_10m * 1.25                  # ERA5 bias correction
        ws_hub = ws_10m * (100 / 10) ** 0.20    # 10m -> 100m hub height
        return float(power_curve(ws_hub).mean())
    except Exception as e:
        print(f'    Wind error ({lat:.1f},{lon:.1f}): {e}')
        return float('nan')

# ---------------------------------------------------------------------------
# SAMPLE GRID
# ---------------------------------------------------------------------------
lats = np.linspace(LAT_MIN + 1.5, LAT_MAX - 1.5, GRID_STEPS)
lons = np.linspace(LON_MIN + 2.0, LON_MAX - 2.0, GRID_STEPS)
grid_points = [(round(lat, 2), round(lon, 2)) for lat in lats for lon in lons]
total = len(grid_points)

print(f'Sampling PVGIS at {total} points ({GRID_STEPS}x{GRID_STEPS} grid) ...')
results = []
for i, (lat, lon) in enumerate(grid_points, 1):
    print(f'  [{i:02d}/{total}] ({lat:.2f}, {lon:.2f}) ...', end=' ', flush=True)
    s = fetch_solar_cf(lat, lon)
    time.sleep(0.4)
    w = fetch_wind_cf(lat, lon)
    time.sleep(0.4)
    print(f'solar={s:.3f}  wind={w:.3f}')
    results.append({'lat': lat, 'lon': lon, 'solar_cf_mean': s, 'wind_cf_mean': w})

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / 'renewable_potential_grid.csv', index=False)
print('Grid data saved -> results/renewable_potential_grid.csv')

best_solar = df.loc[df['solar_cf_mean'].idxmax()]
best_wind  = df.loc[df['wind_cf_mean'].idxmax()]
print(f'Best Solar: ({best_solar.lat:.2f}N, {best_solar.lon:.2f}E)  CF={best_solar.solar_cf_mean:.3f}')
print(f'Best Wind:  ({best_wind.lat:.2f}N,  {best_wind.lon:.2f}E)   CF={best_wind.wind_cf_mean:.3f}')

# ---------------------------------------------------------------------------
# LOAD MAP BOUNDARIES
# ---------------------------------------------------------------------------
print('Loading map boundaries ...')

kz_only = None
context_countries = None
oblasts = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

try:
    raw = requests.get(
        'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
        'master/geojson/ne_10m_admin_0_countries.geojson', timeout=30)
    raw.raise_for_status()
    countries = gpd.read_file(io.StringIO(raw.text))
    neighbours = ['Russia', 'China', 'Uzbekistan', 'Kyrgyzstan',
                  'Turkmenistan', 'Tajikistan', 'Azerbaijan', 'Kazakhstan']
    context_countries = countries[countries['NAME'].isin(neighbours)].copy()
    kz_only = countries[countries['ISO_A2'] == 'KZ'].copy()
    print(f'  Country boundaries loaded ({len(countries)} countries)')
except Exception as e:
    print(f'  Country download failed: {e}')

try:
    raw2 = requests.get(
        'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
        'master/geojson/ne_10m_admin_1_states_provinces.geojson', timeout=30)
    raw2.raise_for_status()
    provinces = gpd.read_file(io.StringIO(raw2.text))
    oblasts = provinces[provinces['iso_a2'] == 'KZ'].copy()
    print(f'  {len(oblasts)} Kazakhstan oblasts loaded')
except Exception as e:
    print(f'  Oblast download failed: {e}')

# ---------------------------------------------------------------------------
# BUBBLE SIZE SCALING
# ---------------------------------------------------------------------------
def cf_to_size(cf_values, min_size=80, max_size=1800):
    arr  = np.array(cf_values, dtype=float)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        return np.full_like(arr, (min_size + max_size) / 2)
    scaled = (arr - vmin) / (vmax - vmin)
    return min_size + scaled * (max_size - min_size)

# ---------------------------------------------------------------------------
# STATIC PNG  (bubble map, PyPSA-Earth style)
# ---------------------------------------------------------------------------
def plot_bubble_map(column, title, cmap_name, fname, note):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160, facecolor='#d6e8f5')

    # neighbouring countries
    if context_countries is not None:
        non_kz = context_countries[context_countries['ISO_A2'] != 'KZ']
        non_kz.plot(ax=ax, color='#e8e8e8', edgecolor='#aaaaaa',
                    linewidth=0.6, zorder=1)

    # Kazakhstan white fill
    if kz_only is not None and not kz_only.empty:
        kz_only.plot(ax=ax, color='white', edgecolor='#333333',
                     linewidth=1.4, zorder=2)

    # oblast borders
    if not oblasts.empty:
        oblasts.plot(ax=ax, color='none', edgecolor='#999999',
                     linewidth=0.55, linestyle='--', zorder=3)

    # bubbles
    valid = df.dropna(subset=[column]).copy()
    sizes = cf_to_size(valid[column].values)
    norm  = Normalize(vmin=valid[column].min(), vmax=valid[column].max())
    sc = ax.scatter(
        valid['lon'], valid['lat'],
        s=sizes, c=valid[column], cmap=cmap_name, norm=norm,
        alpha=0.82, edgecolors='white', linewidths=0.9, zorder=5)

    # CF label inside each bubble
    for _, row in valid.iterrows():
        ax.text(row['lon'], row['lat'], f"{row[column]:.2f}",
                ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

    # best site marker
    best      = df.loc[df[column].idxmax()]
    best_size = cf_to_size([best[column]])[0]
    ax.scatter(best['lon'], best['lat'], s=best_size * 1.7,
               facecolors='none', edgecolors='red', linewidths=2.5, zorder=7)
    ax.scatter(best['lon'], best['lat'], s=120, marker='*',
               color='red', edgecolors='white', linewidths=0.5, zorder=8)

    # city labels
    for name, clat, clon in CITY_LABELS:
        if LON_MIN <= clon <= LON_MAX and LAT_MIN <= clat <= LAT_MAX:
            ax.plot(clon, clat, 'o', ms=3, color='#444', zorder=9)
            ax.text(clon + 0.3, clat + 0.3, name,
                    fontsize=7.5, color='#333', fontstyle='italic', zorder=9)

    # colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.018, pad=0.01, shrink=0.75)
    cbar.set_label('Mean Capacity Factor (0-1)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # bubble size legend
    vmin = valid[column].min()
    vmax = valid[column].max()
    vmid = (vmin + vmax) / 2
    leg_cfs   = [vmin, vmid, vmax]
    leg_sizes = cf_to_size(leg_cfs)
    leg_elems = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='steelblue', alpha=0.75,
               markersize=np.sqrt(s) * 0.55,
               label=f'CF = {v:.2f}')
        for s, v in zip(leg_sizes, leg_cfs)
    ]
    leg_elems.append(
        Line2D([0], [0], marker='o', color='red', markerfacecolor='none',
               markersize=12, linewidth=2,
               label=f'Best site  CF={best[column]:.3f}'))
    ax.legend(handles=leg_elems, title='Bubble size = CF value',
              title_fontsize=9, fontsize=9,
              loc='lower left', framealpha=0.9, edgecolor='#cccccc')

    ax.set_xlim(LON_MIN - 1, LON_MAX + 1)
    ax.set_ylim(LAT_MIN - 1, LAT_MAX + 1)
    ax.set_xlabel('Longitude (E)', fontsize=11)
    ax.set_ylabel('Latitude (N)', fontsize=11)
    ax.set_title(title, fontsize=15, weight='bold', pad=14)
    ax.grid(True, alpha=0.12, linestyle=':', color='#555')
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.text(0.99, 0.01, f'Source: PVGIS ERA5 {PVGIS_YEAR}  |  {note}',
             ha='right', va='bottom', fontsize=8, color='#888')
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, bbox_inches='tight', dpi=160)
    plt.close(fig)
    print(f'PNG saved -> results/{fname}')

plot_bubble_map(
    'solar_cf_mean',
    'Kazakhstan - Annual Mean Solar Capacity Factor\n(larger bubble = higher potential)',
    'YlOrRd', 'solar_potential_map.png',
    'Fixed-tilt PV, tilt=30, loss=14%')

plot_bubble_map(
    'wind_cf_mean',
    'Kazakhstan - Annual Mean Wind Capacity Factor\n(larger bubble = higher potential)',
    'YlGnBu', 'wind_potential_map.png',
    '100m hub height, ERA5 bias-corrected, cubic power curve')

# ---------------------------------------------------------------------------
# INTERACTIVE PLOTLY HTML
# ---------------------------------------------------------------------------
def make_bubble_trace(col, colorscale, label, visible):
    valid    = df.dropna(subset=[col])
    raw_vals = valid[col].values
    smin, smax = raw_vals.min(), raw_vals.max()
    sizes_px = 8 + (raw_vals - smin) / (smax - smin + 1e-9) * 32
    return go.Scattermapbox(
        lat=valid['lat'], lon=valid['lon'],
        mode='markers',
        marker=dict(
            size=sizes_px,
            color=valid[col],
            colorscale=colorscale,
            cmin=round(float(smin), 3),
            cmax=round(float(smax), 3),
            colorbar=dict(title=dict(text=label, side='right'),
                          thickness=14, len=0.65),
            opacity=0.82,
            sizemode='diameter',
        ),
        text=[
            f'<b>{r.lat:.2f}N  {r.lon:.2f}E</b><br>{label}: <b>{r[col]:.3f}</b>'
            for _, r in valid.iterrows()],
        hovertemplate='%{text}<extra></extra>',
        name=label, visible=visible,
    )

solar_tr = make_bubble_trace('solar_cf_mean', 'YlOrRd', 'Solar CF', True)
wind_tr  = make_bubble_trace('wind_cf_mean',  'Blues',  'Wind CF',  False)

best_s_m = go.Scattermapbox(
    lat=[best_solar.lat], lon=[best_solar.lon], mode='markers+text',
    marker=dict(size=18, symbol='star', color='red'),
    text=[f'Best Solar CF={best_solar.solar_cf_mean:.3f}'],
    textposition='top right', textfont=dict(size=11, color='red'),
    name='Best Solar', visible=True,
    hovertemplate=f'Best Solar: {best_solar.lat:.2f}N {best_solar.lon:.2f}E<extra></extra>',
)
best_w_m = go.Scattermapbox(
    lat=[best_wind.lat], lon=[best_wind.lon], mode='markers+text',
    marker=dict(size=18, symbol='star', color='darkblue'),
    text=[f'Best Wind CF={best_wind.wind_cf_mean:.3f}'],
    textposition='top right', textfont=dict(size=11, color='darkblue'),
    name='Best Wind', visible=False,
    hovertemplate=f'Best Wind: {best_wind.lat:.2f}N {best_wind.lon:.2f}E<extra></extra>',
)

fig = go.Figure(data=[solar_tr, best_s_m, wind_tr, best_w_m])
fig.update_layout(
    title=dict(text='Kazakhstan Renewable Energy Potential - Bubble Map (PVGIS ERA5)',
               font=dict(size=17), x=0.5),
    mapbox=dict(style='carto-positron',
                center=dict(lat=48.5, lon=66.5), zoom=3.9),
    height=680, margin=dict(l=0, r=0, t=65, b=0),
    updatemenus=[dict(
        type='buttons', direction='left',
        x=0.5, xanchor='center', y=1.10,
        buttons=[
            dict(label='Solar Potential', method='update',
                 args=[{'visible': [True,  True,  False, False]},
                       {'title.text': 'Kazakhstan Solar CF - larger bubble = more potential'}]),
            dict(label='Wind Potential', method='update',
                 args=[{'visible': [False, False, True,  True]},
                       {'title.text': 'Kazakhstan Wind CF - larger bubble = more potential'}]),
        ],
        showactive=True, bgcolor='#f0f4ff',
        bordercolor='#9999cc', font=dict(size=13),
    )],
    annotations=[dict(
        text='Larger bubble = higher CF | Hover for value | Star = best site',
        align='left', showarrow=False,
        xref='paper', yref='paper', x=0.01, y=0.01,
        xanchor='left', yanchor='bottom',
        bgcolor='rgba(255,255,255,0.88)',
        bordercolor='#cccccc', borderwidth=1,
        font=dict(size=10),
    )],
)
fig.write_html(OUT_DIR / 'renewable_potential_map.html', include_plotlyjs='cdn')
print('Interactive HTML map saved -> results/renewable_potential_map.html')
print('Done! Check your results/ folder.')