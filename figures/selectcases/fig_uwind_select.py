#
# Vertical cross sections of zonal-mean zonal wind: Selected Cases
# Layout: 3 rows (Cases 1, 4, 16) x 5 columns (models)
# Black contour at U = 0
#
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean


def area_mean(arr2d, lat):
    """Area-weighted mean of a (lat,) or (nlev, lat) array over the lat axis."""
    cos_lat = np.cos(np.radians(lat))
    w = cos_lat / cos_lat.sum()
    return (arr2d * w).sum(axis=-1)


# ── ExoCAM ───────────────────────────────────────────────────────────
_d = '/models/data/samosa/exocam'

def read_exocam_wind(case):
    with netCDF4.Dataset(f'{_d}/samosa{case}.cam.h0.avg.nc') as ds:
        U    = np.squeeze(np.array(ds.variables['U']))   # (lev, lat, lon)
        PS   = np.squeeze(np.array(ds.variables['PS']))  # Pa (lat, lon)
        hyam = np.array(ds.variables['hyam'])
        hybm = np.array(ds.variables['hybm'])
        P0   = float(ds.variables['P0'][:])
        lat  = np.array(ds.variables['lat'])
    U_bar   = U.mean(axis=-1)                            # (lev, lat)
    PS_area = area_mean(PS.mean(axis=-1), lat)           # scalar Pa
    lev_hPa = (hyam * P0 + hybm * PS_area) / 100.0      # (lev,) hPa
    return U_bar, lev_hPa, lat


# ── ExoPlaSim ────────────────────────────────────────────────────────
_d_plasim = '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day'
_plasim_files = [
    f'{_d_plasim}/t21_synchronous_0.70pn2_flux500_400.0co2_3000teff_15day.nc',
    f'{_d_plasim}/t21_synchronous_2.34pn2_flux1200_400.0co2_3000teff_15day.nc',
    f'{_d_plasim}/t21_synchronous_10.00pn2_flux1400_400.0co2_3000teff_15day.nc',
]

def read_plasim_wind(path):
    with netCDF4.Dataset(path) as ds:
        ua    = np.average(np.array(ds.variables['ua']), axis=0)  # (lev, lat, lon)
        ps    = np.average(np.array(ds.variables['ps']), axis=0)  # (lat, lon) hPa
        lat   = np.array(ds.variables['lat'])
        sigma = np.array(ds.variables['lev'])
    U_bar   = ua.mean(axis=-1)               # (lev, lat)
    ps_area = area_mean(ps.mean(axis=-1), lat)
    lev_hPa = sigma * ps_area               # (lev,) hPa
    return U_bar, lev_hPa, lat


# ── ROCKE-3D ─────────────────────────────────────────────────────────
_d_r3d = '/models/data/samosa/rocke3d'

def read_r3d_wind(case):
    with netCDF4.Dataset(f'{_d_r3d}/rocke_{case:02d}q.nc') as ds:
        ub  = np.ma.filled(ds.variables['ub'][:], np.nan).astype(float)  # (plm, lat2, lon2)
        lat = np.array(ds.variables['lat2'])
        plm = np.array(ds.variables['plm'])   # hPa, surface→TOA
    U_bar = np.nanmean(ub, axis=-1)           # (plm, lat2)
    return U_bar, plm, lat


# ── Generic PCM ──────────────────────────────────────────────────────
_d_pcm = '/models/data/samosa/genericpcm/OHT_off'

def read_pcm_wind(case):
    path = f'{_d_pcm}/case-{case}/SAMOSA_output_file_Generic_PCM_case-{case}_OHT_off.nc'
    with netCDF4.Dataset(path) as ds:
        u = ds.variables['u_wind_speed'][:].data         # (alt, lat, lon)
        T = ds.variables['atmospheric_temperature'][:].data
        P = ds.variables['atmospheric_pressure'][:].data  # Pa (alt, lat, lon)
        lat = np.array(ds.variables['latitude'])
    valid   = np.isfinite(T).all(axis=(1, 2))
    u, P    = u[valid], P[valid]
    U_bar   = u.mean(axis=-1)                            # (alt_valid, lat)
    P_area  = area_mean(P.mean(axis=-1), lat) / 100.0   # (alt_valid,) hPa
    return U_bar, P_area, lat


# ── LFRic ────────────────────────────────────────────────────────────
_d_lfric = '/models/data/samosa/lfric'

def read_lfric_wind(fname):
    with netCDF4.Dataset(fname) as ds:
        u      = np.array(ds.variables['u_in_w3'])           # (half_levels=41, lat, lon)
        P_full = np.array(ds.variables['pressure_in_wth'])   # (full_levels=42, lat, lon) Pa
        lat    = np.array(ds.variables['lat'])
    P_half = 0.5 * (P_full[:-1] + P_full[1:])               # (41, lat, lon)
    U_bar  = u.mean(axis=-1)                                 # (41, lat)
    P_area = area_mean(P_half.mean(axis=-1), lat) / 100.0   # (41,) hPa
    return U_bar, P_area, lat


# ── Load data ────────────────────────────────────────────────────────
exocam_data = [read_exocam_wind(c) for c in [1, 4, 16]]
plasim_data = [read_plasim_wind(f) for f in _plasim_files]
r3d_data    = [read_r3d_wind(c)   for c in [1, 4, 16]]
# Case 16 is not converged in the Generic PCM (see SAMOSA_summary.pdf) and is omitted
pcm_data    = [read_pcm_wind(1), read_pcm_wind(4), None]
_lf1        = read_lfric_wind(f'{_d_lfric}/lfric_samosa_case01.nc')
_lf4        = read_lfric_wind(f'{_d_lfric}/lfric_samosa_case04.nc')
lfric_data  = [_lf1, _lf4, None]

# panels[col][row] = (U_bar, P_hPa, lat) or None
panels = [
    plasim_data,   # ExoPlaSim
    exocam_data,   # ExoCAM
    r3d_data,      # ROCKE-3D
    pcm_data,      # Generic PCM
    lfric_data,    # LFRic
]

col_titles  = ['ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic']
case_labels = ['Case 1\n500 W/m²\n0.70 bar',
               'Case 4\n1200 W/m²\n2.34 bar',
               'Case 16\n1400 W/m²\n10.00 bar']

# ── Per-row pressure limits ──────────────────────────────────────────
row_Pmax, row_Pmin = [], []
for row in range(3):
    pvals = [col[row][1] for col in panels if col[row] is not None]
    row_Pmax.append(max(p.max() for p in pvals) * 1.05)
    row_Pmin.append(min(p.min() for p in pvals) * 0.7)

# ── Figure ────────────────────────────────────────────────────────────
UMIN, UMAX = -80.0, 80.0
levels     = np.linspace(UMIN, UMAX, 27)
cm         = cmocean.cm.balance

TITLE_FS = 11
LABEL_FS = 9
TICK_FS  = 8
NA_FS    = 10
CB_FS    = 9

def lat_fmt(x, _):
    if x == 0:
        return '0°'
    return f'{int(abs(x))}°{"S" if x < 0 else "N"}'

fig = plt.figure(layout='constrained', figsize=(13, 7))
fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72, wspace=0.04, hspace=0.08)
ax_array = fig.subplots(3, 5, squeeze=False)

im = None
for col, (col_panels, title) in enumerate(zip(panels, col_titles)):
    for row, panel in enumerate(col_panels):
        ax = ax_array[row, col]
        if panel is None:
            ax.set_facecolor('#cccccc')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=NA_FS, color='#555555')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            U_bar, P_hPa, lat = panel
            im = ax.contourf(lat, P_hPa, U_bar, cmap=cm,
                             vmin=UMIN, vmax=UMAX, levels=levels, extend='both')
            ax.contour(lat, P_hPa, U_bar, levels=[0.0],
                       colors='k', linewidths=0.8)
            ax.set_yscale('log')
            ax.set_ylim(row_Pmax[row], row_Pmin[row])
            ax.set_xlim(-90, 90)
            ax.set_xticks([-60, -30, 0, 30, 60])
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lat_fmt))
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
            ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10, labelOnlyBase=True))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.tick_params(axis='both', labelsize=TICK_FS)
            if col > 0:
                ax.set_yticklabels([])

        if row == 0:
            ax.set_title(title, fontsize=TITLE_FS)
        if col == 0 and panel is not None:
            ax.set_ylabel('Pressure (hPa)', fontsize=LABEL_FS)
        if col == 0:
            ax.text(-0.30, 0.5, case_labels[row], transform=ax.transAxes,
                    ha='right', va='center', fontsize=LABEL_FS, linespacing=1.5)

cb = fig.colorbar(im, ax=ax_array, extend='both',
                  ticks=np.arange(-80, 81, 20),
                  orientation='vertical', shrink=0.8, pad=0.02)
cb.ax.tick_params(labelsize=CB_FS)
cb.set_label('Zonal mean zonal wind (m s⁻¹)', fontsize=CB_FS)

fig.savefig('fig_uwind_select.png', bbox_inches='tight', dpi=150)
fig.savefig('fig_uwind_select.eps', bbox_inches='tight')
