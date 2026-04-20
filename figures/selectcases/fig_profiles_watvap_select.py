#
# Vertical profiles of specific humidity: Selected Cases
# Layout: 1 row x 3 columns (Cases 1, 4, 16)
# Solid lines: substellar hemisphere average; dashed: anti-stellar hemisphere average.
#
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

tfreeze  = 273.16
M_N2     = 0.028   # kg/mol
R_gas    = 8.314   # J/(mol·K)
g_accel  = 9.8     # m/s²

# Nominal N2 surface pressures per case (hPa)
P_SURFACE = {1: 700., 4: 2340., 16: 10000.}

MODEL_STYLES = {
    'ExoCAM':    dict(color='#1f77b4', lw=1.6, ls='-'),
    'ExoPlaSim': dict(color='#ff7f0e', lw=1.6, ls='-'),
    'ROCKE-3D':  dict(color='#2ca02c', lw=1.6, ls='-'),
    'PCM':       dict(color='#d62728', lw=1.6, ls='-'),
    'LFRic':     dict(color='#9467bd', lw=1.6, ls='-'),
}
MODEL_LABELS = {
    'ExoCAM':    'ExoCAM',
    'ExoPlaSim': 'ExoPlaSim',
    'ROCKE-3D':  'ROCKE-3D',
    'PCM':       'Generic PCM',
    'LFRic':     'LFRic',
}


# ── helpers ──────────────────────────────────────────────────────────

def roll_to_180(data, lon):
    """Roll a (..., lon) array and its lon vector from [0, 360) to [-180, 180)."""
    idx = np.searchsorted(lon, 180.0)
    return (np.concatenate([data[..., idx:], data[..., :idx]], axis=-1),
            np.concatenate([lon[idx:] - 360.0, lon[:idx]]))


def hemi_mean(data, lat, lon, lon_ss=0.0, day=True):
    """Area-weighted hemispheric mean; NaN-safe.

    data : (nlev, nlat, nlon)  – last two axes are lat / lon
    day  : True → substellar hemisphere, False → anti-stellar hemisphere
    Returns (nlev,)
    """
    cos_lat = np.cos(np.radians(lat))
    cos_d   = np.cos(np.radians(lon - lon_ss))
    mask    = cos_d > 0 if day else cos_d < 0
    geo_w   = cos_lat[:, np.newaxis] * mask[np.newaxis, :]
    data    = np.asarray(data, dtype=float)
    valid_w = geo_w[np.newaxis, :, :] * np.isfinite(data)
    w_sum   = valid_w.sum(axis=(1, 2))
    return (np.where(np.isfinite(data), data, 0.0) * valid_w).sum(axis=(1, 2)) / w_sum

def ss_hemi_mean(data, lat, lon, lon_ss=0.0):
    return hemi_mean(data, lat, lon, lon_ss=lon_ss, day=True)


def area_mean_2d(data2d, lat, lon, lon_ss=0.0):
    """Same, but for a (nlat, nlon) field."""
    cos_lat = np.cos(np.radians(lat))
    mask    = np.cos(np.radians(lon - lon_ss)) > 0
    weights = cos_lat[:, np.newaxis] * mask[np.newaxis, :]
    return np.sum(data2d * weights) / np.sum(weights)


def pcm_pressure_from_alt(T_K, alt_km, P_s_hPa):
    """Integrate pressure downward from surface using hydrostatic eq.

    T_K    : (nlev,) mean temperature profile in K (surface→TOA order)
    alt_km : (nlev,) altitude in km (surface→TOA)
    Returns (nlev,) pressure in hPa
    """
    P = np.empty_like(alt_km)
    P[0] = P_s_hPa
    alt_m = alt_km * 1000.0
    for k in range(1, len(alt_km)):
        dz    = alt_m[k] - alt_m[k - 1]
        T_mid = 0.5 * (T_K[k - 1] + T_K[k])
        P[k]  = P[k - 1] * np.exp(-M_N2 * g_accel * dz / (R_gas * T_mid))
    return P


# ── ExoCAM ───────────────────────────────────────────────────────────
# Hybrid sigma-pressure: P(k) = hyam(k)*P0 + hybm(k)*PS_mean
# Substellar point at lon = 180°

_d = '/models/data/samosa/exocam'

def read_exocam(case):
    with netCDF4.Dataset(f'{_d}/samosa{case}.cam.h0.avg.nc') as ds:
        T    = np.squeeze(np.array(ds.variables['T']))   # (lev, lat, lon)
        Q    = np.squeeze(np.array(ds.variables['Q']))
        PS   = np.squeeze(np.array(ds.variables['PS']))  # Pa
        hyam = np.array(ds.variables['hyam'])
        hybm = np.array(ds.variables['hybm'])
        P0   = float(ds.variables['P0'][:])              # Pa
        lat  = np.array(ds.variables['lat'])
        lon  = np.array(ds.variables['lon'])
    PS_mean  = area_mean_2d(PS, lat, lon, lon_ss=180.)
    lev_hPa  = (hyam * P0 + hybm * PS_mean) / 100.0
    T_ss     = hemi_mean(T, lat, lon, lon_ss=180., day=True)
    Q_ss     = hemi_mean(Q, lat, lon, lon_ss=180., day=True)
    T_as     = hemi_mean(T, lat, lon, lon_ss=180., day=False)
    Q_as     = hemi_mean(Q, lat, lon, lon_ss=180., day=False)
    return T_ss, Q_ss, T_as, Q_as, lev_hPa

T_exocam, Q_exocam, T_exocam_as, Q_exocam_as, P_exocam = zip(*[read_exocam(c) for c in [1, 4, 16]])


# ── ExoPlaSim ────────────────────────────────────────────────────────
# Pure sigma levels; substellar point at lon = 0° (after roll_to_180)

_d = '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day'
_plasim_files = [
    f'{_d}/t21_synchronous_0.70pn2_flux500_400.0co2_3000teff_15day.nc',
    f'{_d}/t21_synchronous_2.34pn2_flux1200_400.0co2_3000teff_15day.nc',
    f'{_d}/t21_synchronous_10.00pn2_flux1400_400.0co2_3000teff_15day.nc',
]

def read_plasim(path):
    with netCDF4.Dataset(path) as ds:
        T   = np.average(np.array(ds.variables['ta']),  axis=0)  # (lev, lat, lon)
        hus = np.average(np.array(ds.variables['hus']), axis=0)
        ps  = np.average(np.array(ds.variables['ps']),  axis=0)  # (lat, lon) hPa
        lat = np.array(ds.variables['lat'])
        lon = np.array(ds.variables['lon'])
        sigma = np.array(ds.variables['lev'])
    T,   lon_s = roll_to_180(T,   lon)
    hus, _     = roll_to_180(hus, lon)
    ps,  _     = roll_to_180(ps,  lon)
    ps_mean = area_mean_2d(ps, lat, lon_s)
    lev_hPa = sigma * ps_mean
    T_ss    = hemi_mean(T,   lat, lon_s, day=True)
    Q_ss    = hemi_mean(hus, lat, lon_s, day=True)
    T_as    = hemi_mean(T,   lat, lon_s, day=False)
    Q_as    = hemi_mean(hus, lat, lon_s, day=False)
    return T_ss, Q_ss, T_as, Q_as, lev_hPa

T_plasim, Q_plasim, T_plasim_as, Q_plasim_as, P_plasim = zip(*[read_plasim(f) for f in _plasim_files])


# ── ROCKE-3D ─────────────────────────────────────────────────────────
# plm: surface → TOA in mb; temp in °C; SS point at lon=0° after rotation

_d = '/models/data/samosa/rocke3d'

def read_r3d(fname):
    with netCDF4.Dataset(fname) as ds:
        T   = np.ma.filled(ds.variables['temp'][:], np.nan).astype(float) + tfreeze
        q   = np.ma.filled(ds.variables['q'][:],    np.nan).astype(float)
        lat = np.array(ds.variables['lat'])
        lon = np.array(ds.variables['lon'])
        plm = np.array(ds.variables['plm'])             # mb = hPa
    T, lon_s = roll_to_180(T, lon)
    q, _     = roll_to_180(q, lon)
    # Rotate SS point from lon=180° to lon=0°
    _lon = lon_s + 180.0
    _lon[_lon >= 180.0] -= 360.0
    idx  = np.argsort(_lon)
    lon_s, T, q = _lon[idx], T[:, :, idx], q[:, :, idx]
    T_ss = hemi_mean(T, lat, lon_s, day=True)
    Q_ss = hemi_mean(q, lat, lon_s, day=True)
    T_as = hemi_mean(T, lat, lon_s, day=False)
    Q_as = hemi_mean(q, lat, lon_s, day=False)
    return T_ss, Q_ss, T_as, Q_as, plm

T_r3d, Q_r3d, T_r3d_as, Q_r3d_as, P_r3d = zip(*[
    read_r3d(f'/models/data/samosa/rocke3d/rocke_{c:02d}q.nc') for c in [1, 4, 16]])


# ── Generic PCM ──────────────────────────────────────────────────────
# Altitude coordinate (km, surface→TOA); pressure computed from T profile
# atmospheric_pressure is absent (all NaN); SS at lon=0°

_d = '/models/data/samosa/genericpcm/OHT_off'
_pcm_files = [
    (f'{_d}/case-1/SAMOSA_output_file_Generic_PCM_case-1_OHT_off.nc',   1),
    (f'{_d}/case-4/SAMOSA_output_file_Generic_PCM_case-4_OHT_off.nc',   4),
    (f'{_d}/case-16/SAMOSA_output_file_Generic_PCM_case-16_OHT_off.nc', 16),
]

def read_pcm(path, case_num):
    with netCDF4.Dataset(path) as ds:
        T_raw = ds.variables['atmospheric_temperature'][:].data  # bypass mask; (alt, lat, lon)
        q_raw = ds.variables['specific_humidity'][:].data
        lat   = np.array(ds.variables['latitude'])
        lon   = np.array(ds.variables['longitude'])
        alt   = np.array(ds.variables['altitude'])               # km
    # Some altitude levels are entirely NaN (model fill); keep only valid ones
    valid = np.isfinite(T_raw).all(axis=(1, 2))
    T_raw, q_raw, alt = T_raw[valid], q_raw[valid], alt[valid]
    T_ss   = hemi_mean(T_raw, lat, lon, day=True)
    Q_ss   = hemi_mean(q_raw, lat, lon, day=True)
    T_as   = hemi_mean(T_raw, lat, lon, day=False)
    Q_as   = hemi_mean(q_raw, lat, lon, day=False)
    P_prof = pcm_pressure_from_alt(T_ss, alt, P_SURFACE[case_num])
    return T_ss, Q_ss, T_as, Q_as, P_prof

T_pcm, Q_pcm, T_pcm_as, Q_pcm_as, P_pcm = zip(*[read_pcm(f, c) for f, c in _pcm_files])


# ── LFRic ────────────────────────────────────────────────────────────
# Full levels, surface (lev 0) → TOA (lev -1); SS at lon=0° after roll
# Case 16 not available

_d = '/models/data/samosa/lfric'

def read_lfric(fname):
    with netCDF4.Dataset(fname) as ds:
        T   = np.array(ds.variables['temperature'])       # (full_levels, lat, lon)
        qv  = np.array(ds.variables['qv'])
        P3d = np.array(ds.variables['pressure_in_wth'])   # Pa
        lat = np.array(ds.variables['lat'])
        lon = np.array(ds.variables['lon'])
    T,   lon_s = roll_to_180(T,   lon)
    qv,  _     = roll_to_180(qv,  lon)
    P3d, _     = roll_to_180(P3d, lon)
    T_ss   = hemi_mean(T,   lat, lon_s, day=True)
    Q_ss   = hemi_mean(qv,  lat, lon_s, day=True)
    T_as   = hemi_mean(T,   lat, lon_s, day=False)
    Q_as   = hemi_mean(qv,  lat, lon_s, day=False)
    P_prof = hemi_mean(P3d, lat, lon_s, day=True) / 100.0   # Pa → hPa
    return T_ss, Q_ss, T_as, Q_as, P_prof

_lfric1 = read_lfric(f'{_d}/lfric_samosa_case01.nc')
_lfric4 = read_lfric(f'{_d}/lfric_samosa_case04.nc')

T_lfric    = [_lfric1[0], _lfric4[0], None]
Q_lfric    = [_lfric1[1], _lfric4[1], None]
T_lfric_as = [_lfric1[2], _lfric4[2], None]
Q_lfric_as = [_lfric1[3], _lfric4[3], None]
P_lfric    = [_lfric1[4], _lfric4[4], None]


# ── Figure: Specific Humidity ─────────────────────────────────────────

_all_P = [p for plist in [P_exocam, P_plasim, P_r3d, P_pcm, P_lfric]
          for p in plist if p is not None]
P_GLOBAL_LO = min(p.min() for p in _all_P) * 0.7
P_GLOBAL_HI = max(p.max() for p in _all_P) * 1.05

case_labels = ['Case 1\n500 W/m², 0.70 bar',
               'Case 4\n1200 W/m², 2.34 bar',
               'Case 16\n1400 W/m², 10.00 bar']

NCASES    = 3
FS_TITLE  = 10
FS_LABEL  = 9
FS_TICK   = 8
FS_LEGEND = 10

fig, axes = plt.subplots(1, NCASES, figsize=(11, 5), layout='constrained')
fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72, wspace=0.06)

for ci, ax in enumerate(axes):
    profiles_ss = [
        ('ExoCAM',    Q_exocam[ci],    P_exocam[ci]),
        ('ExoPlaSim', Q_plasim[ci],    P_plasim[ci]),
        ('ROCKE-3D',  Q_r3d[ci],       P_r3d[ci]),
        ('PCM',       Q_pcm[ci],       P_pcm[ci]),
        ('LFRic',     Q_lfric[ci],     P_lfric[ci]),
    ]
    profiles_as = [
        ('ExoCAM',    Q_exocam_as[ci], P_exocam[ci]),
        ('ExoPlaSim', Q_plasim_as[ci], P_plasim[ci]),
        ('ROCKE-3D',  Q_r3d_as[ci],    P_r3d[ci]),
        ('PCM',       Q_pcm_as[ci],    P_pcm[ci]),
        ('LFRic',     Q_lfric_as[ci],  P_lfric[ci]),
    ]

    panel_handles, panel_labels = [], []
    for name, Q_prof, P_prof in profiles_ss:
        if Q_prof is None:
            continue
        h, = ax.plot(Q_prof * 1e3, P_prof, **MODEL_STYLES[name])
        panel_handles.append(h)
        panel_labels.append(MODEL_LABELS[name])

    for name, Q_prof, P_prof in profiles_as:
        if Q_prof is None:
            continue
        ax.plot(Q_prof * 1e3, P_prof, color=MODEL_STYLES[name]['color'], lw=1.6, ls='--')

    ax.legend(panel_handles, panel_labels, loc='best', fontsize=FS_LEGEND - 1,
              frameon=False, handlelength=1.5, borderpad=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(P_GLOBAL_HI, P_GLOBAL_LO)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10, labelOnlyBase=True))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(False)

    ax.set_title(case_labels[ci], fontsize=FS_TITLE, linespacing=1.5)
    ax.set_xlabel('Specific Humidity (g kg$^{-1}$)', fontsize=FS_LABEL)
    ax.set_ylabel('Pressure (hPa)', fontsize=FS_LABEL)

from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color='k', lw=1.6, ls='-',  label='Substellar hemi.'),
    Line2D([0], [0], color='k', lw=1.6, ls='--', label='Anti-stellar hemi.'),
]
fig.legend(handles=style_handles, loc='outside lower center', ncols=2,
           fontsize=FS_LEGEND, frameon=True, framealpha=0.9,
           handlelength=2.5, handleheight=1.2, handletextpad=0.6,
           borderpad=0.6, labelspacing=0.5)

fig.savefig('fig_profiles_watvap_select.png', bbox_inches='tight', dpi=150)
fig.savefig('fig_profiles_watvap_select.eps', bbox_inches='tight')
