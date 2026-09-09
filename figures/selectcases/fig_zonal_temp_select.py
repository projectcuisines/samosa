#
# Zonal mean surface temperature: Selected Cases
# Layout: 1 row x 3 columns (Cases 1, 4, 16)
# Solid lines: substellar hemisphere zonal mean; dashed: anti-stellar hemisphere.
#
# Companion to fig_profiles_temp_select.py, which shows the same three cases in
# the vertical. Here the average is taken along longitude at each latitude, so
# the figure carries the meridional structure the profile figure integrates
# away, and the day/night convention is the same in both.
#
# PlaHab is included here although it is absent from the vertical profile
# figure: it is two-dimensional and so has no profile to plot, but it does
# resolve latitude, and its caseN_tsurf.out files exist for exactly the three
# selected cases. HEXTOR and ExoColumn are not included. ExoColumn is a single
# column and has no latitude at all. HEXTOR does resolve latitude -- 18 belts --
# but submitted only the global summary file, so no per-belt output exists in
# the archive; its belts are also in the tidally locked coordinate rather than
# the geographic latitude used here, so they would need transforming even once
# supplied.
#
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import netCDF4
import numpy as np
import matplotlib.pyplot as plt

tfreeze = 273.16

# Checked against the reported global means: recombining the two hemispheric
# curves and area weighting in latitude returns each model's submitted global
# mean to 0.03 K for ROCKE-3D, LFRic and PlaHab. ExoCAM, ExoPlaSim and the
# Generic PCM come back about 1 K low, which is the flat half-and-half
# recombination, not the curves: those grids either carry a column exactly on
# the terminator, which belongs to neither hemisphere and is dropped, or split
# unevenly. Averaging their fields over all longitudes instead reproduces the
# reported means to 0.1 K.

# Colors follow fig_profiles_temp_select.py, with PlaHab taking the color it
# carries in the allcases figures.
MODEL_STYLES = {
    'ExoCAM':    dict(color='#1f77b4', lw=1.6),
    'ExoPlaSim': dict(color='#ff7f0e', lw=1.6),
    'ROCKE-3D':  dict(color='#2ca02c', lw=1.6),
    'PCM':       dict(color='#d62728', lw=1.6),
    'LFRic':     dict(color='#9467bd', lw=1.6),
    'PlaHab':    dict(color='#8c564b', lw=1.6),
}
MODEL_LABELS = {
    'ExoCAM':    'ExoCAM',
    'ExoPlaSim': 'ExoPlaSim',
    'ROCKE-3D':  'ROCKE-3D',
    'PCM':       'Generic PCM',
    'LFRic':     'LFRic',
    'PlaHab':    'PlaHab',
}


# -- helpers ---------------------------------------------------------

def roll_to_180(data, lon):
    """Roll a (..., lon) field and its lon vector from [0, 360) to [-180, 180)."""
    idx = np.searchsorted(lon, 180.0)
    return (np.concatenate([data[..., idx:], data[..., :idx]], axis=-1),
            np.concatenate([lon[idx:] - 360.0, lon[:idx]]))


def read_nc(path, var, lat_var='lat', lon_var='lon', avg_axis=None, offset=0.0):
    with netCDF4.Dataset(path) as ds:
        raw  = ds.variables[var]
        data = (np.average(raw, axis=avg_axis) if avg_axis is not None else np.array(raw))
        data = np.squeeze(np.asarray(data, dtype=float)) + offset
        lat  = np.array(ds.variables[lat_var])
        lon  = np.array(ds.variables[lon_var])
    return data, lat, lon


def hemi_zonal_mean(data2d, lon, lon_ss=0.0, day=True):
    """Mean along longitude over one hemisphere, at each latitude.

    data2d : (nlat, nlon)
    day    : True -> substellar hemisphere, False -> anti-stellar
    Returns (nlat,). NaN-safe, and unweighted in longitude because every grid
    used here is uniform in longitude, so a cell-width weight is a constant.
    """
    cos_d = np.cos(np.radians(lon - lon_ss))
    mask  = (cos_d > 0) if day else (cos_d < 0)
    data  = np.asarray(data2d, dtype=float)
    w     = mask[np.newaxis, :] * np.isfinite(data)
    return np.where(np.isfinite(data), data, 0.0).dot(mask * 1.0) / w.sum(axis=1)


def both_hemis(data2d, lat, lon, lon_ss=0.0):
    return (lat,
            hemi_zonal_mean(data2d, lon, lon_ss, day=True),
            hemi_zonal_mean(data2d, lon, lon_ss, day=False))


# -- ExoCAM ----------------------------------------------------------
# Substellar point at native lon = 180, so the grid is left unrolled.
_d = '/models/data/samosa/exocam'
Ts_exocam, lat_exocam, lon_exocam = zip(*[
    read_nc(f'{_d}/samosa{c}.cam.h0.avg.nc', 'TS', avg_axis=0) for c in (1, 4, 16)])
exocam = [both_hemis(t, lat_exocam[0], lon_exocam[0], lon_ss=180.) for t in Ts_exocam]

# -- ExoPlaSim -------------------------------------------------------
_d = '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day'
_plasim_files = [
    f'{_d}/t21_synchronous_0.70pn2_flux500_400.0co2_3000teff_15day.nc',
    f'{_d}/t21_synchronous_2.34pn2_flux1200_400.0co2_3000teff_15day.nc',
    f'{_d}/t21_synchronous_10.00pn2_flux1400_400.0co2_3000teff_15day.nc',
]
plasim = []
for f in _plasim_files:
    ts, lat, lon = read_nc(f, 'ts', avg_axis=0)
    ts, lon_s = roll_to_180(ts, lon)
    plasim.append(both_hemis(ts, lat, lon_s))

# -- ROCKE-3D --------------------------------------------------------
# Celsius, and the substellar point is at native lon = 180: roll, then rotate
# it to lon = 0 so the hemisphere mask matches the other models.
_d = '/models/data/samosa/rocke3d'
r3d = []
for c in (1, 4, 16):
    ts, lat, lon = read_nc(f'{_d}/rocke_{c:02d}q.nc', 'tsurf', offset=tfreeze)
    ts, lon_s = roll_to_180(ts, lon)
    _lon = lon_s + 180.0
    _lon[_lon >= 180.0] -= 360.0
    idx = np.argsort(_lon)
    r3d.append(both_hemis(ts[:, idx], lat, _lon[idx]))

# -- Generic PCM -----------------------------------------------------
# Case 16 is not converged (see SAMOSA_summary.pdf) and is omitted.
_d = '/models/data/samosa/genericpcm/OHT_off'
pcm = []
for c in (1, 4):
    ts, lat, lon = read_nc(
        f'{_d}/case-{c}/SAMOSA_output_file_Generic_PCM_case-{c}_OHT_off.nc',
        'surface_temperature', lat_var='latitude', lon_var='longitude')
    pcm.append(both_hemis(ts, lat, lon))
pcm.append(None)

# -- LFRic -----------------------------------------------------------
_d = '/models/data/samosa/lfric'
lfric = []
for c in (1, 4, 16):
    ts, lat, lon = read_nc(f'{_d}/lfric_samosa_case{c:02d}.nc', 'grid_surface_temperature')
    ts, lon_s = roll_to_180(ts, lon)
    lfric.append(both_hemis(ts, lat, lon_s))

# -- PlaHab ----------------------------------------------------------
# Plain text: column 0 is latitude, the remaining 20 are longitude.
lon_plahab = np.array([-171., -153., -135., -117.,  -99.,  -81.,  -63.,  -45.,  -27.,  -9.,
                          9.,   27.,   45.,   63.,   81.,   99.,  117.,  135.,  153.,  171.])
_d = '/models/data/samosa/plahab/simulations'
plahab = []
for c, s in ((1, 'sample1'), (4, 'sample4'), (16, 'sample16')):
    raw = np.loadtxt(f'{_d}/{s}/case{c}_tsurf.out')
    plahab.append(both_hemis(raw[:, 1:], raw[:, 0], lon_plahab))


# -- Figure ----------------------------------------------------------
case_labels = ['Case 1\n500 W/m², 0.70 bar',
               'Case 4\n1200 W/m², 2.34 bar',
               'Case 16\n1400 W/m², 10.00 bar']

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 10, 9, 8, 10

fig, axes = plt.subplots(1, 3, figsize=(11, 5), layout='constrained')
fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72, wspace=0.06)

for ci, ax in enumerate(axes):
    series = [
        ('ExoCAM',    exocam[ci]),
        ('ExoPlaSim', plasim[ci]),
        ('ROCKE-3D',  r3d[ci]),
        ('PCM',       pcm[ci]),
        ('LFRic',     lfric[ci]),
        ('PlaHab',    plahab[ci]),
    ]

    # Freezing reference, drawn first so the model curves sit over it
    ax.axhline(tfreeze, color='0.6', lw=0.8, ls=(0, (4, 3)), zorder=1)

    for name, entry in series:
        if entry is None:
            continue
        lat, T_ss, T_as = entry
        ax.plot(lat, T_ss, ls='-',  zorder=3, **MODEL_STYLES[name])
        ax.plot(lat, T_as, ls='--', zorder=3, **MODEL_STYLES[name])

    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(False)

    ax.set_title(case_labels[ci], fontsize=FS_TITLE, linespacing=1.5)
    ax.set_xlabel('Latitude (°)', fontsize=FS_LABEL)
    ax.set_ylabel('Zonal mean surface temperature (K)', fontsize=FS_LABEL)

from matplotlib.lines import Line2D
# A six-entry legend inside a panel covers the curves it is labelling, so the
# models go above the panels, as in fig_energy_balance.py, and the line-style
# key stays below. Two legends cannot share 'outside lower center': constrained
# layout gives them the same slot and the second one hides the first. Only five
# curves appear in the Case 16 panel, the Generic PCM having no converged
# solution there.
model_handles = [Line2D([0], [0], ls='-', label=MODEL_LABELS[m], **MODEL_STYLES[m])
                 for m in ('ExoCAM', 'ExoPlaSim', 'ROCKE-3D', 'PCM', 'LFRic', 'PlaHab')]
style_handles = [
    Line2D([0], [0], color='k', lw=1.6, ls='-',  label='Substellar hemi.'),
    Line2D([0], [0], color='k', lw=1.6, ls='--', label='Anti-stellar hemi.'),
    Line2D([0], [0], color='0.6', lw=0.8, ls=(0, (4, 3)), label='273.16 K'),
]
fig.legend(handles=model_handles, loc='outside upper center', ncols=6,
           fontsize=FS_LEGEND, frameon=False,
           handlelength=2.0, handletextpad=0.6, columnspacing=1.8)
fig.legend(handles=style_handles, loc='outside lower center', ncols=3,
           fontsize=FS_LEGEND, frameon=True, framealpha=1.0,
           handlelength=2.5, handleheight=1.2, handletextpad=0.6,
           borderpad=0.6, labelspacing=0.5)

fig.savefig('fig_zonal_temp_select.png', bbox_inches='tight', dpi=150)
fig.savefig('fig_zonal_temp_select.eps', bbox_inches='tight')
