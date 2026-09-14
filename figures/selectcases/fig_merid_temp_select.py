#
# Meridional mean surface temperature: Selected Cases
# Layout: 1 row x 3 columns (Cases 1, 4, 16)
# x axis is longitude with the substellar point at the centre.
#
# Companion to fig_profiles_temp_select.py, which shows the same three cases in
# the vertical. Here the surface temperature is averaged over latitude at each
# longitude, weighted by cos(lat) for cell area, so the x axis runs from the
# anti-stellar meridian through the substellar point and back, and the day-night
# structure that the hemispheric averaging of the profile figure collapses into
# two numbers is resolved as a curve. Longitude is measured from the substellar
# point, so the terminators are at +/-90 degrees.
#
# There is no substellar / anti-stellar line style here as there is in the
# profile figures: longitude is itself the day-night axis, so each model is a
# single curve and the contrast is read off the curve directly.
#
# PlaHab is included although it is absent from the vertical profile figures:
# it is two-dimensional and so has no profile to plot, but it does resolve
# longitude, and its caseN_tsurf.out files exist for exactly the three selected
# cases.
#
# HEXTOR is included at all three cases. It is one-dimensional in the tidally
# locked coordinate, so its 18 belts are indexed by theta, the angle from the
# substellar point, which is very nearly the variable plotted here. Its first
# submission had no steady state at Case 16; with CO2 corrected to the
# protocol's 400 ubar partial pressure (2026-09-14) that case converges at a
# 465 K global mean, its day side on the runaway plateau of the outgoing
# longwave, and it sets the vertical scale of that panel.
#
# ExoColumn is a single column and cannot appear at all.
#
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import netCDF4
import numpy as np
import matplotlib.pyplot as plt

tfreeze = 273.16

# Checked against the reported global means: averaging each curve uniformly in
# longitude returns the model's submitted global mean, exactly, because the
# longitude grids are uniform and the latitude weighting is already applied.
# Unlike a hemispheric average this involves no terminator convention, so there
# is no residual for the grids that carry a column on the terminator itself.

# Colors follow fig_profiles_temp_select.py, with PlaHab taking the color it
# carries in the allcases figures.
MODEL_STYLES = {
    'ExoCAM':    dict(color='#1f77b4', lw=1.6),
    'ExoPlaSim': dict(color='#ff7f0e', lw=1.6),
    'ROCKE-3D':  dict(color='#2ca02c', lw=1.6),
    'PCM':       dict(color='#d62728', lw=1.6),
    'LFRic':     dict(color='#9467bd', lw=1.6),
    'PlaHab':    dict(color='#8c564b', lw=1.6),
    'HEXTOR':    dict(color='#17becf', lw=1.6),
}
MODEL_LABELS = {
    'ExoCAM':    'ExoCAM',
    'ExoPlaSim': 'ExoPlaSim',
    'ROCKE-3D':  'ROCKE-3D',
    'PCM':       'Generic PCM',
    'LFRic':     'LFRic',
    'PlaHab':    'PlaHab',
    'HEXTOR':    'HEXTOR',
}


# -- helpers ---------------------------------------------------------

def read_nc(path, var, lat_var='lat', lon_var='lon', avg_axis=None, offset=0.0):
    with netCDF4.Dataset(path) as ds:
        raw  = ds.variables[var]
        data = (np.average(raw, axis=avg_axis) if avg_axis is not None else np.array(raw))
        data = np.squeeze(np.asarray(data, dtype=float)) + offset
        lat  = np.array(ds.variables[lat_var])
        lon  = np.array(ds.variables[lon_var])
    return data, lat, lon


def merid_mean(data2d, lat):
    """Area-weighted mean along latitude, at each longitude. NaN-safe.

    data2d : (nlat, nlon) -> returns (nlon,)
    """
    data = np.asarray(data2d, dtype=float)
    w    = np.cos(np.radians(np.asarray(lat)))[:, np.newaxis] * np.isfinite(data)
    return (np.where(np.isfinite(data), data, 0.0) * w).sum(axis=0) / w.sum(axis=0)


def centre_on_substellar(data2d, lat, lon, lon_ss):
    """Meridional mean on a longitude axis running -180..180 about the substellar
    point, with the wrap point duplicated at both ends so the curve closes."""
    rel = ((np.asarray(lon, dtype=float) - lon_ss + 180.0) % 360.0) - 180.0
    idx = np.argsort(rel)
    rel, prof = rel[idx], merid_mean(data2d, lat)[idx]
    # Repeat the first point at +360 so the curve reaches the right-hand edge
    rel  = np.concatenate([rel, [rel[0] + 360.0]])
    prof = np.concatenate([prof, [prof[0]]])
    return rel, prof


# -- ExoCAM ----------------------------------------------------------
# Substellar point at native lon = 180.
_d = '/models/data/samosa/exocam'
exocam = []
for c in (1, 4, 16):
    ts, lat, lon = read_nc(f'{_d}/samosa{c}.cam.h0.avg.nc', 'TS', avg_axis=0)
    exocam.append(centre_on_substellar(ts, lat, lon, lon_ss=180.))

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
    plasim.append(centre_on_substellar(ts, lat, lon, lon_ss=0.))

# -- ROCKE-3D --------------------------------------------------------
# Celsius, and the substellar point is at native lon = 180.
_d = '/models/data/samosa/rocke3d'
r3d = []
for c in (1, 4, 16):
    ts, lat, lon = read_nc(f'{_d}/rocke_{c:02d}q.nc', 'tsurf', offset=tfreeze)
    r3d.append(centre_on_substellar(ts, lat, lon, lon_ss=180.))

# -- Generic PCM -----------------------------------------------------
# Case 16 is not converged (see SAMOSA_summary.pdf) and is omitted.
_d = '/models/data/samosa/genericpcm/OHT_off'
pcm = []
for c in (1, 4):
    ts, lat, lon = read_nc(
        f'{_d}/case-{c}/SAMOSA_output_file_Generic_PCM_case-{c}_OHT_off.nc',
        'surface_temperature', lat_var='latitude', lon_var='longitude')
    pcm.append(centre_on_substellar(ts, lat, lon, lon_ss=0.))
pcm.append(None)

# -- LFRic -----------------------------------------------------------
_d = '/models/data/samosa/lfric'
lfric = []
for c in (1, 4, 16):
    ts, lat, lon = read_nc(f'{_d}/lfric_samosa_case{c:02d}.nc', 'grid_surface_temperature')
    lfric.append(centre_on_substellar(ts, lat, lon, lon_ss=0.))

# -- PlaHab ----------------------------------------------------------
# Plain text: column 0 is latitude, the remaining 20 are longitude.
lon_plahab = np.array([-171., -153., -135., -117.,  -99.,  -81.,  -63.,  -45.,  -27.,  -9.,
                          9.,   27.,   45.,   63.,   81.,   99.,  117.,  135.,  153.,  171.])
_d = '/models/data/samosa/plahab/simulations'
plahab = []
for c, s in ((1, 'sample1'), (4, 'sample4'), (16, 'sample16')):
    raw = np.loadtxt(f'{_d}/{s}/case{c}_tsurf.out')
    plahab.append(centre_on_substellar(raw[:, 1:], raw[:, 0], lon_plahab, lon_ss=0.))


# -- HEXTOR ----------------------------------------------------------
# One-dimensional in the tidally locked coordinate: T is a function of theta,
# the angle from the substellar point, over 18 belts centred at 5..175 degrees.
#
# The belts are NOT plotted against longitude directly. Every other curve here
# is a cos(lat)-weighted mean over latitude at fixed longitude, and along a
# meridian at longitude L the angle from the substellar point varies with
# latitude as theta = arccos(cos(lat) cos(L)); only on the equator does theta
# equal |L|. HEXTOR is therefore averaged over the same meridian so that it is
# the same quantity as the rest of the ensemble. The mapping matters: it puts
# the Case 1 substellar value at 195.9 K rather than the 202.9 K of the belt
# itself, because the substellar meridian reaches to both poles.
#
# The belt files were added to the archive on 2026-09-09, alongside a
# README_zonal.txt giving the column layout and the mapping used below; before
# that the archive held only global_output_HEXTOR.dat and this figure had to
# read the model's run directory.
_d = '/models/data/samosa/hextor'

def read_hextor(case, lon_deg, nlat=721):
    d = np.loadtxt(f'{_d}/zonal_output_HEXTOR_case{case:02d}.dat')
    theta, T_belt = d[:, 0], d[:, 1]
    lat = np.radians(np.linspace(-90.0, 90.0, nlat))
    w   = np.cos(lat)
    prof = np.array([
        np.average(np.interp(np.degrees(np.arccos(np.clip(np.cos(lat) * np.cos(L), -1.0, 1.0))),
                             theta, T_belt), weights=w)
        for L in np.radians(np.asarray(lon_deg, dtype=float))])
    return np.asarray(lon_deg, dtype=float), prof

_hextor_lon = np.linspace(-180.0, 180.0, 361)
hextor = [read_hextor(1, _hextor_lon), read_hextor(4, _hextor_lon), read_hextor(16, _hextor_lon)]


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
        ('HEXTOR',    hextor[ci]),
    ]

    # Terminators and the freezing point, drawn under the model curves
    for x in (-90, 90):
        ax.axvline(x, color='0.75', lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.axhline(tfreeze, color='0.6', lw=0.8, ls=(0, (4, 3)), zorder=1)

    for name, entry in series:
        if entry is None:
            continue
        lon, T = entry
        ax.plot(lon, T, ls='-', zorder=3, **MODEL_STYLES[name])

    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(False)

    ax.set_title(case_labels[ci], fontsize=FS_TITLE, linespacing=1.5)
    ax.set_xlabel('Longitude from substellar point (°)', fontsize=FS_LABEL)
    ax.set_ylabel('Meridional mean surface temperature (K)', fontsize=FS_LABEL)

from matplotlib.lines import Line2D
# A six-entry legend inside a panel covers the curves it is labelling, so the
# models go above the panels, as in fig_energy_balance.py, and the reference
# lines are keyed below. Two legends cannot share 'outside lower center':
# constrained layout gives them the same slot and the second hides the first.
# Case 16 carries six curves rather than seven, the Generic PCM having no
# converged solution there.
model_handles = [Line2D([0], [0], ls='-', label=MODEL_LABELS[m], **MODEL_STYLES[m])
                 for m in ('ExoCAM', 'ExoPlaSim', 'ROCKE-3D', 'PCM', 'LFRic', 'PlaHab', 'HEXTOR')]
style_handles = [
    Line2D([0], [0], color='0.75', lw=0.8, ls=(0, (2, 2)), label='Terminator'),
    Line2D([0], [0], color='0.6',  lw=0.8, ls=(0, (4, 3)), label='273.16 K'),
]

fig.legend(handles=model_handles, loc='outside upper center', ncols=7,
           fontsize=FS_LEGEND, frameon=False,
           handlelength=2.0, handletextpad=0.6, columnspacing=1.8)
fig.legend(handles=style_handles, loc='outside lower center', ncols=2,
           fontsize=FS_LEGEND, frameon=True, framealpha=1.0,
           handlelength=2.5, handleheight=1.2, handletextpad=0.6,
           borderpad=0.6, labelspacing=0.5)

fig.savefig('fig_merid_temp_select.png', bbox_inches='tight', dpi=150)
fig.savefig('fig_merid_temp_select.eps', bbox_inches='tight')
