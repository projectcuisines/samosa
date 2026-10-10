#
# Comparison for SAMOSA Selected Cases: the CLERO emulator beside the two
# models it is conditioned on
#
# CLERO is a statistical emulator of GCM climates, run in two configurations,
# one conditioned on the UM and one on ExoCAM. Each configuration is drawn
# directly beneath the model it emulates, for Cases 1, 4 and 16, so the figure
# reads as two emulator-versus-parent pairs. Surface temperature by default;
# --wv draws the water vapor column instead.
#
import os, sys
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'allcases'))
import um_clero as uc

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelpad'] = 8

tfreeze = 273.16
WV      = '--wv' in sys.argv
CASES   = (1, 4, 16)


def read_exocam(case_n, kind):
    """TS or TMQ, gw-weighted in the global mean; substellar point at lon=180."""
    var = {'ts': 'TS', 'wv': 'TMQ'}[kind]
    with netCDF4.Dataset(f'/models/data/samosa/exocam/samosa{case_n}.cam.h0.avg.nc') as ds:
        data = np.average(np.array(ds.variables[var]), axis=0)
        lat, lon = np.array(ds.variables['lat']), np.array(ds.variables['lon'])
        gw = np.array(ds.variables['gw'])
    return lon, lat, data, 180.0, np.average(np.mean(data, axis=1), weights=gw)


def read_um(case_n, kind):
    lon, lat, data = uc.um_map(case_n, kind)
    return lon, lat, data, 0.0, np.average(np.mean(data, axis=1), weights=np.cos(np.radians(lat)))


def read_clero(cfg):
    def read(case_n, kind):
        lon, lat, data = uc.clero_map(cfg, case_n, kind)
        with netCDF4.Dataset(f'{uc.ROOT}/clero/CLERO-{cfg}/samosa_CLERO-{cfg}_case{case_n:02d}.nc') as ds:
            w = np.array(ds.variables['area_weight'])
        return lon, lat, data, 0.0, np.sum(np.mean(data, axis=1) * w)
    return read


kind = 'wv' if WV else 'ts'
rows = [('UM', read_um), ('CLERO-UM', read_clero('UM')),
        ('ExoCAM', read_exocam), ('CLERO-ExoCAM', read_clero('ExoCAM'))]
panels = [[reader(c, kind) for c in CASES] for _, reader in rows]
row_titles = [name for name, _ in rows]

# ---- Figure --------------------------------------------------------
if WV:
    contourmin, contourmax = 1e-3, 1e4
    levels = np.logspace(np.log10(contourmin), np.log10(contourmax), 25)
    norm   = colors.LogNorm(vmin=contourmin, vmax=contourmax)
    cm     = cmocean.cm.rain
    cticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    cb_label, mean_fmt = 'Water Vapor Column (kg m$^{-2}$)', lambda x: (f'{x:.0f}' if x >= 1000 else f'{x:.3g}') + ' kg m$^{-2}$'
else:
    contourmin, contourmax = 175.0, 400.0
    levels = np.linspace(contourmin, contourmax, 21)
    norm   = None
    cm     = cmocean.cm.thermal
    cticks = np.arange(200, contourmax + 1, 50)
    cb_label, mean_fmt = 'Surface Temperature (K)', lambda x: f'{x:.1f} K'

case_labels = ['Case 1\n500 W/m², 0.70 bar',
               'Case 4\n1200 W/m², 2.34 bar',
               'Case 16\n1400 W/m², 10.00 bar']

TITLE_FS = 12   # case headers and model labels
MEAN_FS  = 10   # per-panel global mean xlabel
CB_FS    = 10   # colorbar ticks; its label is set at TITLE_FS

# The layout of Figures 8-11: models in rows, cases in columns, 2:1 boxes placed
# by hand in inches at about print size, the colorbar on its own axes at right.
# A wider gap separates the UM pair from the ExoCAM pair.
MAP_W, MAP_H     = 2.8, 1.4
COL_GAP, ROW_GAP = 0.12, 0.32
PAIR_GAP         = 0.62
LEFT, TOP, BOT   = 1.5, 0.6, 0.35
CB_GAP, CB_W     = 0.2, 0.16
nrows, ncols     = len(panels), len(case_labels)
fig_w = LEFT + ncols * MAP_W + (ncols - 1) * COL_GAP + CB_GAP + CB_W + 1.0
fig_h = TOP + nrows * MAP_H + (nrows - 2) * ROW_GAP + PAIR_GAP + BOT

def row_top(row):
    return TOP + row * (MAP_H + ROW_GAP) + (PAIR_GAP - ROW_GAP if row >= 2 else 0.0)

def rect(x, y, w, h):
    """Figure-fraction rectangle from inches, with y measured down from the top."""
    return [x / fig_w, 1.0 - (y + h) / fig_h, w / fig_w, h / fig_h]

fig = plt.figure(figsize=(fig_w, fig_h))
im = None
for row, (row_panels, title) in enumerate(zip(panels, row_titles)):
    for col, (lon, lat, data, ssp_lon, gmean) in enumerate(row_panels):
        ax = fig.add_axes(rect(LEFT + col * (MAP_W + COL_GAP), row_top(row), MAP_W, MAP_H))
        im = ax.contourf(lon, lat, data, cmap=cm, levels=levels, norm=norm,
                         vmin=None if WV else contourmin, vmax=None if WV else contourmax,
                         extend='both')
        if not WV:
            ax.contour(lon, lat, data, levels=[tfreeze], colors='white', linewidths=1.0)
        ax.plot(ssp_lon, 0, marker='*', color='white', markersize=9,
                markeredgecolor='gray', markeredgewidth=0.5)
        ax.set_xlabel(mean_fmt(gmean), fontsize=MEAN_FS, fontweight='normal', labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(case_labels[col], fontsize=TITLE_FS, linespacing=1.5)
        if col == 0:
            ax.text(-0.04, 0.5, title, transform=ax.transAxes,
                    ha='right', va='center', fontsize=TITLE_FS)

cax = fig.add_axes(rect(LEFT + ncols * MAP_W + (ncols - 1) * COL_GAP + CB_GAP, TOP,
                        CB_W, row_top(nrows - 1) + MAP_H - TOP))
cb = fig.colorbar(im, cax=cax, extend='both', ticks=cticks)
cb.ax.tick_params(labelsize=CB_FS)
cb.ax.get_yaxis().labelpad = 16
cb.set_label(cb_label, rotation=270, fontsize=TITLE_FS)

outname = 'fig_compare_clero_select' + ('_wv' if WV else '')
fig.savefig(f'{outname}.png', bbox_inches='tight')
fig.savefig(f'{outname}.eps', bbox_inches='tight')
