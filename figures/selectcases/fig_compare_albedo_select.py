#
# Comparison for SAMOSA Selected Cases: Planetary Albedo
#
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cmocean

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelpad'] = 8


def roll_to_180(data, lon):
    """Roll a (lat, lon) field and its lon vector from [0, 360) to [-180, 180)."""
    idx = np.searchsorted(lon, 180.0)
    lon_out  = np.concatenate([lon[idx:] - 360.0, lon[:idx]])
    data_out = np.concatenate([data[..., idx:], data[..., :idx]], axis=-1)
    return data_out, lon_out


def read_nc(path, var, lat_var='lat', lon_var='lon', avg_axis=None, offset=0.0, scale=1.0):
    """Read a variable from NetCDF, optionally time-averaging along avg_axis."""
    with netCDF4.Dataset(path) as ds:
        raw  = ds.variables[var]
        data = (np.average(raw, axis=avg_axis) if avg_axis is not None else np.array(raw))
        data = data * scale + offset
        lat  = np.array(ds.variables[lat_var])
        lon  = np.array(ds.variables[lon_var])
    return data, lat, lon


S_CASE = {1: 500.0, 4: 1200.0, 16: 1400.0}   # protocol instellation, W/m2

# Local albedo is reflected over incident shortwave at the top of the
# atmosphere. On the night side both fluxes are exactly zero in every model,
# so the albedo is undefined there and those cells are left unfilled (gray).
# Every lit cell is drawn, up to the terminator; there the four models that
# report their own incident flux stay within 16-55%, while LFRic, whose
# incident flux is reconstructed (see read_lfric), does not.


def local_albedo(reflected, incident, S):
    """Percent albedo per cell, NaN where the cell is unlit."""
    with np.errstate(divide='ignore', invalid='ignore'):
        alb = 100.0 * reflected / incident
    return np.where(incident > 0.0, alb, np.nan)


def planetary_albedo(reflected, incident, lat):
    """Area-weighted reflected over incident: the planetary albedo of Figure 6."""
    w = np.cos(np.radians(lat))[:, None]
    return 100.0 * np.sum(reflected * w) / np.sum(incident * w)


def panel(lon, lat, reflected, incident, S, ssp=0.0):
    return (lon, lat, local_albedo(reflected, incident, S), ssp,
            planetary_albedo(reflected, incident, lat))


# Each loader returns ( reflected, incident, lat, lon ), both fluxes at the top
# of the atmosphere in W/m2 and time-averaged. Every one reproduces the
# planetary albedo of Figure 6 to within 0.1 percentage points.

def read_exocam(case_n):
    """Resolved shortwave profiles at the topmost interface, the top of the
    model; FSNTOA is archived as zero (see fig_interpolation_albedo.py)."""
    path = f'/models/data/samosa/exocam/samosa{case_n}.cam.h0.avg.nc'
    fus, lat, lon = read_nc(path, 'FUS', avg_axis=0)
    fds, _, _     = read_nc(path, 'FDS', avg_axis=0)
    return fus[0], fds[0], lat, lon


def read_plasim(case_n):
    """rsut is stored negative (upward) and rst is net, so incident = rst - rsut."""
    _d = '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day'
    tag = {1: '0.70pn2_flux500', 4: '2.34pn2_flux1200', 16: '10.00pn2_flux1400'}[case_n]
    path = f'{_d}/t21_synchronous_{tag}_400.0co2_3000teff_15day.nc'
    rsut, lat, lon = read_nc(path, 'rsut', avg_axis=0)
    rst, _, _      = read_nc(path, 'rst',  avg_axis=0)
    refl, lon_s = roll_to_180(-rsut, lon)
    inc, _      = roll_to_180(rst - rsut, lon)
    return refl, inc, lat, lon_s


def read_rocke3d(case_n):
    """Incident and net solar at the TOA; substellar point rotated from native 180 to 0."""
    path = f'/models/data/samosa/rocke3d/rocke_{case_n:02d}q.nc'
    inc, lat, lon = read_nc(path, 'incsw_toa')
    net, _, _     = read_nc(path, 'srnf_toa')
    # Each pole row is a single polar-cap cell repeated across every longitude,
    # lit even where the row beside it is in darkness; keep it only where that
    # row is lit. The rows carry no area weight, so the planetary albedo is
    # unchanged.
    for pole, beside in ((0, 1), (-1, -2)):
        dark = inc[beside] == 0.0
        inc[pole, dark] = 0.0
        net[pole, dark] = 0.0
    refl, lon_s = roll_to_180(inc - net, lon)
    inc, _      = roll_to_180(inc, lon)
    _lon = lon_s + 180.0
    _lon[_lon >= 180.0] -= 360.0
    idx = np.argsort(_lon)
    return refl[:, idx], inc[:, idx], lat, _lon[idx]


def read_lfric(case_n):
    """LFRic submitted only the net TOA shortwave, so the incident flux is the
    protocol instellation on a synchronous planet with the substellar point at
    lon=0: S cos(lat) cos(lon) on the day side. Its area mean is S/4 to 0.01 W/m2
    on this grid, and the net flux on the night side is at most 0.005 W/m2, so
    the planetary albedo is exact. The local albedo is not: near the terminator
    the net flux exceeds this geometric incident flux in some cells, which the
    model's own incident flux, not submitted, would have to resolve. Those cells
    come out below zero, down to about -7000% where the incident flux nearly
    vanishes, and take the lowest color. It affects the band where less than 20%
    of the substellar flux arrives, within about 12 degrees of the terminator."""
    path = f'/models/data/samosa/lfric/lfric_samosa_case{case_n:02d}.nc'
    net, lat, lon = read_nc(path, 'sw_net_toa')
    L, Lo = np.meshgrid(lat, lon, indexing='ij')
    inc = S_CASE[case_n] * np.clip(np.cos(np.radians(L)) * np.cos(np.radians(Lo)), 0.0, None)
    refl, lon_s = roll_to_180(inc - net, lon)
    inc, _      = roll_to_180(inc, lon)
    return refl, inc, lat, lon_s


def read_pcm(case_n):
    """Incoming and absorbed shortwave from the NetCDF, which are sound; only the
    flux columns of the .dat summary are unusable (see the data audit)."""
    _d = '/models/data/samosa/genericpcm/OHT_off'
    path = f'{_d}/case-{case_n}/SAMOSA_output_file_Generic_PCM_case-{case_n}_OHT_off.nc'
    inc, lat, lon = read_nc(path, 'incoming_stellar_radiation', lat_var='latitude', lon_var='longitude')
    asr, _, _     = read_nc(path, 'absorbed_shortwave_radiation', lat_var='latitude', lon_var='longitude')
    return inc - asr, inc, lat, lon


def model_row(reader, cases=(1, 4, 16), ssp=0.0):
    row = []
    for c in (1, 4, 16):
        if c in cases:
            refl, inc, lat, lon = reader(c)
            row.append(panel(lon, lat, refl, inc, S_CASE[c], ssp))
        else:
            row.append(None)
    return row


# ---- Figure --------------------------------------------------------
contourmin, contourmax, numcontours = 0.0, 60.0, 13
levels = np.linspace(contourmin, contourmax, numcontours)
cm     = cmocean.cm.haline   # as in Figure 6
night  = '#e6e6e6'           # unlit cells, where albedo is undefined

# PlaHab submitted maps of surface temperature only, and HEXTOR and ExoColumn
# have no horizontal dimension, so the rows are the five 3-D models, as for
# cloud fraction
row_titles  = ['ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic']
case_labels = ['Case 1\n500 W/m², 0.70 bar',
               'Case 4\n1200 W/m², 2.34 bar',
               'Case 16\n1400 W/m², 10.00 bar']

# panels[row][col] = (lon, lat, local albedo, substellar lon, planetary albedo),
# or None where the case is unavailable
panels = [
    model_row(read_plasim),
    model_row(read_exocam, ssp=180.0),   # substellar point at lon=180
    model_row(read_rocke3d),
    model_row(read_pcm, cases=(1, 4)),   # Case 16 did not converge
    model_row(read_lfric),
]

TITLE_FS = 12   # case headers and model labels
MEAN_FS  = 10   # per-panel global mean xlabel
NA_FS    = 12   # N/A placeholder text
CB_FS    = 10   # colorbar ticks; its label is set at TITLE_FS

# Models in rows and cases in columns, as in Figures 11-13, so each map gets a
# third of the width rather than a sixth. Placed by hand in inches, at about
# print size, with the colorbar on its own axes at right as in Figures 3-7.
# The boxes are 2:1, so each map fills its box rather than taking equal
# aspect: grids that stop at cell centres (PlaHab, ExoCAM) would otherwise
# shrink out of line, and the stretch is at most 3%.
MAP_W, MAP_H     = 2.8, 1.4
COL_GAP, ROW_GAP = 0.12, 0.40
LEFT, TOP, BOT   = 1.2, 0.6, 0.35
CB_GAP, CB_W     = 0.2, 0.16
nrows, ncols     = len(panels), len(case_labels)
fig_w = LEFT + ncols * MAP_W + (ncols - 1) * COL_GAP + CB_GAP + CB_W + 1.0
fig_h = TOP + nrows * MAP_H + (nrows - 1) * ROW_GAP + BOT

def rect(x, y, w, h):
    """Figure-fraction rectangle from inches, with y measured down from the top."""
    return [x / fig_w, 1.0 - (y + h) / fig_h, w / fig_w, h / fig_h]

fig = plt.figure(figsize=(fig_w, fig_h))
ax_array = np.array([[fig.add_axes(rect(LEFT + col * (MAP_W + COL_GAP),
                                        TOP + row * (MAP_H + ROW_GAP), MAP_W, MAP_H))
                      for col in range(ncols)] for row in range(nrows)])

im = None
for row, (row_panels, title) in enumerate(zip(panels, row_titles)):
    for col, panel in enumerate(row_panels):
        ax = ax_array[row, col]
        if panel is None:
            ax.set_facecolor('#cccccc')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=NA_FS, color='#555555')
        else:
            lon, lat, data, ssp_lon, alb_planet = panel
            ax.set_facecolor(night)
            im = ax.contourf(lon, lat, data, cmap=cm,
                             vmin=contourmin, vmax=contourmax,
                             levels=levels, extend='both')
            ax.set_xlim(lon.min(), lon.max())
            ax.set_ylim(-90, 90)
            ax.plot(ssp_lon, 0, marker='*', color='white', markersize=9,
                    markeredgecolor='gray', markeredgewidth=0.5)
            ax.set_xlabel(f'{alb_planet:.1f}%', fontsize=MEAN_FS, fontweight='normal', labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(case_labels[col], fontsize=TITLE_FS, linespacing=1.5)
        if col == 0:
            ax.text(-0.04, 0.5, title, transform=ax.transAxes,
                    ha='right', va='center', fontsize=TITLE_FS)

cax = fig.add_axes(rect(LEFT + ncols * MAP_W + (ncols - 1) * COL_GAP + CB_GAP, TOP,
                        CB_W, nrows * MAP_H + (nrows - 1) * ROW_GAP))
cb = fig.colorbar(im, cax=cax, extend='both',
                  ticks=np.arange(contourmin, contourmax + 1, 20))
cb.ax.tick_params(labelsize=CB_FS)
cb.ax.get_yaxis().labelpad = 16
cb.set_label('Local Albedo (%)', rotation=270, fontsize=TITLE_FS)

fig.savefig('fig_compare_albedo_select.png', bbox_inches='tight')
fig.savefig('fig_compare_albedo_select.eps', bbox_inches='tight')
