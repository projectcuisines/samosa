#
# Comparison for SAMOSA Selected Cases: Surface Temperature
#
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cmocean

tfreeze = 273.16


def roll_to_180(data, lon):
    """Roll a (lat, lon) field and its lon vector from [0, 360) to [-180, 180)."""
    idx = np.searchsorted(lon, 180.0)
    lon_out  = np.concatenate([lon[idx:] - 360.0, lon[:idx]])
    data_out = np.concatenate([data[..., idx:], data[..., :idx]], axis=-1)
    return data_out, lon_out


def read_nc(path, var, lat_var='lat', lon_var='lon', avg_axis=None, offset=0.0):
    """Read a variable from NetCDF, optionally time-averaging along avg_axis."""
    with netCDF4.Dataset(path) as ds:
        raw  = ds.variables[var]
        data = (np.average(raw, axis=avg_axis) if avg_axis is not None else np.array(raw)) + offset
        lat  = np.array(ds.variables[lat_var])
        lon  = np.array(ds.variables[lon_var])
    return data, lat, lon


def read_plahab(path, lon_grid):
    """Parse a PlaHab plain-text surface temperature file into a (lat, lon) array."""
    return np.loadtxt(path, usecols=range(1, lon_grid.size + 1))


# ---- ExoCAM --------------------------------------------------------
_d = '/models/data/samosa/exocam'
Ts_exocam1,  lat_exocam, lon_exocam = read_nc(f'{_d}/samosa1.cam.h0.avg.nc',  'TS', avg_axis=0)
Ts_exocam4,           _,          _ = read_nc(f'{_d}/samosa4.cam.h0.avg.nc',  'TS', avg_axis=0)
Ts_exocam16,          _,          _ = read_nc(f'{_d}/samosa16.cam.h0.avg.nc', 'TS', avg_axis=0)

# ---- ExoPlaSim -----------------------------------------------------
_d = '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day'
Ts_plasim1,  lat_plasim, lon_plasim = read_nc(
    f'{_d}/t21_synchronous_0.70pn2_flux500_400.0co2_3000teff_15day.nc',   'ts', avg_axis=0)
Ts_plasim4,           _,          _ = read_nc(
    f'{_d}/t21_synchronous_2.34pn2_flux1200_400.0co2_3000teff_15day.nc',  'ts', avg_axis=0)
Ts_plasim16,          _,          _ = read_nc(
    f'{_d}/t21_synchronous_10.00pn2_flux1400_400.0co2_3000teff_15day.nc', 'ts', avg_axis=0)

Ts_plasim1,  lon_plasim_s = roll_to_180(Ts_plasim1,  lon_plasim)
Ts_plasim4,            _  = roll_to_180(Ts_plasim4,  lon_plasim)
Ts_plasim16,           _  = roll_to_180(Ts_plasim16, lon_plasim)

# ---- ROCKE-3D ------------------------------------------------------
_d = '/models/data/samosa/rocke3d'
Ts_r3d1,  lat_r3d, lon_r3d = read_nc(f'{_d}/rocke_01q.nc', 'tsurf', offset=tfreeze)
Ts_r3d4,         _,       _ = read_nc(f'{_d}/rocke_04q.nc', 'tsurf', offset=tfreeze)
Ts_r3d16,        _,       _ = read_nc(f'{_d}/rocke_16q.nc', 'tsurf', offset=tfreeze)

Ts_r3d1,  lon_r3d_s = roll_to_180(Ts_r3d1,  lon_r3d)
Ts_r3d4,          _ = roll_to_180(Ts_r3d4,  lon_r3d)
Ts_r3d16,         _ = roll_to_180(Ts_r3d16, lon_r3d)

# Substellar point is at native lon=180; rotate 180° to center it at lon=0
_lon = lon_r3d_s + 180.0
_lon[_lon >= 180.0] -= 360.0
_idx = np.argsort(_lon)
lon_r3d_s = _lon[_idx]
Ts_r3d1, Ts_r3d4, Ts_r3d16 = Ts_r3d1[:, _idx], Ts_r3d4[:, _idx], Ts_r3d16[:, _idx]

# ---- PlaHab --------------------------------------------------------
lon_plahab = np.array([-171., -153., -135., -117.,  -99.,  -81.,  -63.,  -45.,  -27.,  -9.,
                           9.,   27.,   45.,   63.,   81.,   99.,  117.,  135.,  153.,  171.])
lat_plahab = np.array([ -88.,  -82.,  -77.,  -72.,  -68.,  -62.,  -58.,  -52.,  -47.,  -43.,
                         -37.,  -32.,  -28.,  -23.,  -18.,  -13.,   -7.,   -2.,    2.,    7.,
                          13.,   18.,   23.,   28.,   32.,   37.,   43.,   47.,   52.,   58.,
                          62.,   68.,   72.,   77.,   82.,   88.])

_d = '/models/data/samosa/plahab/simulations'
Ts_plahab1  = read_plahab(f'{_d}/sample1/case1_tsurf.out',   lon_plahab)
Ts_plahab4  = read_plahab(f'{_d}/sample4/case4_tsurf.out',   lon_plahab)
Ts_plahab16 = read_plahab(f'{_d}/sample16/case16_tsurf.out', lon_plahab)

# ---- LFRic ---------------------------------------------------------
_d = '/models/data/samosa/lfric'
Ts_lfric1, lat_lfric, lon_lfric = read_nc(f'{_d}/lfric_samosa_case01.nc', 'grid_surface_temperature')
Ts_lfric4,          _,        _ = read_nc(f'{_d}/lfric_samosa_case04.nc', 'grid_surface_temperature')

Ts_lfric1, lon_lfric_s = roll_to_180(Ts_lfric1, lon_lfric)
Ts_lfric4,           _ = roll_to_180(Ts_lfric4, lon_lfric)

# ---- Generic PCM (no OHT) -----------------------------------------
_d = '/models/data/samosa/genericpcm/OHT_off'
Ts_pcm1,  lat_pcm, lon_pcm = read_nc(
    f'{_d}/case-1/SAMOSA_output_file_Generic_PCM_case-1_OHT_off.nc',
    'surface_temperature', lat_var='latitude', lon_var='longitude')
Ts_pcm4,         _,       _ = read_nc(
    f'{_d}/case-4/SAMOSA_output_file_Generic_PCM_case-4_OHT_off.nc',
    'surface_temperature', lat_var='latitude', lon_var='longitude')
Ts_pcm16,        _,       _ = read_nc(
    f'{_d}/case-16/SAMOSA_output_file_Generic_PCM_case-16_OHT_off.nc',
    'surface_temperature', lat_var='latitude', lon_var='longitude')

# ---- Figure --------------------------------------------------------
contourmin, contourmax, numcontours = 175.0, 400.0, 21
levels = np.linspace(contourmin, contourmax, numcontours)
cm     = cmocean.cm.thermal

# panels[col][row] = (lon, lat, data) or None for unavailable cases; col = model, row = case
col_titles  = ['ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic', 'PlaHab']
case_labels = ['Case 1\n500 W/m²\n0.70 bar',
               'Case 4\n1200 W/m²\n2.34 bar',
               'Case 16\n1400 W/m²\n10.00 bar']

panels = [
    # ExoPlaSim
    [(lon_plasim_s, lat_plasim,  Ts_plasim1),  (lon_plasim_s, lat_plasim,  Ts_plasim4),  (lon_plasim_s, lat_plasim,  Ts_plasim16) ],
    # ExoCAM (substellar point at lon=180)
    [(lon_exocam,   lat_exocam,  Ts_exocam1,  180.), (lon_exocam,   lat_exocam,  Ts_exocam4,  180.), (lon_exocam,   lat_exocam,  Ts_exocam16, 180.) ],
    # ROCKE-3D
    [(lon_r3d_s,    lat_r3d,     Ts_r3d1),     (lon_r3d_s,    lat_r3d,     Ts_r3d4),     (lon_r3d_s,    lat_r3d,     Ts_r3d16)    ],
    # Generic PCM (no OHT)
    [(lon_pcm,      lat_pcm,     Ts_pcm1),     (lon_pcm,      lat_pcm,     Ts_pcm4),     (lon_pcm,      lat_pcm,     Ts_pcm16)    ],
    # LFRic (case 16 not available)
    [(lon_lfric_s,  lat_lfric,   Ts_lfric1),   (lon_lfric_s,  lat_lfric,   Ts_lfric4),   None                                     ],
    # PlaHab
    [(lon_plahab,   lat_plahab,  Ts_plahab1),  (lon_plahab,   lat_plahab,  Ts_plahab4),  (lon_plahab,   lat_plahab,  Ts_plahab16) ],
]

fig = plt.figure(layout='constrained', figsize=(15, 5))
ax_array = fig.subplots(3, 6, squeeze=False)

im = None
for col, (col_panels, title) in enumerate(zip(panels, col_titles)):
    for row, panel in enumerate(col_panels):
        ax = ax_array[row, col]
        if panel is None:
            ax.set_facecolor('#cccccc')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='#555555')
        else:
            lon, lat, data = panel[:3]
            ssp_lon = panel[3] if len(panel) > 3 else 0.0
            im = ax.contourf(lon, lat, data, cmap=cm,
                             vmin=contourmin, vmax=contourmax,
                             levels=levels, extend='both')
            ax.contour(lon, lat, data, levels=[tfreeze],
                       colors='white', linewidths=0.8)
            ax.plot(ssp_lon, 0, marker='*', color='white', markersize=6,
                    markeredgecolor='gray', markeredgewidth=0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(title, fontsize=13)
        if col == 0:
            ax.text(-0.12, 0.5, case_labels[row], transform=ax.transAxes,
                    ha='right', va='center', fontsize=10, linespacing=1.5)

fig.colorbar(im, ax=ax_array, extend='both',
             ticks=np.arange(200, contourmax + 1, 50),
             shrink=0.8, pad=0.01, label='Surface Temperature (K)')

fig.savefig('fig_compare_temp_select.png', bbox_inches='tight')
fig.savefig('fig_compare_temp_select.eps', bbox_inches='tight')
