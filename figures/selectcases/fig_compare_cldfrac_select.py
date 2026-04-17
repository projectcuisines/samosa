#
# Comparison for SAMOSA Selected Cases: Cloud Fraction
#
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cmocean


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


def read_pcm_cf(case_n):
    """Read Generic PCM ice cloud fraction (3D), project to 2D with max-overlap."""
    _d = '/models/data/samosa/genericpcm/OHT_off'
    path = f'{_d}/case-{case_n}/SAMOSA_output_file_Generic_PCM_case-{case_n}_OHT_off.nc'
    data, lat, lon = read_nc(path, 'ice_cloud_fraction', lat_var='latitude', lon_var='longitude')
    return np.nanmax(data, axis=0) * 100., lat, lon


# ---- ExoCAM --------------------------------------------------------
_d = '/models/data/samosa/exocam'
cf_exocam1,  lat_exocam, lon_exocam = read_nc(f'{_d}/samosa1.cam.h0.avg.nc',  'CLDTOT', avg_axis=0, scale=100.)
cf_exocam4,           _,          _ = read_nc(f'{_d}/samosa4.cam.h0.avg.nc',  'CLDTOT', avg_axis=0, scale=100.)
cf_exocam16,          _,          _ = read_nc(f'{_d}/samosa16.cam.h0.avg.nc', 'CLDTOT', avg_axis=0, scale=100.)

# ---- ExoPlaSim -----------------------------------------------------
_d = '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day'
cf_plasim1,  lat_plasim, lon_plasim = read_nc(
    f'{_d}/t21_synchronous_0.70pn2_flux500_400.0co2_3000teff_15day.nc',   'clt', avg_axis=0, scale=100.)
cf_plasim4,           _,          _ = read_nc(
    f'{_d}/t21_synchronous_2.34pn2_flux1200_400.0co2_3000teff_15day.nc',  'clt', avg_axis=0, scale=100.)
cf_plasim16,          _,          _ = read_nc(
    f'{_d}/t21_synchronous_10.00pn2_flux1400_400.0co2_3000teff_15day.nc', 'clt', avg_axis=0, scale=100.)

cf_plasim1,  lon_plasim_s = roll_to_180(cf_plasim1,  lon_plasim)
cf_plasim4,            _  = roll_to_180(cf_plasim4,  lon_plasim)
cf_plasim16,           _  = roll_to_180(cf_plasim16, lon_plasim)

# ---- ROCKE-3D ------------------------------------------------------
_d = '/models/data/samosa/rocke3d'
cf_r3d1,  lat_r3d, lon_r3d = read_nc(f'{_d}/rocke_01q.nc', 'pcldt')
cf_r3d4,         _,       _ = read_nc(f'{_d}/rocke_04q.nc', 'pcldt')
cf_r3d16,        _,       _ = read_nc(f'{_d}/rocke_16q.nc', 'pcldt')

cf_r3d1,  lon_r3d_s = roll_to_180(cf_r3d1,  lon_r3d)
cf_r3d4,          _ = roll_to_180(cf_r3d4,  lon_r3d)
cf_r3d16,         _ = roll_to_180(cf_r3d16, lon_r3d)

# Substellar point is at native lon=180; rotate 180° to center it at lon=0
_lon = lon_r3d_s + 180.0
_lon[_lon >= 180.0] -= 360.0
_idx = np.argsort(_lon)
lon_r3d_s = _lon[_idx]
cf_r3d1, cf_r3d4, cf_r3d16 = cf_r3d1[:, _idx], cf_r3d4[:, _idx], cf_r3d16[:, _idx]

# ---- LFRic ---------------------------------------------------------
_d = '/models/data/samosa/lfric'
cf_lfric1, lat_lfric, lon_lfric = read_nc(f'{_d}/lfric_samosa_case01.nc', 'cloud_amount_maxrnd', scale=100.)
cf_lfric4,          _,        _ = read_nc(f'{_d}/lfric_samosa_case04.nc', 'cloud_amount_maxrnd', scale=100.)

cf_lfric1, lon_lfric_s = roll_to_180(cf_lfric1, lon_lfric)
cf_lfric4,           _ = roll_to_180(cf_lfric4, lon_lfric)

# ---- Generic PCM ---------------------------------------------------
cf_pcm1,  lat_pcm, lon_pcm = read_pcm_cf(1)
cf_pcm4,         _,       _ = read_pcm_cf(4)
cf_pcm16,        _,       _ = read_pcm_cf(16)

# ---- Figure --------------------------------------------------------
contourmin, contourmax, numcontours = 0.0, 100.0, 21
levels = np.linspace(contourmin, contourmax, numcontours)
cm     = cmocean.cm.ice

# panels[col][row] = (lon, lat, data) or None for unavailable cases; col = model, row = case
col_titles  = ['ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic']
case_labels = ['Case 1\n500 W/m²\n0.70 bar',
               'Case 4\n1200 W/m²\n2.34 bar',
               'Case 16\n1400 W/m²\n10.00 bar']

panels = [
    # ExoPlaSim
    [(lon_plasim_s, lat_plasim,  cf_plasim1),  (lon_plasim_s, lat_plasim,  cf_plasim4),  (lon_plasim_s, lat_plasim,  cf_plasim16) ],
    # ExoCAM (substellar point at lon=180)
    [(lon_exocam,   lat_exocam,  cf_exocam1,  180.), (lon_exocam,   lat_exocam,  cf_exocam4,  180.), (lon_exocam,   lat_exocam,  cf_exocam16, 180.) ],
    # ROCKE-3D
    [(lon_r3d_s,    lat_r3d,     cf_r3d1),     (lon_r3d_s,    lat_r3d,     cf_r3d4),     (lon_r3d_s,    lat_r3d,     cf_r3d16)    ],
    # Generic PCM (no OHT)
    [(lon_pcm,      lat_pcm,     cf_pcm1),     (lon_pcm,      lat_pcm,     cf_pcm4),     (lon_pcm,      lat_pcm,     cf_pcm16)    ],
    # LFRic (case 16 not available)
    [(lon_lfric_s,  lat_lfric,   cf_lfric1),   (lon_lfric_s,  lat_lfric,   cf_lfric4),   None                                     ],
]

TITLE_FS = 11
LABEL_FS = 9
MEAN_FS  = 8
NA_FS    = 10
CB_FS    = 9

fig = plt.figure(layout='constrained', figsize=(13, 5))
fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72, wspace=0.03, hspace=0.08)
ax_array = fig.subplots(3, 5, squeeze=False)

im = None
for col, (col_panels, title) in enumerate(zip(panels, col_titles)):
    for row, panel in enumerate(col_panels):
        ax = ax_array[row, col]
        if panel is None:
            ax.set_facecolor('#cccccc')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=NA_FS, color='#555555')
        else:
            lon, lat, data = panel[:3]
            ssp_lon = panel[3] if len(panel) > 3 else 0.0
            im = ax.contourf(lon, lat, data, cmap=cm,
                             vmin=contourmin, vmax=contourmax,
                             levels=levels, extend='neither')
            ax.plot(ssp_lon, 0, marker='*', color='white', markersize=6,
                    markeredgecolor='gray', markeredgewidth=0.5)
            weights = np.cos(np.radians(lat))
            cf_mean = np.average(np.mean(data, axis=1), weights=weights)
            ax.set_xlabel(f'{cf_mean:.1f}%', fontsize=MEAN_FS)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(title, fontsize=TITLE_FS)
        if col == 0:
            ax.text(-0.12, 0.5, case_labels[row], transform=ax.transAxes,
                    ha='right', va='center', fontsize=LABEL_FS, linespacing=1.5)

cb = fig.colorbar(im, ax=ax_array, extend='neither',
                  ticks=np.arange(contourmin, contourmax + 1, 20),
                  orientation='horizontal', shrink=0.6, pad=0.04)
cb.ax.tick_params(labelsize=CB_FS)
cb.set_label('Cloud Fraction (%)', fontsize=CB_FS)

fig.savefig('fig_compare_cldfrac_select.png', bbox_inches='tight')
fig.savefig('fig_compare_cldfrac_select.eps', bbox_inches='tight')
