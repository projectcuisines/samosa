import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean

from pykrige.ok import OrdinaryKriging

# ─── Variable configuration ──────────────────────────────────────────────────
cm              = cmocean.cm.rain
contourmin      = 1.e-3
contourmax      = 1.e3
cinterval       = 40
sigma_threshold = 3.5       # log-units; hatch where kriging σ exceeds this
cbar_label      = 'Average Water Vapor Column (kg m$^{-2}$)'
cbar_ticks      = [ 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3 ]
# ─────────────────────────────────────────────────────────────────────────────

_s        = 0.1     # unit conversion: raw model output → kg m⁻²
runaway   = 1.e4    # sentinel (kg m⁻²) for runaway/unavailable cases
fluxscale = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Water Vapor Column (kg m⁻²)
plasim  = _s * np.array( [ 0.059, 5113.363, 271.733, 7.509, 37.305, 1452.570, 76.876, 0.351, 7.048, 0.007, 3.042, 1258.514, 59.214, 1.955, 0.416, 1111.793 ] )
exocam  = _s * np.array( [ 0.2335, runaway/_s, runaway/_s, 19.2138, runaway/_s, runaway/_s, runaway/_s, 1.7490, 7.9707, 0.0102, 5.8914, 1430.7613, runaway/_s, 4.2041, 0.9534, 1295.6428 ] )
rocke3d = _s * np.array( [ 0.25980374, runaway/_s, runaway/_s, 14.9693165, 31.839989, runaway/_s, 32.45563, 1.5794185, 5.687923, 0.031635746, 4.2059116, 271.75916, 46.070984, 2.964403, 0.52949935, 132.49808 ] )
# PlaHab: no water vapor data (2D model)
pcm       = _s * np.array( [ 0.37484651163423993, 47.86514350558743, 2.1906175203429563, 25.650192288432617, 0.06080825624148188, 5.93210602524276, 0.8447688576786204 ] )
pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 0.41, 8.18, 7.37, 829.15, 2.34, 0.86 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]


# Normalize both axes to [0, 1] for kriging so distance metric is balanced
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

fig, axd = plt.subplot_mosaic( [[ 'P1', 'P2', 'P3' ],
                                 [ 'P4', 'P5', 'P6' ]],
                               figsize=(18, 9) )

norm           = mcolors.LogNorm( vmin=contourmin, vmax=contourmax )
contour_levels = np.logspace( np.log10(contourmin), np.log10(contourmax), cinterval )

#--------------------------------------------------------------------
# ExoCAM Kriging

OK = OrdinaryKriging(
    norm_pres( pres1 ),
    norm_flux( flux1 ),
    np.log( exocam ),
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

z1, z1_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
xv, yv = np.meshgrid( pn2, flux )

#--------------------------------------------------------------------
# ROCKE-3D Kriging

OK = OrdinaryKriging(
    norm_pres( pres1 ),
    norm_flux( flux1 ),
    np.log( rocke3d ),
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

R3_z1, R3_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# ExoPlaSim Kriging

OK = OrdinaryKriging(
    norm_pres( pres1 ),
    norm_flux( flux1 ),
    np.log( plasim ),
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

PlaSim_z1, PlaSim_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# Generic PCM Kriging

OK = OrdinaryKriging(
    norm_pres( pcm_pres1 ),
    norm_flux( pcm_flux1 ),
    np.log( pcm ),
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

pcm_z1, pcm_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# LFRic Kriging

OK = OrdinaryKriging(
    norm_pres( lfric_pres1 ),
    norm_flux( lfric_flux1 ),
    np.log( lfric ),
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

lfric_z1, lfric_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

# Shared axis limits
xlim = [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ]
ylim = [ min( pn2 )*0.9, max( pn2 )*1.1 ]
marker_edge = 'k'

def setup_panel( ax, title ):
    ax.set_title( title, fontsize=14 )
    ax.set_xlabel( 'Instellation (W m$^{-2}$)', fontsize=12 )
    ax.set_ylabel( 'Surface pressure (bar)', fontsize=12 )
    ax.tick_params( axis='x', labelsize=11 )
    ax.tick_params( axis='y', labelsize=11 )
    ax.set_yscale( 'log' )
    ax.set_xlim( xlim )
    ax.set_ylim( ylim )

#--------------------------------------------------------------------
# Panel 1

cf1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, np.exp(PlaSim_z1), cmap=cm, levels=contour_levels, norm=norm, extend='both' )
axd[ 'P1' ].contourf( yv*fluxscale, xv, np.sqrt(PlaSim_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, norm=norm, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P1' ], f'ExoPlaSim (n={len(plasim)})' )

#--------------------------------------------------------------------
# Panel 2

cf2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, np.exp(z1), cmap=cm, levels=contour_levels, norm=norm, extend='both' )
axd[ 'P2' ].contourf( yv*fluxscale, xv, np.sqrt(z1_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P2' ].scatter( exocam_flux1*fluxscale, exocam_pres1, c=exocam_stable, cmap=cm, norm=norm, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P2' ], f'ExoCAM (n={len(exocam_stable)})' )

#--------------------------------------------------------------------
# Panel 3

cf3 = axd[ 'P3' ].contourf( yv*fluxscale, xv, np.exp(R3_z1), cmap=cm, levels=contour_levels, norm=norm, extend='both' )
axd[ 'P3' ].contourf( yv*fluxscale, xv, np.sqrt(R3_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P3' ].scatter( rocke3d_flux1*fluxscale, rocke3d_pres1, c=rocke3d_stable, cmap=cm, norm=norm, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P3' ], f'ROCKE-3D (n={len(rocke3d_stable)})' )

#--------------------------------------------------------------------
# Panel 4

cf4 = axd[ 'P4' ].contourf( yv*fluxscale, xv, np.exp(pcm_z1), cmap=cm, levels=contour_levels, norm=norm, extend='both' )
axd[ 'P4' ].contourf( yv*fluxscale, xv, np.sqrt(pcm_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P4' ].scatter( pcm_flux1*fluxscale, pcm_pres1, c=pcm, cmap=cm, norm=norm, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P4' ], f'Generic PCM (n={len(pcm)})' )

#--------------------------------------------------------------------
# Panel 5

cf5 = axd[ 'P5' ].contourf( yv*fluxscale, xv, np.exp(lfric_z1), cmap=cm, levels=contour_levels, norm=norm, extend='both' )
axd[ 'P5' ].contourf( yv*fluxscale, xv, np.sqrt(lfric_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P5' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, norm=norm, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P5' ], f'LFRic (n={len(lfric)})' )

#--------------------------------------------------------------------
# Panel 6 — PlaHab (no water vapor data)

axd[ 'P6' ].set_axis_off()
axd[ 'P6' ].set_title( 'PlaHab', fontsize=14 )
axd[ 'P6' ].text( 0.5, 0.5, 'No data\n(2D model)', ha='center', va='center',
                  transform=axd[ 'P6' ].transAxes, fontsize=13, style='italic', color='gray' )

#--------------------------------------------------------------------
# Finalize

fig.subplots_adjust( wspace=0.3, hspace=0.4, right=0.88 )
cax = fig.add_axes( [0.91, 0.1, 0.015, 0.8] )
cb = fig.colorbar( cf1, cax=cax, extend='both', ticks=cbar_ticks )
cb.ax.tick_params( labelsize=11 )
cb.ax.get_yaxis().labelpad = 15
cb.set_label( cbar_label, rotation=270, fontsize=12 )

fig.savefig( "fig_interpolation_watvap.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_watvap.eps", bbox_inches='tight' )
#plt.show()
