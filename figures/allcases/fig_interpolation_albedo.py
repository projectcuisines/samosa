import numpy as np
import matplotlib.pyplot as plt
import cmocean

from pykrige.ok import OrdinaryKriging

# ─── Variable configuration ──────────────────────────────────────────────────
cm              = cmocean.cm.ice
contourmin      = 10.0
contourmax      = 45.0
cinterval       = 36
sigma_threshold = 1.0       # logit-units; hatch where kriging σ exceeds this
cbar_label      = 'Planetary Albedo (%)'
cbar_ticks      = np.arange( 10, 46, 5 )
# ─────────────────────────────────────────────────────────────────────────────

runaway   = 200.0   # sentinel (%) for runaway/unavailable cases
fluxscale = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Planetary albedo (%), from the standardized SAMOSA global output.
#   ROCKE-3D  plan_alb_hemis[2] as reported by the modeling group
#   ExoPlaSim rsut / ( rst + rsut ) from the area-weighted TOA fluxes
#   ExoCAM    1 - FSNT / ( S / 4 ), gw-weighted from samosaN.cam.h0.avg.nc
#   others    1 - ASR / ( S / 4 ), using the incident flux fixed by the protocol
# Cross-checks against the primary NetCDF:
#   ROCKE-3D   reported plan_alb 23.08% vs 1 - ASR/(S/4) = 23.08% at Case 1.
#   LFRic      sw_net_toa / lw_up_toa reproduce the .txt global diagnostics
#              exactly, so the derived albedos here are confirmed.
#   ExoCAM     FSNTOA and every clear-sky field are archived as identically
#              zero in the submitted files, so the summary TOAALB cannot be
#              reproduced from the primary output and FSNT (top of model) is
#              the only usable shortwave flux. We derive ExoCAM from the NetCDF
#              like every other model rather than mixing sources; this runs
#              0.3 pp above TOAALB on average and 0.9 pp at Case 12. The
#              gw-weighted TS reproduces the summary TS exactly for all 11
#              files, which validates the weighting.
plasim  = np.array( [ 40.77, 25.00, 21.85, 39.99, 31.52, 18.87, 29.82, 40.23, 33.58, 42.23, 38.32, 17.26, 31.10, 36.40, 39.34, 32.58 ] )
exocam  = np.array( [ 27.31, runaway, runaway, 31.08, runaway, runaway, runaway, 20.74, 34.54, 31.78, 22.69, 19.05, runaway, 28.91, 20.38, 16.95 ] )
rocke3d = np.array( [ 23.08, runaway, runaway, 31.56, 39.86, runaway, 44.31, 22.14, 37.63, 21.21, 28.72, 17.96, 40.84, 30.09, 22.26, 12.78 ] )
plahab  = np.array( [ 19.11, runaway, runaway, 33.03, 33.02, runaway, 34.90, 29.62, 30.82, 0.89, 63.70, 35.51, 34.85, 31.12, 19.93, 35.54 ] )
pcm     = np.array( [ 25.57, 14.42, 22.69, 17.72, 28.53, 19.54, 21.31 ] )

pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 26.03, 35.74, 34.16, 23.22, 34.41, 28.25, 21.35 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43, 10.00 ] )

# HEXTOR, cases 1, 4, 8, 9, 11, 14, 15. Clear-sky by construction: HEXTOR has no
# clouds, so these are surface-plus-Rayleigh albedos and sit well below the rest
# of the ensemble wherever the surface is ice-free.
hextor       = np.array( [ 20.18, 2.24, 14.54, 2.21, 11.36, 7.42, 19.62 ] )
hextor_flux1 = np.array( [ 500, 1200, 800, 1100, 900, 900, 600 ] ) / fluxscale
hextor_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 0.10, 1.44, 0.43 ] )

# ExoColumn, cases 1, 4, 8, 9, 10, 11, 14, 15. Cloud-free, but its fixed surface
# albedo of 0.2736 stands in for the shortwave effect of clouds, so unlike
# HEXTOR its albedo lands inside the range spanned by the GCMs.
exocolumn       = np.array( [ 26.01, 15.51, 23.11, 19.68, 27.26, 23.95, 21.97, 25.53 ] )
exocolumn_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600 ] ) / fluxscale
exocolumn_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43 ] )

exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway
plahab_mask  = plahab  != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

# Kriging anisotropy, fitted per model by leave-one-out cross-validation in
# fit_anisotropy.py. pykrige scales the second coordinate, which here is
# normalized instellation, so a value of s means one unit of normalized
# instellation counts s times a unit of normalized log-pressure. Isotropic
# kriging (s = 1) asserts the two axes are equally informative, which is false
# for albedo: rerun fit_anisotropy.py after any resubmission.
# LFRic is pinned at 1 for the same reason: its albedo fit resolves no
# surface at any ratio, so anisotropy would only flatten it further.
ANISO = {
    'ExoCAM':       15,
    'ROCKE-3D':     1.5,
    'ExoPlaSim':    1.5,
    'Generic PCM':  5,
    'PlaHab':       4,
    'LFRic':        1,
    'HEXTOR':       15,
    'ExoColumn':    5,
}

# Normalize both axes to [0, 1] for kriging so distance metric is balanced
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

def logit( x ):
    x = np.clip( x, 1.0, 99.0 )
    return np.log( x / ( 100.0 - x ) )

def sigmoid( y ):
    return 100.0 / ( 1.0 + np.exp( -y ) )

# A linear variogram whose fitted slope is zero is a pure nugget: ordinary
# kriging then weights every sample point equally regardless of distance, so
# the interpolated surface collapses to the sample mean and carries no spatial
# information. Those panels are stippled to distinguish that case from a
# genuinely flat but resolved field.
slope_eps = 1.0e-8

# Two rows of four, with the colorbar alongside rather than occupying a panel
# slot. Panels are ordered by model class, ending with the two one-dimensional
# models.
fig, axd = plt.subplot_mosaic( [[ 'P1', 'P2', 'P3', 'P4' ],
                                 [ 'P5', 'P6', 'P7', 'P8' ]],
                               figsize=(22, 9) )

#--------------------------------------------------------------------
# ExoCAM Kriging

OK = OrdinaryKriging(
    norm_pres( exocam_pres1 ),
    norm_flux( exocam_flux1 ),
    logit( exocam_stable ),
    anisotropy_scaling=ANISO[ 'ExoCAM' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

z1, z1_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
slope_exocam = OK.variogram_model_parameters[ 0 ]
xv, yv = np.meshgrid( pn2, flux )

#--------------------------------------------------------------------
# ROCKE-3D Kriging

OK = OrdinaryKriging(
    norm_pres( rocke3d_pres1 ),
    norm_flux( rocke3d_flux1 ),
    logit( rocke3d_stable ),
    anisotropy_scaling=ANISO[ 'ROCKE-3D' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

R3_z1, R3_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
slope_rocke3d = OK.variogram_model_parameters[ 0 ]

#--------------------------------------------------------------------
# ExoPlaSim Kriging

OK = OrdinaryKriging(
    norm_pres( pres1 ),
    norm_flux( flux1 ),
    logit( plasim ),
    anisotropy_scaling=ANISO[ 'ExoPlaSim' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

PlaSim_z1, PlaSim_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
slope_plasim = OK.variogram_model_parameters[ 0 ]

#--------------------------------------------------------------------
# Generic PCM Kriging

OK = OrdinaryKriging(
    norm_pres( pcm_pres1 ),
    norm_flux( pcm_flux1 ),
    logit( pcm ),
    anisotropy_scaling=ANISO[ 'Generic PCM' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

pcm_z1, pcm_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
slope_pcm = OK.variogram_model_parameters[ 0 ]

#--------------------------------------------------------------------
# LFRic Kriging

OK = OrdinaryKriging(
    norm_pres( lfric_pres1 ),
    norm_flux( lfric_flux1 ),
    logit( lfric ),
    anisotropy_scaling=ANISO[ 'LFRic' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

lfric_z1, lfric_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
slope_lfric = OK.variogram_model_parameters[ 0 ]

#--------------------------------------------------------------------
# PlaHab Kriging

OK = OrdinaryKriging(
    norm_pres( plahab_pres1 ),
    norm_flux( plahab_flux1 ),
    logit( plahab_stable ),
    anisotropy_scaling=ANISO[ 'PlaHab' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

PlaHab_z1, PlaHab_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# HEXTOR Kriging

OK = OrdinaryKriging(
    norm_pres( hextor_pres1 ),
    norm_flux( hextor_flux1 ),
    logit( hextor ),
    anisotropy_scaling=ANISO[ 'HEXTOR' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

hextor_z1, hextor_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# ExoColumn Kriging

OK = OrdinaryKriging(
    norm_pres( exocolumn_pres1 ),
    norm_flux( exocolumn_flux1 ),
    logit( exocolumn ),
    anisotropy_scaling=ANISO[ 'ExoColumn' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

exocolumn_z1, exocolumn_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
slope_plahab = OK.variogram_model_parameters[ 0 ]

# Shared axis limits
xlim = [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ]
ylim = [ min( pn2 )*0.9, max( pn2 )*1.1 ]
contour_levels = np.linspace( contourmin, contourmax, cinterval )
marker_edge = 'k'

def flag_degenerate( ax, slope ):
    """Stipple a panel whose variogram fit collapsed to a pure nugget."""
    if slope > slope_eps:
        return
    ax.contourf( yv*fluxscale, xv, np.ones_like( xv ), levels=[0.5, 1.5],
                 hatches=['....'], colors='none', alpha=0 )
    ax.text( 0.5, 0.06, 'no resolvable spatial structure', transform=ax.transAxes,
             ha='center', va='bottom', fontsize=10, style='italic', color='0.15',
             bbox=dict( facecolor='white', edgecolor='none', alpha=0.75, pad=2.0 ) )

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

cf1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, sigmoid(PlaSim_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P1' ].contourf( yv*fluxscale, xv, np.sqrt(PlaSim_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P1' ], f'ExoPlaSim (n={len(plasim)})' )
flag_degenerate( axd[ 'P1' ], slope_plasim )

#--------------------------------------------------------------------
# Panel 2

cf2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, sigmoid(z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P2' ].contourf( yv*fluxscale, xv, np.sqrt(z1_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P2' ].scatter( exocam_flux1*fluxscale, exocam_pres1, c=exocam_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P2' ], f'ExoCAM (n={len(exocam_stable)})' )
flag_degenerate( axd[ 'P2' ], slope_exocam )

#--------------------------------------------------------------------
# Panel 3

cf3 = axd[ 'P3' ].contourf( yv*fluxscale, xv, sigmoid(R3_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P3' ].contourf( yv*fluxscale, xv, np.sqrt(R3_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P3' ].scatter( rocke3d_flux1*fluxscale, rocke3d_pres1, c=rocke3d_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P3' ], f'ROCKE-3D (n={len(rocke3d_stable)})' )
flag_degenerate( axd[ 'P3' ], slope_rocke3d )

#--------------------------------------------------------------------
# Panel 4

cf4 = axd[ 'P4' ].contourf( yv*fluxscale, xv, sigmoid(pcm_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P4' ].contourf( yv*fluxscale, xv, np.sqrt(pcm_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P4' ].scatter( pcm_flux1*fluxscale, pcm_pres1, c=pcm, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P4' ], f'Generic PCM (n={len(pcm)})' )
flag_degenerate( axd[ 'P4' ], slope_pcm )

#--------------------------------------------------------------------
# Panel 5

cf5 = axd[ 'P5' ].contourf( yv*fluxscale, xv, sigmoid(lfric_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P5' ].contourf( yv*fluxscale, xv, np.sqrt(lfric_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P5' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P5' ], f'LFRic (n={len(lfric)})' )
flag_degenerate( axd[ 'P5' ], slope_lfric )

#--------------------------------------------------------------------
# Panel 6

cf6 = axd[ 'P6' ].contourf( yv*fluxscale, xv, sigmoid(PlaHab_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P6' ].contourf( yv*fluxscale, xv, np.sqrt(PlaHab_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P6' ].scatter( plahab_flux1*fluxscale, plahab_pres1, c=plahab_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P6' ], f'PlaHab (n={len(plahab_stable)})' )
flag_degenerate( axd[ 'P6' ], slope_plahab )

#--------------------------------------------------------------------
# Panel 7

cf7 = axd[ 'P7' ].contourf( yv*fluxscale, xv, sigmoid(hextor_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P7' ].contourf( yv*fluxscale, xv, np.sqrt(hextor_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P7' ].scatter( hextor_flux1*fluxscale, hextor_pres1, c=hextor, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P7' ], f'HEXTOR (n={len(hextor)})' )

#--------------------------------------------------------------------
# Panel 8

cf8 = axd[ 'P8' ].contourf( yv*fluxscale, xv, sigmoid(exocolumn_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='both' )
axd[ 'P8' ].contourf( yv*fluxscale, xv, np.sqrt(exocolumn_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P8' ].scatter( exocolumn_flux1*fluxscale, exocolumn_pres1, c=exocolumn, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P8' ], f'ExoColumn (n={len(exocolumn)})' )

#--------------------------------------------------------------------
# Finalize

fig.subplots_adjust( wspace=0.3, hspace=0.4, right=0.88 )
cax = fig.add_axes( [ 0.905, 0.12, 0.013, 0.76 ] )
cb = fig.colorbar( cf1, cax=cax, extend='both', ticks=cbar_ticks )
cb.ax.tick_params( labelsize=11 )
cb.ax.get_yaxis().labelpad = 15
cb.set_label( cbar_label, rotation=270, fontsize=12 )

fig.savefig( "fig_interpolation_albedo.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_albedo.eps", bbox_inches='tight' )
