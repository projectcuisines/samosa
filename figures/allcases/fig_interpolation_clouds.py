import numpy as np
import matplotlib.pyplot as plt
import cmocean

from pykrige.ok import OrdinaryKriging

# ─── Variable configuration ──────────────────────────────────────────────────
cm              = cmocean.cm.ice_r
contourmin      = 0.0
contourmax      = 100.0
cinterval       = 40
sigma_threshold = 1.0       # logit-units; hatch where kriging σ exceeds this
cbar_label      = 'Average Total Cloud Fraction (%)'
cbar_ticks      = np.arange( 0, 101, 20 )
# ─────────────────────────────────────────────────────────────────────────────

runaway   = 200.0   # sentinel (%) for runaway/unavailable cases
fluxscale = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Total Cloud Fraction (%)
# Generic PCM values are the Cldfrac column reported by the modeling group in
# samosa_gcm_output_case-N_OHT_off.dat, not a projection of the 3-D cloud field
# (max-overlap on ice_cloud_fraction gives 43.2% at Case 1 against 25.6% reported).
plasim  = np.array( [ 42.7, 68.2, 70.6, 56.2, 80.1, 32.1, 58.1, 25.7, 58.0, 25.2, 76.2, 30.1, 85.3, 52.7, 48.4, 53.3 ] )
exocam  = np.array( [ 68.75, runaway, runaway, 43.98, runaway, runaway, runaway, 16.34, 75.82, 15.85, 83.08, 56.79, runaway, 34.01, 78.96, 61.40 ] )
rocke3d = np.array( [ 68.20222, runaway, runaway, 51.043224, 81.88261, runaway, 88.81546, 58.24044, 61.535275, 98.8356, 68.16493, 68.01357, 85.71091, 43.637707, 74.40385, 48.08909 ] )
plahab  = np.array( [ 11.11879, runaway, runaway, 35.74597, 48.29323, runaway, 70.91280, 26.24803, 31.64522, 8.4692545, 4.3572873, 76.20874, 57.27629, 28.12309, 16.64636, 72.36285 ] )
pcm     = np.array( [ 25.5674468009485, 27.31228828919005, 16.96672860199983, 25.15276275245855, 24.55454268845772, 16.696470834684884, 32.84789893586739 ] )

pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 31.0, 61.0, 58.0, 81.0, 44.0, 36.0, 83.0 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43, 10.00 ] )

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
# for cloud fraction: rerun fit_anisotropy.py after any resubmission.
# Generic PCM is pinned at 1 because no scaling resolves a surface for it
# here -- its fit is near-degenerate at every ratio, so the least distorted
# metric is kept and the flatness left visible rather than tuned away.
ANISO = {
    'ExoCAM':       1,
    'ROCKE-3D':     3,
    'ExoPlaSim':    1,
    'Generic PCM':  1,
    'PlaHab':       3,
    'LFRic':        15,
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

# Shared axis limits
xlim = [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ]
ylim = [ min( pn2 )*0.9, max( pn2 )*1.1 ]
contour_levels = np.linspace( contourmin, contourmax, cinterval )
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

cf1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, sigmoid(PlaSim_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='neither' )
axd[ 'P1' ].contourf( yv*fluxscale, xv, np.sqrt(PlaSim_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P1' ], f'ExoPlaSim (n={len(plasim)})' )

#--------------------------------------------------------------------
# Panel 2

cf2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, sigmoid(z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='neither' )
axd[ 'P2' ].contourf( yv*fluxscale, xv, np.sqrt(z1_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P2' ].scatter( exocam_flux1*fluxscale, exocam_pres1, c=exocam_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P2' ], f'ExoCAM (n={len(exocam_stable)})' )

#--------------------------------------------------------------------
# Panel 3

cf3 = axd[ 'P3' ].contourf( yv*fluxscale, xv, sigmoid(R3_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='neither' )
axd[ 'P3' ].contourf( yv*fluxscale, xv, np.sqrt(R3_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P3' ].scatter( rocke3d_flux1*fluxscale, rocke3d_pres1, c=rocke3d_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P3' ], f'ROCKE-3D (n={len(rocke3d_stable)})' )

#--------------------------------------------------------------------
# Panel 4

cf4 = axd[ 'P4' ].contourf( yv*fluxscale, xv, sigmoid(pcm_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='neither' )
axd[ 'P4' ].contourf( yv*fluxscale, xv, np.sqrt(pcm_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P4' ].scatter( pcm_flux1*fluxscale, pcm_pres1, c=pcm, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P4' ], f'Generic PCM (n={len(pcm)})' )

#--------------------------------------------------------------------
# Panel 5

cf5 = axd[ 'P5' ].contourf( yv*fluxscale, xv, sigmoid(lfric_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='neither' )
axd[ 'P5' ].contourf( yv*fluxscale, xv, np.sqrt(lfric_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P5' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P5' ], f'LFRic (n={len(lfric)})' )

#--------------------------------------------------------------------
# Panel 6

cf6 = axd[ 'P6' ].contourf( yv*fluxscale, xv, sigmoid(PlaHab_z1), cmap=cm, levels=contour_levels, vmin=contourmin, vmax=contourmax, extend='neither' )
axd[ 'P6' ].contourf( yv*fluxscale, xv, np.sqrt(PlaHab_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P6' ].scatter( plahab_flux1*fluxscale, plahab_pres1, c=plahab_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P6' ], f'PlaHab (n={len(plahab_stable)})' )

#--------------------------------------------------------------------
# Panel 7 — HEXTOR (clear-sky model, no cloud fraction)

axd[ 'P7' ].set_axis_off()
axd[ 'P7' ].set_title( 'HEXTOR', fontsize=14 )
axd[ 'P7' ].text( 0.5, 0.5, 'No data\n(clear-sky model)', ha='center', va='center',
                  transform=axd[ 'P7' ].transAxes, fontsize=13, style='italic', color='gray' )

#--------------------------------------------------------------------
# Panel 8 — ExoColumn (cloud-free by construction)

axd[ 'P8' ].set_axis_off()
axd[ 'P8' ].set_title( 'ExoColumn', fontsize=14 )
axd[ 'P8' ].text( 0.5, 0.5, 'No data\n(cloud-free model)', ha='center', va='center',
                  transform=axd[ 'P8' ].transAxes, fontsize=13, style='italic', color='gray' )

#--------------------------------------------------------------------
# Finalize

fig.subplots_adjust( wspace=0.3, hspace=0.4, right=0.88 )
cax = fig.add_axes( [ 0.905, 0.12, 0.013, 0.76 ] )
cb = fig.colorbar( cf1, cax=cax, extend='neither', ticks=cbar_ticks )
cb.ax.tick_params( labelsize=11 )
cb.ax.get_yaxis().labelpad = 15
cb.set_label( cbar_label, rotation=270, fontsize=12 )

fig.savefig( "fig_interpolation_clouds.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_clouds.eps", bbox_inches='tight' )
#plt.show()
