import numpy as np
import matplotlib.pyplot as plt
import cmocean

from pykrige.ok import OrdinaryKriging

# ─── Variable configuration ──────────────────────────────────────────────────
cm              = cmocean.cm.thermal
contourmin      = 175.0
contourmax      = 370.0
cinterval       = 40
sigma_threshold = 45.0      # K; hatch where kriging σ exceeds this
cbar_label      = 'Average Surface Temperature (K)'
cbar_ticks      = np.arange( 200, 370, 50 )
# ─────────────────────────────────────────────────────────────────────────────

runawaytemp = 600.0
fluxscale   = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Surface Temperature (K)
plasim  = np.array( [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9, 239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ] )
exocam  = np.array( [ 196.8, runawaytemp, runawaytemp, 260.0, runawaytemp, runawaytemp, runawaytemp, 243.8, 244.8, 194.1, 234.0, 350.9, runawaytemp, 236.8, 211.5, 356.7 ] )
rocke3d = np.array( [ 202.8284, runawaytemp, runawaytemp, 260.1185, 265.88116, runawaytemp, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ] )
plahab  = np.array( [ 196.3, runawaytemp, runawaytemp, 273.2, 281.4, runawaytemp, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )

exocam_mask  = exocam  != runawaytemp
rocke3d_mask = rocke3d != runawaytemp
plahab_mask  = plahab  != runawaytemp

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

pcm = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218, 266.5987224285321, 210.69131033681012, 246.04296230476365, 217.2519558970929 ] )
pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 195.37, 251.48, 241.35, 333.20, 228.84, 203.64, 361.70 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43, 10.00 ] )

# HEXTOR, cases 1, 4, 8, 9, 11, 14, 15. The other nine are excluded: eight are
# runaways beyond the radiative lookup table and Case 10 is CO2 condensing.
hextor       = np.array( [ 173.92, 312.24, 225.08, 277.17, 228.72, 242.30, 189.11 ] )
hextor_flux1 = np.array( [ 500, 1200, 800, 1100, 900, 900, 600 ] ) / fluxscale
hextor_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 0.10, 1.44, 0.43 ] )

# ExoColumn, cases 1, 4, 8, 9, 10, 11, 14, 15. The other eight are incipient
# runaways: no steady state exists at that (S, p), so the RCE loop never closes.
exocolumn       = np.array( [ 206.98, 293.26, 248.49, 269.66, 201.36, 242.60, 251.63, 216.92 ] )
exocolumn_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600 ] ) / fluxscale
exocolumn_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43 ] )

# Kriging anisotropy, fitted per model by leave-one-out cross-validation in
# fit_anisotropy.py. pykrige scales the second coordinate, which here is
# normalized instellation, so a value of s means one unit of normalized
# instellation counts s times a unit of normalized log-pressure. Isotropic
# kriging (s = 1) asserts the two axes are equally informative, which is false
# for temperature: rerun fit_anisotropy.py after any resubmission.
ANISO = {
    'ExoCAM':       10,
    'ROCKE-3D':     4,
    'ExoPlaSim':    2,
    'PlaHab':       3,
    'LFRic':        15,
    'Generic PCM':  5,
    'HEXTOR':       7,
    'ExoColumn':    7,
}

# Normalize both axes to [0, 1] for kriging so distance metric is balanced
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

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
    exocam_stable,
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
    rocke3d_stable,
    anisotropy_scaling=ANISO[ 'ROCKE-3D' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

R3_z1, R3_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# Generic PCM Kriging

OK = OrdinaryKriging(
    norm_pres( pcm_pres1 ),
    norm_flux( pcm_flux1 ),
    pcm,
    anisotropy_scaling=ANISO[ 'Generic PCM' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

pcm_z1, pcm_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# ExoPlaSim Kriging

OK = OrdinaryKriging(
    norm_pres( pres1 ),
    norm_flux( flux1 ),
    plasim,
    anisotropy_scaling=ANISO[ 'ExoPlaSim' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

PlaSim_z1, PlaSim_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# PlaHab Kriging

OK = OrdinaryKriging(
    norm_pres( plahab_pres1 ),
    norm_flux( plahab_flux1 ),
    plahab_stable,
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
    lfric,
    anisotropy_scaling=ANISO[ 'LFRic' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

lfric_z1, lfric_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# HEXTOR Kriging

OK = OrdinaryKriging(
    norm_pres( hextor_pres1 ),
    norm_flux( hextor_flux1 ),
    hextor,
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
    exocolumn,
    anisotropy_scaling=ANISO[ 'ExoColumn' ],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    exact_values=True,
)

exocolumn_z1, exocolumn_var = OK.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )

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

cf1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, PlaSim_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P1' ].contourf( yv*fluxscale, xv, np.sqrt(PlaSim_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P1' ], f'ExoPlaSim (n={len(plasim)})' )

#--------------------------------------------------------------------
# Panel 2

cf2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P2' ].contourf( yv*fluxscale, xv, np.sqrt(z1_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P2' ].scatter( exocam_flux1*fluxscale, exocam_pres1, c=exocam_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P2' ], f'ExoCAM (n={len(exocam_stable)})' )

#--------------------------------------------------------------------
# Panel 3

cf3 = axd[ 'P3' ].contourf( yv*fluxscale, xv, R3_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P3' ].contourf( yv*fluxscale, xv, np.sqrt(R3_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P3' ].scatter( rocke3d_flux1*fluxscale, rocke3d_pres1, c=rocke3d_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P3' ], f'ROCKE-3D (n={len(rocke3d_stable)})' )

#--------------------------------------------------------------------
# Panel 4

cf4 = axd[ 'P4' ].contourf( yv*fluxscale, xv, pcm_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P4' ].contourf( yv*fluxscale, xv, np.sqrt(pcm_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P4' ].scatter( pcm_flux1*fluxscale, pcm_pres1, c=pcm, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P4' ], f'Generic PCM (n={len(pcm)})' )

#--------------------------------------------------------------------
# Panel 5

cf5 = axd[ 'P5' ].contourf( yv*fluxscale, xv, lfric_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P5' ].contourf( yv*fluxscale, xv, np.sqrt(lfric_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P5' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P5' ], f'LFRic (n={len(lfric)})' )

#--------------------------------------------------------------------
# Panel 6

cf6 = axd[ 'P6' ].contourf( yv*fluxscale, xv, PlaHab_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P6' ].contourf( yv*fluxscale, xv, np.sqrt(PlaHab_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P6' ].scatter( plahab_flux1*fluxscale, plahab_pres1, c=plahab_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P6' ], f'PlaHab (n={len(plahab_stable)})' )

#--------------------------------------------------------------------
# Panel 7

cf7 = axd[ 'P7' ].contourf( yv*fluxscale, xv, hextor_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P7' ].contourf( yv*fluxscale, xv, np.sqrt(hextor_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P7' ].scatter( hextor_flux1*fluxscale, hextor_pres1, c=hextor, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P7' ], f'HEXTOR (n={len(hextor)})' )

#--------------------------------------------------------------------
# Panel 8

cf8 = axd[ 'P8' ].contourf( yv*fluxscale, xv, exocolumn_z1, cmap=cm, levels=contour_levels, extend='both' )
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

fig.savefig( "fig_interpolation_temp.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_temp.eps", bbox_inches='tight' )
#plt.show()
