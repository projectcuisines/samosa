import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean

from pykrige.ok import OrdinaryKriging

runawaytemp = 600.0
contourmin  = 175.0
contourmax  = 370.0
cinterval   = 40
fluxscale   = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1  = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Temperature from ExoPlaSim, ExoCAM, ROCKE-3D
plasim = np.array( [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9, 239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ] )
exocam = np.array( [ 196.8, runawaytemp, runawaytemp, 260.0, runawaytemp, runawaytemp, runawaytemp, 243.8, 244.8, 194.1, 234.0, 350.9, runawaytemp, 236.8, 211.5, 356.7 ] )
rocke3d = np.array( [ 202.8284, runawaytemp, runawaytemp, 260.1185, 265.88116, runawaytemp, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ] )
plahab = np.array( [ 196.3, runawaytemp, runawaytemp, 273.2, 281.4, runawaytemp, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )

exocam_mask  = exocam  != runawaytemp
rocke3d_mask = rocke3d != runawaytemp
plahab_mask  = plahab  != runawaytemp

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

lfric = np.array( [ 195.37, 251.48, 241.35, 333.20, 228.84, 203.64 ] )
lfric_flux1  = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1  = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

pcm = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218, 266.5987224285321, 210.69131033681012, 246.04296230476365, 217.2519558970929 ] )
pcm_flux1  = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1  = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )


# Normalize both axes to [0, 1] for kriging so distance metric is balanced
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

fig, axd = plt.subplot_mosaic([['P1', 'P2', 'P3' ],
                               ['P4', 'P5', 'P6' ]],
                              figsize=(18, 9))

cm = cmocean.cm.thermal

#--------------------------------------------------------------------
# ExoCAM Kriging

OK = OrdinaryKriging(
    norm_pres( exocam_pres1 ),
    norm_flux( exocam_flux1 ),
    exocam_stable,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

z1, z1_var = OK.execute("grid", norm_pres( pn2 ), norm_flux( flux ))
xv, yv = np.meshgrid( pn2, flux )

#--------------------------------------------------------------------
# ROCKE-3D Kriging

OK = OrdinaryKriging(
    norm_pres( rocke3d_pres1 ),
    norm_flux( rocke3d_flux1 ),
    rocke3d_stable,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

R3_z1, R3_var = OK.execute("grid", norm_pres( pn2 ), norm_flux( flux ))

#--------------------------------------------------------------------
# ExoPlaSim Kriging

OK = OrdinaryKriging(
    norm_pres( pres1 ),
    norm_flux( flux1 ),
    plasim,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

PlaSim_z1, PlaSim_var = OK.execute("grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# PlaHab Kriging

OK = OrdinaryKriging(
    norm_pres( plahab_pres1 ),
    norm_flux( plahab_flux1 ),
    plahab_stable,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

PlaHab_z1, PlaHab_var = OK.execute("grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# LFRic Kriging

OK = OrdinaryKriging(
    norm_pres( lfric_pres1 ),
    norm_flux( lfric_flux1 ),
    lfric,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

lfric_z1, lfric_var = OK.execute("grid", norm_pres( pn2 ), norm_flux( flux ) )

#--------------------------------------------------------------------
# Generic PCM Kriging

OK = OrdinaryKriging(
    norm_pres( pcm_pres1 ),
    norm_flux( pcm_flux1 ),
    pcm,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

pcm_z1, pcm_var = OK.execute("grid", norm_pres( pn2 ), norm_flux( flux ) )

# Shared axis limits
xlim = [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ]
ylim = [ min( pn2 )*0.9, max( pn2 )*1.1 ]
cbar_ticks = np.arange( 200, contourmax, 50 )
contour_levels = np.linspace( contourmin, contourmax, cinterval )
marker_edge = 'k'
sigma_threshold = 45.0  # K; hatch regions where kriging std dev exceeds this

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

cf1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P1' ].contourf( yv*fluxscale, xv, np.sqrt(z1_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P1' ].scatter( exocam_flux1*fluxscale, exocam_pres1, c=exocam_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P1' ], f'ExoCAM (n={len(exocam_stable)})' )

#--------------------------------------------------------------------
# Panel 2

cf2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, R3_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P2' ].contourf( yv*fluxscale, xv, np.sqrt(R3_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P2' ].scatter( rocke3d_flux1*fluxscale, rocke3d_pres1, c=rocke3d_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P2' ], f'ROCKE-3D (n={len(rocke3d_stable)})' )

#--------------------------------------------------------------------
# Panel 3

cf3 = axd[ 'P3' ].contourf( yv*fluxscale, xv, pcm_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P3' ].contourf( yv*fluxscale, xv, np.sqrt(pcm_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P3' ].scatter( pcm_flux1*fluxscale, pcm_pres1, c=pcm, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P3' ], f'Generic PCM (n={len(pcm)})' )

#--------------------------------------------------------------------
# Panel 4

cf4 = axd[ 'P4' ].contourf( yv*fluxscale, xv, PlaSim_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P4' ].contourf( yv*fluxscale, xv, np.sqrt(PlaSim_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P4' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P4' ], f'ExoPlaSim (n={len(plasim)})' )

#--------------------------------------------------------------------
# Panel 5

cf5 = axd[ 'P5' ].contourf( yv*fluxscale, xv, PlaHab_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P5' ].contourf( yv*fluxscale, xv, np.sqrt(PlaHab_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P5' ].scatter( plahab_flux1*fluxscale, plahab_pres1, c=plahab_stable, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P5' ], f'PlaHab (n={len(plahab_stable)})' )

#--------------------------------------------------------------------
# Panel 6

cf6 = axd[ 'P6' ].contourf( yv*fluxscale, xv, lfric_z1, cmap=cm, levels=contour_levels, extend='both' )
axd[ 'P6' ].contourf( yv*fluxscale, xv, np.sqrt(lfric_var), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
axd[ 'P6' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, vmin=contourmin, vmax=contourmax, marker='o', s=70, edgecolors=marker_edge )
setup_panel( axd[ 'P6' ], f'LFric (n={len(lfric)})' )

#--------------------------------------------------------------------
# Finalize

fig.subplots_adjust( wspace=0.3, hspace=0.4, right=0.88 )
cax = fig.add_axes( [0.91, 0.1, 0.015, 0.8] )
cb = fig.colorbar( cf1, cax=cax, extend='both', ticks=cbar_ticks )
cb.ax.tick_params( labelsize=11 )
cb.ax.get_yaxis().labelpad = 15
cb.set_label( 'Average Surface Temperature (K)', rotation=270, fontsize=12 )

fig.savefig( "fig_interpolation_temp.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_temp.eps", bbox_inches='tight' )
#plt.show()
