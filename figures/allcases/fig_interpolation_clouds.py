import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import CloughTocher2DInterpolator

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

runaway = 2.0
contourmax  = 1.0
cinterval   = 40 
fluxscale   = 100
cloudscale  = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1  = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Cloud Cover percentage from ExoPlaSim, ExoCAM, ROCKE-3D
plasim  = np.array( [ 0.427, 0.682, 0.706, 0.562, 0.801, 0.321, 0.581, 0.257, 0.580, 0.252, 0.762, 0.301, 0.853, 0.527, 0.484, 0.533 ] )
exocam  = np.array( [ 0.6875, runaway, runaway, 0.4398, runaway, runaway, runaway, 0.1634, 0.7582, 0.1585, 0.8308, 0.5679, runaway, 0.3401, 0.7896,0.6140 ] )
rocke3d = np.array( [ 0.6820222, runaway, runaway, 0.51043224, 0.8188261, runaway, 0.8881546, 0.5824044, 0.61535275, 0.988356, 0.6816493, 0.6801357, 0.8571091, 0.43637707, 0.7440385, 0.4808909 ] )
plahab  = np.array( [ 0.1111879, runaway, runaway, 0.3574597, 0.4829323, runaway, 0.7091280, 0.2624803, 0.3164522, 8.4692545E-02, 4.3572873E-02, 0.7620874, 0.5727629, 0.2812309, 0.1664636, 0.7236285 ] )

pcm     = np.array( [ 0.255674468009485, 0.1789595000126142, 0.35156315788637776, 0.2731228828919005, 0.7920630548839582, 0.2212486628937806, 0.4930302490315669, 0.1696672860199983, 0.2515276275245855, 0.2455454268845772, 0.4415314026459278, 0.15979634438771312, 0.7680075170136155, 0.16696470834684884, 0.3284789893586739, 0.11882094919440202 ] )

lfric = np.array( [ 0.31, 0.61, 0.58, 0.81, 0.44, 0.36 ] )
lfric_flux1  = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1  = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )


#fig, axd = plt.subplot_mosaic([['P1', 'P2' ],
#                               ['P4', 'P5' ]],
#                              figsize=(15.375, 11.375))
fig, axd = plt.subplot_mosaic([['P1', 'P2', 'P3' ],
                               ['P4', 'P5', 'P6' ]],
                              figsize=(18, 9))

cm = mpl.colormaps.get_cmap('PuBuGn_r')
#cm = mpl.colormaps.get_cmap('cool')
#cm = mpl.colormaps.get_cmap('terrain')

#--------------------------------------------------------------------
# ExoCAM Kriging

# OPTIONAL: set boundary condition for runaway at hot/high-pressure conditions 
#pres1  = np.append( pres1, 10.0 )
#flux1  = np.append( flux1, 2600 )
#exocam = np.append( exocam, runawaytemp )
#exocam = np.append( exocam, runawaytemp+100 )

OK = OrdinaryKriging(
    pres1,
    flux1,
    exocam,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

z1, ss = OK.execute("grid", pn2, flux)
xv, yv = np.meshgrid( pn2, flux )

#OK.display_variogram_model()

#--------------------------------------------------------------------
# ROCKE-3D Kriging

OK = OrdinaryKriging(
    pres1,
    flux1,
    rocke3d,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

R3_z1, R3_ss = OK.execute("grid", pn2, flux)

#--------------------------------------------------------------------
# ExoPlaSim Kriging

OK = OrdinaryKriging(
    pres1,
    flux1,
    plasim * cloudscale,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

PlaSim_z1, PlaSim_ss = OK.execute("grid", pn2, flux)

#--------------------------------------------------------------------
# PlaHab Kriging

OK = OrdinaryKriging(
    pres1,
    flux1,
    plahab,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

PlaHab_z1, PlaHab_ss = OK.execute("grid", pn2, flux)

#--------------------------------------------------------------------
# Generic PCM (no OHT) Kriging

OK = OrdinaryKriging(
    pres1,
    flux1,
    pcm,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

pcm_z1, pcm_ss = OK.execute("grid", pn2, flux)

#--------------------------------------------------------------------
# LFric Kriging

OK = OrdinaryKriging(
    lfric_pres1,
    lfric_flux1,
    lfric,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

lfric_z1, lfric_ss = OK.execute("grid", pn2, flux)

#OK.display_variogram_model()

#--------------------------------------------------------------------
# Panel 1

im1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, z1, cmap=cm, levels=np.linspace(0,contourmax,cinterval), extend='both' )
im1 = axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=exocam, cmap=cm, vmin=0, vmax=contourmax, marker='o', s=70, edgecolors='k' )
axd[ 'P1' ].tick_params( axis='x', labelsize=12 )
axd[ 'P1' ].tick_params( axis='y', labelsize=12 )
axd[ 'P1' ].set( title='ExoCAM' )
axd[ 'P1' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P1' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P1' ].set_yscale('log')
axd[ 'P1' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P1' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )

#--------------------------------------------------------------------
# Panel 2

im2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, R3_z1, cmap=cm, levels=np.linspace(0,contourmax,cinterval), extend='both' )
im2 = axd[ 'P2' ].scatter( flux1*fluxscale, pres1, c=rocke3d, cmap=cm, vmin=0, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P2' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P2' ].tick_params( axis='x', labelsize=12 )
axd[ 'P2' ].tick_params( axis='y', labelsize=12 )
axd[ 'P2' ].set( title='ROCKE-3D' )
axd[ 'P2' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P2' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P2' ].set_yscale('log')
axd[ 'P2' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P2' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )

#--------------------------------------------------------------------
# Panel 3

im2 = axd[ 'P4' ].contourf( yv*fluxscale, xv, PlaSim_z1/cloudscale, cmap=cm, levels=np.linspace(0,contourmax,cinterval), extend='both' )
im2 = axd[ 'P4' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=0, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P4' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P4' ].tick_params( axis='x', labelsize=12 )
axd[ 'P4' ].tick_params( axis='y', labelsize=12 )
axd[ 'P4' ].set( title='ExoPlaSim' )
axd[ 'P4' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P4' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P4' ].set_yscale('log')
axd[ 'P4' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P4' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )

#--------------------------------------------------------------------
# Panel 4

im2 = axd[ 'P3' ].contourf( yv*fluxscale, xv, pcm_z1, cmap=cm, levels=np.linspace(0,contourmax,cinterval), extend='both' )
im2 = axd[ 'P3' ].scatter( flux1*fluxscale, pres1, c=pcm, cmap=cm, vmin=0, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P3' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P3' ].tick_params( axis='x', labelsize=12 )
axd[ 'P3' ].tick_params( axis='y', labelsize=12 )
axd[ 'P3' ].set( title='Generic PCM (no OHT)' )
axd[ 'P3' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P3' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P3' ].set_yscale('log')
axd[ 'P3' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P3' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )

#--------------------------------------------------------------------
# Panel 5

im2 = axd[ 'P5' ].contourf( yv*fluxscale, xv, PlaHab_z1, cmap=cm, levels=np.linspace(0,contourmax,cinterval), extend='both' )
im2 = axd[ 'P5' ].scatter( flux1*fluxscale, pres1, c=plahab, cmap=cm, vmin=0, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P5' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P5' ].tick_params( axis='x', labelsize=12 )
axd[ 'P5' ].tick_params( axis='y', labelsize=12 )
axd[ 'P5' ].set( title='PlaHab' )
axd[ 'P5' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P5' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P5' ].set_yscale('log')
axd[ 'P5' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P5' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )

#--------------------------------------------------------------------
# Panel 5

im2 = axd[ 'P6' ].contourf( yv*fluxscale, xv, lfric_z1, cmap=cm, levels=np.linspace(0,contourmax,cinterval), extend='both' )
im2 = axd[ 'P6' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, vmin=0, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P6' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P6' ].tick_params( axis='x', labelsize=12 )
axd[ 'P6' ].tick_params( axis='y', labelsize=12 )
axd[ 'P6' ].set( title='LFric' )
axd[ 'P6' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P6' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P6' ].set_yscale('log')
axd[ 'P6' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P6' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )

#--------------------------------------------------------------------
# Finalize

cb1 = fig.colorbar( im1, ax=axd[ 'P1' ], extend='both' )
cb2 = fig.colorbar( im2, ax=axd[ 'P2' ], extend='both' )
cb3 = fig.colorbar( im1, ax=axd[ 'P3' ], extend='both' )
cb4 = fig.colorbar( im1, ax=axd[ 'P4' ], extend='both' )
cb5 = fig.colorbar( im2, ax=axd[ 'P5' ], extend='both' )
cb6 = fig.colorbar( im2, ax=axd[ 'P6' ], extend='both' )

cb1.ax.get_yaxis().labelpad = 15
cb1.set_label( 'Average Cloud Cover (%)', rotation=270 )
cb2.ax.get_yaxis().labelpad = 15
cb2.set_label( 'Average Cloud Cover (%)', rotation=270 )
cb3.ax.get_yaxis().labelpad = 15
cb3.set_label( 'Average Cloud Cover (%)', rotation=270 )
cb4.ax.get_yaxis().labelpad = 15
cb4.set_label( 'Average Cloud Cover (%)', rotation=270 )
cb5.ax.get_yaxis().labelpad = 15
cb5.set_label( 'Average Cloud Cover (%)', rotation=270 )
cb6.ax.get_yaxis().labelpad = 15
cb6.set_label( 'Average Cloud Cover (%)', rotation=270 )

fig.subplots_adjust( wspace = 0.25, hspace = 0.25 )

fig.savefig( "fig_interpolation_clouds.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_clouds.eps", bbox_inches='tight' )
#plt.show()
