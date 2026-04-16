import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import CloughTocher2DInterpolator

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

runawaytemp = 600.0
contourmax  = 500.0
fluxscale   = 100

#flux = np.log( np.arange( 400, 2700, 100 ) )
#pn2  = np.log( np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] ) )
flux = np.arange( 400, 2700, 100 ) / fluxscale
#flux = np.arange( 400, 2700, 100 )
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
#flux1  = np.log( np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) )
#pres1  = np.log( np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] ) )
flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
#flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] )
pres1  = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Minimum Temperature from ExoPlaSim, ExoCAM, ROCKE-3D
plasim  = np.array( [ 124.6, 318.2, 259.5, 200.6, 219.0, 333.9, 245.5, 158.1, 182.1, 132.2, 133.5, 337.2, 229.0, 158.1, 122.5, 306.3 ] )
exocam  = np.array( [ 147.4, runawaytemp, runawaytemp, 211.1, runawaytemp, runawaytemp, runawaytemp, 218.2, 200.4, 166.5, 195.2, 348.2, runawaytemp, 183.2, 157.9, 353.8 ] )
rocke3d = 273.16 + np.array( [ -129.406, runawaytemp, runawaytemp, -64.612, -33.342, runawaytemp, -26.935, -51.890, -70.497, -97.390, -85.507, -0.835, -29.809, -85.369, -117.976, 20.372 ] )
plahab  = np.array( [ 150.0000, runawaytemp, runawaytemp, 247.8903, 269.2430, runawaytemp, 287.6475, 177.7665, 215.4244, 150.0000, 150.0000, 291.2116, 277.1293, 184.7641, 150.0000, 286.7990 ] )


# OPTIONAL: remove runaway cases
#runawaycases = np.where( exocam == runawaytemp )
#exocam = np.delete( exocam, runawaycases )
#plasim = np.delete( plasim, runawaycases )
#flux1  = np.delete( flux1,  runawaycases )
#pres1  = np.delete( pres1,  runawaycases )
#flux   = np.delete( flux, np.where( flux > 1500 ) )

#numplots = 2
#fig, (ax, ax2) = plt.subplots( 1, numplots, sharex=False, figsize = (15.0,4.75) )
#fig, (ax, ax2) = plt.subplots( 1, numplots, sharex=False, figsize = (14.0,4.75) )
#plt.rc('axes', titlesize=15)

fig, axd = plt.subplot_mosaic([['P1', 'P2' ],
                               ['P4', 'P5' ]],
                              figsize=(15.375, 11.375))
#fig, axd = plt.subplot_mosaic([['P1', 'P2', 'P3' ],
#                               ['P4', 'P5', 'P6' ]],
#                              figsize=(15.375, 9.375))

#cm = mpl.colormaps.get_cmap('cool')
cm = mpl.colormaps.get_cmap('terrain')

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
    plasim,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

#PlaSim_z1, PlaSim_ss = OK.execute("grid", pn2, np.log( flux ) )
PlaSim_z1, PlaSim_ss = OK.execute("grid", pn2, flux )

#xvPS, yvPS = np.meshgrid( pn2, flux )
#xvPS, yvPS = np.meshgrid( pn2, np.log( flux ) )

#OK.display_variogram_model()

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

PlaHab_z1, PlaHab_ss = OK.execute("grid", pn2, flux )

#--------------------------------------------------------------------
# Panel 1

#im1 = axd[ 'P1' ].contourf( yv, xv, z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im1 = axd[ 'P1' ].scatter( flux1, pres1, c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
im1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
im1 = axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
#im1 = axd[ 'P1' ].contourf( yv, np.exp( xv ), z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im1 = axd[ 'P1' ].scatter( flux1, np.exp( pres1 ), c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
#im1 = axd[ 'P1' ].contourf( np.exp( yv ), np.exp( xv ), np.exp( z1 ), cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im1 = axd[ 'P1' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
#im1 = axd[ 'P1' ].contourf( np.exp( yv ), xv, z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im1 = axd[ 'P1' ].scatter( np.exp( flux1 ), pres1, c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
axd[ 'P1' ].tick_params( axis='x', labelsize=12 )
axd[ 'P1' ].tick_params( axis='y', labelsize=12 )
axd[ 'P1' ].set( title='ExoCAM' )
axd[ 'P1' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P1' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P1' ].set_yscale('log')
#axd[ 'P1' ].set_xlim( [ max( flux ) + 50, min( flux ) - 50 ] )
axd[ 'P1' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )
axd[ 'P1' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
#axd[ 'P1' ].set_xlim( [ max( np.exp( flux ) ) + 50, min( np.exp( flux ) ) - 50 ] )
#axd[ 'P1' ].set_ylim( [ min( np.exp( pn2 ) )*0.9,  max( np.exp( pn2 ) )*1.1 ] )

#--------------------------------------------------------------------
# Panel 2

#im2 = axd[ 'P2' ].contourf( yv, xv, R3_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P2' ].scatter( flux1, pres1, c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
im2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, R3_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
im2 = axd[ 'P2' ].scatter( flux1*fluxscale, pres1, c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P2' ].contourf( yv, np.exp( xv ), R3_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P2' ].scatter( flux1, np.exp( pres1 ), c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P2' ].contourf( np.exp( yv ), np.exp( xv ), R3_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P2' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P2' ].contourf( np.exp( yv ), xv, R3_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P2' ].scatter( np.exp( flux1 ), pres1, c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P2' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P2' ].tick_params( axis='x', labelsize=12 )
axd[ 'P2' ].tick_params( axis='y', labelsize=12 )
axd[ 'P2' ].set( title='ROCKE-3D' )
axd[ 'P2' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P2' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P2' ].set_yscale('log')
#axd[ 'P2' ].set_xlim( [ max( flux ) + 50, min( flux ) - 50 ] )
axd[ 'P2' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P2' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )
#axd[ 'P2' ].set_xlim( [ max( np.exp( flux ) ) + 50, min( np.exp( flux ) ) - 50 ] )
#axd[ 'P2' ].set_ylim( [ min( np.exp( pn2 ) )*0.9,  max( np.exp( pn2 ) )*1.1 ] )

#--------------------------------------------------------------------
# Panel 3

#im2 = axd[ 'P4' ].contourf( yv, xv, PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P4' ].scatter( flux1, pres1, c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
im2 = axd[ 'P4' ].contourf( yv*fluxscale, xv, PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
im2 = axd[ 'P4' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P4' ].contourf( yv, np.exp( xv ), PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P4' ].scatter( flux1, np.exp( pres1 ), c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P4' ].contourf( np.exp( yv ), np.exp( xv ), PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P4' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P4' ].contourf( np.exp( yv ), xv, PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P4' ].scatter( np.exp( flux1 ), pres1, c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P4' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P4' ].tick_params( axis='x', labelsize=12 )
axd[ 'P4' ].tick_params( axis='y', labelsize=12 )
axd[ 'P4' ].set( title='ExoPlaSim' )
axd[ 'P4' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P4' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P4' ].set_yscale('log')
#axd[ 'P4' ].set_xlim( [ max( flux ) + 50, min( flux ) - 50 ] )
axd[ 'P4' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P4' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )
#axd[ 'P4' ].set_xlim( [ max( np.exp( flux ) ) + 50, min( np.exp( flux ) ) - 50 ] )
#axd[ 'P4' ].set_ylim( [ min( np.exp( pn2 ) )*0.9,  max( np.exp( pn2 ) )*1.1 ] )

#--------------------------------------------------------------------
# Panel 4

#im2 = axd[ 'P5' ].contourf( yv, xv, PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P5' ].scatter( flux1, pres1, c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
im2 = axd[ 'P5' ].contourf( yv*fluxscale, xv, PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
im2 = axd[ 'P5' ].scatter( flux1*fluxscale, pres1, c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P5' ].contourf( yv, np.exp( xv ), PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P5' ].scatter( flux1, np.exp( pres1 ), c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P5' ].contourf( np.exp( yv ), np.exp( xv ), PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P5' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P5' ].contourf( np.exp( yv ), xv, PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,20), extend='both' )
#im2 = axd[ 'P5' ].scatter( np.exp( flux1 ), pres1, c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
axd[ 'P5' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P5' ].tick_params( axis='x', labelsize=12 )
axd[ 'P5' ].tick_params( axis='y', labelsize=12 )
axd[ 'P5' ].set( title='PlaHab' )
axd[ 'P5' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P5' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P5' ].set_yscale('log')
#axd[ 'P5' ].set_xlim( [ max( flux ) + 50, min( flux ) - 50 ] )
axd[ 'P5' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
axd[ 'P5' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )
#axd[ 'P5' ].set_xlim( [ max( np.exp( flux ) ) + 50, min( np.exp( flux ) ) - 50 ] )
#axd[ 'P5' ].set_ylim( [ min( np.exp( pn2 ) )*0.9,  max( np.exp( pn2 ) )*1.1 ] )

#--------------------------------------------------------------------
# Finalize

cb1 = fig.colorbar( im1, ax=axd[ 'P1' ], extend='both' )
cb2 = fig.colorbar( im2, ax=axd[ 'P2' ], extend='both', ticks=np.arange( 200, contourmax + 50, 50 ) )
cb4 = fig.colorbar( im1, ax=axd[ 'P4' ], extend='both' )
cb5 = fig.colorbar( im2, ax=axd[ 'P5' ], extend='both', ticks=np.arange( 200, contourmax + 50, 50 ) )

cb1.ax.get_yaxis().labelpad = 15
cb1.set_label( 'Minimum Surface Temperature (K)', rotation=270 )
cb2.ax.get_yaxis().labelpad = 15
cb2.set_label( 'Minimum Surface Temperature (K)', rotation=270 )
cb4.ax.get_yaxis().labelpad = 15
cb4.set_label( 'Minimum Surface Temperature (K)', rotation=270 )
cb5.ax.get_yaxis().labelpad = 15
cb5.set_label( 'Minimum Surface Temperature (K)', rotation=270 )

fig.subplots_adjust( wspace = 0.25, hspace = 0.25 )

fig.savefig( "fig_interpolation_tempmin.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_tempmin.eps", bbox_inches='tight' )
#plt.show()
