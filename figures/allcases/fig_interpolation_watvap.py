import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import CloughTocher2DInterpolator

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

import matplotlib.colors as colors

runaway = 1.e4
contourmax  = 1.e3
contourmin  = 1.e-4
numcontours = 40

fluxscale   = 100
valuescale  = 0.1

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1  = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Water Vapor Column
plasim  = valuescale * np.array( [ 0.059, 5113.363, 271.733, 7.509, 37.305, 1452.570, 76.876, 0.351, 7.048, 0.007, 3.042, 1258.514, 59.214, 1.955, 0.416, 1111.793 ] )
exocam  = valuescale * np.array( [ 0.2335, runaway, runaway, 19.2138, runaway, runaway, runaway, 1.7490, 7.9707, 0.0102, 5.8914, 1430.7613, runaway, 4.2041, 0.9534, 1295.6428 ] )
rocke3d = valuescale * np.array( [ 0.25980374, runaway, runaway, 14.9693165, 31.839989, runaway, 32.45563, 1.5794185, 5.687923, 0.031635746, 4.2059116, 271.75916, 46.070984, 2.964403, 0.52949935, 132.49808 ] )
#plahab  = np.array( [ contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin, contourmin ] )
pcm = valuescale * np.array( [ 0.37484651163423993, 1687.0922440321456, 904.4863537156082, 47.86514350558743, 1466.6319240274788, 807.4897987980322, 1336.7879757824508, 2.1906175203429563, 25.650192288432617, 0.06080825624148188, 14.360702582472284, 1628.2873883303755, 1417.47476458977, 5.93210602524276, 0.8447688576786204, 2155.903633545497 ] )

lfric = np.array( [ 0.41, 8.18, 7.37, 829.15, 2.34, 0.86 ] )
lfric_flux1  = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1  = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )


#fig, axd = plt.subplot_mosaic([['P1', 'P2' ],
#                               ['P4', 'P5' ]],
#                              figsize=(15.375, 11.375))
fig, axd = plt.subplot_mosaic([['P1', 'P2', 'P3' ],
                               ['P4', 'P5', 'P6' ]],
                              figsize=(18, 9))

cm = mpl.colormaps.get_cmap('YlGnBu_r')
#cm = mpl.colormaps.get_cmap('PuBuGn_r')
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
    np.log(exocam),
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
    np.log(rocke3d),
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
    np.log(plasim),
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

PlaSim_z1, PlaSim_ss = OK.execute("grid", pn2, flux)

#--------------------------------------------------------------------
# Generic PCM (no OHT) Kriging

OK = OrdinaryKriging(
    pres1,
    flux1,
    np.log(pcm),
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
    np.log(lfric),
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

lfric_z1, lfric_ss = OK.execute("grid", pn2, flux)

#--------------------------------------------------------------------
# PlaHab Kriging
#
#OK = OrdinaryKriging(
#    pres1,
#    flux1,
#    np.log(plahab),
#    variogram_model="linear",
#    verbose=True,
#    enable_plotting=False,
#    exact_values=True,
#)
#
#PlaHab_z1, PlaHab_ss = OK.execute("grid", pn2, flux)
#
#OK.display_variogram_model()
#
#--------------------------------------------------------------------
# Panel 1

im1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, np.exp(z1), cmap=cm, levels=np.logspace( np.log10(contourmin), np.log10(contourmax) , numcontours ), extend='both', norm=colors.LogNorm() )
im1 = axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=exocam/valuescale, cmap=cm, marker='o', s=70, edgecolors='k', norm=colors.LogNorm() )
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

im2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, np.exp(R3_z1), cmap=cm, levels=np.logspace( np.log10(contourmin), np.log10(contourmax) , numcontours ), extend='both', norm=colors.LogNorm() )
im2 = axd[ 'P2' ].scatter( flux1*fluxscale, pres1, c=rocke3d/valuescale, cmap=cm, marker='o', s=70, facecolors='k', edgecolors='k', norm=colors.LogNorm() )
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

im2 = axd[ 'P4' ].contourf( yv*fluxscale, xv, np.exp(PlaSim_z1), cmap=cm, levels=np.logspace( np.log10(contourmin), np.log10(contourmax) , numcontours ), extend='both', norm=colors.LogNorm() )
im2 = axd[ 'P4' ].scatter( flux1*fluxscale, pres1, c=plasim/valuescale, cmap=cm, marker='o', s=70, facecolors='k', edgecolors='k', norm=colors.LogNorm() )
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

im2 = axd[ 'P3' ].contourf( yv*fluxscale, xv, np.exp(pcm_z1), cmap=cm, levels=np.logspace( np.log10(contourmin), np.log10(contourmax) , numcontours ), extend='both', norm=colors.LogNorm() )
im2 = axd[ 'P3' ].scatter( flux1*fluxscale, pres1, c=pcm, cmap=cm, marker='o', s=70, facecolors='k', edgecolors='k', norm=colors.LogNorm() )
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

im2 = axd[ 'P6' ].contourf( yv*fluxscale, xv, np.exp(lfric_z1), cmap=cm, levels=np.logspace( np.log10(contourmin), np.log10(contourmax) , numcontours ), extend='both', norm=colors.LogNorm() )
im2 = axd[ 'P6' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, marker='o', s=70, facecolors='k', edgecolors='k', norm=colors.LogNorm() )
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

cb1 = fig.colorbar( im1, ax=axd[ 'P1' ], extend='both', norm=colors.LogNorm( vmin=np.log10(contourmin), vmax=np.log10(contourmax) ), ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3] )
cb2 = fig.colorbar( im2, ax=axd[ 'P2' ], extend='both', norm=colors.LogNorm( vmin=np.log10(contourmin), vmax=np.log10(contourmax) ), ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3] )
cb4 = fig.colorbar( im1, ax=axd[ 'P4' ], extend='both', norm=colors.LogNorm( vmin=np.log10(contourmin), vmax=np.log10(contourmax) ), ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3] )
cb5 = fig.colorbar( im2, ax=axd[ 'P3' ], extend='both', norm=colors.LogNorm( vmin=np.log10(contourmin), vmax=np.log10(contourmax) ), ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3] )
cb6 = fig.colorbar( im2, ax=axd[ 'P6' ], extend='both', norm=colors.LogNorm( vmin=np.log10(contourmin), vmax=np.log10(contourmax) ), ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3] )

cb1.ax.get_yaxis().labelpad = 15
cb1.set_label( 'Average Water Vapor Column (kg m$^{-2}$)', rotation=270 )
cb2.ax.get_yaxis().labelpad = 15
cb2.set_label( 'Average Water Vapor Column (kg m$^{-2}$)', rotation=270 )
cb4.ax.get_yaxis().labelpad = 15
cb4.set_label( 'Average Water Vapor Column (kg m$^{-2}$)', rotation=270 )
cb5.ax.get_yaxis().labelpad = 15
cb5.set_label( 'Average Water Vapor Column (kg m$^{-2}$)', rotation=270 )
cb6.ax.get_yaxis().labelpad = 15
cb6.set_label( 'Average Water Vapor Column (kg m$^{-2}$)', rotation=270 )

fig.subplots_adjust( wspace = 0.25, hspace = 0.25 )

fig.savefig( "fig_interpolation_watvap.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_watvap.eps", bbox_inches='tight' )
#plt.show()
