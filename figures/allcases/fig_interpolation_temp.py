import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

from pykrige.ok import OrdinaryKriging

runawaytemp = 600.0
contourmax  = 500.0
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

lfric = np.array( [ 195.37, 251.48, 241.35, 333.20, 228.84, 203.64 ] )
lfric_flux1  = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1  = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

pcm = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218, 266.5987224285321, 210.69131033681012, 246.04296230476365, 217.2519558970929 ] )
pcm_flux1  = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1  = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )


fig, axd = plt.subplot_mosaic([['P1', 'P2', 'P3' ],
                               ['P4', 'P5', 'P6' ]],
                              figsize=(18, 9))

cm = mpl.colormaps.get_cmap('terrain')

#--------------------------------------------------------------------
# ExoCAM Kriging

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

PlaSim_z1, PlaSim_ss = OK.execute("grid", pn2, flux )

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
# LFRic Kriging

OK = OrdinaryKriging(
    lfric_pres1,
    lfric_flux1,
    lfric,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

lfric_z1, lfric_ss = OK.execute("grid", pn2, flux )

#--------------------------------------------------------------------
# Generic PCM (no OHT) Kriging

OK = OrdinaryKriging(
    pcm_pres1,
    pcm_flux1,
    pcm,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

pcm_z1, pcm_ss = OK.execute("grid", pn2, flux )

#--------------------------------------------------------------------
# Panel 1

im1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im1 = axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
axd[ 'P1' ].tick_params( axis='x', labelsize=12 )
axd[ 'P1' ].tick_params( axis='y', labelsize=12 )
axd[ 'P1' ].set( title='ExoCAM' )
axd[ 'P1' ].set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
axd[ 'P1' ].set_ylabel( 'Surface pressure (bar)', fontsize = 12 )
axd[ 'P1' ].set_yscale('log')
axd[ 'P1' ].set_ylim( [ min( pn2 )*0.9,  max( pn2  )*1.1 ] )
axd[ 'P1' ].set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )

#--------------------------------------------------------------------
# Panel 2

im2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, R3_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im2 = axd[ 'P2' ].scatter( flux1*fluxscale, pres1, c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
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

im3 = axd[ 'P3' ].contourf( yv*fluxscale, xv, pcm_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im3 = axd[ 'P3' ].scatter( pcm_flux1*fluxscale, pcm_pres1, c=pcm, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
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
# Panel 4

im4 = axd[ 'P4' ].contourf( yv*fluxscale, xv, PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im4 = axd[ 'P4' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
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
# Panel 5

im5 = axd[ 'P5' ].contourf( yv*fluxscale, xv, PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im5 = axd[ 'P5' ].scatter( flux1*fluxscale, pres1, c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
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
# Panel 6

im6 = axd[ 'P6' ].contourf( yv*fluxscale, xv, lfric_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im6 = axd[ 'P6' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
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
cb3 = fig.colorbar( im3, ax=axd[ 'P3' ], extend='both', ticks=np.arange( 200, contourmax + 50, 50 ) )
cb4 = fig.colorbar( im4, ax=axd[ 'P4' ], extend='both' )
cb5 = fig.colorbar( im5, ax=axd[ 'P5' ], extend='both')
cb6 = fig.colorbar( im6, ax=axd[ 'P6' ], extend='both', ticks=np.arange( 200, contourmax + 50, 50 ) )

cb1.ax.get_yaxis().labelpad = 15
cb1.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb2.ax.get_yaxis().labelpad = 15
cb2.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb3.ax.get_yaxis().labelpad = 15
cb3.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb4.ax.get_yaxis().labelpad = 15
cb4.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb5.ax.get_yaxis().labelpad = 15
cb5.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb6.ax.get_yaxis().labelpad = 15
cb6.set_label( 'Average Surface Temperature (K)', rotation=270 )

fig.subplots_adjust( wspace = 0.33, hspace = 0.33 )

fig.savefig( "fig_interpolation_temp.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_temp.eps", bbox_inches='tight' )
#plt.show()
