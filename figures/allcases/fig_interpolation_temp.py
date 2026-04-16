import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import CloughTocher2DInterpolator

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

runawaytemp = 600.0
contourmax  = 500.0
cinterval   = 40
#cinterval   = 20.0
fluxscale   = 100


#flux = np.log( np.arange( 400, 2700, 100 ) )
#pn2  = np.log( np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] ) )
flux = np.arange( 400, 2700, 100 ) / fluxscale
#flux = np.arange( 400, 2700, 20 ) / fluxscale
#flux = np.arange( 400, 2700, 100 )

pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )
#pn2 = np.linspace( 0.1, 10.0, 100 )

# QMC sequence 1 + sequence 2
#flux1  = np.log( np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) )
#pres1  = np.log( np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] ) )
flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
#flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] )
pres1  = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Temperature from ExoPlaSim, ExoCAM, ROCKE-3D
plasim = np.array( [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9, 239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ] )
exocam = np.array( [ 196.8, runawaytemp, runawaytemp, 260.0, runawaytemp, runawaytemp, runawaytemp, 243.8, 244.8, 194.1, 234.0, 350.9, runawaytemp, 236.8, 211.5, 356.7 ] )
rocke3d = np.array( [ 202.8284, runawaytemp, runawaytemp, 260.1185, 265.88116, runawaytemp, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ] )
plahab = np.array( [ 196.3, runawaytemp, runawaytemp, 273.2, 281.4, runawaytemp, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )
#lfric = np.array( [ 195.37, runawaytemp, runawaytemp, 251.48, runawaytemp, runawaytemp, runawaytemp, runawaytemp, 241.35, runawaytemp, runawaytemp, 333.20, runawaytemp, 228.84, 203.64, runawaytemp ] )
#pcm = np.array( [ 210.9195445942203, 335.4116248917576, 313.56687204690985, 286.7294656230531, 330.1368829509338, 314.2140155031567, 325.73076835950087, 246.76730657647218, 266.5987224285321, 210.69131033681012, 255.82577832394563, 343.9473544143145, 325.7290522210099, 246.04296230476365, 217.2519558970929, 361.4170017678916 ] )

#rocke3d = np.array( [ 202.8284, 260.1185, 265.88116, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ] )
#rocke3d_flux1  = np.array( [ 500, 1200, 1500, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
#rocke3d_pres1  = np.array( [ 0.70, 2.34, 0.16, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )
#plahab = np.array( [ 196.3, 273.2, 281.4, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )
#plahab_flux1  = np.array( [ 500, 1200, 1500, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
#plahab_pres1  = np.array( [ 0.70, 2.34, 0.16, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )
lfric = np.array( [ 195.37, 251.48, 241.35, 333.20, 228.84, 203.64 ] )
lfric_flux1  = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1  = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

pcm = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218, 266.5987224285321, 210.69131033681012, 246.04296230476365, 217.2519558970929 ] )
pcm_flux1  = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1  = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )


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

#fig, axd = plt.subplot_mosaic([['P1', 'P2' ],
#                               ['P4', 'P5' ]],
#                              figsize=(15.375, 11.375))
fig, axd = plt.subplot_mosaic([['P1', 'P2', 'P3' ],
                               ['P4', 'P5', 'P6' ]],
                              figsize=(18, 9))

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

#im1 = axd[ 'P1' ].contourf( yv, xv, z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im1 = axd[ 'P1' ].scatter( flux1, pres1, c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
im1 = axd[ 'P1' ].contourf( yv*fluxscale, xv, z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im1 = axd[ 'P1' ].scatter( flux1*fluxscale, pres1, c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
#im1 = axd[ 'P1' ].contourf( yv, np.exp( xv ), z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im1 = axd[ 'P1' ].scatter( flux1, np.exp( pres1 ), c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
#im1 = axd[ 'P1' ].contourf( np.exp( yv ), np.exp( xv ), np.exp( z1 ), cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im1 = axd[ 'P1' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=exocam, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, edgecolors='k' )
#im1 = axd[ 'P1' ].contourf( np.exp( yv ), xv, z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
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

#im2 = axd[ 'P2' ].contourf( yv, xv, R3_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P2' ].scatter( flux1, pres1, c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
im2 = axd[ 'P2' ].contourf( yv*fluxscale, xv, R3_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im2 = axd[ 'P2' ].scatter( flux1*fluxscale, pres1, c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P2' ].contourf( yv, np.exp( xv ), R3_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P2' ].scatter( flux1, np.exp( pres1 ), c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P2' ].contourf( np.exp( yv ), np.exp( xv ), R3_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P2' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=rocke3d, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P2' ].contourf( np.exp( yv ), xv, R3_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
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

#im2 = axd[ 'P4' ].contourf( yv, xv, PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P4' ].scatter( flux1, pres1, c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
im2 = axd[ 'P4' ].contourf( yv*fluxscale, xv, PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im2 = axd[ 'P4' ].scatter( flux1*fluxscale, pres1, c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P4' ].contourf( yv, np.exp( xv ), PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P4' ].scatter( flux1, np.exp( pres1 ), c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P4' ].contourf( np.exp( yv ), np.exp( xv ), PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P4' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=plasim, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P4' ].contourf( np.exp( yv ), xv, PlaSim_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
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

#im2 = axd[ 'P5' ].contourf( yv, xv, PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P5' ].scatter( flux1, pres1, c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
im2 = axd[ 'P5' ].contourf( yv*fluxscale, xv, PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im2 = axd[ 'P5' ].scatter( flux1*fluxscale, pres1, c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P5' ].contourf( yv, np.exp( xv ), PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P5' ].scatter( flux1, np.exp( pres1 ), c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P5' ].contourf( np.exp( yv ), np.exp( xv ), PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
#im2 = axd[ 'P5' ].scatter( np.exp( flux1 ), np.exp( pres1 ), c=plahab, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
#im2 = axd[ 'P5' ].contourf( np.exp( yv ), xv, PlaHab_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
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
# Panel 5

im2 = axd[ 'P3' ].contourf( yv*fluxscale, xv, pcm_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im2 = axd[ 'P3' ].scatter( pcm_flux1*fluxscale, pcm_pres1, c=pcm, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
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
# Panel 6

im2 = axd[ 'P6' ].contourf( yv*fluxscale, xv, lfric_z1, cmap=cm, levels=np.linspace(200,contourmax,cinterval), extend='both' )
im2 = axd[ 'P6' ].scatter( lfric_flux1*fluxscale, lfric_pres1, c=lfric, cmap=cm, vmin=200, vmax=contourmax, marker='o', s=70, facecolors='k', edgecolors='k' )
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
cb3 = fig.colorbar( im2, ax=axd[ 'P3' ], extend='both', ticks=np.arange( 200, contourmax + 50, 50 ) )
cb4 = fig.colorbar( im1, ax=axd[ 'P4' ], extend='both' )
cb5 = fig.colorbar( im2, ax=axd[ 'P5' ], extend='both')
cb6 = fig.colorbar( im2, ax=axd[ 'P6' ], extend='both', ticks=np.arange( 200, contourmax + 50, 50 ) )

cb1.ax.get_yaxis().labelpad = 15
cb1.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb2.ax.get_yaxis().labelpad = 15
cb2.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb3.ax.get_yaxis().labelpad = 15
cb3.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb5.ax.get_yaxis().labelpad = 15
cb5.set_label( 'Average Surface Temperature (K)', rotation=270 )
cb6.ax.get_yaxis().labelpad = 15
cb6.set_label( 'Average Surface Temperature (K)', rotation=270 )

fig.subplots_adjust( wspace = 0.33, hspace = 0.33 )

fig.savefig( "fig_interpolation_temp.png", bbox_inches='tight' )
fig.savefig( "fig_interpolation_temp.eps", bbox_inches='tight' )
#plt.show()
