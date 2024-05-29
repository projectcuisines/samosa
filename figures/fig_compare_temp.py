#
# Comparison for SAMOSA Case 4: Temperature
#
import netCDF4
from mpl_toolkits.basemap import Basemap, shiftgrid

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from scipy.interpolate import CubicSpline

#--------------------------------------------------------------------
# Read ExoPlaSim data
f1 = netCDF4.Dataset( '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day/t21_synchronous_2.34pn2_flux1200_400.0co2_3000teff_15day.nc' )

Ts_plasim4 = np.average( f1.variables[ 'ts' ], axis=0 )
lat_plasim = np.array( f1.variables[ 'lat' ] )
lon_plasim = np.array( f1.variables[ 'lon' ] )

Ts_plasim4, lon_plasim = shiftgrid( 180., Ts_plasim4, lon_plasim, start=False )

lats_plasim = np.zeros( lat_plasim.size * lon_plasim.size )
lons_plasim = np.zeros( lat_plasim.size * lon_plasim.size )
Tss_plasim  = np.zeros( lat_plasim.size * lon_plasim.size )
count = 0

for i in range( 0, lon_plasim.size ):
	for j in range( 0, lat_plasim.size ):
		lons_plasim[ count ] = lon_plasim[i]
		lats_plasim[ count ] = lat_plasim[j]
		Tss_plasim[  count ] = Ts_plasim4[j,i]
		count += 1

#--------------------------------------------------------------------
# Read ExoCAM data
f2 = netCDF4.Dataset( '/models/data/samosa/exocam/samosa4.cam.h0.avg.nc' )

Ts_exocam4 = np.average( f2.variables[ 'TS' ], axis=0 )
lat_exocam = np.array( f2.variables[ 'lat' ] )
lon_exocam = np.array( f2.variables[ 'lon' ] )

lats_exocam = np.zeros( lat_exocam.size * lon_exocam.size )
lons_exocam = np.zeros( lat_exocam.size * lon_exocam.size )
Tss_exocam  = np.zeros( lat_exocam.size * lon_exocam.size )
count = 0

for i in range( 0, lon_exocam.size ):
	for j in range( 0, lat_exocam.size ):
		lons_exocam[ count ] = lon_exocam[i]
		lats_exocam[ count ] = lat_exocam[j]
		Tss_exocam[  count ] = Ts_exocam4[j,i]
		count += 1

#--------------------------------------------------------------------
# Read Generic PCM data
f3 = netCDF4.Dataset( '/home/jacob/research/samosa/data/martin/SAMOSA_output_file_Generic_PCM_case-4_static_ocean_TESTCASE.nc' )

#Ts_pcm4 = np.average( f3.variables[ 'surface_temperature' ], axis=0 )
Ts_pcm4 = np.array( f3.variables[ 'surface_temperature' ] )
lat_pcm = np.array( f3.variables[ 'latitude' ] )
lon_pcm = np.array( f3.variables[ 'longitude' ] )

lats_pcm = np.zeros( lat_pcm.size * lon_pcm.size )
lons_pcm = np.zeros( lat_pcm.size * lon_pcm.size )
Tss_pcm  = np.zeros( lat_pcm.size * lon_pcm.size )
count = 0

for i in range( 0, lon_pcm.size ):
	for j in range( 0, lat_pcm.size ):
		lons_pcm[ count ] = lon_pcm[i]
		lats_pcm[ count ] = lat_pcm[j]
		Tss_pcm[  count ] = Ts_pcm4[j,i]
		count += 1

#--------------------------------------------------------------------
# Set Up Figure

cm = mpl.colormaps.get_cmap('cool')
contourmax  = 300.0
contourmin  = 200.0

fig, axd = plt.subplot_mosaic([['upper left', 'upper right'],
                               ['lower left', 'lower right']],
                              figsize=(15.5, 4.75))

#--------------------------------------------------------------------
# Panel 1

im1 = axd[ 'upper left' ].contourf( lon_exocam, lat_exocam, Ts_exocam4, cmap=cm, levels=np.linspace(contourmin,contourmax,20), extend='both' )
#axd[ 'upper left' ].set( title='ExoCAM' )
axd[ 'upper left' ].set_title( 'ExoCAM', fontsize=15 )
axd[ 'upper left' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'upper left' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'upper left' ].set_xticks( [ 90, 180, 270 ], labels=[] )
axd[ 'upper left' ].set_yticks( [ -45, 0, 45 ], labels=[] )


#--------------------------------------------------------------------
# Panel 2

im2 = axd[ 'lower left' ].contourf( lon_plasim, lat_plasim, Ts_plasim4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )
axd[ 'lower left' ].set_title( 'ExoPlaSim', fontsize=15 )
axd[ 'lower left' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'lower left' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'lower left' ].set_xticks( [ -90, 0, 90 ], labels=[] )
axd[ 'lower left' ].set_yticks( [ -45, 0, 45 ], labels=[] )

#--------------------------------------------------------------------
# Panel 3

im3 = axd[ 'upper right' ].contourf( lon_pcm, lat_pcm, Ts_pcm4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )
axd[ 'upper right' ].set_title( 'Generic PCM', fontsize=15 )
axd[ 'upper right' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'upper right' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'upper right' ].set_xticks( [ -90, 0, 90 ], labels=[] )
axd[ 'upper right' ].set_yticks( [ -45, 0, 45 ], labels=[] )

#--------------------------------------------------------------------
# Panel 4

ZEROVALS = np.zeros( ( np.size( lat_pcm ), np.size( lon_pcm ) ) )
im4 = axd[ 'lower right' ].contourf( lon_pcm, lat_pcm, ZEROVALS, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )

#--------------------------------------------------------------------
# Finalize

cb1 = fig.colorbar( im1, ax=axd[ 'upper left' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb2 = fig.colorbar( im2, ax=axd[ 'lower left' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb3 = fig.colorbar( im3, ax=axd[ 'upper right' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb4 = fig.colorbar( im4, ax=axd[ 'lower right' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )

cb1.ax.get_yaxis().labelpad = 15
cb1.set_label( 'Temperature (K)', rotation=270 )
cb2.ax.get_yaxis().labelpad = 15
cb2.set_label( 'Temperature (K)', rotation=270 )
cb3.ax.get_yaxis().labelpad = 15
cb3.set_label( 'Temperature (K)', rotation=270 )
cb4.ax.get_yaxis().labelpad = 15
cb4.set_label( 'Temperature (K)', rotation=270 )

fig.subplots_adjust( hspace = 0.50 )

fig.savefig( "fig_compare_temp.png", bbox_inches='tight' )
fig.savefig( "fig_compare_temp.eps", bbox_inches='tight' )
#plt.show()
