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

Ts_plasim4 = np.abs( np.average( f1.variables[ 'rlut' ], axis=0 ) )
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

Ts_exocam4 = np.average( f2.variables[ 'FLUT' ], axis=0 )
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
#f3 = netCDF4.Dataset( '/models/data/samosa/genericpcm/SAMOSA_output_file_Generic_PCM_case-4_OHT_on.nc' )
f3 = netCDF4.Dataset( '/models/data/samosa/genericpcm/SAMOSA_output_file_Generic_PCM_case-4_OHT_off.nc' )

Ts_pcm4 = np.array( f3.variables[ 'outgoing_longwave_radiation' ] )
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
# Read ROCKE-3D data
f4 = netCDF4.Dataset( '/models/data/samosa/rocke3d/rocke_04q.nc' )

Ts_r3d4 = np.abs( np.array( f4.variables[ 'trnf_toa' ] ) )
lat_r3d = np.array( f4.variables[ 'lat' ] )
lon_r3d = np.array( f4.variables[ 'lon' ] )

Ts_r3d4, lon_r3d = shiftgrid( 0, Ts_r3d4, lon_r3d, start=False )

lats_r3d = np.zeros( lat_r3d.size * lon_r3d.size )
lons_r3d = np.zeros( lat_r3d.size * lon_r3d.size )
Tss_r3d  = np.zeros( lat_r3d.size * lon_r3d.size )
count = 0

for i in range( 0, lon_r3d.size ):
	for j in range( 0, lat_r3d.size ):
		lons_r3d[ count ] = lon_r3d[i]
		lats_r3d[ count ] = lat_r3d[j]
		Tss_r3d[  count ] = Ts_r3d4[j,i]
		count += 1

#--------------------------------------------------------------------
# Read PlaHab data
f5 = open( '/models/data/samosa/plahab/case4_OLR.out' )

lon_plahab = np.array( [ -171., -153. , -135. , -117., -99., -81.,  -63.,  -45.,  -27., -9., 9.,  27.0 , 45., 63., 81., 99., 117., 135., 153., 171. ] )
lat_plahab = np.array( [ -88., -82., -77., -72., -68., -62., -58., -52., -47., -43., -37., -32., -28., -23., -18., -13., -7., -2., 2., 7., 13., 18., 23., 28., 32., 37., 43., 47., 52., 58., 62., 68., 72., 77., 82., 88. ] )
Ts_plahab4 = np.zeros( ( lat_plahab.size, lon_plahab.size ) )

lats_plahab = np.zeros( lat_plahab.size * lon_plahab.size )
lons_plahab = np.zeros( lat_plahab.size * lon_plahab.size )
Tss_plahab  = np.zeros( lat_plahab.size * lon_plahab.size )
count = 0
j = 0

for line in f5.readlines():
	line = line.strip()
	columns = line.split()
	for i in range( 1, lon_plahab.size + 1 ):
		lons_plahab[ count ] = lon_plahab[i-1]
		lats_plahab[ count ] = columns[0]
		Tss_plahab[  count ] = columns[i]
		Ts_plahab4[ j, i-1 ] = columns[i]
		count += 1
	j += 1

#--------------------------------------------------------------------
# Read LFRic data
f6 = netCDF4.Dataset( '/models/data/samosa/lfric/samosa_case04__lfric_c24.nc' )

Ts_lfric4 = f6.variables[ 'lw_up_toa' ]
lat_lfric = np.array( f6.variables[ 'lat' ] )
lon_lfric = np.array( f6.variables[ 'lon' ] )
nlev_lfric = len( np.array( f6.variables[ 'level_height' ] ) )

Ts_lfric4, lon_lfric = shiftgrid( 180., Ts_lfric4, lon_lfric, start=False )

lats_lfric = np.zeros( lat_lfric.size * lon_lfric.size )
lons_lfric = np.zeros( lat_lfric.size * lon_lfric.size )
Tss_lfric  = np.zeros( lat_lfric.size * lon_lfric.size )
count = 0

for i in range( 0, lon_lfric.size ):
	for j in range( 0, lat_lfric.size ):
		lons_lfric[ count ] = lon_lfric[i]
		lats_lfric[ count ] = lat_lfric[j]
		Tss_lfric[  count ] = Ts_lfric4[j,i]
		count += 1

#--------------------------------------------------------------------
# Set Up Figure

cm = mpl.colormaps.get_cmap('cool')
contourmax  = 300.0
contourmin  = 150.0

fig, axd = plt.subplot_mosaic([['P1', 'P2'],
                               ['P3', 'P4'],
                               ['P5', 'P6']],
                              figsize=(15.5, 7.125))

#--------------------------------------------------------------------
# Panel 1

im1 = axd[ 'P1' ].contourf( lon_exocam, lat_exocam, Ts_exocam4, cmap=cm, levels=np.linspace(contourmin,contourmax,20), extend='both' )
#axd[ 'P1' ].set( title='ExoCAM' )
axd[ 'P1' ].set_title( 'ExoCAM', fontsize=15 )
axd[ 'P1' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'P1' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'P1' ].set_xticks( [ 90, 180, 270 ], labels=[] )
axd[ 'P1' ].set_yticks( [ -45, 0, 45 ], labels=[] )


#--------------------------------------------------------------------
# Panel 2

im2 = axd[ 'P3' ].contourf( lon_plasim, lat_plasim, Ts_plasim4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )
axd[ 'P3' ].set_title( 'ExoPlaSim', fontsize=15 )
axd[ 'P3' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'P3' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'P3' ].set_xticks( [ -90, 0, 90 ], labels=[] )
axd[ 'P3' ].set_yticks( [ -45, 0, 45 ], labels=[] )

#--------------------------------------------------------------------
# Panel 3

im3 = axd[ 'P2' ].contourf( lon_pcm, lat_pcm, Ts_pcm4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )
#axd[ 'P2' ].set_title( 'Generic PCM (with OHT)', fontsize=15 )
axd[ 'P2' ].set_title( 'Generic PCM (no OHT)', fontsize=15 )
axd[ 'P2' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'P2' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'P2' ].set_xticks( [ -90, 0, 90 ], labels=[] )
axd[ 'P2' ].set_yticks( [ -45, 0, 45 ], labels=[] )

#--------------------------------------------------------------------
# Panel 4

im4 = axd[ 'P4' ].contourf( lon_r3d, lat_r3d, Ts_r3d4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )
axd[ 'P4' ].set_title( 'ROCKE-3D', fontsize=15 )
axd[ 'P4' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'P4' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'P4' ].set_xticks( [ -90, 0, 90 ], labels=[] )
axd[ 'P4' ].set_yticks( [ -45, 0, 45 ], labels=[] )
axd[ 'P4' ].set_xlim( [ -5, -365 ] )
axd[ 'P4' ].set_ylim( [ -90,  90 ] )

#--------------------------------------------------------------------
# Panel 5

im5 = axd[ 'P5' ].contourf( lon_plahab, lat_plahab, Ts_plahab4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )
axd[ 'P5' ].set_title( 'PlaHab', fontsize=15 )
axd[ 'P5' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'P5' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'P5' ].set_xticks( [ -90, 0, 90 ], labels=[] )
axd[ 'P5' ].set_yticks( [ -45, 0, 45 ], labels=[] )

#--------------------------------------------------------------------
# Panel 6

im6 = axd[ 'P6' ].contourf( lon_lfric, lat_lfric, Ts_lfric4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,20), extend='both' )
axd[ 'P6' ].set_title( 'LFRic', fontsize=15 )
axd[ 'P6' ].set_xlabel( 'Longitude', fontsize = 10 )
axd[ 'P6' ].set_ylabel( 'Latitude', fontsize = 10 )
axd[ 'P6' ].set_xticks( [ -90, 0, 90 ], labels=[] )
axd[ 'P6' ].set_yticks( [ -45, 0, 45 ], labels=[] )

#--------------------------------------------------------------------
# ZEROVAL Panel
#ZEROVALS = np.zeros( ( np.size( lat_pcm ), np.size( lon_pcm ) ) )
#im6 = axd[ 'P6' ].contourf( lon_pcm, lat_pcm, ZEROVAL

#--------------------------------------------------------------------
# Finalize

cb1 = fig.colorbar( im1, ax=axd[ 'P1' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb2 = fig.colorbar( im2, ax=axd[ 'P3' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb3 = fig.colorbar( im3, ax=axd[ 'P2' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb4 = fig.colorbar( im4, ax=axd[ 'P4' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb5 = fig.colorbar( im5, ax=axd[ 'P5' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )
cb6 = fig.colorbar( im6, ax=axd[ 'P6' ], extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ) )

cb1.ax.get_yaxis().labelpad = 15
cb1.set_label( 'OLR (W m$^{-2}$)', rotation=270 )
cb2.ax.get_yaxis().labelpad = 15
cb2.set_label( 'OLR (W m$^{-2}$)', rotation=270 )
cb3.ax.get_yaxis().labelpad = 15
cb3.set_label( 'OLR (W m$^{-2}$)', rotation=270 )
cb4.ax.get_yaxis().labelpad = 15
cb4.set_label( 'OLR (W m$^{-2}$)', rotation=270 )
cb5.ax.get_yaxis().labelpad = 15
cb5.set_label( 'OLR (W m$^{-2}$)', rotation=270 )
cb6.ax.get_yaxis().labelpad = 15
cb6.set_label( 'OLR (W m$^{-2}$)', rotation=270 )

fig.subplots_adjust( hspace = 0.50 )
fig.suptitle( "CASE 4: S = 1200 W m$^{-2}$, p = 2.34 bar", fontsize=15 )

fig.savefig( "fig_compare_OLR.png", bbox_inches='tight' )
fig.savefig( "fig_compare_OLR.eps", bbox_inches='tight' )
#plt.show()
