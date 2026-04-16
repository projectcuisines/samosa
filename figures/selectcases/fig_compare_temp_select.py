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

tfreeze = 273.16

#--------------------------------------------------------------------
# Read ExoPlaSim data
f_exoplasim1 = netCDF4.Dataset( '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day/t21_synchronous_0.70pn2_flux500_400.0co2_3000teff_15day.nc' )
f_exoplasim4 = netCDF4.Dataset( '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day/t21_synchronous_2.34pn2_flux1200_400.0co2_3000teff_15day.nc' )
f_exoplasim16 = netCDF4.Dataset( '/models/data/samosa/exoplasim/full_t21_synchronous__3000teff_15day/t21_synchronous_10.00pn2_flux1400_400.0co2_3000teff_15day.nc' )

Ts_plasim1  = np.average( f_exoplasim1.variables[ 'ts' ], axis=0 )
Ts_plasim4  = np.average( f_exoplasim4.variables[ 'ts' ], axis=0 )
Ts_plasim16 = np.average( f_exoplasim16.variables[ 'ts' ], axis=0 )

lat_plasim = np.array( f_exoplasim4.variables[ 'lat' ] )
lon_plasim = np.array( f_exoplasim4.variables[ 'lon' ] )

Ts_plasim1, lon_plasim1   = shiftgrid( 180., Ts_plasim1, lon_plasim, start=False )
Ts_plasim4, lon_plasim4   = shiftgrid( 180., Ts_plasim4, lon_plasim, start=False )
Ts_plasim16, lon_plasim16 = shiftgrid( 180., Ts_plasim16, lon_plasim, start=False )

lats_plasim  = np.zeros( lat_plasim.size * lon_plasim4.size )
lons_plasim  = np.zeros( lat_plasim.size * lon_plasim4.size )
Tss_plasim1  = np.zeros( lat_plasim.size * lon_plasim4.size )
Tss_plasim4  = np.zeros( lat_plasim.size * lon_plasim4.size )
Tss_plasim16 = np.zeros( lat_plasim.size * lon_plasim4.size )
count = 0

for i in range( 0, lon_plasim.size ):
	for j in range( 0, lat_plasim.size ):
		lons_plasim[ count ]  = lon_plasim4[i]
		lats_plasim[ count ]  = lat_plasim[j]
		Tss_plasim1[  count ] = Ts_plasim1[j,i]
		Tss_plasim4[  count ] = Ts_plasim4[j,i]
		Tss_plasim16[ count ] = Ts_plasim16[j,i]
		count += 1

#--------------------------------------------------------------------
# Read ExoCAM data
f_exocam1  = netCDF4.Dataset( '/models/data/samosa/exocam/samosa1.cam.h0.avg.nc' )
f_exocam4  = netCDF4.Dataset( '/models/data/samosa/exocam/samosa4.cam.h0.avg.nc' )
f_exocam16 = netCDF4.Dataset( '/models/data/samosa/exocam/samosa16.cam.h0.avg.nc' )

Ts_exocam1  = np.average( f_exocam1.variables[ 'TS' ], axis=0 )
Ts_exocam4  = np.average( f_exocam4.variables[ 'TS' ], axis=0 )
Ts_exocam16 = np.average( f_exocam16.variables[ 'TS' ], axis=0 )

lat_exocam  = np.array( f_exocam4.variables[ 'lat' ] )
lon_exocam  = np.array( f_exocam4.variables[ 'lon' ] )

lats_exocam  = np.zeros( lat_exocam.size * lon_exocam.size )
lons_exocam  = np.zeros( lat_exocam.size * lon_exocam.size )
Tss_exocam1  = np.zeros( lat_exocam.size * lon_exocam.size )
Tss_exocam4  = np.zeros( lat_exocam.size * lon_exocam.size )
Tss_exocam16 = np.zeros( lat_exocam.size * lon_exocam.size )
count = 0

for i in range( 0, lon_exocam.size ):
	for j in range( 0, lat_exocam.size ):
		lons_exocam[ count ]   = lon_exocam[i]
		lats_exocam[ count ]   = lat_exocam[j]
		Tss_exocam1[  count ]  = Ts_exocam1[j,i]
		Tss_exocam4[  count ]  = Ts_exocam4[j,i]
		Tss_exocam16[  count ] = Ts_exocam16[j,i]
		count += 1

#--------------------------------------------------------------------
# Read Generic PCM data (with OHT)
#f3 = netCDF4.Dataset( '/models/data/samosa/genericpcm/SAMOSA_output_file_Generic_PCM_case-4_OHT_on.nc' )
#
#Ts_pcm4 = np.array( f3.variables[ 'surface_temperature' ] )
#lat_pcm = np.array( f3.variables[ 'latitude' ] )
#lon_pcm = np.array( f3.variables[ 'longitude' ] )
#
#lats_pcm = np.zeros( lat_pcm.size * lon_pcm.size )
#lons_pcm = np.zeros( lat_pcm.size * lon_pcm.size )
#Tss_pcm  = np.zeros( lat_pcm.size * lon_pcm.size )
#count = 0
#
#for i in range( 0, lon_pcm.size ):
#	for j in range( 0, lat_pcm.size ):
#		lons_pcm[ count ] = lon_pcm[i]
#		lats_pcm[ count ] = lat_pcm[j]
#		Tss_pcm[  count ] = Ts_pcm4[j,i]
#		count += 1

#--------------------------------------------------------------------
# Read Generic PCM data (without OHT)
f_PCM_noOHT1 = netCDF4.Dataset( '/models/data/samosa/genericpcm/OHT_off/case-1/SAMOSA_output_file_Generic_PCM_case-1_OHT_off.nc' )
f_PCM_noOHT4 = netCDF4.Dataset( '/models/data/samosa/genericpcm/OHT_off/case-4/SAMOSA_output_file_Generic_PCM_case-4_OHT_off.nc' )
f_PCM_noOHT16 = netCDF4.Dataset( '/models/data/samosa/genericpcm/OHT_off/case-16/SAMOSA_output_file_Generic_PCM_case-16_OHT_off.nc' )

Ts_pcm_noOHT1  = np.array( f_PCM_noOHT1.variables[ 'surface_temperature' ] )
Ts_pcm_noOHT4  = np.array( f_PCM_noOHT4.variables[ 'surface_temperature' ] )
Ts_pcm_noOHT16  = np.array( f_PCM_noOHT16.variables[ 'surface_temperature' ] )
lat_pcm = np.array( f_PCM_noOHT4.variables[ 'latitude' ] )
lon_pcm = np.array( f_PCM_noOHT4.variables[ 'longitude' ] )

lats_pcm = np.zeros( lat_pcm.size * lon_pcm.size )
lons_pcm = np.zeros( lat_pcm.size * lon_pcm.size )
Tss_pcm_noOHT1 = np.zeros( lat_pcm.size * lon_pcm.size )
Tss_pcm_noOHT4 = np.zeros( lat_pcm.size * lon_pcm.size )
Tss_pcm_noOHT16 = np.zeros( lat_pcm.size * lon_pcm.size )
count = 0

for i in range( 0, lon_pcm.size ):
	for j in range( 0, lat_pcm.size ):
		lons_pcm[ count ] = lon_pcm[i]
		lats_pcm[ count ] = lat_pcm[j]
		Tss_pcm_noOHT1[  count ] = Ts_pcm_noOHT1[j,i]
		Tss_pcm_noOHT4[  count ] = Ts_pcm_noOHT4[j,i]
		Tss_pcm_noOHT16[  count ] = Ts_pcm_noOHT16[j,i]
		count += 1

#--------------------------------------------------------------------
# Read ROCKE-3D data
f_r3d1  = netCDF4.Dataset( '/models/data/samosa/rocke3d/rocke_01q.nc' )
f_r3d4  = netCDF4.Dataset( '/models/data/samosa/rocke3d/rocke_04q.nc' )
f_r3d16 = netCDF4.Dataset( '/models/data/samosa/rocke3d/rocke_16q.nc' )

Ts_r3d1  = np.array( f_r3d1.variables[ 'tsurf' ] ) + tfreeze
Ts_r3d4  = np.array( f_r3d4.variables[ 'tsurf' ] ) + tfreeze
Ts_r3d16 = np.array( f_r3d16.variables[ 'tsurf' ] ) + tfreeze

lat_r3d = np.array( f_r3d4.variables[ 'lat' ] )
lon_r3d = np.array( f_r3d4.variables[ 'lon' ] )

Ts_r3d1, lon_r3d1   = shiftgrid( 0, Ts_r3d1, lon_r3d, start=False )
Ts_r3d4, lon_r3d4   = shiftgrid( 0, Ts_r3d4, lon_r3d, start=False )
Ts_r3d16, lon_r3d16 = shiftgrid( 0, Ts_r3d16, lon_r3d, start=False )

lats_r3d  = np.zeros( lat_r3d.size * lon_r3d4.size )
lons_r3d  = np.zeros( lat_r3d.size * lon_r3d4.size )
Tss_r3d1  = np.zeros( lat_r3d.size * lon_r3d4.size )
Tss_r3d4  = np.zeros( lat_r3d.size * lon_r3d4.size )
Tss_r3d16 = np.zeros( lat_r3d.size * lon_r3d4.size )
count = 0

for i in range( 0, lon_r3d.size ):
	for j in range( 0, lat_r3d.size ):
		lons_r3d[ count ]  = lon_r3d4[i]
		lats_r3d[ count ]  = lat_r3d[j]
		Tss_r3d1[  count ] = Ts_r3d1[j,i]
		Tss_r3d4[  count ] = Ts_r3d4[j,i]
		Tss_r3d16[ count ] = Ts_r3d16[j,i]
		count += 1

#--------------------------------------------------------------------
# Read PlaHab data
f_plahab1  = open( '/models/data/samosa/plahab/simulations/sample1/case1_tsurf.out' )
f_plahab4  = open( '/models/data/samosa/plahab/simulations/sample4/case4_tsurf.out' )
f_plahab16 = open( '/models/data/samosa/plahab/simulations/sample16/case16_tsurf.out' )

lon_plahab  = np.array( [ -171., -153. , -135. , -117., -99., -81.,  -63.,  -45.,  -27., -9., 9.,  27.0 , 45., 63., 81., 99., 117., 135., 153., 171. ] )
lat_plahab  = np.array( [ -88., -82., -77., -72., -68., -62., -58., -52., -47., -43., -37., -32., -28., -23., -18., -13., -7., -2., 2., 7., 13., 18., 23., 28., 32., 37., 43., 47., 52., 58., 62., 68., 72., 77., 82., 88. ] )
Ts_plahab1  = np.zeros( ( lat_plahab.size, lon_plahab.size ) )
Ts_plahab4  = np.zeros( ( lat_plahab.size, lon_plahab.size ) )
Ts_plahab16 = np.zeros( ( lat_plahab.size, lon_plahab.size ) )

lats_plahab  = np.zeros( lat_plahab.size * lon_plahab.size )
lons_plahab  = np.zeros( lat_plahab.size * lon_plahab.size )
Tss_plahab1  = np.zeros( lat_plahab.size * lon_plahab.size )
Tss_plahab4  = np.zeros( lat_plahab.size * lon_plahab.size )
Tss_plahab16 = np.zeros( lat_plahab.size * lon_plahab.size )

count = 0
j = 0
for line in f_plahab4.readlines():
	line = line.strip()
	columns = line.split()
	for i in range( 1, lon_plahab.size + 1 ):
		lons_plahab[ count ] = lon_plahab[i-1]
		lats_plahab[ count ] = columns[0]
		Tss_plahab4[  count ] = columns[i]
		Ts_plahab4[ j, i-1 ] = columns[i]
		count += 1
	j += 1

count = 0
j = 0
for line in f_plahab1.readlines():
	line = line.strip()
	columns = line.split()
	for i in range( 1, lon_plahab.size + 1 ):
		Tss_plahab1[  count ] = columns[i]
		Ts_plahab1[ j, i-1 ] = columns[i]
		count += 1
	j += 1

count = 0
j = 0
for line in f_plahab16.readlines():
	line = line.strip()
	columns = line.split()
	for i in range( 1, lon_plahab.size + 1 ):
		Tss_plahab16[  count ] = columns[i]
		Ts_plahab16[ j, i-1 ] = columns[i]
		count += 1
	j += 1

#--------------------------------------------------------------------
# Read LFRic data
f_lfric1 = netCDF4.Dataset( '/models/data/samosa/lfric/lfric_samosa_case01.nc' )
f_lfric4 = netCDF4.Dataset( '/models/data/samosa/lfric/lfric_samosa_case04.nc' )

Ts_lfric1 = f_lfric1.variables[ 'grid_surface_temperature' ]
Ts_lfric4 = f_lfric4.variables[ 'grid_surface_temperature' ]
lat_lfric = np.array( f_lfric4.variables[ 'lat' ] )
lon_lfric = np.array( f_lfric4.variables[ 'lon' ] )

Ts_lfric1, lon_lfric1 = shiftgrid( 180., Ts_lfric1, lon_lfric, start=False )
Ts_lfric4, lon_lfric4 = shiftgrid( 180., Ts_lfric4, lon_lfric, start=False )

lats_lfric = np.zeros( lat_lfric.size * lon_lfric4.size )
lons_lfric = np.zeros( lat_lfric.size * lon_lfric4.size )
Tss_lfric1  = np.zeros( lat_lfric.size * lon_lfric4.size )
Tss_lfric4  = np.zeros( lat_lfric.size * lon_lfric4.size )
count = 0

for i in range( 0, lon_lfric4.size ):
	for j in range( 0, lat_lfric.size ):
		lons_lfric[ count ] = lon_lfric4[i]
		lats_lfric[ count ] = lat_lfric[j]
		Tss_lfric1[  count ] = Ts_lfric4[j,i]
		Tss_lfric4[  count ] = Ts_lfric4[j,i]
		count += 1


#--------------------------------------------------------------------
# Set Up Figure

cm = mpl.colormaps.get_cmap( 'plasma' )
#cm = mpl.colormaps.get_cmap('cool')
contourmax  = 375.0
contourmin  = 175.0
numcontours = 25

#fig, axd = plt.subplot_mosaic( [ [ 'P1.1', 'P2.1', 'P3.1' 'P4.1', 'P5.1', 'P6.1' ],
#                                 [ 'P1.2', 'P2.2', 'P3.2' 'P4.2', 'P5.2', 'P6.2' ],
#                                 [ 'P1.3', 'P2.3', 'P3.3' 'P4.3', 'P5.3', 'P6.3' ] ] )

fig = plt.figure( layout="constrained", figsize=(18, 6) )
ax_array = fig.subplots( 3, 7, squeeze="False" )

#--------------------------------------------------------------------
# Column 1: ExoCAM

im1 = ax_array[0, 0].contourf( lon_exocam, lat_exocam, Ts_exocam1, cmap=cm, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[0, 0].set_title( 'ExoCAM', fontsize=15 )
ax_array[0, 0].set_xlabel( '', fontsize = 10 )
ax_array[0, 0].set_ylabel( 'CASE 1', fontsize = 15 )
#ax_array[0, 0].set_xticks( [ 90, 180, 270 ], labels=[] )
#ax_array[0, 0].set_yticks( [ -45, 0, 45 ], labels=[] )
ax_array[0, 0].set_xticks( [ ], labels=[] )
ax_array[0, 0].set_yticks( [ ], labels=[] )

im1 = ax_array[1, 0].contourf( lon_exocam, lat_exocam, Ts_exocam4, cmap=cm, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[1, 0].set_title( '', fontsize=15 )
ax_array[1, 0].set_xlabel( '', fontsize = 10 )
ax_array[1, 0].set_ylabel( 'CASE 4', fontsize = 15 )
ax_array[1, 0].set_xticks( [ ], labels=[] )
ax_array[1, 0].set_yticks( [ ], labels=[] )

im1 = ax_array[2, 0].contourf( lon_exocam, lat_exocam, Ts_exocam16, cmap=cm, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[2, 0].set_title( '', fontsize=15 )
ax_array[2, 0].set_xlabel( '', fontsize = 10 )
ax_array[2, 0].set_ylabel( 'CASE 16', fontsize = 15 )
ax_array[2, 0].set_xticks( [ ], labels=[] )
ax_array[2, 0].set_yticks( [ ], labels=[] )


#--------------------------------------------------------------------
# Column 2: ExoPlaSim

im2 = ax_array[0, 1].contourf( lon_plasim1, lat_plasim, Ts_plasim1, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[0, 1].set_title( 'ExoPlaSim', fontsize=15 )
ax_array[0, 1].set_xlabel( '', fontsize = 10 )
ax_array[0, 1].set_ylabel( '', fontsize = 10 )
ax_array[0, 1].set_xticks( [ ], labels=[] )
ax_array[0, 1].set_yticks( [ ], labels=[] )

im2 = ax_array[1, 1].contourf( lon_plasim4, lat_plasim, Ts_plasim4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[1, 1].set_title( '', fontsize=15 )
ax_array[1, 1].set_xlabel( '', fontsize = 10 )
ax_array[1, 1].set_ylabel( '', fontsize = 10 )
ax_array[1, 1].set_xticks( [ ], labels=[] )
ax_array[1, 1].set_yticks( [ ], labels=[] )

im2 = ax_array[2, 1].contourf( lon_plasim16, lat_plasim, Ts_plasim16, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[2, 1].set_title( '', fontsize=15 )
ax_array[2, 1].set_xlabel( '', fontsize = 10 )
ax_array[2, 1].set_ylabel( '', fontsize = 10 )
ax_array[2, 1].set_xticks( [ ], labels=[] )
ax_array[2, 1].set_yticks( [ ], labels=[] )

#--------------------------------------------------------------------
# Column 3: ROCKE-3D

im4 = ax_array[0, 2].contourf( lon_r3d, lat_r3d, Ts_r3d1, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[0, 2].set_title( 'ROCKE-3D', fontsize=15 )
ax_array[0, 2].set_xlabel( '', fontsize = 10 )
ax_array[0, 2].set_ylabel( '', fontsize = 10 )
ax_array[0, 2].set_xticks( [ ], labels=[] )
ax_array[0, 2].set_yticks( [ ], labels=[] )

im4 = ax_array[1, 2].contourf( lon_r3d, lat_r3d, Ts_r3d4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[1, 2].set_title( '', fontsize=15 )
ax_array[1, 2].set_xlabel( '', fontsize = 10 )
ax_array[1, 2].set_ylabel( '', fontsize = 10 )
ax_array[1, 2].set_xticks( [ ], labels=[] )
ax_array[1, 2].set_yticks( [ ], labels=[] )

im4 = ax_array[2, 2].contourf( lon_r3d, lat_r3d, Ts_r3d16, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[2, 2].set_title( '', fontsize=15 )
ax_array[2, 2].set_xlabel( '', fontsize = 10 )
ax_array[2, 2].set_ylabel( '', fontsize = 10 )
ax_array[2, 2].set_xticks( [ ], labels=[] )
ax_array[2, 2].set_yticks( [ ], labels=[] )

#--------------------------------------------------------------------
# Column 4: PlaHab

im5 = ax_array[0, 3].contourf( lon_plahab, lat_plahab, Ts_plahab1, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[0, 3].set_title( 'PlaHab', fontsize=15 )
ax_array[0, 3].set_xlabel( '', fontsize = 10 )
ax_array[0, 3].set_ylabel( '', fontsize = 10 )
ax_array[0, 3].set_xticks( [ ], labels=[] )
ax_array[0, 3].set_yticks( [ ], labels=[] )

im5 = ax_array[1, 3].contourf( lon_plahab, lat_plahab, Ts_plahab4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[1, 3].set_title( '', fontsize=15 )
ax_array[1, 3].set_xlabel( '', fontsize = 10 )
ax_array[1, 3].set_ylabel( '', fontsize = 10 )
ax_array[1, 3].set_xticks( [ ], labels=[] )
ax_array[1, 3].set_yticks( [ ], labels=[] )

im5 = ax_array[2, 3].contourf( lon_plahab, lat_plahab, Ts_plahab16, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[2, 3].set_title( '', fontsize=15 )
ax_array[2, 3].set_xlabel( '', fontsize = 10 )
ax_array[2, 3].set_ylabel( '', fontsize = 10 )
ax_array[2, 3].set_xticks( [ ], labels=[] )
ax_array[2, 3].set_yticks( [ ], labels=[] )

#--------------------------------------------------------------------
# Column 5: LFRic

im6 = ax_array[0, 4].contourf( lon_lfric4, lat_lfric, Ts_lfric1, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[0, 4].set_title( 'LFRic', fontsize=15 )
ax_array[0, 4].set_xlabel( '', fontsize = 10 )
ax_array[0, 4].set_ylabel( '', fontsize = 10 )
ax_array[0, 4].set_xticks( [ ], labels=[] )
ax_array[0, 4].set_yticks( [ ], labels=[] )

im6 = ax_array[1, 4].contourf( lon_lfric4, lat_lfric, Ts_lfric4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[1, 4].set_title( '', fontsize=15 )
ax_array[1, 4].set_xlabel( '', fontsize = 10 )
ax_array[1, 4].set_ylabel( '', fontsize = 10 )
ax_array[1, 4].set_xticks( [ ], labels=[] )
ax_array[1, 4].set_yticks( [ ], labels=[] )

ZEROVALS = np.zeros( ( np.size( lat_pcm ), np.size( lon_pcm ) ) )
im6 = ax_array[2, 4].contourf( lon_pcm, lat_pcm, ZEROVALS, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[2, 4].set_title( '', fontsize=15 )
ax_array[2, 4].set_xticks( [ ], labels=[] )
ax_array[2, 4].set_yticks( [ ], labels=[] )

#--------------------------------------------------------------------
# Column 6: Generic PCM (no OHT)

im3 = ax_array[0, 5].contourf( lon_pcm, lat_pcm, Ts_pcm_noOHT1, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[0, 5].set_title( 'Generic PCM', fontsize=15 )
ax_array[0, 5].set_xlabel( '', fontsize = 10 )
ax_array[0, 5].set_ylabel( '', fontsize = 10 )
ax_array[0, 5].set_xticks( [ ], labels=[] )
ax_array[0, 5].set_yticks( [ ], labels=[] )

im3 = ax_array[1, 5].contourf( lon_pcm, lat_pcm, Ts_pcm_noOHT4, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[1, 5].set_title( '', fontsize=15 )
ax_array[1, 5].set_xlabel( '', fontsize = 10 )
ax_array[1, 5].set_ylabel( '', fontsize = 10 )
ax_array[1, 5].set_xticks( [ ], labels=[] )
ax_array[1, 5].set_yticks( [ ], labels=[] )

im3 = ax_array[2, 5].contourf( lon_pcm, lat_pcm, Ts_pcm_noOHT16, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[2, 5].set_title( '', fontsize=15 )
ax_array[2, 5].set_xlabel( '', fontsize = 10 )
ax_array[2, 5].set_ylabel( '', fontsize = 10 )
ax_array[2, 5].set_xticks( [ ], labels=[] )
ax_array[2, 5].set_yticks( [ ], labels=[] )

#--------------------------------------------------------------------
# Column 7: Generic PCM (with OHT)

ZEROVALS = np.zeros( ( np.size( lat_pcm ), np.size( lon_pcm ) ) )
im3 = ax_array[0, 6].contourf( lon_pcm, lat_pcm, ZEROVALS, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[0, 6].set_title( '', fontsize=15 )
ax_array[0, 6].set_xticks( [ ], labels=[] )
ax_array[0, 6].set_yticks( [ ], labels=[] )

ZEROVALS = np.zeros( ( np.size( lat_pcm ), np.size( lon_pcm ) ) )
im3 = ax_array[1, 6].contourf( lon_pcm, lat_pcm, ZEROVALS, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[1, 6].set_title( '', fontsize=15 )
ax_array[1, 6].set_xticks( [ ], labels=[] )
ax_array[1, 6].set_yticks( [ ], labels=[] )

ZEROVALS = np.zeros( ( np.size( lat_pcm ), np.size( lon_pcm ) ) )
im3 = ax_array[2, 6].contourf( lon_pcm, lat_pcm, ZEROVALS, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
ax_array[2, 6].set_title( '', fontsize=15 )
ax_array[2, 6].set_xticks( [ ], labels=[] )
ax_array[2, 6].set_yticks( [ ], labels=[] )

#--------------------------------------------------------------------
# ZEROVAL Panel
#ZEROVALS = np.zeros( ( np.size( lat_pcm ), np.size( lon_pcm ) ) )
#im6 = axd[ 'P6' ].contourf( lon_pcm, lat_pcm, ZEROVALS, cmap=cm, vmin=contourmin, vmax=contourmax, levels=np.linspace(contourmin,contourmax,numcontours), extend='both' )
#axd[ 'P6' ].set_title( 'TBD', fontsize=15 )


#--------------------------------------------------------------------
# Finalize

cb0 = fig.colorbar( im1, ax=ax_array, extend='both', ticks=np.arange( contourmin, contourmax + 50, 50 ), shrink=0.8, pad=0.01, label='Surface Temperature (K)' )

#fig.subplots_adjust( hspace = 0.65 )
#fig.suptitle( "Surface Temperature", fontsize=20 )

fig.savefig( "fig_compare_temp_select.png", bbox_inches='tight' )
fig.savefig( "fig_compare_temp_select.eps", bbox_inches='tight' )
#plt.show()
