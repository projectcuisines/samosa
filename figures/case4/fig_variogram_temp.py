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
f3 = netCDF4.Dataset( '/models/data/samosa/genericpcm/SAMOSA_output_file_Generic_PCM_case-4_OHT_on.nc' )
#f3 = netCDF4.Dataset( '/models/data/samosa/genericpcm/SAMOSA_output_file_Generic_PCM_case-4_OHT_off.nc' )

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
# Read ROCKE-3D data
f4 = netCDF4.Dataset( '/models/data/samosa/rocke3d/rocke_04q.nc' )

Ts_r3d4 = np.array( f4.variables[ 'tsurf' ] ) + tfreeze
uwind_r3d4 = np.array( f4.variables[ 'ub' ] )
vwind_r3d4 = np.array( f4.variables[ 'vb' ] )
lat_r3d = np.array( f4.variables[ 'lat' ] )
lon_r3d = np.array( f4.variables[ 'lon' ] )
lat2_r3d = np.array( f4.variables[ 'lat2' ] )
lon2a_r3d = np.array( f4.variables[ 'lon2' ] )
#nlev_r3d = len( np.array( f4.variables[ 'plm' ] ) )
nlev_r3d = 1

Ts_r3d4, lon_r3d = shiftgrid( 0, Ts_r3d4, lon_r3d, start=False )
uwind_r3d4, lon2_r3d = shiftgrid( 0, uwind_r3d4, lon2a_r3d, start=False )
vwind_r3d4, lon2_r3d = shiftgrid( 0, vwind_r3d4, lon2a_r3d, start=False )

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
f5 = open( '/models/data/samosa/plahab/case4_tsurf.out' )

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

Ts_lfric4 = f6.variables[ 'grid_surface_temperature' ]
lat_lfric = np.array( f6.variables[ 'lat' ] )
lon_lfric = np.array( f6.variables[ 'lon' ] )

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
contourmin  = 200.0

fig, axd = plt.subplot_mosaic([['right', 'right'],
                               ['right', 'right']],
                              figsize=(7.5, 4.75))

#--------------------------------------------------------------------
# Kriging: ExoPlaSim
OK = OrdinaryKriging(
    lons_plasim,
    lats_plasim,
    Tss_plasim,
    variogram_model="exponential",
    coordinates_type="geographic",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)
#OK.display_variogram_model()

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 160
variymin  = 0
variymax  = 1100
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='#666666' )
axd[ 'right' ].text( 135, 630, 'ExoPlaSim', fontsize=12, color='#666666' )

axd[ 'right' ].tick_params( axis='x', labelsize=12 )
axd[ 'right' ].tick_params( axis='y', labelsize=12 )
axd[ 'right' ].set_title( 'Case 4 Temperature Variogram', fontsize=15 )
axd[ 'right' ].set_xlabel( 'Spatial separation |h|', fontsize = 12 )
axd[ 'right' ].set_ylabel( 'Dissimilarities $γ^*$', fontsize = 12 )
axd[ 'right' ].set_xlim( [ varixmin, varixmax ] )
axd[ 'right' ].set_ylim( [ variymin, variymax ] )

#--------------------------------------------------------------------
# Kriging: ExoCAM
OK = OrdinaryKriging(
    lons_exocam,
    lats_exocam,
    Tss_exocam,
    variogram_model="exponential",
    coordinates_type="geographic",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 160
variymin  = 0
variymax  = 1100
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='tab:blue' )
axd[ 'right' ].text( 100, 920, 'ExoCAM', fontsize=12, color='tab:blue' )

#--------------------------------------------------------------------
# Kriging: Generic PCM
OK = OrdinaryKriging(
    lons_pcm,
    lats_pcm,
    Tss_pcm,
    variogram_model="exponential",
    coordinates_type="geographic",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 160
variymin  = 0
variymax  = 1100
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='tab:green' )
axd[ 'right' ].text( 100, 200, 'Generic PCM (with OHT)', fontsize=12, color='tab:green' )
#axd[ 'right' ].text( 100, 200, 'Generic PCM (no OHT)', fontsize=12, color='tab:green' )

#--------------------------------------------------------------------
# Kriging: ROCKE-3D
OK = OrdinaryKriging(
    lons_r3d,
    lats_r3d,
    Tss_r3d,
    variogram_model="exponential",
    coordinates_type="geographic",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 160
variymin  = 0
variymax  = 1100
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='tab:orange' )
axd[ 'right' ].text( 110, 770, 'ROCKE-3D', fontsize=12, color='tab:orange' )

#--------------------------------------------------------------------
# Kriging: PlaHab
OK = OrdinaryKriging(
    lons_plahab,
    lats_plahab,
    Tss_plahab,
    variogram_model="exponential",
    coordinates_type="geographic",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 160
variymin  = 0
variymax  = 1100
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='tab:purple' )
axd[ 'right' ].text( 130, 310, 'PlaHab', fontsize=12, color='tab:purple' )

#--------------------------------------------------------------------
# Kriging: LFRic
OK = OrdinaryKriging(
    lons_lfric,
    lats_lfric,
    Tss_lfric,
    variogram_model="exponential",
    coordinates_type="geographic",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 160
variymin  = 0
variymax  = 1100
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='k' )
axd[ 'right' ].text( 110, 1020, 'LFRic', fontsize=12, color='k' )


#--------------------------------------------------------------------
# Finalize

fig.savefig( "fig_variogram_temp.png", bbox_inches='tight' )
fig.savefig( "fig_variogram_temp.eps", bbox_inches='tight' )
#plt.show()
