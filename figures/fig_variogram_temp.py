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
variymax  = 1000
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='#666666' )
axd[ 'right' ].text( 130, 700, 'ExoPlaSim', fontsize=12, color='#666666' )

axd[ 'right' ].tick_params( axis='x', labelsize=12 )
axd[ 'right' ].tick_params( axis='y', labelsize=12 )
axd[ 'right' ].set_title( 'Variogram', fontsize=15 )
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
variymax  = 1000
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='tab:blue' )
axd[ 'right' ].text( 130, 850, 'ExoCAM', fontsize=12, color='tab:blue' )

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
variymax  = 1000
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd[ 'right' ].plot( varifuncx, varifuncy, marker='none', color='tab:green' )
axd[ 'right' ].text( 130, 500, 'Generic PCM', fontsize=12, color='tab:green' )

#--------------------------------------------------------------------
# Finalize

fig.savefig( "fig_variogram_temp.png", bbox_inches='tight' )
fig.savefig( "fig_variogram_temp.eps", bbox_inches='tight' )
#plt.show()
