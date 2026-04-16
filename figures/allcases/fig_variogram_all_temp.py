import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import CubicSpline

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

runawaytemp = 600.0
contourmax  = 500.0

flux = np.arange( 400, 2700, 100 )
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] )
pres1  = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Temperature from ExoPlaSim, ExoCAM, ROCKE-3D
plasim = np.array( [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9, 239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ] )
exocam = np.array( [ 196.8, runawaytemp, runawaytemp, 260.0, runawaytemp, runawaytemp, runawaytemp, 243.8, 244.8, 194.1, 234.0, 350.9, runawaytemp, 236.8, 211.5, 356.7 ] )
rocke3d = np.array( [ 202.8284, runawaytemp, runawaytemp, 260.1185, 265.88116, runawaytemp, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ] )
plahab = np.array( [ 196.3, runawaytemp, runawaytemp, 273.2, 281.4, runawaytemp, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )


numplots = 1
fig, axd = plt.subplots( 1, numplots, sharex=False, figsize = (7.5,4.75) )
#fig, (ax, ax2) = plt.subplots( 1, numplots, sharex=False, figsize = (14.0,4.75) )
plt.rc('axes', titlesize=15)

#cm = mpl.colormaps.get_cmap('cool')
cm = mpl.colormaps.get_cmap('terrain')

#--------------------------------------------------------------------
# ExoCAM Variogram

OK = OrdinaryKriging(
    pres1,
    flux1,
    exocam,
    variogram_model="linear",
    verbose=True,
    enable_plotting=False,
    exact_values=True,
)
#OK.display_variogram_model()

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 2000
variymin  = 0
variymax  = 100000
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd.plot( varifuncx, varifuncy, marker='none', color='tab:blue' )
axd.text( 100, 90000, 'ExoCAM', fontsize=12, color='tab:blue' )

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

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 2000
variymin  = 0
variymax  = 100000
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd.plot( varifuncx, varifuncy, marker='none', color='tab:orange' )
axd.text( 100, 85000, 'ROCKE-3D', fontsize=12, color='tab:orange' )

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

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 2000
variymin  = 0
variymax  = 100000
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd.plot( varifuncx, varifuncy, marker='none', color='#666666' )
axd.text( 100, 80000, 'ExoPlaSim', fontsize=12, color='#666666' )

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

varix, variy = OK.get_variogram_points()
varixmin  = 0
varixmax  = 2000
variymin  = 0
variymax  = 100000
varifunc  = CubicSpline( varix, variy )
varifuncx = np.arange( varixmin, varixmax, 1 )
varifuncy = varifunc( varifuncx )

im1 = axd.plot( varifuncx, varifuncy, marker='none', color='tab:purple' )
axd.text( 100, 75000, 'PlaHab', fontsize=12, color='tab:purple' )

#--------------------------------------------------------------------
# Finalize

axd.tick_params( axis='x', labelsize=12 )
axd.tick_params( axis='y', labelsize=12 )
axd.set_title( 'Temperature Variogram (all 16 cases)', fontsize=15 )
axd.set_xlabel( 'Spatial separation |h|', fontsize = 12 )
axd.set_ylabel( 'Dissimilarities $γ^*$', fontsize = 12 )
axd.set_xlim( [ varixmin, varixmax ] )
axd.set_ylim( [ variymin, variymax ] )

fig.savefig( "fig_variogram_all_temp.png", bbox_inches='tight' )
fig.savefig( "fig_variogram_all_temp.eps", bbox_inches='tight' )
#plt.show()
