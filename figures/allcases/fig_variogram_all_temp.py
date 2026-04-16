import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pykrige.ok import OrdinaryKriging

runawaytemp = 600.0
fluxscale   = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

plasim  = np.array( [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9, 239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ] )
exocam  = np.array( [ 196.8, runawaytemp, runawaytemp, 260.0, runawaytemp, runawaytemp, runawaytemp, 243.8, 244.8, 194.1, 234.0, 350.9, runawaytemp, 236.8, 211.5, 356.7 ] )
rocke3d = np.array( [ 202.8284, runawaytemp, runawaytemp, 260.1185, 265.88116, runawaytemp, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ] )
plahab  = np.array( [ 196.3, runawaytemp, runawaytemp, 273.2, 281.4, runawaytemp, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )

pcm = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218, 266.5987224285321, 210.69131033681012, 246.04296230476365, 217.2519558970929 ] )
pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 195.37, 251.48, 241.35, 333.20, 228.84, 203.64 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

exocam_mask  = exocam  != runawaytemp
rocke3d_mask = rocke3d != runawaytemp
plahab_mask  = plahab  != runawaytemp

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

# Normalization (matches fig_interpolation_temp.py)
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

fig, ax = plt.subplots( figsize=(7.5, 4.75) )

varixmin  = 0.0
varixmax  = 1.5
variymin  = 0.0
variymax  = 10000.0
varifuncx = np.linspace( varixmin, varixmax, 500 )

models = [
    ( 'ExoPlaSim',   norm_pres(pres1),         norm_flux(flux1),         plasim,         'tab:blue'   ),
    ( 'ExoCAM',      norm_pres(exocam_pres1),   norm_flux(exocam_flux1),  exocam_stable,  'tab:orange' ),
    ( 'ROCKE-3D',    norm_pres(rocke3d_pres1),  norm_flux(rocke3d_flux1), rocke3d_stable, 'tab:green'  ),
    ( 'Generic PCM', norm_pres(pcm_pres1),      norm_flux(pcm_flux1),     pcm,            'tab:red'    ),
    ( 'LFRic',       norm_pres(lfric_pres1),    norm_flux(lfric_flux1),   lfric,          'tab:purple' ),
    ( 'PlaHab',      norm_pres(plahab_pres1),   norm_flux(plahab_flux1),  plahab_stable,  'tab:brown'  ),
]

for name, np_, nf, vals, color in models:
    OK = OrdinaryKriging(
        np_, nf, vals,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        exact_values=True,
    )
    varix, variy = OK.get_variogram_points()
    varifunc  = CubicSpline( varix, variy )
    ax.plot( varifuncx, varifunc(varifuncx), color=color, label=name )

ax.tick_params( axis='x', labelsize=12 )
ax.tick_params( axis='y', labelsize=12 )
ax.set_title( 'Surface Temperature Variogram', fontsize=15 )
ax.set_xlabel( 'Normalized separation $|h|$', fontsize=12 )
ax.set_ylabel( 'Dissimilarity $γ^*$ (K$^2$)', fontsize=12 )
ax.set_xlim( [ varixmin, varixmax ] )
ax.set_ylim( [ variymin, variymax ] )
ax.legend( fontsize=11 )

fig.savefig( "fig_variogram_all_temp.png", bbox_inches='tight' )
fig.savefig( "fig_variogram_all_temp.eps", bbox_inches='tight' )
#plt.show()
