import numpy as np
from pykrige.ok import OrdinaryKriging

#--------------------------------------------------------------------
# Data setup (mirrors fig_interpolation_clouds.py)

runaway   = 200.0   # sentinel (%) for runaway/unavailable cases
fluxscale = 100

flux  = np.arange( 400, 2700, 100 ) / fluxscale
pn2   = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

plasim  = np.array( [ 42.7, 68.2, 70.6, 56.2, 80.1, 32.1, 58.1, 25.7, 58.0, 25.2, 76.2, 30.1, 85.3, 52.7, 48.4, 53.3 ] )
exocam  = np.array( [ 68.75, runaway, runaway, 43.98, runaway, runaway, runaway, 16.34, 75.82, 15.85, 83.08, 56.79, runaway, 34.01, 78.96, 61.40 ] )
rocke3d = np.array( [ 68.20222, runaway, runaway, 51.043224, 81.88261, runaway, 88.81546, 58.24044, 61.535275, 98.8356, 68.16493, 68.01357, 85.71091, 43.637707, 74.40385, 48.08909 ] )
plahab  = np.array( [ 11.11879, runaway, runaway, 35.74597, 48.29323, runaway, 70.91280, 26.24803, 31.64522, 8.4692545, 4.3572873, 76.20874, 57.27629, 28.12309, 16.64636, 72.36285 ] )
pcm     = np.array( [ 25.5674468009485, 27.31228828919005, 16.96672860199983, 25.15276275245855, 24.55454268845772, 16.696470834684884, 32.84789893586739 ] )
pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )
lfric       = np.array( [ 31.0, 61.0, 58.0, 81.0, 44.0, 36.0 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway
plahab_mask  = plahab  != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

#--------------------------------------------------------------------
# Normalization (mirrors fig_interpolation_clouds.py)

log_pn2 = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

def logit( x ):
    x = np.clip( x, 1.0, 99.0 )
    return np.log( x / ( 100.0 - x ) )

#--------------------------------------------------------------------
# Leave-one-out cross-validation in logit space

def loo_rmse( pres, flx, vals_logit, vmodel ):
    n = len( vals_logit )
    residuals = np.zeros( n )
    for i in range( n ):
        idx = np.arange( n ) != i
        try:
            OK = OrdinaryKriging(
                norm_pres( pres[ idx ] ),
                norm_flux( flx[ idx ] ),
                vals_logit[ idx ],
                variogram_model=vmodel,
                verbose=False,
                enable_plotting=False,
                exact_values=True,
            )
            pred, _ = OK.execute( "points",
                                  norm_pres( pres[ [i] ] ),
                                  norm_flux( flx[ [i] ] ) )
            residuals[ i ] = vals_logit[ i ] - pred[ 0 ]
        except Exception:
            residuals[ i ] = np.nan
    return np.sqrt( np.nanmean( residuals**2 ) )

variogram_models = [ 'linear', 'power', 'gaussian', 'spherical', 'exponential' ]

climate_models = [
    ( 'ExoCAM',      exocam_pres1,  exocam_flux1,  logit( exocam_stable  ) ),
    ( 'ROCKE-3D',    rocke3d_pres1, rocke3d_flux1, logit( rocke3d_stable ) ),
    ( 'ExoPlaSim',   pres1,         flux1,          logit( plasim          ) ),
    ( 'Generic PCM', pcm_pres1,     pcm_flux1,     logit( pcm             ) ),
    ( 'PlaHab',      plahab_pres1,  plahab_flux1,  logit( plahab_stable  ) ),
    ( 'LFRic',       lfric_pres1,   lfric_flux1,   logit( lfric           ) ),
]

#--------------------------------------------------------------------
# Run and print results

col_w = 13
print( f"\nLeave-one-out cross-validation RMSE (logit-units)\n" )
print( f"{'Model':<14}", end='' )
for vm in variogram_models:
    print( f"{vm:>{col_w}}", end='' )
print()
print( '-' * ( 14 + col_w * len( variogram_models ) ) )

all_rmse = { vm: [] for vm in variogram_models }

for name, pres, flx, vals_logit in climate_models:
    print( f"{name:<14}", end='' )
    for vm in variogram_models:
        rmse = loo_rmse( pres, flx, vals_logit, vm )
        all_rmse[ vm ].append( rmse )
        if np.isnan( rmse ):
            print( f"{'fail':>{col_w}}", end='' )
        else:
            print( f"{rmse:>{col_w}.3f}", end='' )
    print()

print( '-' * ( 14 + col_w * len( variogram_models ) ) )
print( f"{'Mean RMSE':<14}", end='' )
for vm in variogram_models:
    vals = [ v for v in all_rmse[ vm ] if not np.isnan( v ) ]
    mean = np.mean( vals ) if vals else np.nan
    print( f"{mean:>{col_w}.3f}", end='' )
print( '\n' )
