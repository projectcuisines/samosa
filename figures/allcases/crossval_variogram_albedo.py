import numpy as np
from pykrige.ok import OrdinaryKriging

#--------------------------------------------------------------------
# Data setup (mirrors fig_interpolation_albedo.py)

runaway   = 200.0   # sentinel (%) for runaway/unavailable cases
fluxscale = 100

flux  = np.arange( 400, 2700, 100 ) / fluxscale
pn2   = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

plasim  = np.array( [ 40.77, 25.00, 21.85, 39.99, 31.52, 18.87, 29.82, 40.23, 33.58, 42.23, 38.32, 17.26, 31.10, 36.40, 39.34, 32.58 ] )
exocam  = np.array( [ 27.31, runaway, runaway, 31.08, runaway, runaway, runaway, 20.74, 34.54, 31.78, 22.69, 19.05, runaway, 28.91, 20.38, 16.95 ] )
rocke3d = np.array( [ 23.08, runaway, runaway, 31.56, 39.86, runaway, 44.31, 22.14, 37.63, 21.21, 28.72, 17.96, 40.84, 30.09, 22.26, 12.78 ] )
plahab  = np.array( [ 19.11, runaway, runaway, 33.03, 33.02, runaway, 34.90, 29.62, 30.82, 0.89, 63.70, 35.51, 34.85, 31.12, 19.93, 35.54 ] )
pcm     = np.array( [ 25.57, 14.42, 22.69, 17.72, 28.53, 19.54, 21.31 ] )
pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )
lfric       = np.array( [ 26.03, 35.74, 34.16, 23.22, 34.41, 28.25, 21.35 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43, 10.00 ] )

# ExoColumn, cases 1, 4, 8, 9, 10, 11, 14, 15. Cloud-free, but with a fixed
# surface albedo of 0.2736 standing in for the shortwave effect of clouds, so
# unlike HEXTOR its albedo sits inside the GCM range.
exocolumn       = np.array( [ 26.01, 15.51, 23.11, 19.68, 27.26, 23.95, 21.97, 25.53 ] )
exocolumn_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600 ] ) / fluxscale
exocolumn_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43 ] )

# HEXTOR, cases 1, 4, 8, 9, 11, 14, 15. Clear-sky by construction, so these are
# surface-plus-Rayleigh albedos with no cloud contribution.
hextor       = np.array( [ 20.18, 2.24, 14.54, 2.21, 11.36, 7.42, 19.62 ] )
hextor_flux1 = np.array( [ 500, 1200, 800, 1100, 900, 900, 600 ] ) / fluxscale
hextor_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 0.10, 1.44, 0.43 ] )


exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway
plahab_mask  = plahab  != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

#--------------------------------------------------------------------
# Normalization (mirrors fig_interpolation_albedo.py)

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

def fitted_slope( pres, flx, vals_logit, vmodel ):
    """Slope (or equivalent structure term) of the fitted variogram. A slope of
    zero means a pure-nugget fit, for which ordinary kriging degenerates to the
    global mean and the interpolated field carries no spatial information."""
    try:
        OK = OrdinaryKriging(
            norm_pres( pres ), norm_flux( flx ), vals_logit,
            variogram_model=vmodel, verbose=False,
            enable_plotting=False, exact_values=True,
        )
        return OK.variogram_model_parameters[ 0 ]
    except Exception:
        return np.nan

variogram_models = [ 'linear', 'power', 'gaussian', 'spherical', 'exponential' ]

climate_models = [
    ( 'ExoCAM',      exocam_pres1,  exocam_flux1,  logit( exocam_stable  ) ),
    ( 'ROCKE-3D',    rocke3d_pres1, rocke3d_flux1, logit( rocke3d_stable ) ),
    ( 'ExoPlaSim',   pres1,         flux1,          logit( plasim          ) ),
    ( 'Generic PCM', pcm_pres1,     pcm_flux1,     logit( pcm             ) ),
    ( 'PlaHab',      plahab_pres1,  plahab_flux1,  logit( plahab_stable  ) ),
    ( 'LFRic',       lfric_pres1,   lfric_flux1,   logit( lfric           ) ),
    ( 'HEXTOR',      hextor_pres1,  hextor_flux1,  logit( hextor          ) ),
    ( 'ExoColumn',   exocolumn_pres1, exocolumn_flux1, logit( exocolumn     ) ),
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

#--------------------------------------------------------------------
# Degeneracy check: which fits collapse to a pure nugget?

print( "Fitted variogram structure term (0 => pure nugget => flat kriged field)\n" )
print( f"{'Model':<14}", end='' )
for vm in variogram_models:
    print( f"{vm:>{col_w}}", end='' )
print()
print( '-' * ( 14 + col_w * len( variogram_models ) ) )
for name, pres, flx, vals_logit in climate_models:
    print( f"{name:<14}", end='' )
    for vm in variogram_models:
        s = fitted_slope( pres, flx, vals_logit, vm )
        print( f"{s:>{col_w}.4f}" if not np.isnan( s ) else f"{'fail':>{col_w}}", end='' )
    print()
print()
