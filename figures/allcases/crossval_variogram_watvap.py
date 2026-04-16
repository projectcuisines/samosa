import numpy as np
from pykrige.ok import OrdinaryKriging

#--------------------------------------------------------------------
# Data setup (mirrors fig_interpolation_watvap.py)

_s        = 0.1     # unit conversion: raw model output → kg m⁻²
runaway   = 1.e4    # sentinel (kg m⁻²) for runaway/unavailable cases
fluxscale = 100

flux  = np.arange( 400, 2700, 100 ) / fluxscale
pn2   = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

plasim  = _s * np.array( [ 0.059, 5113.363, 271.733, 7.509, 37.305, 1452.570, 76.876, 0.351, 7.048, 0.007, 3.042, 1258.514, 59.214, 1.955, 0.416, 1111.793 ] )
exocam  = _s * np.array( [ 0.2335, runaway/_s, runaway/_s, 19.2138, runaway/_s, runaway/_s, runaway/_s, 1.7490, 7.9707, 0.0102, 5.8914, 1430.7613, runaway/_s, 4.2041, 0.9534, 1295.6428 ] )
rocke3d = _s * np.array( [ 0.25980374, runaway/_s, runaway/_s, 14.9693165, 31.839989, runaway/_s, 32.45563, 1.5794185, 5.687923, 0.031635746, 4.2059116, 271.75916, 46.070984, 2.964403, 0.52949935, 132.49808 ] )
pcm     = _s * np.array( [ 0.37484651163423993, 47.86514350558743, 2.1906175203429563, 25.650192288432617, 0.06080825624148188, 5.93210602524276, 0.8447688576786204 ] )

pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 0.41, 8.18, 7.37, 829.15, 2.34, 0.86 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900, 600 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]

# Runaway BC arrays (always kept in training set)
exocam_bc_pres  = pres1[ ~exocam_mask ];  exocam_bc_flux  = flux1[ ~exocam_mask ]
exocam_bc_log   = np.log( exocam[ ~exocam_mask ] )
rocke3d_bc_pres = pres1[ ~rocke3d_mask ]; rocke3d_bc_flux = flux1[ ~rocke3d_mask ]
rocke3d_bc_log  = np.log( rocke3d[ ~rocke3d_mask ] )

#--------------------------------------------------------------------
# Normalization (mirrors fig_interpolation_watvap.py)

log_pn2 = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

#--------------------------------------------------------------------
# Leave-one-out cross-validation

def loo_rmse( pres, flx, vals_log, vmodel ):
    """Standard LOO for models without runaway boundary conditions."""
    n = len( vals_log )
    residuals = np.zeros( n )
    for i in range( n ):
        idx = np.arange( n ) != i
        try:
            OK = OrdinaryKriging(
                norm_pres( pres[ idx ] ),
                norm_flux( flx[ idx ] ),
                vals_log[ idx ],
                variogram_model=vmodel,
                verbose=False,
                enable_plotting=False,
                exact_values=True,
            )
            pred, _ = OK.execute( "points",
                                  norm_pres( pres[ [i] ] ),
                                  norm_flux( flx[ [i] ] ) )
            residuals[ i ] = vals_log[ i ] - pred[ 0 ]
        except Exception:
            residuals[ i ] = np.nan
    return np.sqrt( np.nanmean( residuals**2 ) )


def loo_rmse_with_bc( pres_stable, flx_stable, vals_log_stable,
                      pres_bc, flx_bc, vals_log_bc, vmodel ):
    """LOO for models with runaway BCs: stable points are left out one at a time,
    BC points always remain in the training set."""
    n = len( vals_log_stable )
    residuals = np.zeros( n )
    for i in range( n ):
        idx = np.arange( n ) != i
        pres_train = np.concatenate( [ pres_stable[ idx ], pres_bc ] )
        flx_train  = np.concatenate( [ flx_stable[ idx ],  flx_bc  ] )
        vals_train = np.concatenate( [ vals_log_stable[ idx ], vals_log_bc ] )
        try:
            OK = OrdinaryKriging(
                norm_pres( pres_train ),
                norm_flux( flx_train ),
                vals_train,
                variogram_model=vmodel,
                verbose=False,
                enable_plotting=False,
                exact_values=True,
            )
            pred, _ = OK.execute( "points",
                                  norm_pres( pres_stable[ [i] ] ),
                                  norm_flux( flx_stable[ [i] ] ) )
            residuals[ i ] = vals_log_stable[ i ] - pred[ 0 ]
        except Exception:
            residuals[ i ] = np.nan
    return np.sqrt( np.nanmean( residuals**2 ) )


variogram_models = [ 'linear', 'power', 'gaussian', 'spherical', 'exponential' ]

#--------------------------------------------------------------------
# Run and print results

col_w = 13
print( f"\nLeave-one-out cross-validation RMSE (log kg m⁻²)\n" )
print( f"{'Model':<14}", end='' )
for vm in variogram_models:
    print( f"{vm:>{col_w}}", end='' )
print()
print( '-' * ( 14 + col_w * len( variogram_models ) ) )

all_rmse = { vm: [] for vm in variogram_models }

climate_models = [
    ( 'ExoCAM',      'bc',     exocam_pres1,  exocam_flux1,  np.log(exocam_stable),  exocam_bc_pres,  exocam_bc_flux,  exocam_bc_log  ),
    ( 'ROCKE-3D',    'bc',     rocke3d_pres1, rocke3d_flux1, np.log(rocke3d_stable), rocke3d_bc_pres, rocke3d_bc_flux, rocke3d_bc_log ),
    ( 'ExoPlaSim',   'std',    pres1,         flux1,         np.log(plasim),         None, None, None ),
    ( 'Generic PCM', 'std',    pcm_pres1,     pcm_flux1,     np.log(pcm),            None, None, None ),
    ( 'LFRic',       'std',    lfric_pres1,   lfric_flux1,   np.log(lfric),          None, None, None ),
]

for row in climate_models:
    name, mode = row[0], row[1]
    print( f"{name:<14}", end='' )
    for vm in variogram_models:
        if mode == 'bc':
            pres_s, flx_s, vals_s, pres_b, flx_b, vals_b = row[2], row[3], row[4], row[5], row[6], row[7]
            rmse = loo_rmse_with_bc( pres_s, flx_s, vals_s, pres_b, flx_b, vals_b, vm )
        else:
            rmse = loo_rmse( row[2], row[3], row[4], vm )
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
