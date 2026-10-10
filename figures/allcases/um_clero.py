"""Summary arrays for the UM and the two CLERO configurations, shared by every
figure script, so that these three models are held in one place rather than
copied into each script as the eight earlier ones are.

Both arrived on 2026-10-09 (contact Ed Stevenson), and both packages pass their
MANIFEST sha256 checks.

UM       Met Office Unified Model 13.9, run by Mei Ting Mak for ThousandWorlds.
         The UM is LFRic's predecessor. It ran nine cases, HEXTOR's set; the
         other seven were not run, rather than being runaways. It follows the
         protocol except for its PHOENIX 3000 K spectrum, which replaces the
         blackbody. Source: um/global_UM.dat, which the cos(lat)-weighted
         fields of um_samosa_caseNN.nc reproduce to 2 dp. No total cloud
         fraction is reported.

CLERO    A statistical emulator of GCM climates, conditioned on either the UM
         (CLERO-UM) or ExoCAM (CLERO-ExoCAM), and entered here as two models,
         as its author suggests. global_CLERO.dat is the mean of the two
         configurations, so it is NOT used. Values are area_weight-weighted
         means of clero/CLERO-*/samosa_CLERO-*_caseNN.nc. The emulator
         delivers 14 of Cases 1-16, dropping Cases 3 and 13, where the mean
         runaway probability of the two configurations exceeds 0.5. The
         CLERO-ExoCAM probability alone is 0.59 at Case 6, which is kept as
         delivered. The water vapor column runs from the surface to 10 hPa,
         CLERO's top level. No total cloud fraction is reported. CLERO is
         plotted but kept out of every ensemble statistic: it emulates two of
         the ensemble's own models rather than being independent of them.

Regenerate with extract_fluxes.py (albedo, imbalance) and extract_maxtemp.py
(Tmax, Tmin). Instellation is in W/m2, NOT divided by any fluxscale.
"""
import numpy as np

# ── UM: Cases 1, 4, 8, 9, 10, 11, 14, 15, 16 ────────────────────────────────
um_cases = [ 1, 4, 8, 9, 10, 11, 14, 15, 16 ]
um_flux  = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600, 1400 ], float )
um_pres  = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43, 10.00 ] )
um_temp  = np.array( [ 191.67, 260.62, 224.27, 253.33, 192.59, 228.14, 232.28, 201.97, 286.49 ] )
um_tmax  = np.array( [ 269.7, 299.1, 267.3, 297.6, 244.4, 286.1, 285.0, 275.8, 308.4 ] )
um_tmin  = np.array( [ 132.3, 221.1, 188.9, 202.0, 157.5, 182.6, 170.7, 140.3, 269.5 ] )
um_wv    = np.array( [ 0.3962, 15.7, 0.6129, 15.52, 0.04881, 5.418, 2.386, 0.8748, 42.49 ] )    # kg/m2
um_alb   = np.array( [ 26.99, 28.52, 34.51, 23.16, 19.60, 28.64, 30.21, 26.43, 29.00 ] )        # %

# ── CLERO: Cases 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16 ─────────────
clero_cases = [ 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16 ]
clero_flux  = np.array( [ 500, 1900, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 900, 600, 1400 ], float )
clero_pres  = np.array( [ 0.70, 7.85, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )

clero_um_temp = np.array( [ 191.01, 335.69, 260.27, 280.20, 335.21, 294.50, 224.97, 253.38, 189.58, 227.76, 295.57, 233.13, 201.67, 289.68 ] )
clero_um_tmax = np.array( [ 268.8, 341.8, 298.3, 308.5, 341.7, 315.1, 270.0, 297.1, 244.1, 284.2, 317.5, 286.3, 277.0, 310.9 ] )
clero_um_tmin = np.array( [ 133.7, 323.8, 221.1, 253.7, 324.1, 269.1, 194.3, 205.7, 155.6, 182.8, 270.3, 185.7, 144.0, 273.0 ] )
clero_um_wv   = np.array( [ 0.2692, 1023.0, 13.38, 314.5, 2308.0, 257.9, 0.68, 14.09, 0.05221, 5.121, 96.63, 2.758, 0.7092, 44.68 ] )
clero_um_alb  = np.array( [ 27.61, 20.66, 28.92, 20.50, 19.16, 20.87, 32.66, 23.83, 23.86, 26.68, 21.97, 30.88, 27.01, 27.67 ] )

clero_exocam_temp = np.array( [ 193.81, 377.65, 273.84, 277.83, 341.68, 290.21, 244.13, 248.05, 193.94, 231.75, 334.23, 233.86, 204.29, 348.45 ] )
clero_exocam_tmax = np.array( [ 265.0, 386.3, 306.9, 307.9, 348.7, 312.0, 282.4, 292.4, 237.8, 283.3, 342.0, 285.4, 274.0, 353.6 ] )
clero_exocam_tmin = np.array( [ 137.5, 366.0, 237.0, 248.6, 335.3, 266.2, 211.3, 203.6, 160.3, 195.2, 324.6, 183.2, 149.5, 343.6 ] )
clero_exocam_wv   = np.array( [ 0.2334, 11550.0, 25.66, 290.5, 4032.0, 259.9, 1.295, 10.18, 0.02101, 5.197, 701.7, 3.165, 0.6488, 944.1 ] )
clero_exocam_alb  = np.array( [ 27.04, 18.60, 28.67, 24.50, 27.38, 30.78, 21.90, 30.92, 28.93, 23.03, 21.56, 28.63, 25.10, 17.05 ] )

# Figure 2's marker and color for each, used wherever per-model points appear
STYLE = {
    'UM':           dict( color='#e377c2', marker='>' ),
    'CLERO-UM':     dict( color='#bcbd22', marker='h' ),
    'CLERO-ExoCAM': dict( color='#e7ba52', marker='p' ),
}

# ( flux, pres, value ) per model and variable; flux in W/m2
def samples( var ):
    """{ name: ( flux, pres, values ) } for var in 'temp', 'tmax', 'tmin', 'wv', 'alb'."""
    g = globals()
    return { 'UM':           ( um_flux,    um_pres,    g[ f'um_{var}' ] ),
             'CLERO-UM':     ( clero_flux, clero_pres, g[ f'clero_um_{var}' ] ),
             'CLERO-ExoCAM': ( clero_flux, clero_pres, g[ f'clero_exocam_{var}' ] ) }


# ── Fields for the select-case maps and profiles ────────────────────────────
# Every reader returns its fields with the substellar point at longitude 0 and
# longitudes in [-180, 180).
ROOT = '/models/data/samosa'
G    = 9.80665

def _roll_to_180( data, lon ):
    idx = np.searchsorted( lon, 180.0 )
    return ( np.concatenate( [ data[ ..., idx: ], data[ ..., :idx ] ], axis=-1 ),
             np.concatenate( [ lon[ idx: ] - 360.0, lon[ :idx ] ] ) )

def um_map( case, kind ):
    """( lon, lat, field ) from um_samosa_caseNN.nc. kind is 'ts' (K), 'wv'
    (kg/m2, the trapezoidal integral of specific humidity in pressure over the
    39 native levels, which reproduces Qmass of global_UM.dat to 4 figures), or
    'sw' (( reflected, incident ) at the top of the atmosphere, W/m2)."""
    import netCDF4
    with netCDF4.Dataset( f'{ROOT}/um/um_samosa_case{case:02d}.nc' ) as d:
        lat, lon = np.array( d[ 'latitude' ][ : ] ), np.array( d[ 'longitude' ][ : ] )
        if kind == 'ts':
            f = np.array( d[ 'surface_temperature' ][ : ], float )
        elif kind == 'wv':
            q = np.array( d[ 'specific_humidity' ][ : ], float )
            p = np.array( d[ 'air_pressure' ][ : ], float )
            f = np.sum( 0.5*( q[ 1: ] + q[ :-1 ] )*( p[ :-1 ] - p[ 1: ] ), axis=0 ) / G
        elif kind == 'sw':
            f = np.stack( [ np.array( d[ 'toa_outgoing_shortwave_flux' ][ : ], float ),
                            np.array( d[ 'toa_incoming_shortwave_flux' ][ : ], float ) ] )
    f, lon = _roll_to_180( f, lon )
    return lon, lat, f

def clero_map( cfg, case, kind ):
    """( lon, lat, field ) from CLERO-<cfg>; kind 'ts' or 'wv'. CLERO's grid is
    already centred on the substellar point."""
    import netCDF4
    var = { 'ts': 'surface_temperature', 'wv': 'water_vapor_path' }[ kind ]
    with netCDF4.Dataset( f'{ROOT}/clero/CLERO-{cfg}/samosa_CLERO-{cfg}_case{case:02d}.nc' ) as d:
        return np.array( d[ 'lon' ][ : ] ), np.array( d[ 'lat' ][ : ] ), np.array( d[ var ][ : ], float )

def _hemi_mean( data, lat, lon, day ):
    """Area-weighted mean of a (lev, lat, lon) field over the substellar (day)
    or anti-stellar hemisphere, the substellar point at lon 0; returns (lev,)."""
    cd = np.cos( np.radians( lon ) )
    w  = np.cos( np.radians( lat ) )[ :, None ] * ( ( cd > 0 ) if day else ( cd < 0 ) )[ None, : ]
    return ( data * w ).sum( axis=( 1, 2 ) ) / w.sum()

def um_profiles( case ):
    """( T_ss, Q_ss, T_as, Q_as, P_hPa ), surface upward. Temperature is on the
    38 theta levels; humidity and pressure carry the surface as level 0 as well,
    which is dropped so all three share the 38 levels."""
    import netCDF4
    with netCDF4.Dataset( f'{ROOT}/um/um_samosa_case{case:02d}.nc' ) as d:
        lat, lon = np.array( d[ 'latitude' ][ : ] ), np.array( d[ 'longitude' ][ : ] )
        T = np.array( d[ 'air_temperature' ][ : ], float )
        q = np.array( d[ 'specific_humidity' ][ : ], float )[ 1: ]
        p = np.array( d[ 'air_pressure' ][ : ], float )[ 1: ]
    T, lon_s = _roll_to_180( T, lon )
    q, _     = _roll_to_180( q, lon )
    p, _     = _roll_to_180( p, lon )
    return ( _hemi_mean( T, lat, lon_s, True ),  _hemi_mean( q, lat, lon_s, True ),
             _hemi_mean( T, lat, lon_s, False ), _hemi_mean( q, lat, lon_s, False ),
             _hemi_mean( p, lat, lon_s, True ) / 100.0 )

def clero_profiles( cfg, case ):
    """As um_profiles, for CLERO-<cfg>: ten relative isobars, 0.925 of the
    surface pressure up to 10 hPa."""
    import netCDF4
    with netCDF4.Dataset( f'{ROOT}/clero/CLERO-{cfg}/samosa_CLERO-{cfg}_case{case:02d}.nc' ) as d:
        lat, lon = np.array( d[ 'lat' ][ : ] ), np.array( d[ 'lon' ][ : ] )
        T = np.array( d[ 'temperature' ][ : ], float )
        q = np.array( d[ 'specific_humidity' ][ : ], float )
        p = np.array( d[ 'pressure' ][ : ], float ) / 100.0
    return ( _hemi_mean( T, lat, lon, True ),  _hemi_mean( q, lat, lon, True ),
             _hemi_mean( T, lat, lon, False ), _hemi_mean( q, lat, lon, False ), p )

def um_uwind( case ):
    """( U_bar (lev, lat), P_hPa (lev,), lat ): zonal-mean zonal wind. x_wind is
    on the 38 rho levels, between the theta levels on which air_pressure is
    given (level 0 the surface), so each rho level takes the mean pressure of
    the theta levels either side, as LFRic's half levels do in fig_uwind_select.py."""
    import netCDF4
    with netCDF4.Dataset( f'{ROOT}/um/um_samosa_case{case:02d}.nc' ) as d:
        u   = np.array( d[ 'x_wind' ][ : ], float )
        p   = np.array( d[ 'air_pressure' ][ : ], float )
        lat = np.array( d[ 'latitude' ][ : ] )
    p_rho = 0.5*( p[ :-1 ] + p[ 1: ] )
    w     = np.cos( np.radians( lat ) ); w = w / w.sum()
    return u.mean( axis=-1 ), ( p_rho.mean( axis=-1 ) * w ).sum( axis=-1 ) / 100.0, lat

def clero_uwind( cfg, case ):
    """As um_uwind, for CLERO-<cfg> on its ten relative isobars."""
    import netCDF4
    with netCDF4.Dataset( f'{ROOT}/clero/CLERO-{cfg}/samosa_CLERO-{cfg}_case{case:02d}.nc' ) as d:
        return ( np.array( d[ 'u' ][ : ], float ).mean( axis=-1 ),
                 np.array( d[ 'pressure' ][ : ], float ) / 100.0, np.array( d[ 'lat' ][ : ] ) )
