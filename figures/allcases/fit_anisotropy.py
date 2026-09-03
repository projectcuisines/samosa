"""Fit the kriging anisotropy ratio for each model and variable by cross-validation.

The SAMOSA parameter space has two axes with no common unit: instellation and
log surface pressure. Kriging them with an isotropic distance metric silently
asserts that one normalized unit of each is equally informative, and for this
ensemble that is false -- surface temperature and albedo vary far more sharply
with instellation than with pressure. Left isotropic, HEXTOR degenerates
completely: its two most widely separated sample points (Case 8 at 6.16 bar and
Case 11 at 0.10 bar) differ by only 3.6 K, so the fitted variogram slope
collapses to zero and the panel becomes the sample mean.

pykrige's anisotropy_scaling multiplies the second coordinate, which in every
SAMOSA call is normalized instellation. A scaling of s therefore means one unit
of normalized instellation counts s times a unit of normalized log-pressure.
This script picks s per model and per variable by leave-one-out RMSE, the same
way crossval_variogram*.py already picks the variogram family per model.

    cd figures/allcases && python fit_anisotropy.py

Paste the printed ANISOTROPY dict into the figure scripts.
"""
import contextlib, io, runpy, warnings
import numpy as np
from pykrige.ok import OrdinaryKriging

warnings.filterwarnings( 'ignore' )

SCALINGS = [ 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50 ]

# Ceiling on the fitted ratio. If the LOO minimum sits at the top of the search
# range the fit is asking to drop the pressure axis altogether; that is a
# statement about coverage, not a resolved anisotropy, so it is capped and the
# residual uncertainty is left to show through the kriging variance instead.
CAP = 15

# Each crossval script already holds the arrays and the per-model registry, so
# they are the single source of truth rather than a fourth copy of the data.
SOURCES = {
    'temperature':    ( 'crossval_variogram.py',         'dict' ),
    'water vapor':    ( 'crossval_variogram_watvap.py',  'wv'   ),
    'cloud fraction': ( 'crossval_variogram_clouds.py',  'list' ),
    'albedo':         ( 'crossval_variogram_albedo.py',  'list' ),
}


def registry( path, kind ):
    # The crossval scripts print their own variogram-family tables on import;
    # swallow that so this script's output is only the anisotropy fit.
    with contextlib.redirect_stdout( io.StringIO() ):
        g = runpy.run_path( path )
    cm = g[ 'climate_models' ]
    if kind == 'dict':
        return g, { k: v for k, v in cm.items() }
    if kind == 'wv':                      # ( name, tag, pres, flux, vals, ... )
        return g, { r[ 0 ]: ( r[ 2 ], r[ 3 ], r[ 4 ] ) for r in cm }
    return g, { r[ 0 ]: ( r[ 1 ], r[ 2 ], r[ 3 ] ) for r in cm }


def resolution( npres, nflux, pres, flux, vals, scaling, grid ):
    """Std of the kriged field over the std of the data.

    Leave-one-out error alone is not a sufficient criterion. A variogram can fit
    a near-zero structure term, collapse to pure nugget and return the sample
    mean everywhere, yet still score well on LOO because exact_values pins the
    surface at the sample points. Such a fit produces a flat panel, which is a
    failure for a figure whose job is to show a surface. This measures whether
    the fitted surface actually varies.
    """
    gp, gf = grid
    try:
        ok = OrdinaryKriging( npres( pres ), nflux( flux ), vals,
                              variogram_model='linear', anisotropy_scaling=scaling,
                              verbose=False, enable_plotting=False, exact_values=True )
        z, _ = ok.execute( 'grid', npres( gp ), nflux( gf ) )
    except Exception:
        return 0.0
    sd = np.std( vals )
    return float( np.std( np.asarray( z ) ) / sd ) if sd > 0 else 0.0


MIN_RESOLUTION = 0.25   # kriged field must vary at least this much relative to the data


def loo_rmse( npres, nflux, pres, flux, vals, scaling ):
    res = np.full( len( vals ), np.nan )
    for i in range( len( vals ) ):
        m = np.ones( len( vals ), bool ); m[ i ] = False
        try:
            ok = OrdinaryKriging( npres( pres[ m ] ), nflux( flux[ m ] ), vals[ m ],
                                  variogram_model='linear', anisotropy_scaling=scaling,
                                  verbose=False, enable_plotting=False, exact_values=True )
            pred, _ = ok.execute( 'points', npres( pres[ [ i ] ] ), nflux( flux[ [ i ] ] ) )
            res[ i ] = vals[ i ] - pred[ 0 ]
        except Exception:
            pass
    return np.sqrt( np.nanmean( res**2 ) )


fitted = {}
for var, ( path, kind ) in SOURCES.items():
    g, models = registry( path, kind )
    npres, nflux = g[ 'norm_pres' ], g[ 'norm_flux' ]
    grid = ( g[ 'pn2' ], g[ 'flux' ] )
    print( f'\n=== {var} : leave-one-out RMSE vs anisotropy scaling ===' )
    print( f"{'model':<14}" + ''.join( f'{s:>8g}' for s in SCALINGS ) + '    pick  note' )
    for name, ( pres, flux, vals ) in models.items():
        pres, flux, vals = np.asarray( pres ), np.asarray( flux ), np.asarray( vals )
        rm  = [ loo_rmse( npres, nflux, pres, flux, vals, s ) for s in SCALINGS ]
        res = [ resolution( npres, nflux, pres, flux, vals, s, grid ) for s in SCALINGS ]

        # Only scalings that produce a genuinely varying surface are eligible.
        ok_i = [ i for i in range( len( SCALINGS ) ) if res[ i ] >= MIN_RESOLUTION
                                                    and np.isfinite( rm[ i ] ) ]
        note = ''
        if not ok_i:
            # Nothing resolves: keep the most structured fit and let the panel's
            # kriging variance report the problem rather than hiding it.
            pick = SCALINGS[ int( np.argmax( res ) ) ]
            note = 'no scaling resolves a surface'
        else:
            b    = min( ok_i, key=lambda i: rm[ i ] )
            # The LOO curve is flat-bottomed, so take the smallest eligible
            # scaling within 2% of the best: a smaller ratio distorts less.
            tol  = [ i for i in ok_i if rm[ i ] <= rm[ b ] * 1.02 ]
            pick = SCALINGS[ min( tol ) ]
            if b == len( SCALINGS ) - 1:
                pick, note = CAP, 'LOO min at range edge, capped'
        fitted[ ( var, name ) ] = pick
        row = ''.join( ( f'{v:>8.2f}' if res[ i ] >= MIN_RESOLUTION else f'{v:>7.2f}~' )
                       for i, v in enumerate( rm ) )
        print( f'{name:<14}' + row + f'  {pick:>6g}  {note}' )

print( '\n\n# ── Fitted anisotropy, for the figure scripts ──' )
print( 'ANISOTROPY = {' )
for var in SOURCES:
    entries = { n: s for ( v, n ), s in fitted.items() if v == var }
    print( f"    {var!r:18}: {entries!r}," )
print( '}' )
