import warnings; warnings.filterwarnings( 'ignore' )
import numpy as np
from pykrige.ok import OrdinaryKriging
flux = np.arange( 400, 2700, 100 ) / 100
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )
lp = np.log( pn2 ); np_ = lambda p: ( np.log( p ) - lp.min() ) / np.ptp( lp ); nf = lambda f: ( f - flux.min() ) / np.ptp( flux )
T  = np.array( [ 195.37, 251.48, 400.52, 231.83, 241.35, 197.81, 227.52, 333.20, 228.84, 203.64, 361.70 ] )
F  = np.array( [ 500, 1200, 1600, 800, 1100, 400, 900, 1500, 900, 600, 1400 ] ) / 100
P  = np.array( [ 0.70, 2.34, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )
C  = np.array( [ 1, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16 ] )
S  = [ 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50 ]
def loo( p, f, v, s, fam ):
    r = []
    for i in range( len( v ) ):
        m = np.arange( len( v ) ) != i
        try:
            ok = OrdinaryKriging( np_( p[m] ), nf( f[m] ), v[m], variogram_model=fam, anisotropy_scaling=s, verbose=False, enable_plotting=False, exact_values=True )
            r.append( v[i] - ok.execute( 'points', np_( p[[i]] ), nf( f[[i]] ) )[0][0] )
        except Exception: r.append( np.nan )
    return np.sqrt( np.nanmean( np.square( r ) ) ), np.array( r )
for label, keep in ( ( 'all 11 cases', np.ones( 11, bool ) ), ( 'without Case 7', C != 7 ) ):
    print( f'--- LFRic temperature, {label}: LOO RMSE (K) by anisotropy ---' )
    print( f"{'family':<12}" + ''.join( f'{s:>7g}' for s in S ) )
    for fam in ( 'linear', 'spherical', 'exponential', 'gaussian' ):
        print( f'{fam:<12}' + ''.join( f'{loo( P[keep], F[keep], T[keep], s, fam )[0]:7.1f}' for s in S ) )
# Which case dominates the LOO error at s = 4 and 15, linear, all cases
for s in ( 4, 15 ):
    rm, r = loo( P, F, T, s, 'linear' )
    print( f's={s}: per-case LOO residual (K):', ' '.join( f'C{c}:{x:+.0f}' for c, x in zip( C, r ) ) )
