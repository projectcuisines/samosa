"""Recompute every manuscript statistic that HEXTOR Case 16 can touch.

Usage: python case16_stats.py [figures/allcases dir, default the repo's]
Run against the HEAD copy first: its numbers must match the text before the
same code is trusted on the edited scripts.
"""
import contextlib, io, os, runpy, sys, warnings
import numpy as np
import matplotlib
matplotlib.use( 'Agg' )
warnings.filterwarnings( 'ignore' )

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
from _paths import FIG_ALL, scratch_copy
os.chdir( scratch_copy( sys.argv[ 1 ] if len( sys.argv ) > 1 else FIG_ALL ) )

def load( path ):
    with contextlib.redirect_stdout( io.StringIO() ):
        return runpy.run_path( path )

RESOLVED = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab', 'Generic PCM', 'LFRic' ]
ONE_D    = [ 'HEXTOR', 'ExoColumn' ]

# ── A. Sample-point temperature ranges ────────────────────────────────────────
cv = load( 'crossval_variogram.py' )
flux1, pres1 = cv[ 'flux1' ], cv[ 'pres1' ]
def per_case( models ):
    out = {}
    for c in range( 16 ):
        v = {}
        for n in models:
            p, f, t = cv[ 'climate_models' ][ n ]
            m = np.isclose( f, flux1[ c ] ) & np.isclose( p, pres1[ c ] )
            if m.any():
                v[ n ] = float( np.asarray( t )[ m ][ 0 ] )
        out[ c + 1 ] = v
    return out
T8 = per_case( RESOLVED + ONE_D )
T6 = per_case( RESOLVED )
def ranges( T ):
    return { c: max( v.values() ) - min( v.values() ) for c, v in T.items() if len( v ) >= 3 }
r8, r6 = ranges( T8 ), ranges( T6 )
print( '=== A. sample-point Tglob range, cases with >=3 models ===' )
print( f'  all eight: {min(r8.values()):.1f}-{max(r8.values()):.1f} K, median {np.median(list(r8.values())):.1f} K over {len(r8)} cases' )
print( f'  six resolved: median {np.median(list(r6.values())):.1f} K over {len(r6)} cases' )
print( '  widest (all eight):', ', '.join( f'C{c} {r:.1f}' for c, r in sorted( r8.items(), key=lambda x: -x[1] )[ :6 ] ) )
print( '  six-model range at those:', ', '.join( f'C{c} {r6.get(c, float("nan")):.1f}' for c, _ in sorted( r8.items(), key=lambda x: -x[1] )[ :6 ] ) )
allv = [ ( t, n, c ) for c, v in T8.items() for n, t in v.items() ]
print( '  coldest', min( allv ), ' warmest', max( allv ) )
print( f'  Case 16 models: ' + ', '.join( f'{n} {t:.1f}' for n, t in sorted( T8[16].items(), key=lambda x: x[1] ) ) )
print( '  HEXTOR minus resolved median:', ', '.join(
    f'C{c} {T8[c]["HEXTOR"] - np.median([t for n, t in T8[c].items() if n in RESOLVED]):+.1f}'
    for c in sorted( T8 ) if 'HEXTOR' in T8[ c ] ) )
n5 = [ c for c, v in T8.items() if len( v ) >= 5 ]; n8 = [ c for c, v in T8.items() if len( v ) == 8 ]
print( f'  cases with >=5 models: {n5}; with all 8: {n8}' )

# ── C. Albedo ranges ──────────────────────────────────────────────────────────
al = load( 'fig_interpolation_albedo.py' )
def alb_models():
    run = al[ 'runaway' ]
    reg = { 'ExoPlaSim': ( flux1, pres1, al[ 'plasim' ] ) }
    for n, k in ( ( 'ExoCAM', 'exocam' ), ( 'ROCKE-3D', 'rocke3d' ), ( 'PlaHab', 'plahab' ) ):
        v = al[ k ]; m = v != run
        reg[ n ] = ( flux1[ m ], pres1[ m ], v[ m ] )
    for n, k in ( ( 'Generic PCM', 'pcm' ), ( 'LFRic', 'lfric' ), ( 'HEXTOR', 'hextor' ), ( 'ExoColumn', 'exocolumn' ) ):
        reg[ n ] = ( al[ k + '_flux1' ], al[ k + '_pres1' ], al[ k ] )
    return reg
AR = alb_models()
def alb_case( models ):
    out = {}
    for c in range( 16 ):
        v = {}
        for n in models:
            f, p, a = AR[ n ]
            m = np.isclose( f, flux1[ c ] ) & np.isclose( p, pres1[ c ] )
            if m.any():
                v[ n ] = float( a[ m ][ 0 ] )
        out[ c + 1 ] = v
    return out
A8, A6 = alb_case( RESOLVED + ONE_D ), alb_case( RESOLVED )
ra8, ra6 = ranges( A8 ), ranges( A6 )
print( '\n=== C. albedo range, cases with >=3 models ===' )
print( f'  all eight: {min(ra8.values()):.1f}-{max(ra8.values()):.1f} points, median {np.median(list(ra8.values())):.1f}' )
print( f'  six resolved: {min(ra6.values()):.1f}-{max(ra6.values()):.1f} points, median {np.median(list(ra6.values())):.1f}' )
dark = [ c for c, v in A8.items() if 'HEXTOR' in v and min( v, key=v.get ) == 'HEXTOR' ]
print( f'  HEXTOR darkest at {len(dark)} of {sum("HEXTOR" in v for v in A8.values())} cases: {dark}' )
print( '  HEXTOR albedo:', { c: v[ 'HEXTOR' ] for c, v in A8.items() if 'HEXTOR' in v } )

# ── D. Spread ─────────────────────────────────────────────────────────────────
sp = load( 'fig_spread.py' )
# fig_spread.py no longer keeps each model's kriged field at module level, so
# krige them here from its TS_MODELS table, in the order the spread is accumulated.
ws, krige = sp[ 'weighted_std' ], sp[ 'krige' ]
_kr = [ krige( *sp[ 'TS_MODELS' ][ n ], sp[ 'ANISO_TS' ][ n ] ) for n in sp[ 'TS_MODELS' ] ]
z, var = [ np.asarray( k[ 0 ] ) for k in _kr ], [ np.asarray( k[ 1 ] ) for k in _kr ]
assert list( sp[ 'TS_MODELS' ] ) == [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab', 'Generic PCM', 'LFRic', 'HEXTOR', 'ExoColumn' ]
F = sp[ 'flux' ] * 100; P = sp[ 'pn2' ]
order = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab', 'Generic PCM', 'LFRic', 'HEXTOR', 'ExoColumn' ]
def sd( names, zz=None ):
    zz = zz or z
    idx = [ order.index( n ) for n in names ]
    return ws( [ zz[ i ] for i in idx ], [ var[ i ] for i in idx ] )
S8 = sd( order ); S6 = sd( RESOLVED ); S7 = sd( RESOLVED + [ 'HEXTOR' ] )
hx_f = sp[ 'hextor_flux1' ].max() * 100
# In-band is the instellation range BOTH one-dimensional models sampled (the
# definition the manuscript uses since HEXTOR Case 16, at 1400 W/m2, was
# accepted on 2026-09-16), i.e. up to 1200 W/m2.
band = ( F <= min( hx_f, sp[ 'exocolumn_flux1' ].max() * 100 ) )[ :, None ] & np.ones( ( 1, len( P ) ), bool )
print( '\n=== D. temperature spread (Figure spread) ===' )
i, j = np.unravel_index( np.argmax( S8 ), S8.shape )
print( f'  all eight: median {np.median(S8):.1f}, min {S8.min():.1f}, max {S8.max():.1f} at {F[i]:.0f} W/m2, {P[j]} bar' )
def at( S, f, p ): return S[ np.argmin( abs( F - f ) ), np.argmin( abs( P - p ) ) ]
print( f'  at Case 7 (1600, 0.55): {at(S8,1600,0.55):.1f}; at Case 16 (1400, 10): {at(S8,1400,10.0):.1f}; corner (1900, 10): {at(S8,1900,10.0):.1f}' )
# local maxima on the grid
loc = [ ( S8[a, b], F[a], P[b] ) for a in range( len( F ) ) for b in range( len( P ) )
        if S8[a, b] == S8[ max(a-1,0):a+2, max(b-1,0):b+2 ].max() ]
print( '  local maxima:', ', '.join( f'{s:.1f} at ({f:.0f}, {p})' for s, f, p in sorted( loc, reverse=True )[ :4 ] ) )
print( f'  whole-plane medians: six {np.median(S6):.1f}, +HEXTOR {np.median(S7):.1f}, +both {np.median(S8):.1f}' )
print( f'  band <= {band.any(axis=1).nonzero()[0].max()*100+400:.0f} W/m2: six {np.median(S6[band]):.1f} -> eight {np.median(S8[band]):.1f}; '
       f'outside: six {np.median(S6[~band]):.1f} -> eight {np.median(S8[~band]):.1f}' )
# LFRic Case 7 and Cases 8/10/11 withheld
lf = sp[ 'lfric_flux1' ] * 100; lp = sp[ 'lfric_pres1' ]; lt = sp[ 'ts_lfric' ]
def lfric_without( drop ):
    keep = np.ones( len( lt ), bool )
    for f, p in drop:
        keep &= ~( np.isclose( lf, f ) & np.isclose( lp, p ) )
    zz = list( z ); zz[ order.index( 'LFRic' ) ] = krige( lp[ keep ], lf[ keep ] / 100, lt[ keep ], sp[ 'ANISO_TS' ][ 'LFRic' ] )[ 0 ]
    return sd( order, zz )
S_no7 = lfric_without( [ ( 1600, 0.55 ) ] )
zz7 = list( z ); _k = ~( np.isclose( lf, 1600 ) & np.isclose( lp, 0.55 ) )
zz7[ order.index( 'LFRic' ) ] = krige( lp[_k], lf[_k] / 100, lt[_k], sp[ 'ANISO_TS' ][ 'LFRic' ] )[ 0 ]
S6_no7 = sd( RESOLVED, zz7 )
print( f'  LFRic Case 7 on six-model medians: in-band {np.median(S6[band])-np.median(S6_no7[band]):+.2f}, out {np.median(S6[~band])-np.median(S6_no7[~band]):+.2f}' )
S_no81011 = lfric_without( [ ( 800, 6.16 ), ( 400, 4.83 ), ( 900, 0.10 ) ] )
for lab, S in ( ( 'Case 7', S_no7 ), ( 'Cases 8/10/11', S_no81011 ) ):
    print( f'  LFRic {lab} moves: whole {np.median(S8)-np.median(S):+.2f}, in-band {np.median(S8[band])-np.median(S[band]):+.2f}, '
           f'out {np.median(S8[~band])-np.median(S[~band]):+.2f}' )
ex = { c: T8[c]['ExoColumn'] - T8[c]['HEXTOR'] for c in T8 if 'ExoColumn' in T8[c] and 'HEXTOR' in T8[c] }
print( f'  ExoColumn minus HEXTOR at {len(ex)} shared cases: {min(ex.values()):+.1f} to {max(ex.values()):+.1f}' )

# ── E. Summary figure ─────────────────────────────────────────────────────────
sm = load( 'fig_summary.py' )
# fig_summary.py now returns each panel's regions from consensus(); the full
# plane is panels[0]. Areas are weighted by cell width in log pressure, as in
# its report_areas(), because the pressure grid is non-uniform.
full = sm[ 'panels' ][ 0 ]
fluxf = np.asarray( full[ 'flux_grid' ] ) * 100; pn2f = np.asarray( full[ 'pres_grid' ] ); close = sm[ 'close_rows' ]
_W = np.broadcast_to( np.gradient( np.log( pn2f ) ), ( len( fluxf ), len( pn2f ) ) )
area = lambda M: float( np.sum( M * _W ) / np.sum( _W ) )
C = full[ 'contested_all' ]
print( '\n=== E. summary figure ===' )
print( f'  blue {100*area(full["band_blue"]):.1f}%, green {100*area(full["band_warm"]):.1f}%, contested {100*area(C):.1f}%, '
       f'3-D-only {100*area(full["contested_3d"]):.1f}%, PlaHab-only {100*area(full["contested_plahab"]):.1f}%' )
# How closely PlaHab's kriged global mean follows the three full-coverage 3-D
# GCMs in sign: the share of the area those three place below (above) 273.16 K
# where PlaHab does too. Unfaded, since all four ran every case. The Figure 17
# TO DO on whether PlaHab should vote quotes these. The old "95% / 98%" there
# dated from 2026-08-19 and no script reproduced it.
_Z = full[ 'Z' ]
_cold = np.all( [ _Z[ n ] <  273.16 for n in ( 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D' ) ], axis=0 )
_warm = np.all( [ _Z[ n ] >= 273.16 for n in ( 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D' ) ], axis=0 )
print( f'  PlaHab agrees in sign with ExoPlaSim/ExoCAM/ROCKE-3D over {100*area(_cold & (_Z["PlaHab"] < 273.16))/area(_cold):.1f}% '
       f'of their frozen area and {100*area(_warm & (_Z["PlaHab"] >= 273.16))/area(_warm):.1f}% of their unfrozen area' )
def edges( M ):
    lo = np.full( len( pn2f ), np.nan ); hi = lo.copy()
    for j in range( len( pn2f ) ):
        w = np.where( M[ :, j ] )[ 0 ]
        if len( w ): lo[ j ], hi[ j ] = fluxf[ w.min() ], fluxf[ w.max() ]
    return lo, hi
lo, hi = edges( C )
for p in ( 0.10, 2.34, 10.0 ):
    j = np.argmin( abs( pn2f - p ) )
    print( f'  band at {pn2f[j]:.2f} bar: {lo[j]:.0f}-{hi[j]:.0f}' )
# The six-resolved consensus: consensus() takes its model lists from the
# script's globals, so narrow them to the six for this one call.
_G = sm[ 'consensus' ].__globals__
_saved = _G[ 'mean_models' ], _G[ 'partial_models' ]
_G[ 'mean_models' ]    = [ n for n in _saved[ 0 ] if n in RESOLVED ]
_G[ 'partial_models' ] = [ n for n in _saved[ 1 ] if n in RESOLVED ]
six = sm[ 'consensus' ]( { n: sm[ 'ts_in' ][ n ] for n in RESOLVED }, sm[ 'pn2f' ], sm[ 'fluxf' ], fade=True )
_G[ 'mean_models' ], _G[ 'partial_models' ] = _saved
C6 = six[ 'contested_all' ]
print( f'  six resolved contested: {100*area(C6):.1f}%' )
lo6, hi6 = edges( C6 )
d = lo6 - lo
k = np.nanargmax( d )
print( f'  1-D cold-edge shift: median {np.nanmedian(d[pn2f>0.1+1e-9]):.0f}, max {d[k]:.0f} at {pn2f[k]:.2f} bar; '
       f'rows with shift>0: {np.sum(d>0)}/{len(d)}; median of positive shifts {np.median(d[d>0]):.0f}; warm-edge shift max {np.nanmax(abs(hi6-hi)):.0f}' )
s_lo, s_hi = np.abs( np.diff( lo ) ), np.abs( np.diff( hi ) )
kk = np.argmax( np.maximum( s_lo, s_hi ) )
print( f'  largest adjacent step: cold {s_lo.max():.0f}, warm {s_hi.max():.0f} (at {pn2f[kk]:.2f}-{pn2f[kk+1]:.2f} bar); '
       f'median {np.median(np.concatenate([s_lo, s_hi])):.0f}' )
# invariants
blue, warm = full[ 'band_blue' ], full[ 'band_warm' ]
part = ( blue.astype( int ) + warm + C )
gaps = sum( 1 for j in range( len( pn2f ) ) if len( np.where( C[:, j] )[0] ) and
            not C[ np.where( C[:, j] )[0].min():np.where( C[:, j] )[0].max()+1, j ].all() )
eb = load( 'fig_energy_balance.py' ) if False else None
reg = open( 'fig_energy_balance.py' ).read()
reg = eval( reg[ reg.index( 'regime = [' ) + len( 'regime = ' ):reg.index( ']', reg.index( 'regime = [' ) ) + 1 ] )
mism = []
for c in range( 16 ):
    a = np.argmin( abs( fluxf - flux1[ c ] * 100 ) ); b = np.argmin( abs( pn2f - pres1[ c ] ) )
    got = 'frozen' if blue[ a, b ] else 'warm' if warm[ a, b ] else 'mixed' if C[ a, b ] else 'none'
    if reg[ c ] != 'runaway' and got != reg[ c ]:
        mism.append( ( c + 1, reg[ c ], got ) )
print( f'  invariants: partition ok {bool(np.all(part == 1))}; row gaps {gaps}; white subset {bool(np.all(~full["contested_3d"] | C))}; '
       f'regime mismatches (non-runaway) {mism}' )
