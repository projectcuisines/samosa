"""ExoCAM surface temperature by universal kriging, with a one-dimensional model
as the external drift (Figure 19, Section 4 of the manuscript).

This is two-fidelity kriging. The low-resolution field is HEXTOR or ExoColumn,
which have run all 64 protocol cases, and the high-resolution field is ExoCAM,
which has 10 stable cases among the 16. The low-resolution field is first kriged
on its own (ordinary kriging of its stable Cases 1-64, as in the right-hand
panels of Figure 18), which gives a value L(S, p) everywhere. ExoCAM is then
kriged with L as a specified drift,

    T_ExoCAM(S, p) = a + b L(S, p) + r(S, p),

where a and b are fitted implicitly by the universal kriging system and r is
the kriged residual. Where ExoCAM behaves like a shifted and scaled copy of the
1-D model, the drift carries the 1-D model's shape into the gaps between
ExoCAM's cases.

Two caveats built into the method. pykrige fits the variogram to the raw
values, not to the residuals from the drift, so the variogram (and σ) is that
of the full field; the regression-kriging check below fits the drift by least
squares and kriges the residuals instead, as a test of how much that matters.
And L at ExoCAM's two warmest cases (12 and 16) is an extrapolation for
ExoColumn, which runs away there; HEXTOR has Case 16 itself.

    cd figures/allcases && python fig_interpolation_temp_uk.py

Prints leave-one-out errors for ordinary kriging of ExoCAM alone and for the
two drifts, over a range of anisotropy ratios, and the drift coefficients.
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from matplotlib.transforms import offset_copy
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from fig_interpolation_temp_1d import ( CASES, MODELS as LOWRES, samples, ANISO_64,
                                        pn2, flux, fluxscale, norm_pres, norm_flux,
                                        kriging as ok_1d,
                                        cm, contourmin, contourmax, cinterval,
                                        cbar_label, cbar_ticks )

warnings.filterwarnings( 'ignore' )
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

# ExoCAM global mean surface temperature by case (fig_interpolation_temp.py);
# Cases 2, 3, 5, 6, 7 and 13 ran away.
EXOCAM = { 1: 196.8, 4: 260.0, 8: 243.8, 9: 244.8, 10: 194.1, 11: 234.0,
           12: 350.9, 14: 236.8, 15: 211.5, 16: 356.7 }
ec  = np.array( sorted( EXOCAM ) )
efs = np.array( [ CASES[ c ][ 0 ] for c in ec ] ) / fluxscale
eps = np.array( [ CASES[ c ][ 1 ] for c in ec ] )
ev  = np.array( [ EXOCAM[ c ] for c in ec ] )

ANISO_OK = 10          # ExoCAM's Figure 3 ratio
SCALINGS = [ 1, 1.5, 2, 3, 4, 5, 7, 10, 15 ]


def lowres_field( name ):
    """The low-resolution model kriged from its 64-case set: ( L at ExoCAM's
    cases, L on the grid ( flux x pressure ), kriging variance on the grid )."""
    fs, ps, vals, _ = samples( LOWRES[ name ], 64 )
    ok = ok_1d( fs, ps, vals, ANISO_64[ name ] )
    at_pts, _ = ok.execute( 'points', norm_pres( eps ), norm_flux( efs ) )
    grid, var = ok.execute( 'grid', norm_pres( pn2 ), norm_flux( flux ) )
    return np.asarray( at_pts ), np.asarray( grid ), np.asarray( var )


def uk( fs, ps, vals, drift, aniso ):
    return UniversalKriging( norm_pres( ps ), norm_flux( fs ), vals,
                             drift_terms=[ 'specified' ], specified_drift=[ drift ],
                             anisotropy_scaling=aniso, variogram_model='linear',
                             verbose=False, enable_plotting=False, exact_values=True )


def ok( fs, ps, vals, aniso ):
    return OrdinaryKriging( norm_pres( ps ), norm_flux( fs ), vals,
                            anisotropy_scaling=aniso, variogram_model='linear',
                            verbose=False, enable_plotting=False, exact_values=True )


def loo( method, aniso, drift=None ):
    """Leave-one-out errors (kriged - ExoCAM) at ExoCAM's stable cases."""
    err = []
    for i in range( len( ev ) ):
        k = np.arange( len( ev ) ) != i
        pi, fi = norm_pres( eps[ [ i ] ] ), norm_flux( efs[ [ i ] ] )
        if method == 'ok':
            z, _ = ok( efs[ k ], eps[ k ], ev[ k ], aniso ).execute( 'points', pi, fi )
        elif method == 'uk':
            z, _ = uk( efs[ k ], eps[ k ], ev[ k ], drift[ k ], aniso ).execute(
                'points', pi, fi, specified_drift_arrays=[ drift[ [ i ] ] ] )
        elif method == 'rk':
            # regression kriging: least-squares drift, residuals kriged
            b, a = np.polyfit( drift[ k ], ev[ k ], 1 )
            r = ev[ k ] - ( a + b*drift[ k ] )
            z, _ = ok( efs[ k ], eps[ k ], r, aniso ).execute( 'points', pi, fi )
            z = np.asarray( z ) + a + b*drift[ i ]
        elif method == 'lin':
            # the drift alone, no kriging of the residual
            b, a = np.polyfit( drift[ k ], ev[ k ], 1 )
            z = [ a + b*drift[ i ] ]
        err.append( float( np.asarray( z ).ravel()[ 0 ] ) - ev[ i ] )
    return np.array( err )


def rmse( e ):
    return np.sqrt( np.mean( e**2 ) )


if __name__ == '__main__':
    fields = { name: lowres_field( name ) for name in LOWRES }

    # ── Cross-validation ────────────────────────────────────────────────────
    print( 'ExoCAM stable cases:', ec.tolist() )
    print( '\nLow-resolution drift L at ExoCAM cases (K):' )
    print( '  case   ExoCAM ' + ''.join( f'{n:>11}' for n in LOWRES ) )
    for i, c in enumerate( ec ):
        tags = ''.join( f'{fields[ n ][ 0 ][ i ]:10.1f}' + ( ' ' if c in LOWRES[ n ] else '*' ) for n in LOWRES )
        print( f'  {c:4d} {ev[ i ]:8.1f} {tags}' )
    print( '  (* = not a stable case of that model; L is kriged there)' )
    for n in LOWRES:
        b, a = np.polyfit( fields[ n ][ 0 ], ev, 1 )
        r = np.corrcoef( fields[ n ][ 0 ], ev )[ 0, 1 ]
        print( f'  {n}: ExoCAM = {a:+.1f} + {b:.3f} L, r = {r:.3f}, residual std {np.std( ev - a - b*fields[ n ][ 0 ] ):.1f} K' )

    print( '\nLeave-one-out RMSE (K) by anisotropy ratio' )
    print( '  ratio   OK ExoCAM' + ''.join( f'  UK {n:<9} RK {n:<9}' for n in LOWRES ) )
    table = {}
    for s in SCALINGS:
        row = { 'ok': rmse( loo( 'ok', s ) ) }
        for n in LOWRES:
            row[ ( 'uk', n ) ] = rmse( loo( 'uk', s, fields[ n ][ 0 ] ) )
            row[ ( 'rk', n ) ] = rmse( loo( 'rk', s, fields[ n ][ 0 ] ) )
        table[ s ] = row
        print( f'  {s:5}   {row[ "ok" ]:9.1f}' + ''.join(
            f'  {row[ ( "uk", n ) ]:12.1f} {row[ ( "rk", n ) ]:12.1f}' for n in LOWRES ) )
    for n in LOWRES:
        print( f'  drift alone ({n}, a + b L, no kriging): {rmse( loo( "lin", 0, fields[ n ][ 0 ] ) ):.1f}' )

    # UK, like OK, keeps improving up to the cap, so both are compared at
    # ExoCAM's Figure 3 ratio rather than each at its own optimum.
    best = { n: ANISO_OK for n in LOWRES }

    print( f'\nPer-case LOO errors (K), all at ratio {ANISO_OK}' )
    e_ok = loo( 'ok', ANISO_OK )
    e_uk = { n: loo( 'uk', best[ n ], fields[ n ][ 0 ] ) for n in LOWRES }
    print( '  case  ExoCAM      OK' + ''.join( f'{"UK " + n:>14}' for n in LOWRES ) )
    for i, c in enumerate( ec ):
        print( f'  {c:4d} {ev[ i ]:7.1f} {e_ok[ i ]:+7.1f}' + ''.join( f'{e_uk[ n ][ i ]:+14.1f}' for n in LOWRES ) )
    print( f'  RMSE          {rmse( e_ok ):7.1f}' + ''.join( f'{rmse( e_uk[ n ] ):14.1f}' for n in LOWRES ) )
    print( f'  RMSE w/o 12,16{rmse( e_ok[ ( ec != 12 ) & ( ec != 16 ) ] ):7.1f}' + ''.join(
        f'{rmse( e_uk[ n ][ ( ec != 12 ) & ( ec != 16 ) ] ):14.1f}' for n in LOWRES ) )

    # ── Figure: one row per low-resolution model ────────────────────────────
    # low-res field | ExoCAM, ordinary | ExoCAM, universal | universal - ordinary
    fig, axs = plt.subplots( 2, 4, figsize=(12.9, 6.6) )
    xv, yv = np.meshgrid( pn2, flux )
    levels = np.linspace( contourmin, contourmax, cinterval )
    dlevels = [ l for l in range( -60, 61, 5 ) if l ]     # K, zero omitted

    z_ok, var_ok = ok( efs, eps, ev, ANISO_OK ).execute( 'grid', norm_pres( pn2 ), norm_flux( flux ) )

    # ExoCAM's cases as circles, the 1-D model's as smaller diamonds beneath
    # them; hollow on the difference panels, whose colors are not temperatures
    def markers( ax, cases, temps, c=True, marker='o' ):
        f = [ CASES[ k ][ 0 ] for k in cases ]; p = [ CASES[ k ][ 1 ] for k in cases ]
        big = marker == 'o'
        ax.scatter( f, p, c=[ temps[ k ] for k in cases ] if c else 'w', cmap=cm,
                    vmin=contourmin, vmax=contourmax, marker=marker, s=40 if big else 22,
                    edgecolors='k', linewidths=1.0 if big else 0.8, zorder=4 if big else 3 )


    for row, name in enumerate( LOWRES ):
        drift_pts, drift_grid, _ = fields[ name ]
        s = best[ name ]
        u = uk( efs, eps, ev, drift_pts, s )
        z_uk, var_uk = u.execute( 'grid', norm_pres( pn2 ), norm_flux( flux ),
                                  specified_drift_arrays=[ drift_grid ] )
        _, _, _, lc = samples( LOWRES[ name ], 64 )

        ax = axs[ row, 0 ]
        cf = ax.contourf( yv*fluxscale, xv, drift_grid, cmap=cm, levels=levels, extend='both' )
        markers( ax, lc, LOWRES[ name ], marker='D' )
        ax.set_title( f'{name} (n={len( lc )})', fontsize=12 )

        ax = axs[ row, 1 ]
        ax.contourf( yv*fluxscale, xv, z_ok, cmap=cm, levels=levels, extend='both' )
        markers( ax, ec, EXOCAM )
        ax.set_title( f'ExoCAM (n={len( ec )})', fontsize=12 )

        ax = axs[ row, 2 ]
        ax.contourf( yv*fluxscale, xv, z_uk, cmap=cm, levels=levels, extend='both' )
        markers( ax, lc, LOWRES[ name ], marker='D' ); markers( ax, ec, EXOCAM )
        ax.set_title( 'Universal kriging', fontsize=12 )

        ax = axs[ row, 3 ]
        # The difference as labeled lines, warming solid red and cooling dashed
        # blue, so the figure needs only the temperature colorbar
        dc = ax.contour( yv*fluxscale, xv, z_uk - z_ok, levels=dlevels, linewidths=1.0,
                         colors=[ 'tab:blue' if l < 0 else 'tab:red' for l in dlevels ],
                         linestyles=[ '--' if l < 0 else '-' for l in dlevels ] )
        ax.clabel( dc, fmt='%+d', fontsize=9, inline_spacing=2 )
        markers( ax, lc, LOWRES[ name ], c=False, marker='D' ); markers( ax, ec, EXOCAM, c=False )
        ax.set_title( 'Universal − ordinary (K)', fontsize=12 )

        print( f'\n{name} drift, ratio {s}: fitted linear variogram slope {u.variogram_model_parameters[ 0 ]:.1f}, '
               f'nugget {u.variogram_model_parameters[ 1 ]:.1f}; UK - OK on the grid: '
               f'{np.min( z_uk - z_ok ):+.1f} to {np.max( z_uk - z_ok ):+.1f} K; '
               f'UK field range {np.min( z_uk ):.1f}-{np.max( z_uk ):.1f} K' )

    for ax in axs.flat:
        ax.tick_params( axis='both', labelsize=10 )
        ax.set_yscale( 'log' )
        ax.set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
        ax.set_xticks( [ 2500, 2000, 1500, 1000, 500 ] )
        ax.set_ylim( [ min( pn2 )*0.9, max( pn2 )*1.1 ] )
        ax.set_box_aspect( 1 )
        ax.apply_aspect()
    for ax in axs[ :, 1: ].flat:
        ax.tick_params( labelleft=False )
    for ax in axs[ 0, : ]:
        ax.tick_params( labelbottom=False )

    fig.subplots_adjust( wspace=0.12, hspace=0.25, right=0.84 )
    top, bot = axs[ 0, 0 ].get_position(), axs[ 1, 0 ].get_position()
    fig.text( top.x0, ( bot.y0 + top.y1 )/2, 'Surface pressure (bar)', rotation=90, ha='right', va='center',
              fontsize=12, fontweight='bold', transform=offset_copy( fig.transFigure, fig=fig, x=-38, units='points' ) )
    fig.text( ( bot.x0 + axs[ 1, -1 ].get_position().x1 )/2, bot.y0, 'Instellation (W m$^{-2}$)', ha='center', va='top',
              fontsize=12, fontweight='bold', transform=offset_copy( fig.transFigure, fig=fig, y=-25, units='points' ) )

    right = axs[ 0, -1 ].get_position().x1
    cax = fig.add_axes( [ right + 0.015, bot.y0, 0.011, top.y1 - bot.y0 ] )
    cb = fig.colorbar( cf, cax=cax, extend='both', ticks=cbar_ticks )
    cb.ax.tick_params( labelsize=10 ); cb.set_label( cbar_label, rotation=270, fontsize=12, labelpad=16 )

    fig.savefig( 'fig_interpolation_temp_uk.png', bbox_inches='tight' )
    fig.savefig( 'fig_interpolation_temp_uk.eps', bbox_inches='tight' )
