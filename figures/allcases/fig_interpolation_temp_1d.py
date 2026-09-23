"""Surface temperature of the two one-dimensional models, kriged from the 16
primary cases and from all 64 cases of the protocol.

HEXTOR and ExoColumn are the only models to have run the protocol's optional
Sequences 1b, 2b and 3 (Cases 17-64, Table 4 of Haqq-Misra et al. 2022), so for
them the sparse sample can be checked against a denser one. For each model the
left panel is kriged as in Figure 3 (fig_interpolation_temp.py), from its
stable Cases 1-16, and the right panel from its stable cases among all 64.
Every panel shows the same markers, so the new cases sit over the 16-case
surface they were not used to build.

    cd figures/allcases && python fig_interpolation_temp_1d.py

Also prints every number Section 4 of the manuscript quotes except the
leave-one-out errors, which come from fit_anisotropy.py: the 16-case kriging
evaluated at the new stable cases, the case counts, the HEXTOR-ExoColumn
comparison and the inner edge each sample brackets. Run fit_anisotropy.py to refit ANISO_64,
which it reads from climate_models below.
"""
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from matplotlib.transforms import offset_copy
from pykrige.ok import OrdinaryKriging

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

# ─── Variable configuration, as in fig_interpolation_temp.py ─────────────────
cm              = cmocean.cm.thermal
contourmin      = 175.0
contourmax      = 370.0
cinterval       = 40
cbar_label      = 'Average Surface Temperature (K)'
cbar_ticks      = np.arange( 200, 370, 50 )
# ─────────────────────────────────────────────────────────────────────────────

fluxscale = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# ( instellation [W/m2], N2 pressure [bar] ) by case. Cases 1-16 are Table 1;
# 17-64 are Table 4 of the protocol as printed. Sequence 3 (33-64) lies off the
# ExoPlaSim grid. The protocol's own generator reproduces every row of Table 4
# except Case 39, which it puts at 2276 W/m2; the published 2279 is what both
# models ran.
CASES = {
    1: ( 500, 0.70 ),    2: ( 1900, 7.85 ),   3: ( 2400, 0.21 ),   4: ( 1200, 2.34 ),
    5: ( 1500, 0.16 ),   6: ( 2100, 1.83 ),   7: ( 1600, 0.55 ),   8: ( 800, 6.16 ),
    9: ( 1100, 0.70 ),  10: ( 400, 4.83 ),   11: ( 900, 0.10 ),   12: ( 1500, 2.98 ),
   13: ( 1600, 0.16 ),  14: ( 900, 1.44 ),   15: ( 600, 0.43 ),   16: ( 1400, 10.00 ),
   17: ( 700, 0.26 ),   18: ( 1700, 2.98 ),  19: ( 2300, 0.89 ),  20: ( 1300, 10.00 ),
   21: ( 900, 0.43 ),   22: ( 2600, 4.83 ),  23: ( 2000, 0.10 ),  24: ( 500, 1.13 ),
   25: ( 1300, 0.16 ),  26: ( 500, 2.34 ),   27: ( 1000, 0.89 ),  28: ( 1700, 3.79 ),
   29: ( 1400, 0.55 ),  30: ( 800, 7.85 ),   31: ( 500, 0.21 ),   32: ( 1200, 1.13 ),
   33: ( 1279, 3.92 ),  34: ( 2455, 1.00 ),  35: ( 1628, 1.07 ),  36: ( 466, 0.27 ),
   37: ( 838, 2.72 ),   38: ( 2013, 0.11 ),  39: ( 2279, 9.97 ),  40: ( 1114, 0.39 ),
   41: ( 981, 1.78 ),   42: ( 2143, 0.22 ),  43: ( 1871, 4.83 ),  44: ( 696, 0.60 ),
   45: ( 599, 6.00 ),   46: ( 1762, 0.48 ),  47: ( 2596, 2.20 ),  48: ( 1421, 0.18 ),
   49: ( 1490, 1.81 ),  50: ( 2528, 0.15 ),  51: ( 1692, 6.58 ),  52: ( 668, 0.53 ),
   53: ( 764, 5.29 ),   54: ( 1803, 0.66 ),  55: ( 2074, 1.46 ),  56: ( 1050, 0.18 ),
   57: ( 1182, 8.18 ),  58: ( 2208, 0.32 ),  59: ( 1944, 2.98 ),  60: ( 907, 0.12 ),
   61: ( 535, 1.17 ),   62: ( 1560, 0.30 ),  63: ( 2385, 3.21 ),  64: ( 1348, 0.82 ),
}

# Global mean surface temperature (K) of every stable case, by case number,
# read from the SAMOSA archive. The _all64 tables hold the protocol's 64 cases;
# their rows for Cases 1-16 are identical to the submitted 16-case tables that
# fig_interpolation_temp.py was transcribed from. Both models were run on the
# configuration of their submissions (see README_all64.txt in each folder).
# Every case not listed ran away. ExoColumn Case 57 is accepted on the
# protocol's stable-trend allowance (309.5 K for thousands of days, with
# +1.4 W/m2 at the top of the atmosphere from a water-budget residual).
ARCHIVE = '/models/data/samosa'
TABLES  = { 'HEXTOR':    f'{ARCHIVE}/hextor/global_output_HEXTOR_all64.dat',
            'ExoColumn': f'{ARCHIVE}/exocolumn/global_output_ExoColumn_a2736_all64.dat' }

def read_tglob( path ):
    """{ case: Tglob } from a SAMOSA global output file, checking each row's
    instellation and pressure against the case table above."""
    temps = {}
    for line in open( path ):
        if line.startswith( '#' ) or not line.strip():
            continue
        f = line.split()
        case, inst, pres = int( f[ 0 ] ), float( f[ 1 ] ), float( f[ 2 ] )
        assert abs( inst - CASES[ case ][ 0 ] ) < 0.5 and abs( pres - CASES[ case ][ 1 ] ) < 0.005, \
            f'{path}: case {case} at ( {inst}, {pres} ), not {CASES[ case ]}'
        temps[ case ] = float( f[ 3 ] )
    return temps

hextor    = read_tglob( TABLES[ 'HEXTOR' ] )
exocolumn = read_tglob( TABLES[ 'ExoColumn' ] )
MODELS = { 'HEXTOR': hextor, 'ExoColumn': exocolumn }
# Figure rows, top to bottom (Figure 20 follows the same order)
ROW_ORDER = [ 'ExoColumn', 'HEXTOR' ]

def samples( temps, last_case ):
    """( instellation / fluxscale, pressure, temperature, case ) of the stable cases up to last_case."""
    cs = np.array( [ c for c in sorted( temps ) if c <= last_case ] )
    return ( np.array( [ CASES[ c ][ 0 ] for c in cs ] ) / fluxscale,
             np.array( [ CASES[ c ][ 1 ] for c in cs ] ),
             np.array( [ temps[ c ] for c in cs ] ), cs )

# The 64-case sets in the registry format of crossval_variogram.py, so that
# fit_anisotropy.py can fit them: name -> ( pressure, instellation, values ).
climate_models = { name: ( ps, fs, vals ) for name, ( fs, ps, vals, _ ) in
                   ( ( n, samples( t, 64 ) ) for n, t in MODELS.items() ) }

# Kriging anisotropy (see fig_interpolation_temp.py). The 16-case panels keep
# Figure 3's ratios, so they reproduce its panels; the 64-case ratios are the
# fit_anisotropy.py result for these sets.
ANISO_16 = { 'HEXTOR': 10, 'ExoColumn': 7 }
ANISO_64 = { 'HEXTOR': 1.5, 'ExoColumn': 10 }

# Normalize both axes to [0, 1] for kriging so distance metric is balanced
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

def kriging( fs, ps, vals, aniso ):
    return OrdinaryKriging( norm_pres( ps ), norm_flux( fs ), vals,
                            anisotropy_scaling=aniso, variogram_model='linear',
                            verbose=False, enable_plotting=False, exact_values=True )


def out_of_sample( name ):
    """The 16-case kriging of one model evaluated at its new stable cases."""
    fs, ps, vals, _ = samples( MODELS[ name ], 16 )
    nf, np_, nv, nc = samples( MODELS[ name ], 64 )
    new = nc > 16
    z, var = kriging( fs, ps, vals, ANISO_16[ name ] ).execute( 'points', norm_pres( np_[ new ] ), norm_flux( nf[ new ] ) )
    err, sig = np.asarray( z ) - nv[ new ], np.sqrt( np.asarray( var ) )
    return nc[ new ], nv[ new ], np.asarray( z ), err, sig


if __name__ == '__main__':
    marker_edge = 'k'
    MARKER = { 'HEXTOR': 'X', 'ExoColumn': '*' }

    # One row per model: ( model, last case used ) for each panel
    PANELS = [ ( name, last ) for name in ROW_ORDER for last in ( 16, 64 ) ]

    # Panels ~2.1 in square, the size of those in Figures 3-6 and 20, with the
    # same fonts; the paper includes this figure at the fraction of \linewidth
    # that gives it their print scale, rather than at full width
    fig, axs = plt.subplots( 2, 2, figsize=(6.6, 6.6) )
    xv, yv = np.meshgrid( pn2, flux )
    levels = np.linspace( contourmin, contourmax, cinterval )
    for ax, ( name, last ) in zip( axs.flat, PANELS ):
        temps = MODELS[ name ]
        fs, ps, vals, _ = samples( temps, last )
        aniso = ( ANISO_16 if last == 16 else ANISO_64 )[ name ]
        z, _ = kriging( fs, ps, vals, aniso ).execute( 'grid', norm_pres( pn2 ), norm_flux( flux ) )
        cf = ax.contourf( yv*fluxscale, xv, z, cmap=cm, levels=levels, extend='both' )

        # The same markers in every panel, in the model's shape from Figure 2
        # (fig_energy_balance.py): large for Cases 1-16, small for 17-64, and
        # thin gray crosses where the model ran away, kept light so that they do
        # not read as HEXTOR's X. The star is drawn larger so that it reads at
        # the same weight as the X.
        af, ap, av, ac = samples( temps, 64 )
        old = ac <= 16
        mk, scale = MARKER[ name ], 1.8 if MARKER[ name ] == '*' else 1.0
        ax.scatter( af[ old ]*fluxscale, ap[ old ], c=av[ old ], cmap=cm, vmin=contourmin, vmax=contourmax,
                    marker=mk, s=75*scale, edgecolors=marker_edge, zorder=3, clip_on=False )
        ax.scatter( af[ ~old ]*fluxscale, ap[ ~old ], c=av[ ~old ], cmap=cm, vmin=contourmin, vmax=contourmax,
                    marker=mk, s=32*scale, edgecolors=marker_edge, linewidths=0.7, zorder=3, clip_on=False )
        runaway = [ c for c in CASES if c not in temps ]
        ax.scatter( [ CASES[ c ][ 0 ] for c in runaway ], [ CASES[ c ][ 1 ] for c in runaway ],
                    marker='x', s=18, c='0.4', linewidths=0.7, zorder=2 )

        ax.set_title( f'Cases 1–{last} (n={len( vals )})', fontsize=12 )
        ax.tick_params( axis='both', labelsize=10 )
        ax.set_yscale( 'log' )
        ax.set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
        ax.set_xticks( [ 2500, 2000, 1500, 1000, 500 ] )
        ax.set_ylim( [ min( pn2 )*0.9, max( pn2 )*1.1 ] )
        ax.set_box_aspect( 1 )

    fig.subplots_adjust( wspace=0.12, hspace=0.36, left=0.14, right=0.82, bottom=0.1, top=0.89 )
    for ax in axs.flat:
        ax.apply_aspect()     # square boxes fixed before positions are read

    # One bold header over each model's row of panels, as over the blocks of
    # Figure 3, and the axes labeled once for the figure
    above = offset_copy( fig.transFigure, fig=fig, y=23, units='points' )
    for i, name in enumerate( ROW_ORDER ):
        left, right = axs[ i, 0 ].get_position(), axs[ i, 1 ].get_position()
        fig.text( ( left.x0 + right.x1 )/2, left.y1, name, transform=above,
                  ha='center', va='bottom', fontsize=14, fontweight='bold' )
    for ax in axs[ :, 1 ]:
        ax.tick_params( labelleft=False )
    for ax in axs[ 0, : ]:
        ax.tick_params( labelbottom=False )
    top, bot, last_ax = axs[ 0, 0 ].get_position(), axs[ 1, 0 ].get_position(), axs[ 1, -1 ].get_position()
    fig.text( top.x0, ( bot.y0 + top.y1 )/2, 'Surface pressure (bar)',
              rotation=90, ha='right', va='center', fontsize=12, fontweight='bold',
              transform=offset_copy( fig.transFigure, fig=fig, x=-38, units='points' ) )
    fig.text( ( bot.x0 + last_ax.x1 )/2, bot.y0, 'Instellation (W m$^{-2}$)',
              ha='center', va='top', fontsize=12, fontweight='bold',
              transform=offset_copy( fig.transFigure, fig=fig, y=-25, units='points' ) )

    cax = fig.add_axes( [ last_ax.x1 + 0.03, bot.y0, 0.022, top.y1 - bot.y0 ] )
    cb = fig.colorbar( cf, cax=cax, extend='both', ticks=cbar_ticks )
    cb.ax.tick_params( labelsize=10 )
    cb.ax.get_yaxis().labelpad = 16
    cb.set_label( cbar_label, rotation=270, fontsize=12 )

    fig.savefig( 'fig_interpolation_temp_1d.png', bbox_inches='tight' )
    fig.savefig( 'fig_interpolation_temp_1d.eps', bbox_inches='tight' )

    # ── Numbers for the manuscript ──────────────────────────────────────────
    print( 'Out-of-sample check: 16-case kriging at the new stable cases' )
    for name in MODELS:
        nc, truth, pred, err, sig = out_of_sample( name )
        print( f'\n{name}: {len( nc )} new stable cases' )
        print( f'  RMSE {np.sqrt( np.mean( err**2 ) ):.1f} K, bias (kriged - model) {np.mean( err ):+.1f} K, '
               f'median |err| {np.median( np.abs( err ) ):.1f} K, max |err| {np.max( np.abs( err ) ):.1f} K '
               f'at Case {nc[ np.argmax( np.abs( err ) ) ]}' )
        print( f'  within 1 sigma: {np.sum( np.abs( err ) <= sig )}/{len( nc )}, within 2 sigma: '
               f'{np.sum( np.abs( err ) <= 2*sig )}/{len( nc )}; median sigma {np.median( sig ):.1f} K' )
        for c, t, p, e, s in sorted( zip( nc, truth, pred, err, sig ), key=lambda r: -abs( r[ 3 ] ) )[ :6 ]:
            print( f'    Case {c:2d} ({CASES[ c ][ 0 ]:4d} W/m2, {CASES[ c ][ 1 ]:5.2f} bar): '
                   f'model {t:6.1f}  kriged {p:6.1f}  err {e:+6.1f}  sigma {s:5.1f}' )

    # How many new cases each model completes, and how the two compare there
    new_stable = { name: [ c for c in t if c > 16 ] for name, t in MODELS.items() }
    for name, cs in new_stable.items():
        print( f'\n{name}: stable at {len( cs )} of the 48 new cases, runaway at {48 - len( cs )}' )
    shared = [ c for c in new_stable[ 'ExoColumn' ] if c in hextor ]
    d = np.array( [ hextor[ c ] - exocolumn[ c ] for c in shared ] )
    print( f'new cases stable in both: {len( shared )}; stable in ExoColumn only: '
           f'{[ c for c in new_stable[ "ExoColumn" ] if c not in hextor ]}; in HEXTOR only: '
           f'{[ c for c in new_stable[ "HEXTOR" ] if c not in exocolumn ]}' )
    print( f'HEXTOR - ExoColumn there: colder at {np.sum( d < 0 )} of {len( d )}, median {np.median( d ):+.1f} K, '
           f'range {d.min():+.1f} to {d.max():+.1f} K' )

    # The inner edge each sample can place: the warmest stable case against the
    # least irradiated runaway, from the primary cases and from all 64
    print( '\nInner edge bracketed by the sample' )
    for name, t in MODELS.items():
        for last in ( 16, 64 ):
            stable  = [ c for c in t if c <= last ]
            runaway = [ c for c in CASES if c <= last and c not in t ]
            hs = max( stable, key=lambda c: CASES[ c ][ 0 ] )
            lr = min( runaway, key=lambda c: CASES[ c ][ 0 ] )
            print( f'  {name:<9} Cases 1-{last}: stable up to {CASES[ hs ][ 0 ]} W/m2 (Case {hs}, {CASES[ hs ][ 1 ]} bar), '
                   f'runaway from {CASES[ lr ][ 0 ]} W/m2 (Case {lr}, {CASES[ lr ][ 1 ]} bar)' )
    print( '  ExoColumn, all 64 cases at 1100-1450 W/m2:' )
    for c in sorted( ( c for c in CASES if 1100 <= CASES[ c ][ 0 ] <= 1450 ), key=lambda c: CASES[ c ] ):
        print( f'    Case {c:2d} {CASES[ c ][ 0 ]:5d} W/m2 {CASES[ c ][ 1 ]:5.2f} bar: '
               + ( f'{exocolumn[ c ]:.1f} K' if c in exocolumn else 'runaway' ) )
