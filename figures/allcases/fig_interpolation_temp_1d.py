"""Surface temperature of the two one-dimensional models, kriged from the 16
primary cases and from all 64 cases of the protocol.

HEXTOR and ExoColumn are the only models to have run the protocol's optional
Sequences 1b, 2b and 3 (Cases 17-64, Table 4 of Haqq-Misra et al. 2022), so for
them the sparse sample can be checked against a denser one. For each model the
left panel is kriged exactly as in Figure 3 (fig_interpolation_temp.py), from
its stable Cases 1-16, and the right panel from its stable cases among all 64.
Every panel shows the same markers, so the new cases sit over the 16-case
surface they were not used to build.

    cd figures/allcases && python fig_interpolation_temp_1d.py

Also prints the out-of-sample check the manuscript quotes: the 16-case kriging
evaluated at the new stable cases. Run fit_anisotropy.py to refit ANISO_64,
which it reads from climate_models below.
"""
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from matplotlib.transforms import offset_copy
from pykrige.ok import OrdinaryKriging
from scipy import ndimage

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

# ─── Variable configuration, as in fig_interpolation_temp.py ─────────────────
cm              = cmocean.cm.thermal
contourmin      = 175.0
contourmax      = 370.0
cinterval       = 40
sigma_threshold = 45.0      # K; hatch where kriging σ exceeds this
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

# Global mean surface temperature (K) of every stable case, by case number.
# Cases 1-16 are the values of fig_interpolation_temp.py; 17-64 are from
# /models/hextor/samosa_all64/global_output_HEXTOR.dat (2026-09-21) and
# exocolumn_samosa/output/global_output_ExoColumn_a2736_all64.dat (2026-09-22),
# both run on the configuration of the submitted Cases 1-16, which they
# reproduce exactly. Every case not listed ran away. ExoColumn Case 57 is
# accepted on the protocol's stable-trend allowance (309.5 K for thousands of
# days, with +1.4 W/m2 at the top of the atmosphere from a water-budget leak).
hextor = {
     1: 173.10,  4: 292.22,  8: 224.40,  9: 267.86, 10: 152.42, 11: 225.52, 14: 241.46, 15: 188.62, 16: 376.07,
    17: 201.80, 20: 332.80, 21: 236.47, 24: 172.94, 25: 282.24, 26: 172.41, 27: 255.35, 29: 313.17, 30: 224.11,
    31: 173.12, 32: 285.24, 33: 312.95, 36: 167.30, 37: 231.84, 40: 266.14, 41: 255.65, 44: 202.07, 45: 187.39,
    48: 308.54, 52: 198.21, 53: 216.56, 56: 254.39, 57: 300.58, 60: 228.01, 61: 178.74, 64: 305.53,
}
exocolumn = {
     1: 206.98,  4: 293.26,  8: 248.49,  9: 269.66, 10: 201.36, 11: 242.60, 14: 251.63, 15: 216.92,
    17: 226.18, 21: 246.35, 24: 208.08, 25: 280.16, 26: 210.37, 27: 259.93, 30: 249.93, 31: 205.39, 32: 285.72,
    36: 201.74, 37: 248.46, 40: 267.18, 41: 262.00, 44: 227.33, 45: 225.63, 52: 224.23, 53: 243.45, 56: 257.10,
    57: 310.12, 60: 243.52, 61: 212.09,
}
MODELS = { 'HEXTOR': hextor, 'ExoColumn': exocolumn }

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

# Of the regions where σ exceeds the threshold, hatch only those reaching the
# highest instellation on the grid, as in fig_interpolation_temp.py.
def warm_edge_sigma( sigma ):
    regions, _ = ndimage.label( sigma > sigma_threshold, structure=np.ones( ( 3, 3 ) ) )
    dropped    = np.setdiff1d( regions, np.append( regions[ -1, : ], 0 ) )
    return np.where( np.isin( regions, dropped ), 0.0, sigma )


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

    # Each panel is ( model, last case used, header over the pair, title )
    PANELS = [ ( name, last ) for name in MODELS for last in ( 16, 64 ) ]

    fig, axs = plt.subplots( 1, 4, figsize=(12.2, 2.95), squeeze=False )
    xv, yv = np.meshgrid( pn2, flux )
    levels = np.linspace( contourmin, contourmax, cinterval )
    for ax, ( name, last ) in zip( axs.flat, PANELS ):
        temps = MODELS[ name ]
        fs, ps, vals, _ = samples( temps, last )
        aniso = ( ANISO_16 if last == 16 else ANISO_64 )[ name ]
        z, var = kriging( fs, ps, vals, aniso ).execute( 'grid', norm_pres( pn2 ), norm_flux( flux ) )
        cf = ax.contourf( yv*fluxscale, xv, z, cmap=cm, levels=levels, extend='both' )
        ax.contourf( yv*fluxscale, xv, warm_edge_sigma( np.sqrt( np.clip( var, 0, None ) ) ), levels=[ sigma_threshold, 1e9 ],
                     hatches=[ '///' ], colors='none', alpha=0 )

        # The same markers in every panel: circles for Cases 1-16, diamonds for
        # 17-64, crosses where the model ran away.
        af, ap, av, ac = samples( temps, 64 )
        old = ac <= 16
        ax.scatter( af[ old ]*fluxscale, ap[ old ], c=av[ old ], cmap=cm, vmin=contourmin, vmax=contourmax,
                    marker='o', s=45, edgecolors=marker_edge, zorder=3 )
        ax.scatter( af[ ~old ]*fluxscale, ap[ ~old ], c=av[ ~old ], cmap=cm, vmin=contourmin, vmax=contourmax,
                    marker='D', s=24, edgecolors=marker_edge, linewidths=0.8, zorder=3 )
        runaway = [ c for c in CASES if c not in temps ]
        ax.scatter( [ CASES[ c ][ 0 ] for c in runaway ], [ CASES[ c ][ 1 ] for c in runaway ],
                    marker='x', s=22, c='k', linewidths=0.9, zorder=3 )

        ax.set_title( f'Cases 1–{last} (n={len( vals )})', fontsize=12 )
        ax.tick_params( axis='both', labelsize=10 )
        ax.set_yscale( 'log' )
        ax.set_xlim( [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ] )
        ax.set_xticks( [ 2500, 2000, 1500, 1000, 500 ] )
        ax.set_ylim( [ min( pn2 )*0.9, max( pn2 )*1.1 ] )
        ax.set_box_aspect( 1 )
        ax.apply_aspect()

    fig.subplots_adjust( wspace=0.12, right=0.88 )

    # One bold header over each model's pair of panels, as over the blocks of
    # Figure 3, and the axes labeled once for the row
    above = offset_copy( fig.transFigure, fig=fig, y=23, units='points' )
    for i, name in enumerate( MODELS ):
        left, right = axs[ 0, 2*i ].get_position(), axs[ 0, 2*i + 1 ].get_position()
        fig.text( ( left.x0 + right.x1 )/2, left.y1, name, transform=above,
                  ha='center', va='bottom', fontsize=14, fontweight='bold' )
    for ax in axs.flat[ 1: ]:
        ax.tick_params( labelleft=False )
    first, last_ax = axs[ 0, 0 ].get_position(), axs[ 0, -1 ].get_position()
    fig.text( first.x0, ( first.y0 + first.y1 )/2, 'Surface pressure (bar)',
              rotation=90, ha='right', va='center', fontsize=12, fontweight='bold',
              transform=offset_copy( fig.transFigure, fig=fig, x=-38, units='points' ) )
    fig.text( ( first.x0 + last_ax.x1 )/2, first.y0, 'Instellation (W m$^{-2}$)',
              ha='center', va='top', fontsize=12, fontweight='bold',
              transform=offset_copy( fig.transFigure, fig=fig, y=-25, units='points' ) )

    cax = fig.add_axes( [ 0.905, first.y0, 0.013, first.y1 - first.y0 ] )
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
