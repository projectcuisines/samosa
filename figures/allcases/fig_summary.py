"""
SAMOSA summary figure: ensemble consensus on the freezing boundary.

Partitions the (instellation, surface pressure) plane into the region where every
model places a frozen surface, the region where every model places an unfrozen
surface, and the band between them where the models disagree.  Each model's own
273.16 K isotherm is drawn on top, so the width of the contested band is the
splay of the individual contours.

Consensus shading is computed from the four models that attempted all 16 cases
(ExoPlaSim, ExoCAM, ROCKE-3D, PlaHab).  The Generic PCM and LFRic contours are
each constrained by a single case above freezing, so they are clipped to the
region each model actually sampled and do not contribute to the shading.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path
from scipy.spatial import ConvexHull

from pykrige.ok import OrdinaryKriging

fluxscale = 100

flux  = np.arange( 400, 2700, 100 ) / fluxscale
pn2   = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89,
                    1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400,
                    900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83,
                    0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

pcm_flux1   = np.array( [ 500, 1200,  800, 1100, 400,  900, 600 ] ) / fluxscale
pcm_pres1   = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )
lfric_flux1 = np.array( [ 500, 1200, 1100, 1500, 900,  600 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.70, 2.98, 1.44, 0.43 ] )

# ── Temperature data (K) ──────────────────────────────────────────────────────
runawaytemp = 600.0

ts_plasim  = np.array( [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9,
                         239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ] )
ts_exocam  = np.array( [ 196.8, runawaytemp, runawaytemp, 260.0, runawaytemp, runawaytemp,
                         runawaytemp, 243.8, 244.8, 194.1, 234.0, 350.9,
                         runawaytemp, 236.8, 211.5, 356.7 ] )
ts_rocke3d = np.array( [ 202.8284, runawaytemp, runawaytemp, 260.1185, 265.88116, runawaytemp,
                         267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902,
                         271.92654, 236.30406, 210.50339, 319.25085 ] )
ts_plahab  = np.array( [ 196.3, runawaytemp, runawaytemp, 273.2, 281.4, runawaytemp,
                         293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )
ts_pcm     = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218,
                         266.5987224285321, 210.69131033681012, 246.04296230476365,
                         217.2519558970929 ] )
ts_lfric   = np.array( [ 195.37, 251.48, 241.35, 333.20, 228.84, 203.64 ] )


ts_exocam_mask  = ts_exocam  != runawaytemp
ts_rocke3d_mask = ts_rocke3d != runawaytemp
ts_plahab_mask  = ts_plahab  != runawaytemp

T_FREEZE     = 273.16
sigma_thresh = 45.0      # K, matches fig_interpolation_temp.py

# ── Kriging onto a fine display grid ──────────────────────────────────────────
fluxf = np.linspace( 400, 2600, 221 ) / fluxscale
# The 20 protocol pressures are rounded to two decimals and so are not exactly
# log-spaced. They are merged into the display grid so that every sample point
# lands on a node: otherwise the shading at a sample point is evaluated up to
# ~1% away in pressure, and with a linear variogram and exact_values=True the
# kriged field has a cusp at data points and can move several K over that
# offset. That put Case 5 in the frozen region although PlaHab reports 281.4 K
# there, because the kriged value at the neighboring node was 272.9 K.
pn2f  = np.unique( np.concatenate( [ np.exp( np.linspace( np.log( 0.10 ), np.log( 10.0 ), 181 ) ),
                                     pn2 ] ) )

log_pn2 = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

def krige( p, f, z ):
    ok = OrdinaryKriging( norm_pres( p ), norm_flux( f ), z,
                          variogram_model="linear", verbose=False,
                          enable_plotting=False, exact_values=True )
    return ok.execute( "grid", norm_pres( pn2f ), norm_flux( fluxf ) )

# Model style, following fig_energy_balance.py
style = { 'ExoPlaSim':   dict( color='#ff7f0e', ls='-'  ),
          'ExoCAM':      dict( color='#1f77b4', ls='-'  ),
          'ROCKE-3D':    dict( color='#2ca02c', ls='-'  ),
          'PlaHab':      dict( color='#8c564b', ls='--' ),
          'Generic PCM': dict( color='#d62728', ls='-'  ),
          'LFRic':       dict( color='#9467bd', ls='-'  ) }

consensus_models = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab' ]

# The Generic PCM and LFRic each have exactly one case above 273.16 K (Case 4 at
# 286.7 K and Case 12 at 333.2 K), and neither group submitted anything above
# 1200 and 1500 W m^-2 respectively.  Their isotherms are therefore drawn only
# inside the convex hull of the cases each model actually ran, so that they
# appear as short segments rather than as curves spanning the whole domain.
# Set to False to omit them entirely.
SHOW_PARTIAL = True
partial_models = [ 'Generic PCM', 'LFRic' ]

ts_in = {
    'ExoPlaSim':   ( pres1,                    flux1,                    ts_plasim ),
    'ExoCAM':      ( pres1[ ts_exocam_mask  ], flux1[ ts_exocam_mask  ], ts_exocam[  ts_exocam_mask  ] ),
    'ROCKE-3D':    ( pres1[ ts_rocke3d_mask ], flux1[ ts_rocke3d_mask ], ts_rocke3d[ ts_rocke3d_mask ] ),
    'PlaHab':      ( pres1[ ts_plahab_mask  ], flux1[ ts_plahab_mask  ], ts_plahab[  ts_plahab_mask  ] ),
    'Generic PCM': ( pcm_pres1,                pcm_flux1,                ts_pcm ),
    'LFRic':       ( lfric_pres1,              lfric_flux1,              ts_lfric ),
}

Z, WELL, SIG = {}, {}, {}
for name, ( p, f, z ) in ts_in.items():
    zz, vv    = krige( p, f, z )
    Z[ name ] = np.asarray( zz )
    SIG[ name ]  = np.sqrt( np.asarray( vv ) )
    WELL[ name ] = SIG[ name ] < sigma_thresh


# The plane is split by the sign of the global mean surface temperature:
#
#   blue    every model places the global mean below 273.16 K
#   green   every model places it above 273.16 K
#   white   the models disagree on the sign
def sampled_region( p_pts, f_pts ):
    """Mask of the display grid lying inside the convex hull of a model's own
    sample points, taken in the same normalized coordinates used for kriging."""
    pts  = np.column_stack( [ norm_pres( p_pts ), norm_flux( f_pts ) ] )
    poly = Path( pts[ ConvexHull( pts ).vertices ] )
    PP, FF = np.meshgrid( norm_pres( pn2f ), norm_flux( fluxf ) )
    return poly.contains_points( np.column_stack( [ PP.ravel(), FF.ravel() ] ) ).reshape( PP.shape )


# The two mean-based bands use every model, including PlaHab, whose global means
# track the GCMs over 95% of the blue band and 98% of the pale band even though its
# extremes are unusable.
#
# A model's influence fades with distance from the cases it actually ran, rather
# than switching on and off at the edge of its convex hull.  A hard cutoff makes the
# band boundary jump wherever a model enters, because the Generic PCM isotherm sits
# 60-170 W m^-2 cold-ward of the others; fading keeps the boundary continuous.
# alpha = 1 within d_full of a sample point and 0 beyond d_none, and a model with
# alpha = 0 neither constrains nor blocks.  The thresholds are set so that the four
# models with full 16-case coverage keep alpha = 1 almost everywhere (ExoPlaSim's
# largest distance to its own nearest case is 0.327) while the Generic PCM and LFRic,
# which reach 0.80, fade out where they have nothing nearby.
mean_models = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab', 'Generic PCM', 'LFRic' ]
d_full, d_none = 0.30, 0.55     # normalized (log p, flux) units
# Large enough to dominate any temperature margin (|T - 273| < 110 K here) but not
# so large that a model at alpha = 0.999 is handed a spurious slack of several K.
BIG = 300.0

_PP, _FF = np.meshgrid( norm_pres( pn2f ), norm_flux( fluxf ) )

def influence( name ):
    p, f, _ = ts_in[ name ]
    d = np.min( np.hypot( _PP[ ..., None ] - norm_pres( p ),
                          _FF[ ..., None ] - norm_flux( f ) ), axis=-1 )
    a = np.clip( ( d_none - d ) / ( d_none - d_full ), 0.0, 1.0 )
    return a * a * ( 3.0 - 2.0 * a )        # smoothstep, C1 continuous

alpha = { n: influence( n ) for n in mean_models }

# A partial model is silenced entirely outside the convex hull of its own cases,
# the same clipping already applied to its drawn isotherm. Without this the
# shading and the curves disagree: the Generic PCM, which never ran Case 16,
# extrapolates to 269.9 K at (1400 W m^-2, 10 bar) and vetoes the ice-free
# consensus there, while its isotherm is not drawn anywhere near that point, so
# Case 16 sits in the contested band with no curve to account for it. A model
# should not block a consensus where its curve is not shown.
# A model's own sample points are vertices of its hull, so a display node a
# fraction of a per cent away from one can fall outside and silence the model
# where it actually has data. Case 4 is the Generic PCM's own case and the
# nearest node sits 0.59% below it in pressure. The hull is therefore unioned
# with a small buffer around each sample point.
HULL_BUF = 0.02       # normalized units, about three grid diagonals

for n in partial_models:
    p_pts, f_pts, _ = ts_in[ n ]
    near = np.min( np.hypot( _PP[ ..., None ] - norm_pres( p_pts ),
                             _FF[ ..., None ] - norm_flux( f_pts ) ), axis=-1 ) < HULL_BUF
    alpha[ n ] = alpha[ n ] * ( sampled_region( p_pts, f_pts ) | near )

# margin > 0 means the model places the global mean below freezing
def consensus_bands( names ):
    cold = [ ( T_FREEZE - Z[ n ] ) + ( 1.0 - alpha[ n ] ) * BIG for n in names ]
    warm = [ ( Z[ n ] - T_FREEZE ) + ( 1.0 - alpha[ n ] ) * BIG for n in names ]
    # At least one model must actually be constraining, or neither band is asserted
    con  = np.max( [ alpha[ n ] for n in names ], axis=0 ) > 0.5
    return ( ( np.min( cold, axis=0 ) >  0.0 ) & con,
             ( np.min( warm, axis=0 ) >= 0.0 ) & con )

band_blue, band_warm = consensus_bands( mean_models )

# The contested band is split by whether PlaHab, the only 2-D model, is needed
# to produce the disagreement. Adding a model can only shrink the two consensus
# bands, so the 3-D contested region is a strict subset of the full one and the
# difference is exactly the area PlaHab alone makes contested.
mean_models_3d = [ n for n in mean_models if n != 'PlaHab' ]
band_blue_3d, band_warm_3d = consensus_bands( mean_models_3d )

# Silencing a partial model at its hull edge can punch an interior hole in the
# contested band: just outside the Generic PCM hull near 2.6-2.8 bar it stops
# blocking the frozen consensus while ExoCAM has not yet crossed 273.16 K, so a
# pocket at 1150-1180 W m^-2 reverts to blue inside an otherwise contested
# region. Such pockets are an artifact of where a mask edge falls, not a
# consensus. At any given pressure the contested band is a single interval in
# instellation -- it is the transition between a frozen and an ice-free
# consensus -- so it is reported as the interval between its extremes at that
# pressure. binary_fill_holes does not do this: the pockets open out to the
# exterior in 2-D and so are notches rather than enclosed islands.
def close_rows( M ):
    out = M.copy()
    for j in range( M.shape[ 1 ] ):
        w = np.where( M[ :, j ] )[ 0 ]
        if len( w ):
            out[ w.min():w.max() + 1, j ] = True
    return out

contested_all = close_rows( ~( band_blue    | band_warm    ) )
contested_3d  = close_rows( ~( band_blue_3d | band_warm_3d ) )
contested_all = contested_all | contested_3d          # keep white inside grey
contested_plahab = contested_all & ~contested_3d

# The three regions must stay a partition of the plane
band_blue = band_blue & ~contested_all
band_warm = band_warm & ~contested_all
# The mean-field WELL mask is not applied to the ice-free band: ExoCAM and
# ROCKE-3D lose their hot cases to runaway, so that mask is false across most of
# the region this band occupies.  The majority-runaway wash drawn on top already
# marks where the hot end is unconstrained.


n_warm = np.array( [ ( Z[n] > T_FREEZE ) & WELL[n] for n in consensus_models ] ).sum( axis=0 )
n_well = np.array( [ WELL[n]                       for n in consensus_models ] ).sum( axis=0 )
frac_warm  = np.where( n_well > 0, n_warm / np.maximum( n_well, 1 ), np.nan )

# Runaway: fraction of the consensus models in runaway, from the kriged 0/1 indicator
run_ind = { 'ExoPlaSim': np.zeros( 16 ),
            'ExoCAM':    ( ~ts_exocam_mask  ).astype( float ),
            'ROCKE-3D':  ( ~ts_rocke3d_mask ).astype( float ),
            'PlaHab':    ( ~ts_plahab_mask  ).astype( float ) }
R = {}
for name, ind in run_ind.items():
    R[ name ] = ( np.zeros_like( Z['ExoPlaSim'] ) if ind.sum() == 0
                  else np.asarray( krige( pres1, flux1, ind )[0] ) )
frac_run = np.array( [ R[n] > 0.5 for n in consensus_models ] ).mean( axis=0 )

# ── Plot ──────────────────────────────────────────────────────────────────────
X, Y = np.meshgrid( pn2f, fluxf * fluxscale )   # X = pressure, Y = instellation

c_frozen = '#d6e6f4'    # every 3-D model: global mean below freezing
c_warm   = '#dcefdb'    # every model: global mean above freezing
c_mixed  = '#ffffff'    # the 3-D models disagree, left white
c_mixed_2d = '#e6e6e6'  # contested only once PlaHab, the 2-D model, is included
c_run    = '#f2d6d8'    # pre-blended: EPS does not support transparency
marker_edge = 'k'

fig, ax = plt.subplots( figsize=( 9.5, 7.5 ) )

# Consensus regions, then the majority-runaway subset of the unfrozen region on
# top of it, then the contested band, which is the subject of the figure.
ax.contourf( Y, X, band_blue.astype( float ),  levels=[ 0.5, 1.5 ], colors=[ c_frozen ], zorder=0 )
ax.contourf( Y, X, band_warm.astype( float ),  levels=[ 0.5, 1.5 ], colors=[ c_warm   ], zorder=0 )

# Majority-runaway region, as a soft wash rather than a contour, so that it is not
# confused with the individual model isotherms
ax.contourf( Y, X, frac_run, levels=[ 0.49, 1.01 ], colors=[ c_run ], zorder=3 )

# Grey first, then the 3-D-only contested band in white on top of it
ax.contourf( Y, X, contested_all.astype( float ),
             levels=[ 0.5, 1.5 ], colors=[ c_mixed_2d ], zorder=2 )
ax.contourf( Y, X, contested_3d.astype( float ),
             levels=[ 0.5, 1.5 ], colors=[ c_mixed ], zorder=2.2 )

drawn = consensus_models + ( partial_models if SHOW_PARTIAL else [] )
for name in drawn:
    mask = WELL[ name ]
    if name in partial_models:
        p_pts, f_pts, _ = ts_in[ name ]
        mask = mask & sampled_region( p_pts, f_pts )
    ax.contour( Y, X, np.where( mask, Z[ name ], np.nan ), levels=[ T_FREEZE ],
                colors=[ style[ name ][ 'color' ] ], linewidths=2.2,
                linestyles=style[ name ][ 'ls' ], zorder=4 )

# Case 10 sits against the right-hand edge and Case 16 against the top, so their
# labels are placed inboard rather than with the default offset.
label_offset = { 10: ( -17, 5 ), 16: ( 7, -14 ) }

# QMC sample points: filled where all consensus models are stable, open cross where
# two or more report a runaway
runaway_count = np.array( [ ( ~ts_exocam_mask ).astype( int ),
                            ( ~ts_rocke3d_mask ).astype( int ),
                            ( ~ts_plahab_mask ).astype( int ) ] ).sum( axis=0 )
for i, ( f, p ) in enumerate( zip( flux1 * fluxscale, pres1 ) ):
    if runaway_count[ i ] >= 2:
        ax.plot( f, p, marker='X', color='k', ms=11, mew=0.0, zorder=6 )
    else:
        ax.scatter( f, p, marker='o', s=55, c='k', edgecolors=marker_edge, zorder=6 )
    ax.annotate( str( i + 1 ), ( f, p ), textcoords='offset points',
                 xytext=label_offset.get( i + 1, ( 7, 5 ) ), fontsize=9, zorder=7 )

# Label positions chosen from the widest, emptiest part of each shaded region
ax.annotate( 'all models\nglobal mean\nbelow freezing', xy=( 950, 0.30 ), fontsize=10.5,
             ha='center', va='center', style='italic', color='#3a6f9c', zorder=5 )
ax.annotate( 'all models\nglobal mean\nabove freezing', xy=( 1575, 1.67 ), fontsize=10.5,
             ha='center', va='center', style='italic', color='#6b9b62', zorder=5 )
ax.annotate( 'runaway\n(all but ExoPlaSim)', xy=( 2320, 0.60 ), fontsize=11,
             ha='center', va='center', style='italic', color='#8a3a42', zorder=5 )

ax.set_yscale( 'log' )
ax.set_xlim( 2650, 350 )
ax.set_ylim( 0.09, 11 )
ax.set_xlabel( 'Instellation (W m$^{-2}$)', fontsize=12 )
ax.set_ylabel( 'N$_2$ surface pressure (bar)', fontsize=12 )
ax.tick_params( axis='both', labelsize=11 )

legend_order = [ n for n in drawn if n != 'PlaHab' ] + [ n for n in drawn if n == 'PlaHab' ]
handles  = [ Line2D( [], [], color=style[n][ 'color' ],
                     lw=2.2, ls=style[n][ 'ls' ], label=n )
             for n in legend_order ]
ax.legend( handles=handles, loc='upper left', fontsize=10, ncol=1,
           framealpha=1, borderpad=0.7, labelspacing=0.5 )

# Area fractions of the plane as drawn: flux is linear and pressure logarithmic.
# The pressure grid is no longer uniform once the protocol pressures are merged
# in, so cells are weighted by their width in log pressure.
_wp = np.gradient( np.log( pn2f ) )
_W  = np.broadcast_to( _wp, ( len( fluxf ), len( pn2f ) ) )
area = lambda M: float( np.sum( M * _W ) / np.sum( _W ) )

print( '=== fraction of the plane ===' )
for label, m in ( ( 'all models below freezing', band_blue ),
                  ( 'all models above freezing', band_warm ),
                  ( 'contested, 3-D models only', contested_3d ),
                  ( 'contested, PlaHab only',     contested_plahab ),
                  ( 'contested, total',           contested_all ) ):
    print( f'  {label:28s} {100.0 * area( m ):5.1f}%' )

suffix = "" if SHOW_PARTIAL else "_nopartial"
fig.savefig( f"fig_summary{suffix}.png", bbox_inches='tight' )
fig.savefig( f"fig_summary{suffix}.eps", bbox_inches='tight' )
