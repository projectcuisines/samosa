"""
SAMOSA summary figure: ensemble consensus on the freezing boundary.

Partitions the (instellation, surface pressure) plane into the region where every
model places a frozen surface, the region where every model places an unfrozen
surface, and the band between them where the models disagree.  Each model's own
273.16 K isotherm is drawn on top, so the width of the contested band is the
splay of the individual contours.

Consensus shading uses all eight models, but each model's influence fades with
distance from the cases it actually ran, and the four partial submissions
(Generic PCM with 7 cases, ExoColumn with 8, HEXTOR with 9, LFRic with 11) are
silenced outside the convex hull of their own cases.  Their isotherms are clipped to that same region, so they read
as short segments rather than as curves spanning the whole domain.  The
runaway wash and the per-model agreement count use only the four models that
attempted all 16 cases (ExoPlaSim, ExoCAM, ROCKE-3D, PlaHab).

With --common, every model is kriged from only the cases stable in all eight
models (1, 4, 8, 9, 10, 14, 15), on a grid zoomed to the range those cases span.
Every model then ran exactly the same cases, so none is partial: no model fades
or is clipped, and every isotherm is drawn across the whole zoomed panel. With
--stacked, the full panel and the common-case panel are drawn side by side.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt
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
lfric_flux1 = np.array( [ 500, 1200, 1600, 800, 1100, 400, 900, 1500, 900,  600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )
hextor_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600, 1400 ] ) / fluxscale
hextor_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43, 10.00 ] )
exocolumn_flux1 = np.array( [ 500, 1200,  800, 1100,  400,  900,  900,  600 ] ) / fluxscale
exocolumn_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43 ] )

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
ts_lfric   = np.array( [ 195.37, 251.48, 400.52, 231.83, 241.35, 197.81, 227.52, 333.20, 228.84, 203.64, 361.70 ] )
ts_hextor  = np.array( [ 173.10, 292.22, 224.40, 267.86, 152.42, 225.52, 241.46, 188.62, 376.07 ] )
ts_exocolumn = np.array( [ 206.98, 293.26, 248.49, 269.66, 201.36, 242.60, 251.63, 216.92 ] )


ts_exocam_mask  = ts_exocam  != runawaytemp
ts_rocke3d_mask = ts_rocke3d != runawaytemp
ts_plahab_mask  = ts_plahab  != runawaytemp

T_FREEZE     = 273.16
sigma_thresh = 45.0      # K, matches fig_interpolation_temp.py

COMMON  = '--common' in sys.argv
STACKED = '--stacked' in sys.argv

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

# Kriging anisotropy, fitted per model in fit_anisotropy.py; the same values the
# fig_interpolation_temp.py panels use. A value of s means one unit of normalized
# instellation counts s times a unit of normalized log-pressure.
ANISO = { 'ExoPlaSim': 2, 'ExoCAM': 10, 'ROCKE-3D': 4, 'PlaHab': 3,
          'Generic PCM': 5, 'LFRic': 15, 'HEXTOR': 10, 'ExoColumn': 7 }

def krige( p, f, z, scaling=1.0, pres_grid=pn2f, flux_grid=fluxf ):
    ok = OrdinaryKriging( norm_pres( p ), norm_flux( f ), z,
                          anisotropy_scaling=scaling,
                          variogram_model="linear", verbose=False,
                          enable_plotting=False, exact_values=True )
    return ok.execute( "grid", norm_pres( pres_grid ), norm_flux( flux_grid ) )

# Model style, following fig_energy_balance.py
style = { 'ExoPlaSim':   dict( color='#ff7f0e', ls='-'  ),
          'ExoCAM':      dict( color='#1f77b4', ls='-'  ),
          'ROCKE-3D':    dict( color='#2ca02c', ls='-'  ),
          'PlaHab':      dict( color='#8c564b', ls='--' ),
          'Generic PCM': dict( color='#d62728', ls='-'  ),
          'LFRic':       dict( color='#9467bd', ls='-'  ),
          'HEXTOR':      dict( color='#17becf', ls='--' ),
          'ExoColumn':   dict( color='#7f7f7f', ls='--' ) }

consensus_models = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab' ]

# The Generic PCM has exactly one case above 273.16 K (Case 4 at 286.7 K) and
# submitted nothing above 1200 W m^-2.  LFRic has three (Case 7 at 400.5 K,
# Case 12 at 333.2 K and Case 16 at 361.7 K) but still only 8 of the 16 cases,
# and nothing above 1600 W m^-2.  Their isotherms are therefore drawn only inside the convex hull
# of the cases each model actually ran, so that they appear as short segments
# rather than as curves spanning the whole domain.
# HEXTOR contributes 9 of the 16 cases and nothing above 1400 W m^-2, so it is
# drawn on the same footing. It is also the only 1-D model in the ensemble and the only
# one without clouds, so its isotherms are not a like-for-like comparison with
# the GCMs even inside its hull; the dashed style marks that.
# Set to False to omit them entirely.
SHOW_PARTIAL = True
partial_models = [ 'Generic PCM', 'LFRic', 'HEXTOR', 'ExoColumn' ]

ts_in = {
    'ExoPlaSim':   ( pres1,                    flux1,                    ts_plasim ),
    'ExoCAM':      ( pres1[ ts_exocam_mask  ], flux1[ ts_exocam_mask  ], ts_exocam[  ts_exocam_mask  ] ),
    'ROCKE-3D':    ( pres1[ ts_rocke3d_mask ], flux1[ ts_rocke3d_mask ], ts_rocke3d[ ts_rocke3d_mask ] ),
    'PlaHab':      ( pres1[ ts_plahab_mask  ], flux1[ ts_plahab_mask  ], ts_plahab[  ts_plahab_mask  ] ),
    'Generic PCM': ( pcm_pres1,                pcm_flux1,                ts_pcm ),
    'LFRic':       ( lfric_pres1,              lfric_flux1,              ts_lfric ),
    'HEXTOR':      ( hextor_pres1,             hextor_flux1,             ts_hextor ),
    'ExoColumn':   ( exocolumn_pres1,          exocolumn_flux1,          ts_exocolumn ),
}


# The plane is split by the sign of the global mean surface temperature:
#
#   blue    every model places the global mean below 273.16 K
#   green   every model places it above 273.16 K
#   white   the models disagree on the sign
def sampled_region( p_pts, f_pts, PP, FF ):
    """Mask of the display grid lying inside the convex hull of a model's own
    sample points, taken in the same normalized coordinates used for kriging."""
    pts  = np.column_stack( [ norm_pres( p_pts ), norm_flux( f_pts ) ] )
    poly = Path( pts[ ConvexHull( pts ).vertices ] )
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
mean_models = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab', 'Generic PCM', 'LFRic', 'HEXTOR', 'ExoColumn' ]
d_full, d_none = 0.30, 0.55     # normalized (log p, flux) units
# A model's margin is blended toward SLACK as its weight falls, so a fully faded
# model contributes +SLACK and never blocks, whatever its temperature, while a
# partially faded one hands over gradually. The old formulation added
# (1 - alpha) * 300 K, which had to be large enough to dominate any temperature
# margin and therefore switched off blocking as soon as alpha left 1, making the
# fade abrupt in effect however smooth it was in space.
SLACK = 5.0

def influence( p, f, PP, FF ):
    d = np.min( np.hypot( PP[ ..., None ] - norm_pres( p ),
                          FF[ ..., None ] - norm_flux( f ) ), axis=-1 )
    a = np.clip( ( d_none - d ) / ( d_none - d_full ), 0.0, 1.0 )
    return a * a * ( 3.0 - 2.0 * a )        # smoothstep, C1 continuous

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
HULL_W = 0.12         # normalized units over which a partial model fades out

def hull_sdist( p_pts, f_pts, PP, FF ):
    """Signed distance to the convex hull of a model's own cases, negative
    inside. Exact on the outward side of each edge, which is all that the
    taper needs."""
    pts = np.column_stack( [ norm_pres( p_pts ), norm_flux( f_pts ) ] )
    V   = pts[ ConvexHull( pts ).vertices ]
    d   = np.full( PP.shape, -1e9 )
    for k in range( len( V ) ):
        a, b = V[ k ], V[ ( k + 1 ) % len( V ) ]
        e = b - a
        L = np.hypot( *e )
        d = np.maximum( d, ( e[1] / L ) * ( PP - a[0] ) - ( e[0] / L ) * ( FF - a[1] ) )
    return d

def smoothstep( x ):
    x = np.clip( x, 0.0, 1.0 )
    return x * x * ( 3.0 - 2.0 * x )

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

def consensus( models, pres_grid, flux_grid, fade ):
    """Kriged global means and consensus regions of one panel. With fade, each
    model's weight falls with distance from its own cases and partial models are
    silenced outside their hulls; without it, every model counts fully
    everywhere, which is right only when all models ran the same cases."""
    PP, FF = np.meshgrid( norm_pres( pres_grid ), norm_flux( flux_grid ) )
    Z, WELL, SIG = {}, {}, {}
    for name, ( p, f, z ) in models.items():
        zz, vv    = krige( p, f, z, ANISO[ name ], pres_grid, flux_grid )
        Z[ name ] = np.asarray( zz )
        SIG[ name ]  = np.sqrt( np.asarray( vv ) )
        WELL[ name ] = SIG[ name ] < sigma_thresh

    if fade:
        alpha = { n: influence( *models[ n ][ :2 ], PP, FF ) for n in mean_models }
        # A hard 0/1 hull mask makes alpha discontinuous, and with it the consensus
        # margins, so the band edge jumps where the hull boundary crosses it. The mask
        # is therefore a taper over HULL_W rather than a step, and the margins below
        # blend toward a small positive slack rather than adding a large penalty, so
        # that a fading model hands over gradually to the next one.
        for n in partial_models:
            p_pts, f_pts, _ = models[ n ]
            alpha[ n ] = alpha[ n ] * smoothstep( ( HULL_W - hull_sdist( p_pts, f_pts, PP, FF ) ) / HULL_W )
    else:
        alpha = { n: np.ones( PP.shape ) for n in mean_models }

    # margin > 0 means the model places the global mean below freezing
    def consensus_bands( names ):
        cold = [ alpha[ n ] * ( T_FREEZE - Z[ n ] ) + ( 1.0 - alpha[ n ] ) * SLACK for n in names ]
        warm = [ alpha[ n ] * ( Z[ n ] - T_FREEZE ) + ( 1.0 - alpha[ n ] ) * SLACK for n in names ]
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

    return dict( Z=Z, WELL=WELL, PP=PP, FF=FF, pres_grid=pres_grid, flux_grid=flux_grid,
                 band_blue=band_blue, band_warm=band_warm, contested_all=contested_all,
                 contested_3d=contested_3d, contested_plahab=contested_plahab )

# Area fractions of the plane as drawn: flux is linear and pressure logarithmic.
# The pressure grid is no longer uniform once the protocol pressures are merged
# in, so cells are weighted by their width in log pressure.
def report_areas( title, B ):
    _wp = np.gradient( np.log( B[ 'pres_grid' ] ) )
    _W  = np.broadcast_to( _wp, ( len( B[ 'flux_grid' ] ), len( B[ 'pres_grid' ] ) ) )
    area = lambda M: float( np.sum( M * _W ) / np.sum( _W ) )
    print( title )
    for label, m in ( ( 'all models below freezing', B[ 'band_blue' ] ),
                      ( 'all models above freezing', B[ 'band_warm' ] ),
                      ( 'contested, 3-D models only', B[ 'contested_3d' ] ),
                      ( 'contested, PlaHab only',     B[ 'contested_plahab' ] ),
                      ( 'contested, total',           B[ 'contested_all' ] ) ):
        print( f'  {label:28s} {100.0 * area( m ):5.1f}%' )

# ── Panels ────────────────────────────────────────────────────────────────────
panels = []

if not COMMON or STACKED:
    full = consensus( ts_in, pn2f, fluxf, fade=True )

    n_warm = np.array( [ ( full['Z'][n] > T_FREEZE ) & full['WELL'][n] for n in consensus_models ] ).sum( axis=0 )
    n_well = np.array( [ full['WELL'][n]                               for n in consensus_models ] ).sum( axis=0 )
    frac_warm  = np.where( n_well > 0, n_warm / np.maximum( n_well, 1 ), np.nan )

    # Runaway: fraction of the consensus models in runaway, from the kriged 0/1 indicator
    run_ind = { 'ExoPlaSim': np.zeros( 16 ),
                'ExoCAM':    ( ~ts_exocam_mask  ).astype( float ),
                'ROCKE-3D':  ( ~ts_rocke3d_mask ).astype( float ),
                'PlaHab':    ( ~ts_plahab_mask  ).astype( float ) }
    R = {}
    for name, ind in run_ind.items():
        R[ name ] = ( np.zeros_like( full['Z']['ExoPlaSim'] ) if ind.sum() == 0
                      else np.asarray( krige( pres1, flux1, ind )[0] ) )
    full[ 'frac_run' ] = np.array( [ R[n] > 0.5 for n in consensus_models ] ).mean( axis=0 )

    # QMC sample points: filled where all consensus models are stable, open cross where
    # two or more report a runaway
    runaway_count = np.array( [ ( ~ts_exocam_mask ).astype( int ),
                                ( ~ts_rocke3d_mask ).astype( int ),
                                ( ~ts_plahab_mask ).astype( int ) ] ).sum( axis=0 )
    # Case 10 sits against the right-hand edge and Case 16 against the top, so their
    # labels are placed inboard rather than with the default offset.
    full.update( header='All Cases',
                 cases=list( range( 16 ) ), crossed=runaway_count >= 2,
                 label_offset={ 10: ( -17, 5 ), 16: ( 7, -14 ) },
                 isotherm_mask=lambda name: full[ 'WELL' ][ name ] & (
                     sampled_region( *ts_in[ name ][ :2 ], full[ 'PP' ], full[ 'FF' ] )
                     if name in partial_models else True ),
                 # Label positions chosen from the widest, emptiest part of each shaded region
                 region_labels=[ ( 'all models\nglobal mean\nbelow freezing', ( 950, 0.30 ) ),
                                 ( 'all models\nglobal mean\nabove freezing', ( 1575, 1.67 ) ) ],
                 xlim=( 2650, 350 ), ylim=( 0.09, 11 ), plain_ticks=False, legend=True )
    report_areas( '=== fraction of the plane ===', full )
    panels.append( full )

if COMMON or STACKED:
    def _at( f, p, fs, ps ):
        return np.isclose( fs, f ) & np.isclose( ps, p )
    keep  = np.array( [ all( _at( f, p, fs, ps ).any() for ps, fs, _ in ts_in.values() )
                        for f, p in zip( flux1, pres1 ) ] )
    cases = np.where( keep )[ 0 ]
    print( 'cases stable in every model:', ( cases + 1 ).tolist() )
    common_in = {}
    for name, ( ps, fs, vals ) in ts_in.items():
        m = np.array( [ keep[ _at( f, p, flux1, pres1 ) ].any() for f, p in zip( fs, ps ) ] )
        common_in[ name ] = ( ps[ m ], fs[ m ], vals[ m ] )

    # Zoomed to the instellation and pressure the common cases span, padded by
    # 50 W m^-2 and 10% as the per-model figures are. Instellation every 10 W m^-2
    # and the protocol pressures merged in, so every sample point is a node.
    lo, hi   = pres1[ keep ].min()*0.9, pres1[ keep ].max()*1.1
    fluxf_c  = np.linspace( flux1[ keep ].min() - 0.5, flux1[ keep ].max() + 0.5, 91 )
    pn2f_c   = np.unique( np.concatenate( [ np.geomspace( lo, hi, 121 ), pn2[ ( pn2 > lo ) & ( pn2 < hi ) ] ] ) )
    common = consensus( common_in, pn2f_c, fluxf_c, fade=False )
    common.update( header='Common Cases',
                   cases=cases.tolist(), crossed=np.zeros( 16, dtype=bool ),
                   label_offset={ 8: ( 7, -14 ), 10: ( -17, 5 ) },
                   isotherm_mask=lambda name: True, region_labels='auto',
                   xlim=( fluxf_c.max()*fluxscale, fluxf_c.min()*fluxscale ), ylim=( lo, hi ),
                   plain_ticks=True, legend=not STACKED )
    report_areas( '=== fraction of the common-case panel ===', common )
    # The same rectangle from each model's full set of cases, faded as in the full panel
    report_areas( '=== same rectangle, all cases ===', consensus( ts_in, pn2f_c, fluxf_c, fade=True ) )
    panels.append( common )

# ── Plot ──────────────────────────────────────────────────────────────────────
c_frozen = '#d6e6f4'    # every 3-D model: global mean below freezing
c_warm   = '#dcefdb'    # every model: global mean above freezing
c_mixed  = '#ffffff'    # the 3-D models disagree, left white
c_mixed_2d = '#e6e6e6'  # contested only once PlaHab, the 2-D model, is included
c_run    = '#f2d6d8'    # pre-blended: EPS does not support transparency
marker_edge = 'k'

drawn = consensus_models + ( partial_models if SHOW_PARTIAL else [] )

def auto_label_position( B, M ):
    """The node of a region farthest from its edge, in display units, or None
    for a region too thin to hold a label."""
    fx = ( B[ 'flux_grid' ] - B[ 'flux_grid' ].min() ) / np.ptp( B[ 'flux_grid' ] )
    lp = np.log( B[ 'pres_grid' ] )
    py = ( lp - lp.min() ) / np.ptp( lp )
    d  = distance_transform_edt( np.pad( M, 1 ), sampling=( fx[ 1 ] - fx[ 0 ], np.median( np.diff( py ) ) ) )[ 1:-1, 1:-1 ]
    if d.max() < 0.12:
        return None
    i, j = np.unravel_index( np.argmax( d ), d.shape )
    return ( B[ 'flux_grid' ][ i ]*fluxscale, B[ 'pres_grid' ][ j ] )

def draw_panel( ax, B ):
    X, Y = np.meshgrid( B[ 'pres_grid' ], B[ 'flux_grid' ] * fluxscale )   # X = pressure, Y = instellation

    # Consensus regions, then the majority-runaway subset of the unfrozen region on
    # top of it, then the contested band, which is the subject of the figure.
    ax.contourf( Y, X, B[ 'band_blue' ].astype( float ),  levels=[ 0.5, 1.5 ], colors=[ c_frozen ], zorder=0 )
    ax.contourf( Y, X, B[ 'band_warm' ].astype( float ),  levels=[ 0.5, 1.5 ], colors=[ c_warm   ], zorder=0 )

    # Majority-runaway region, as a soft wash rather than a contour, so that it is not
    # confused with the individual model isotherms
    if 'frac_run' in B:
        ax.contourf( Y, X, B[ 'frac_run' ], levels=[ 0.49, 1.01 ], colors=[ c_run ], zorder=3 )

    # Grey first, then the 3-D-only contested band in white on top of it
    ax.contourf( Y, X, B[ 'contested_all' ].astype( float ),
                 levels=[ 0.5, 1.5 ], colors=[ c_mixed_2d ], zorder=2 )
    ax.contourf( Y, X, B[ 'contested_3d' ].astype( float ),
                 levels=[ 0.5, 1.5 ], colors=[ c_mixed ], zorder=2.2 )

    for name in drawn:
        mask = B[ 'isotherm_mask' ]( name )
        ax.contour( Y, X, np.where( mask, B[ 'Z' ][ name ], np.nan ), levels=[ T_FREEZE ],
                    colors=[ style[ name ][ 'color' ] ], linewidths=2.2,
                    linestyles=style[ name ][ 'ls' ], zorder=4 )

    for i in B[ 'cases' ]:
        f, p = flux1[ i ] * fluxscale, pres1[ i ]
        if B[ 'crossed' ][ i ]:
            ax.plot( f, p, marker='X', color='k', ms=11, mew=0.0, zorder=6 )
        else:
            ax.scatter( f, p, marker='o', s=55, c='k', edgecolors=marker_edge, zorder=6 )
        ax.annotate( str( i + 1 ), ( f, p ), textcoords='offset points',
                     xytext=B[ 'label_offset' ].get( i + 1, ( 7, 5 ) ), fontsize=9, zorder=7 )

    if B[ 'region_labels' ] == 'auto':
        region_labels = [ ( text, auto_label_position( B, B[ key ] ) )
                          for text, key in ( ( 'all models\nglobal mean\nbelow freezing', 'band_blue' ),
                                             ( 'all models\nglobal mean\nabove freezing', 'band_warm' ) ) ]
    else:
        region_labels = B[ 'region_labels' ]
    for text, xy in region_labels:
        if xy is None:
            continue
        color = '#3a6f9c' if 'below' in text else '#6b9b62'
        ax.annotate( text, xy=xy, fontsize=10.5,
                     ha='center', va='center', style='italic', color=color, zorder=5 )
    if 'frac_run' in B:
        ax.annotate( 'runaway\n(all but ExoPlaSim)', xy=( 2320, 0.60 ), fontsize=11,
                     ha='center', va='center', style='italic', color='#8a3a42', zorder=5 )

    ax.set_yscale( 'log' )
    ax.set_xlim( *B[ 'xlim' ] )
    ax.set_ylim( *B[ 'ylim' ] )
    ax.set_xlabel( 'Instellation (W m$^{-2}$)', fontsize=12 )
    ax.set_ylabel( 'N$_2$ surface pressure (bar)', fontsize=12 )
    ax.tick_params( axis='both', labelsize=11 )
    if B[ 'plain_ticks' ]:
        # Under a decade of pressure holds only one power of ten, so label plain values
        ax.set_yticks( [ 0.5, 1, 2, 5 ], labels=[ '0.5', '1', '2', '5' ] )
        ax.yaxis.set_minor_formatter( plt.NullFormatter() )

    if B[ 'legend' ]:
        legend_order = [ n for n in drawn if n != 'PlaHab' ] + [ n for n in drawn if n == 'PlaHab' ]
        handles  = [ Line2D( [], [], color=style[n][ 'color' ],
                             lw=2.2, ls=style[n][ 'ls' ], label=n )
                     for n in legend_order ]
        ax.legend( handles=handles, loc='upper left', fontsize=10, ncol=1,
                   framealpha=1, borderpad=0.7, labelspacing=0.5 )

if len( panels ) == 1:
    fig, ax = plt.subplots( figsize=( 9.5, 7.5 ) )
    draw_panel( ax, panels[ 0 ] )
else:
    # Side by side rather than stacked: the single panel already fills most of a
    # page with its caption. Each panel under a bold header; the legend, which
    # applies to both, sits in the full panel.
    fig, axs = plt.subplots( 1, len( panels ), figsize=( 9.5*len( panels ), 7.9 ), layout='constrained' )
    for ax, B in zip( axs, panels ):
        draw_panel( ax, B )
        ax.set_title( B[ 'header' ], fontsize=14, fontweight='bold', pad=10 )

suffix = ( "" if SHOW_PARTIAL else "_nopartial" ) + ( "_stacked" if STACKED else "_common" if COMMON else "" )
fig.savefig( f"fig_summary{suffix}.png", bbox_inches='tight' )
fig.savefig( f"fig_summary{suffix}.eps", bbox_inches='tight' )
