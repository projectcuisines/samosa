import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import cmocean

from matplotlib import patheffects
from pykrige.ok import OrdinaryKriging

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

fluxscale = 100

flux  = np.arange( 400, 2700, 100 ) / fluxscale
pn2   = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89,
                    1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400,
                    900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83,
                    0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

pcm_flux1  = np.array( [ 500, 1200,  800, 1100, 400,  900, 600 ] ) / fluxscale
pcm_pres1  = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )
lfric_flux1 = np.array( [ 500, 1200, 1600, 800, 1100, 400, 900, 1500, 900,  600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )
# HEXTOR contributes surface temperature only: it reports no water vapor column
# and no cloud fraction, so it enters the temperature panel and not the others.
hextor_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600, 1400 ] ) / fluxscale
hextor_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43, 10.00 ] )
# ExoColumn contributes surface temperature and water vapor; it is cloud-free,
# so it does not enter the cloud fraction panel.
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
                         293.0, 242.9, 260.8, 190.1, 240.0, 295.3, 286.1, 246.1, 207.9, 292.7 ] )
ts_pcm     = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218,
                         266.5987224285321, 210.69131033681012, 246.04296230476365,
                         217.2519558970929 ] )
ts_lfric   = np.array( [ 195.37, 251.48, 400.52, 231.83, 241.35, 197.81, 227.52, 333.20, 228.84, 203.64, 361.70 ] )
ts_hextor  = np.array( [ 173.10, 292.22, 224.40, 267.86, 152.42, 225.52, 241.46, 188.62, 376.07 ] )
ts_exocolumn = np.array( [ 206.98, 293.26, 248.49, 269.66, 201.36, 242.60, 251.63, 216.92 ] )

ts_exocam_mask  = ts_exocam  != runawaytemp
ts_rocke3d_mask = ts_rocke3d != runawaytemp
ts_plahab_mask  = ts_plahab  != runawaytemp

# ── Water vapor data (kg m⁻²) ─────────────────────────────────────────────────
runaway_wv = 1.e4

wv_plasim  = np.array( [ 0.059, 5113.363, 271.733, 7.509, 37.305, 1452.570,
                          76.876, 0.351, 7.048, 0.007, 3.042, 1258.514,
                          59.214, 1.955, 0.416, 1111.793 ] )
wv_exocam  = np.array( [ 0.2335, runaway_wv, runaway_wv, 19.2138,
                          runaway_wv, runaway_wv, runaway_wv, 1.7490,
                          7.9707, 0.0102, 5.8914, 1430.7613,
                          runaway_wv, 4.2041, 0.9534, 1295.6428 ] )
wv_rocke3d = np.array( [ 0.25980374, runaway_wv, runaway_wv, 14.9693165,
                          31.839989, runaway_wv, 32.45563, 1.5794185,
                          5.687923, 0.031635746, 4.2059116, 271.75916,
                          46.070984, 2.964403, 0.52949935, 132.49808 ] )
wv_pcm     = np.array( [ 0.37484651163423993, 47.86514350558743, 2.1906175203429563,
                          25.650192288432617, 0.06080825624148188,
                          5.93210602524276, 0.8447688576786204 ] )
wv_lfric   = np.array( [ 0.41, 8.18, 1735.22, 1.01, 7.37, 0.04, 5.46, 829.15, 2.34, 0.86, 1863.11 ] )

# ── Cloud fraction data (%) ───────────────────────────────────────────────────
runaway_cf = 200.0

cf_plasim  = np.array( [ 42.7, 68.2, 70.6, 56.2, 80.1, 32.1, 58.1, 25.7, 58.0,
                         25.2, 76.2, 30.1, 85.3, 52.7, 48.4, 53.3 ] )
cf_exocam  = np.array( [ 68.75, runaway_cf, runaway_cf, 43.98, runaway_cf, runaway_cf,
                         runaway_cf, 16.34, 75.82, 15.85, 83.08, 56.79,
                         runaway_cf, 34.01, 78.96, 61.40 ] )
cf_rocke3d = np.array( [ 68.20222, runaway_cf, runaway_cf, 51.043224, 81.88261, runaway_cf,
                         88.81546, 58.24044, 61.535275, 98.8356, 68.16493, 68.01357,
                         85.71091, 43.637707, 74.40385, 48.08909 ] )
cf_plahab  = np.array( [ 11.11879, runaway_cf, runaway_cf, 35.74597, 48.29323, runaway_cf,
                         70.91280, 26.24803, 31.64522, 8.4692545, 25.38866, 76.20874,
                         57.27629, 28.12309, 16.64636, 72.36285 ] )
cf_pcm     = np.array( [ 25.5674468009485, 27.31228828919005, 16.96672860199983,
                         25.15276275245855, 24.55454268845772, 16.696470834684884,
                         32.84789893586739 ] )
cf_lfric   = np.array( [ 31.0, 61.0, 34.0, 42.0, 58.0, 21.0, 87.0, 81.0, 44.0, 36.0, 83.0 ] )

wv_exocam_mask  = wv_exocam  != runaway_wv
wv_rocke3d_mask = wv_rocke3d != runaway_wv

cf_exocam_mask  = cf_exocam  != runaway_cf
cf_rocke3d_mask = cf_rocke3d != runaway_cf
cf_plahab_mask  = cf_plahab  != runaway_cf

# Which of the 16 main QMC points appear in the PCM / LFRic subsets
pcm_in_main   = np.array( [ any( np.isclose( f, pcm_flux1   ) & np.isclose( p, pcm_pres1   ) ) for f, p in zip( flux1, pres1 ) ] )
lfric_in_main = np.array( [ any( np.isclose( f, lfric_flux1 ) & np.isclose( p, lfric_pres1 ) ) for f, p in zip( flux1, pres1 ) ] )

# Number of models with valid data at each of the 16 QMC sample points
wv_exocolumn = np.array( [ 0.00248419, 23.5491, 0.418244, 3.59523, 0.00097159, 0.199347, 0.591193, 0.0102235 ] )

exocolumn_in_main = np.array( [ any( np.isclose( f, exocolumn_flux1 ) & np.isclose( p, exocolumn_pres1 ) ) for f, p in zip( flux1, pres1 ) ] )

hextor_in_main = np.array( [ any( np.isclose( f, hextor_flux1 ) & np.isclose( p, hextor_pres1 ) ) for f, p in zip( flux1, pres1 ) ] )

n_ts = ( np.ones( 16 )           # ExoPlaSim always valid
       + ts_exocam_mask           + ts_rocke3d_mask  + ts_plahab_mask
       + pcm_in_main              + lfric_in_main    + hextor_in_main
       + exocolumn_in_main )
n_wv = ( np.ones( 16 )           # ExoPlaSim always valid; PlaHab has no WV data
       + wv_exocam_mask           + wv_rocke3d_mask
       + pcm_in_main              + lfric_in_main    + exocolumn_in_main )
n_cf = ( np.ones( 16 )           # ExoPlaSim always valid
       + cf_exocam_mask           + cf_rocke3d_mask  + cf_plahab_mask
       + pcm_in_main              + lfric_in_main )

# ── Normalization ─────────────────────────────────────────────────────────────
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

def logit( x ):
    x = np.clip( x, 1.0, 99.0 )
    return np.log( x / ( 100.0 - x ) )

def sigmoid( y ):
    return 100.0 / ( 1.0 + np.exp( -y ) )

# Kriging anisotropy, fitted per model and per variable in fit_anisotropy.py and
# matching the values used by the fig_interpolation_* figures. A value of s means
# one unit of normalized instellation counts s times a unit of normalized
# log-pressure. Model-count surfaces stay isotropic: coverage is a property of
# the sampling design, not of a physical field.
ANISO_TS = { 'ExoPlaSim': 2,  'ExoCAM': 10, 'ROCKE-3D': 4, 'PlaHab': 10,
             'Generic PCM': 5, 'LFRic': 15, 'HEXTOR': 10, 'ExoColumn': 7 }
ANISO_WV = { 'ExoPlaSim': 3,  'ExoCAM': 10, 'ROCKE-3D': 7,
             'Generic PCM': 10, 'LFRic': 10, 'ExoColumn': 10 }
ANISO_CF = { 'ExoPlaSim': 1,  'ExoCAM': 1,  'ROCKE-3D': 3, 'PlaHab': 7,
             'Generic PCM': 1, 'LFRic': 1 }

def krige( p, f, z, scaling=1.0, pres_grid=pn2, flux_grid=flux ):
    ok = OrdinaryKriging( norm_pres( p ), norm_flux( f ), z,
                          anisotropy_scaling=scaling,
                          variogram_model="linear", verbose=False,
                          enable_plotting=False, exact_values=True )
    z_pred, z_var = ok.execute( "grid", norm_pres( pres_grid ), norm_flux( flux_grid ) )
    return z_pred, z_var

def weighted_std( values, variances ):
    """Variance-weighted std: models with low kriging uncertainty get higher weight.
    Floor at 10% of the median variance to cap max weight ratio at ~10:1,
    preventing exact-data sample points (var=0) from dominating."""
    V       = np.array( values )
    var_arr = np.array( variances )
    eps     = np.median( var_arr ) * 0.1
    W       = 1.0 / np.maximum( var_arr, eps )
    W_sum   = W.sum( axis=0 )
    z_mean  = ( W * V ).sum( axis=0 ) / W_sum
    return np.sqrt( ( W * ( V - z_mean ) ** 2 ).sum( axis=0 ) / W_sum )

# ── Models per variable ───────────────────────────────────────────────────────
# Each model's stable samples as ( pressure, instellation, value ), untransformed,
# in the order the spread is accumulated. A model that reports no value for a
# variable is absent from its table.
TS_MODELS = {
    'ExoPlaSim':   ( pres1,                    flux1,                    ts_plasim ),
    'ExoCAM':      ( pres1[ ts_exocam_mask  ], flux1[ ts_exocam_mask  ], ts_exocam[  ts_exocam_mask  ] ),
    'ROCKE-3D':    ( pres1[ ts_rocke3d_mask ], flux1[ ts_rocke3d_mask ], ts_rocke3d[ ts_rocke3d_mask ] ),
    'PlaHab':      ( pres1[ ts_plahab_mask  ], flux1[ ts_plahab_mask  ], ts_plahab[  ts_plahab_mask  ] ),
    'Generic PCM': ( pcm_pres1,                pcm_flux1,                ts_pcm ),
    'LFRic':       ( lfric_pres1,              lfric_flux1,              ts_lfric ),
    'HEXTOR':      ( hextor_pres1,             hextor_flux1,             ts_hextor ),
    'ExoColumn':   ( exocolumn_pres1,          exocolumn_flux1,          ts_exocolumn ),
}
WV_MODELS = {
    'ExoPlaSim':   ( pres1,                    flux1,                    wv_plasim ),
    'ExoCAM':      ( pres1[ wv_exocam_mask  ], flux1[ wv_exocam_mask  ], wv_exocam[  wv_exocam_mask  ] ),
    'ROCKE-3D':    ( pres1[ wv_rocke3d_mask ], flux1[ wv_rocke3d_mask ], wv_rocke3d[ wv_rocke3d_mask ] ),
    'Generic PCM': ( pcm_pres1,                pcm_flux1,                wv_pcm ),
    'LFRic':       ( lfric_pres1,              lfric_flux1,              wv_lfric ),
    'ExoColumn':   ( exocolumn_pres1,          exocolumn_flux1,          wv_exocolumn ),
}
CF_MODELS = {
    'ExoPlaSim':   ( pres1,                    flux1,                    cf_plasim ),
    'ExoCAM':      ( pres1[ cf_exocam_mask  ], flux1[ cf_exocam_mask  ], cf_exocam[  cf_exocam_mask  ] ),
    'ROCKE-3D':    ( pres1[ cf_rocke3d_mask ], flux1[ cf_rocke3d_mask ], cf_rocke3d[ cf_rocke3d_mask ] ),
    'PlaHab':      ( pres1[ cf_plahab_mask  ], flux1[ cf_plahab_mask  ], cf_plahab[  cf_plahab_mask  ] ),
    'Generic PCM': ( pcm_pres1,                pcm_flux1,                cf_pcm ),
    'LFRic':       ( lfric_pres1,              lfric_flux1,              cf_lfric ),
}

# Each variable is kriged in its own transformed coordinate (forward) and its
# spread taken after mapping back (back): K for temperature, dex for water vapor,
# percentage points for cloud fraction. The full color range, levels and ticks
# are those of the published figure; step and tick are the level spacing and
# tick interval a fitted range keeps.
ln10 = np.log( 10 )
VARS = [
    dict( key='ts', title='Surface Temperature', models=TS_MODELS, aniso=ANISO_TS, count=n_ts,
          forward=lambda v: v, back=lambda z: z,
          cm=cmocean.cm.thermal, label='σ(T$_s$) (K)',
          cmax=35, nlev=71, ticks=np.arange( 0, 36, 5 ), step=0.5, tick=5 ),
    dict( key='wv', title='Water Vapor Column', models=WV_MODELS, aniso=ANISO_WV, count=n_wv,
          forward=np.log, back=lambda z: z / ln10,
          cm=cmocean.cm.rain, label='σ(log$_{10}$ WV) (dex)',
          cmax=1.0, nlev=41, ticks=[ 0, 0.25, 0.5, 0.75, 1.0 ], step=0.025, tick=0.25 ),
    dict( key='cf', title='Cloud Fraction', models=CF_MODELS, aniso=ANISO_CF, count=n_cf,
          forward=logit, back=sigmoid,
          cm=cmocean.cm.ice_r, label='σ(CF) (%)',
          cmax=35, nlev=71, ticks=np.arange( 0, 36, 5 ), step=0.5, tick=5 ),
]

def spread( var, models, pres_grid=pn2, flux_grid=flux ):
    results = [ krige( p, f, var[ 'forward' ]( v ), var[ 'aniso' ][ name ], pres_grid, flux_grid )
                for name, ( p, f, v ) in models.items() ]
    z, variances = zip( *results )
    return weighted_std( [ var[ 'back' ]( zz ) for zz in z ], variances )

# With --common, every model is kriged from only the sample points at which all
# models with data for that variable reached a steady state, on a grid zoomed to
# the range those points span, so the spread measures disagreement between
# models and not differences in where each was sampled. The set is computed per
# variable; it is Cases 1, 4, 8, 9, 10, 14 and 15 for all three. The anisotropy
# ratios are left at the values fitted on each model's full set of cases. Every
# model has data at every point of the zoomed region, so nothing is hatched.
#
# With --stacked, the full figure is drawn above the common-case one, each row
# under its own header, the common cases on color ranges fitted to their spread.
COMMON  = '--common' in sys.argv
STACKED = '--stacked' in sys.argv
outname = 'fig_spread' + ( '_stacked' if STACKED else '_common' if COMMON else '' )

def _at( f, p, fs, ps ):
    return np.isclose( fs, f ) & np.isclose( ps, p )

def common_cases( models ):
    return np.array( [ all( _at( f, p, fs, ps ).any() for ps, fs, _ in models.values() )
                       for f, p in zip( flux1, pres1 ) ] )

def restrict( models, keep ):
    out = {}
    for name, ( ps, fs, vals ) in models.items():
        m = np.array( [ keep[ _at( f, p, flux1, pres1 ) ].any() for f, p in zip( fs, ps ) ] )
        out[ name ] = ( ps[ m ], fs[ m ], vals[ m ] )
    return out

def full_block():
    single = ( n_ts == 1 )
    return dict( header='All Cases', hatch=True, plain_ticks=False,
                 pres_grid=pn2, flux_grid=flux,
                 xlim=[ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ], xticks=[ 2500, 2000, 1500, 1000, 500 ],
                 ylim=[ min( pn2 )*0.9, max( pn2 )*1.1 ],
                 open_pts=~single, cross_pts=single,
                 std={ v[ 'key' ]: spread( v, v[ 'models' ] ) for v in VARS },
                 colors={ v[ 'key' ]: ( v[ 'cmax' ], v[ 'nlev' ], v[ 'ticks' ] ) for v in VARS } )

def common_block():
    keep = { v[ 'key' ]: common_cases( v[ 'models' ] ) for v in VARS }
    for v in VARS:
        print( f"{v[ 'title' ]}: cases stable in every model with data:",
               ( np.where( keep[ v[ 'key' ] ] )[ 0 ] + 1 ).tolist() )
    shown = np.logical_or.reduce( list( keep.values() ) )
    flux_grid = np.linspace( flux1[ shown ].min() - 0.5, flux1[ shown ].max() + 0.5, 41 )
    pres_grid = np.geomspace( pres1[ shown ].min()*0.9, pres1[ shown ].max()*1.1, 41 )
    std, colors = {}, {}
    for v in VARS:
        k = v[ 'key' ]
        std[ k ] = spread( v, restrict( v[ 'models' ], keep[ k ] ), pres_grid, flux_grid )
        if STACKED:
            # A range fitted to the common-case spread, rounded up to a tick
            cmax = v[ 'tick' ]*np.ceil( std[ k ].max()/v[ 'tick' ] )
            colors[ k ] = ( cmax, int( round( cmax/v[ 'step' ] ) ) + 1,
                            np.arange( 0, cmax + v[ 'tick' ]/2, v[ 'tick' ] ) )
        else:
            colors[ k ] = ( v[ 'cmax' ], v[ 'nlev' ], v[ 'ticks' ] )
        # The same region from each model's full set of cases, for comparison
        full_here = spread( v, v[ 'models' ], pres_grid, flux_grid )
        print( f"  median spread over the common-case region: {np.median( full_here ):.3g} "
               f"from all cases, {np.median( std[ k ] ):.3g} from the common cases "
               f"(max {std[ k ].max():.3g})" )
    return dict( header='Common Cases',
                 hatch=False, plain_ticks=True, pres_grid=pres_grid, flux_grid=flux_grid,
                 xlim=[ max( flux_grid*fluxscale ), min( flux_grid*fluxscale ) ], xticks=[ 1100, 900, 700, 500 ],
                 ylim=[ min( pres_grid ), max( pres_grid ) ],
                 open_pts=shown, cross_pts=np.zeros( 16, dtype=bool ), std=std, colors=colors )

if STACKED:
    blocks = [ full_block(), common_block() ]
elif COMMON:
    blocks = [ common_block() ]
else:
    blocks = [ full_block() ]

if not COMMON or STACKED:
    for v in VARS:
        print( f"{v[ 'title' ]}: median spread over the full plane {np.median( blocks[ 0 ][ 'std' ][ v[ 'key' ] ] ):.3g}" )

# ── Model-count surfaces (for single-model masking) ───────────────────────────
counts = { v[ 'key' ]: krige( pres1, flux1, v[ 'count' ] )[ 0 ] for v in VARS }

# ── Plot ──────────────────────────────────────────────────────────────────────
# Case numbers beside the sample points, as on the per-model figures. Labels sit
# to the right of each marker, except where that would crowd a neighbor or run
# off the panel.
label_left = { 1, 8, 10, 13, 15 }

def setup_panel( ax, title, block ):
    ax.set_title( title, fontsize=12 )
    ax.tick_params( axis='both', labelsize=10 )
    ax.set_yscale( 'log' )
    ax.set_xlim( block[ 'xlim' ] )
    ax.set_xticks( block[ 'xticks' ] )
    ax.set_ylim( block[ 'ylim' ] )
    ax.set_box_aspect( 1 )
    if block[ 'plain_ticks' ]:
        # Under a decade of pressure holds only one power of ten, so label plain values
        ax.set_yticks( [ 0.5, 1, 2, 5 ], labels=[ '0.5', '1', '2', '5' ] )
        ax.yaxis.set_minor_formatter( plt.NullFormatter() )
    ax.scatter( flux1[ block[ 'open_pts' ]  ]*fluxscale, pres1[ block[ 'open_pts' ]  ],
                color='none', edgecolors='k', s=40, linewidths=0.7, zorder=5 )
    ax.scatter( flux1[ block[ 'cross_pts' ] ]*fluxscale, pres1[ block[ 'cross_pts' ] ],
                color='k', marker='x', s=40, linewidths=0.7, zorder=5 )
    for i in np.where( block[ 'open_pts' ] | block[ 'cross_pts' ] )[ 0 ]:
        left = ( i + 1 ) in label_left
        ax.annotate( str( i + 1 ), ( flux1[ i ]*fluxscale, pres1[ i ] ),
                     xytext=( -5 if left else 5, 0 ), textcoords='offset points',
                     ha='right' if left else 'left', va='center', fontsize=9, zorder=6,
                     path_effects=[ patheffects.withStroke( linewidth=2.0, foreground='w' ) ] )

def draw_row( host, axes, block ):
    xv, yv = np.meshgrid( block[ 'pres_grid' ], block[ 'flux_grid' ] )
    for ax, v in zip( axes, VARS ):
        k = v[ 'key' ]
        cmax, nlev, ticks = block[ 'colors' ][ k ]
        # Spread above the top of the scale saturates at the top color rather than
        # being left unfilled: the temperature spread reaches 51.9 K at Case 7 and
        # 35-37 K in the warm high-pressure corner, above the 35 K scale, and
        # extend='neither' drew those regions as blank white.
        ax.contourf( yv*fluxscale, xv, block[ 'std' ][ k ], cmap=v[ 'cm' ], levels=np.linspace( 0, cmax, nlev ), extend='max' )
        if block[ 'hatch' ]:
            ax.contourf( yv*fluxscale, xv, counts[ k ], levels=[-1e9, 1.5], hatches=['///'], colors='none', alpha=0 )
        sm = mcm.ScalarMappable( cmap=v[ 'cm' ], norm=mcolors.Normalize( vmin=0, vmax=cmax ) )
        cb = host.colorbar( sm, ax=ax, extend='max' )
        cb.set_ticks( ticks )
        cb.ax.tick_params( labelsize=10 )
        cb.set_label( v[ 'label' ], fontsize=12 )
        setup_panel( ax, v[ 'title' ], block )
    # The panels of a row share their axes, so each is labeled once per row
    host.supxlabel( 'Instellation (W m$^{-2}$)', fontsize=12, fontweight='bold' )
    host.supylabel( 'Surface pressure (bar)', fontsize=12, fontweight='bold' )

if len( blocks ) == 1:
    fig, axes = plt.subplots( 1, 3, figsize=( 11.4, 3.7 ), layout='constrained' )
    draw_row( fig, axes, blocks[ 0 ] )
else:
    # Rows one above another, each under a bold header
    fig  = plt.figure( figsize=( 11.4, 7.6 ), layout='constrained' )
    rows = fig.subfigures( len( blocks ), 1, hspace=0.06 )
    for row, block in zip( rows, blocks ):
        row.suptitle( block[ 'header' ], fontsize=14, fontweight='bold' )
        draw_row( row, row.subplots( 1, 3 ), block )

fig.savefig( f"{outname}.png", bbox_inches='tight' )
fig.savefig( f"{outname}.eps", bbox_inches='tight' )
#plt.show()
