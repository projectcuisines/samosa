import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import cmocean

from pykrige.ok import OrdinaryKriging

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

# ── Water vapor data (kg m⁻²) ─────────────────────────────────────────────────
_s         = 0.1
runaway_wv = 1.e4

wv_plasim  = _s * np.array( [ 0.059, 5113.363, 271.733, 7.509, 37.305, 1452.570,
                               76.876, 0.351, 7.048, 0.007, 3.042, 1258.514,
                               59.214, 1.955, 0.416, 1111.793 ] )
wv_exocam  = _s * np.array( [ 0.2335, runaway_wv/_s, runaway_wv/_s, 19.2138,
                               runaway_wv/_s, runaway_wv/_s, runaway_wv/_s, 1.7490,
                               7.9707, 0.0102, 5.8914, 1430.7613,
                               runaway_wv/_s, 4.2041, 0.9534, 1295.6428 ] )
wv_rocke3d = _s * np.array( [ 0.25980374, runaway_wv/_s, runaway_wv/_s, 14.9693165,
                               31.839989, runaway_wv/_s, 32.45563, 1.5794185,
                               5.687923, 0.031635746, 4.2059116, 271.75916,
                               46.070984, 2.964403, 0.52949935, 132.49808 ] )
wv_pcm     = _s * np.array( [ 0.37484651163423993, 47.86514350558743, 2.1906175203429563,
                               25.650192288432617, 0.06080825624148188,
                               5.93210602524276, 0.8447688576786204 ] )
wv_lfric   = np.array( [ 0.41, 8.18, 7.37, 829.15, 2.34, 0.86 ] )

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
                         70.91280, 26.24803, 31.64522, 8.4692545, 4.3572873, 76.20874,
                         57.27629, 28.12309, 16.64636, 72.36285 ] )
cf_pcm     = np.array( [ 25.5674468009485, 27.31228828919005, 16.96672860199983,
                         25.15276275245855, 24.55454268845772, 16.696470834684884,
                         32.84789893586739 ] )
cf_lfric   = np.array( [ 31.0, 61.0, 58.0, 81.0, 44.0, 36.0 ] )

wv_exocam_mask  = wv_exocam  != runaway_wv
wv_rocke3d_mask = wv_rocke3d != runaway_wv

cf_exocam_mask  = cf_exocam  != runaway_cf
cf_rocke3d_mask = cf_rocke3d != runaway_cf
cf_plahab_mask  = cf_plahab  != runaway_cf

# Which of the 16 main QMC points appear in the PCM / LFRic subsets
pcm_in_main   = np.array( [ any( np.isclose( f, pcm_flux1   ) & np.isclose( p, pcm_pres1   ) ) for f, p in zip( flux1, pres1 ) ] )
lfric_in_main = np.array( [ any( np.isclose( f, lfric_flux1 ) & np.isclose( p, lfric_pres1 ) ) for f, p in zip( flux1, pres1 ) ] )

# Number of models with valid data at each of the 16 QMC sample points
n_ts = ( np.ones( 16 )           # ExoPlaSim always valid
       + ts_exocam_mask           + ts_rocke3d_mask  + ts_plahab_mask
       + pcm_in_main              + lfric_in_main )
n_wv = ( np.ones( 16 )           # ExoPlaSim always valid; PlaHab has no WV data
       + wv_exocam_mask           + wv_rocke3d_mask
       + pcm_in_main              + lfric_in_main )
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

def krige( p, f, z ):
    ok = OrdinaryKriging( norm_pres( p ), norm_flux( f ), z,
                          variogram_model="linear", verbose=False,
                          enable_plotting=False, exact_values=True )
    z_pred, z_var = ok.execute( "grid", norm_pres( pn2 ), norm_flux( flux ) )
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

# ── Kriging: Temperature ──────────────────────────────────────────────────────
ts_krige = [
    krige( pres1,                              flux1,                              ts_plasim ),
    krige( pres1[ ts_exocam_mask  ],           flux1[ ts_exocam_mask  ],           ts_exocam[  ts_exocam_mask  ] ),
    krige( pres1[ ts_rocke3d_mask ],           flux1[ ts_rocke3d_mask ],           ts_rocke3d[ ts_rocke3d_mask ] ),
    krige( pres1[ ts_plahab_mask  ],           flux1[ ts_plahab_mask  ],           ts_plahab[  ts_plahab_mask  ] ),
    krige( pcm_pres1,                          pcm_flux1,                          ts_pcm ),
    krige( lfric_pres1,                        lfric_flux1,                        ts_lfric ),
]
z_ts,   var_ts = zip( *ts_krige )
std_ts = weighted_std( z_ts, var_ts )

# ── Kriging: Water vapor (log-space; runaway excluded) ────────────────────────
wv_krige = [
    krige( pres1,                    flux1,                    np.log( wv_plasim ) ),
    krige( pres1[ wv_exocam_mask  ], flux1[ wv_exocam_mask  ], np.log( wv_exocam[  wv_exocam_mask  ] ) ),
    krige( pres1[ wv_rocke3d_mask ], flux1[ wv_rocke3d_mask ], np.log( wv_rocke3d[ wv_rocke3d_mask ] ) ),
    krige( pcm_pres1,                pcm_flux1,                np.log( wv_pcm ) ),
    krige( lfric_pres1,              lfric_flux1,              np.log( wv_lfric ) ),
]
z_wv_log, var_wv = zip( *wv_krige )
ln10    = np.log( 10 )
std_wv  = weighted_std( [ z / ln10 for z in z_wv_log ], var_wv )

# ── Kriging: Cloud fraction (logit-space) ─────────────────────────────────────
cf_krige = [
    krige( pres1,                              flux1,                              logit( cf_plasim ) ),
    krige( pres1[ cf_exocam_mask  ],           flux1[ cf_exocam_mask  ],           logit( cf_exocam[  cf_exocam_mask  ] ) ),
    krige( pres1[ cf_rocke3d_mask ],           flux1[ cf_rocke3d_mask ],           logit( cf_rocke3d[ cf_rocke3d_mask ] ) ),
    krige( pres1[ cf_plahab_mask  ],           flux1[ cf_plahab_mask  ],           logit( cf_plahab[  cf_plahab_mask  ] ) ),
    krige( pcm_pres1,                          pcm_flux1,                          logit( cf_pcm ) ),
    krige( lfric_pres1,                        lfric_flux1,                        logit( cf_lfric ) ),
]
z_cf,   var_cf = zip( *cf_krige )
std_cf  = weighted_std( [ sigmoid( z ) for z in z_cf ], var_cf )

# ── Krige model-count surfaces (for single-model masking) ────────────────────
count_ts, _ = krige( pres1, flux1, n_ts )
count_wv, _ = krige( pres1, flux1, n_wv )
count_cf, _ = krige( pres1, flux1, n_cf )

# ── Plot ──────────────────────────────────────────────────────────────────────
xv, yv = np.meshgrid( pn2, flux )
xlim = [ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ]
ylim = [ min( pn2 )*0.9, max( pn2 )*1.1 ]

cm_ts = cmocean.cm.thermal
cm_wv = cmocean.cm.rain
cm_cf = cmocean.cm.ice_r

fig, axes = plt.subplots( 1, 3, figsize=( 14, 4.5 ) )

single_mask = ( n_ts == 1 )
multi_mask  = ~single_mask

def setup_panel( ax, title ):
    ax.set_title( title, fontsize=13 )
    ax.set_xlabel( 'Instellation (W m$^{-2}$)', fontsize=11 )
    ax.set_ylabel( 'Surface pressure (bar)', fontsize=11 )
    ax.tick_params( axis='x', labelsize=10 )
    ax.tick_params( axis='y', labelsize=10 )
    ax.set_yscale( 'log' )
    ax.set_xlim( xlim )
    ax.set_ylim( ylim )
    ax.scatter( flux1[ multi_mask  ]*fluxscale, pres1[ multi_mask  ],
                color='none', edgecolors='k', s=40, linewidths=0.7, zorder=5 )
    ax.scatter( flux1[ single_mask ]*fluxscale, pres1[ single_mask ],
                color='k', marker='x', s=40, linewidths=0.7, zorder=5 )

# Panel 1: Temperature spread
axes[0].contourf( yv*fluxscale, xv, std_ts, cmap=cm_ts, levels=np.linspace( 0, 35, 71 ), extend='neither' )
axes[0].contourf( yv*fluxscale, xv, count_ts, levels=[-1e9, 1.5], hatches=['///'], colors='none', alpha=0 )
sm1 = mcm.ScalarMappable( cmap=cm_ts, norm=mcolors.Normalize( vmin=0, vmax=35 ) )
cb1 = fig.colorbar( sm1, ax=axes[0], label='σ(T$_s$) (K)', extend='neither' )
cb1.set_ticks( np.arange( 0, 36, 5 ) )
setup_panel( axes[0], 'Surface Temperature' )

# Panel 2: Water vapor spread
axes[1].contourf( yv*fluxscale, xv, std_wv, cmap=cm_wv, levels=np.linspace( 0, 1.0, 41 ), extend='neither' )
axes[1].contourf( yv*fluxscale, xv, count_wv, levels=[-1e9, 1.5], hatches=['///'], colors='none', alpha=0 )
sm2 = mcm.ScalarMappable( cmap=cm_wv, norm=mcolors.Normalize( vmin=0, vmax=1.0 ) )
cb2 = fig.colorbar( sm2, ax=axes[1], label='σ(log$_{10}$ WV) (dex)', extend='neither' )
cb2.set_ticks( [ 0, 0.25, 0.5, 0.75, 1.0 ] )
setup_panel( axes[1], 'Water Vapor Column' )

# Panel 3: Cloud fraction spread
axes[2].contourf( yv*fluxscale, xv, std_cf, cmap=cm_cf, levels=np.linspace( 0, 35, 71 ), extend='neither' )
axes[2].contourf( yv*fluxscale, xv, count_cf, levels=[-1e9, 1.5], hatches=['///'], colors='none', alpha=0 )
sm3 = mcm.ScalarMappable( cmap=cm_cf, norm=mcolors.Normalize( vmin=0, vmax=35 ) )
cb3 = fig.colorbar( sm3, ax=axes[2], label='σ(CF) (%)', extend='neither' )
cb3.set_ticks( np.arange( 0, 36, 5 ) )
setup_panel( axes[2], 'Cloud Fraction' )

fig.tight_layout()
fig.savefig( "fig_spread.png", bbox_inches='tight' )
fig.savefig( "fig_spread.eps", bbox_inches='tight' )
#plt.show()
