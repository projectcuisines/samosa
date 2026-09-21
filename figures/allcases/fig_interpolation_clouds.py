import sys
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from matplotlib.transforms import offset_copy
from matplotlib import patheffects
from pykrige.ok import OrdinaryKriging
from scipy import ndimage

# ─── Variable configuration ──────────────────────────────────────────────────
cm              = cmocean.cm.ice_r
contourmin      = 0.0
contourmax      = 100.0
cinterval       = 40
sigma_threshold = 1.0       # logit-units; hatch where kriging σ exceeds this
cbar_label      = 'Average Total Cloud Fraction (%)'
cbar_ticks      = np.arange( 0, 101, 20 )
# ─────────────────────────────────────────────────────────────────────────────

runaway   = 200.0   # sentinel (%) for runaway/unavailable cases
fluxscale = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Total Cloud Fraction (%)
# Generic PCM values are the Cldfrac column reported by the modeling group in
# samosa_gcm_output_case-N_OHT_off.dat, not a projection of the 3-D cloud field
# (max-overlap on ice_cloud_fraction gives 43.2% at Case 1 against 25.6% reported).
plasim  = np.array( [ 42.7, 68.2, 70.6, 56.2, 80.1, 32.1, 58.1, 25.7, 58.0, 25.2, 76.2, 30.1, 85.3, 52.7, 48.4, 53.3 ] )
exocam  = np.array( [ 68.75, runaway, runaway, 43.98, runaway, runaway, runaway, 16.34, 75.82, 15.85, 83.08, 56.79, runaway, 34.01, 78.96, 61.40 ] )
rocke3d = np.array( [ 68.20222, runaway, runaway, 51.043224, 81.88261, runaway, 88.81546, 58.24044, 61.535275, 98.8356, 68.16493, 68.01357, 85.71091, 43.637707, 74.40385, 48.08909 ] )
plahab  = np.array( [ 11.11879, runaway, runaway, 35.74597, 48.29323, runaway, 70.91280, 26.24803, 31.64522, 8.4692545, 4.3572873, 76.20874, 57.27629, 28.12309, 16.64636, 72.36285 ] )
pcm     = np.array( [ 25.5674468009485, 27.31228828919005, 16.96672860199983, 25.15276275245855, 24.55454268845772, 16.696470834684884, 32.84789893586739 ] )

pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 31.0, 61.0, 34.0, 42.0, 58.0, 21.0, 87.0, 81.0, 44.0, 36.0, 83.0 ] )
lfric_flux1 = np.array( [ 500, 1200, 1600, 800, 1100, 400, 900, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )

exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway
plahab_mask  = plahab  != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

# Each model's stable samples as ( instellation, pressure, cloud fraction ), in
# panel order: by model class, ending with the two one-dimensional models. A
# model that reports no cloud fraction is a string, drawn as a labelled empty
# panel.
MODELS = {
    'ExoPlaSim':   ( flux1,         pres1,         plasim         ),
    'ExoCAM':      ( exocam_flux1,  exocam_pres1,  exocam_stable  ),
    'ROCKE-3D':    ( rocke3d_flux1, rocke3d_pres1, rocke3d_stable ),
    'Generic PCM': ( pcm_flux1,     pcm_pres1,     pcm            ),
    'LFRic':       ( lfric_flux1,   lfric_pres1,   lfric          ),
    'PlaHab':      ( plahab_flux1,  plahab_pres1,  plahab_stable  ),
    'HEXTOR':      'No data\n(clear-sky model)',
    'ExoColumn':   'No data\n(cloud-free model)',
}

# With --common, every model is kriged from only the sample points at which all
# models with data reached a steady state, so the panels differ in the models
# and not in where each one was sampled. The set is computed rather than listed,
# so it follows the arrays above; it is Cases 1, 4, 8, 9, 10, 14 and 15, the
# same set as for surface temperature. The anisotropy ratios are left at the
# values fitted on each model's full set of cases.
#
# With --stacked, the full figure is drawn above the common-case one, each set
# under its own header and with its own colorbar, the common cases on a color
# range fitted to their samples.
COMMON  = '--common' in sys.argv
STACKED = '--stacked' in sys.argv
outname = 'fig_interpolation_clouds' + ( '_stacked' if STACKED else '_common' if COMMON else '' )

def _at( f, p, fs, ps ):
    return np.isclose( fs, f ) & np.isclose( ps, p )

def restrict_to_common( models ):
    with_data = [ m for m in models.values() if not isinstance( m, str ) ]
    common = np.array( [ all( _at( f, p, fs, ps ).any() for fs, ps, _ in with_data )
                         for f, p in zip( flux1, pres1 ) ] )
    cases  = ( np.where( common )[ 0 ] + 1 ).tolist()
    print( 'cases stable in every model with data:', cases )
    restricted = {}
    for name, model in models.items():
        if isinstance( model, str ):
            restricted[ name ] = model
            continue
        fs, ps, vals = model
        m = np.array( [ common[ _at( f, p, flux1, pres1 ) ].any() for f, p in zip( fs, ps ) ] )
        restricted[ name ] = ( fs[ m ], ps[ m ], vals[ m ] )
    return restricted, cases

# A view is the grid a set of panels is kriged on, its axis limits, its color
# range, and whether it is hatched where the kriging σ is large. The full view
# covers the whole parameter space. The zoomed view, used for the common cases,
# is unhatched and spans only the instellation and pressure its samples cover,
# padded by 50 W/m2 and 10% as the full axes are, since beyond that the kriging
# merely carries the edge values outward. It is kriged on a finer grid so the
# contours stay smooth; the normalization below still uses the full grid, which
# the anisotropy ratios were fitted on.
def full_view():
    return dict( flux_grid=flux, pres_grid=pn2, hatch=True, plain_ticks=False,
                 xlim=[ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ],
                 ylim=[ min( pn2 )*0.9, max( pn2 )*1.1 ],
                 cmin=contourmin, cmax=contourmax, cticks=cbar_ticks )

# A color range fitted to the samples shown, rounded out to 10% and ticked
# every 20% (every 10% over a span of 50% or less), for a block that has a
# colorbar of its own
def fitted_colors( models ):
    vals_shown = np.concatenate( [ m[ 2 ] for m in models.values() if not isinstance( m, str ) ] )
    cmin = 10*np.floor( vals_shown.min()/10 )
    cmax = 10*np.ceil( vals_shown.max()/10 )
    step = 10 if cmax - cmin <= 50 else 20
    return dict( cmin=cmin, cmax=cmax, cticks=np.arange( step*np.ceil( cmin/step ), cmax + 1, step ) )

def zoomed_view( models ):
    with_data = [ m for m in models.values() if not isinstance( m, str ) ]
    fs_shown  = np.concatenate( [ fs for fs, _, _ in with_data ] )
    ps_shown  = np.concatenate( [ ps for _, ps, _ in with_data ] )
    flux_grid = np.linspace( fs_shown.min() - 0.5, fs_shown.max() + 0.5, 41 )
    pres_grid = np.geomspace( ps_shown.min()*0.9, ps_shown.max()*1.1, 41 )
    return dict( flux_grid=flux_grid, pres_grid=pres_grid, hatch=False, plain_ticks=True,
                 xlim=[ max( flux_grid*fluxscale ), min( flux_grid*fluxscale ) ],
                 ylim=[ min( pres_grid ), max( pres_grid ) ],
                 cmin=contourmin, cmax=contourmax, cticks=cbar_ticks )

# Each block is ( header, models, view ), drawn as rows of four panels. Stacked,
# each block has its own colorbar and the common cases get a fitted color
# range; alone, they keep the full one so they compare directly with the full
# figure.
if STACKED or COMMON:
    common_models, _ = restrict_to_common( MODELS )
    common_block = ( 'Common Cases',
                     common_models, zoomed_view( common_models ) )
if STACKED:
    common_block[ 2 ].update( fitted_colors( common_models ) )
    blocks = [ ( 'All Cases', MODELS, full_view() ), common_block ]
elif COMMON:
    blocks = [ common_block ]
else:
    blocks = [ ( None, MODELS, full_view() ) ]

# Kriging anisotropy, fitted per model by leave-one-out cross-validation in
# fit_anisotropy.py. pykrige scales the second coordinate, which here is
# normalized instellation, so a value of s means one unit of normalized
# instellation counts s times a unit of normalized log-pressure. Isotropic
# kriging (s = 1) asserts the two axes are equally informative, which is false
# for cloud fraction: rerun fit_anisotropy.py after any resubmission.
# Generic PCM is pinned at 1 because no scaling resolves a surface for it
# here -- its fit is near-degenerate at every ratio, so the least distorted
# metric is kept and the flatness left visible rather than tuned away.
ANISO = {
    'ExoCAM':       1,
    'ROCKE-3D':     3,
    'ExoPlaSim':    1,
    'Generic PCM':  1,
    'PlaHab':       3,
    'LFRic':        1,
}

# Normalize both axes to [0, 1] for kriging so distance metric is balanced
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

# Cloud fraction is bounded, so it is kriged on the logit and mapped back
# through the inverse logit, which keeps the interpolated field within bounds
def krige( name, fs, ps, vals, view ):
    OK = OrdinaryKriging(
        norm_pres( ps ),
        norm_flux( fs ),
        logit( vals ),
        anisotropy_scaling=ANISO[ name ],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        exact_values=True,
    )
    return OK.execute( "grid", norm_pres( view[ 'pres_grid' ] ), norm_flux( view[ 'flux_grid' ] ) )

marker_edge = 'k'

# Of the regions where σ exceeds the threshold, hatch only those reaching the
# highest instellation on the grid, as the temperature figure does. The rest are
# slivers in the cool, thin corner of the plane, just past the outermost samples,
# where σ tops the threshold by at most 0.24 logit units; they run to seven grid
# cells in ExoCAM, ROCKE-3D, LFRic and PlaHab, and hatched they read as holes in
# the sampled region. The regions are 8-connected, so a dropped one never shares
# a grid cell with a kept one and zeroing it leaves the kept boundaries where
# they were.
def warm_edge_sigma( sigma ):
    regions, _ = ndimage.label( sigma > sigma_threshold, structure=np.ones( ( 3, 3 ) ) )
    dropped    = np.setdiff1d( regions, np.append( regions[ -1, : ], 0 ) )
    return np.where( np.isin( regions, dropped ), 0.0, sigma )

# ExoPlaSim is stable at all sixteen cases, so its panels carry the case numbers.
# Labels sit to the right of each marker, except where that would crowd a
# neighbor or run off the panel.
labeled_model = 'ExoPlaSim'
label_left    = { 10, 13 }

def label_cases( ax, fs, ps ):
    for f, p in zip( fs, ps ):
        case = np.where( _at( f, p, flux1, pres1 ) )[ 0 ][ 0 ] + 1
        left = case in label_left
        ax.annotate( str( case ), ( f*fluxscale, p ), xytext=( -6 if left else 6, 0 ), textcoords='offset points',
                     ha='right' if left else 'left', va='center', fontsize=10,
                     path_effects=[ patheffects.withStroke( linewidth=2.5, foreground='w' ) ] )

def setup_panel( ax, title, view ):
    ax.set_title( title, fontsize=14 )
    ax.set_xlabel( 'Instellation (W m$^{-2}$)', fontsize=12 )
    ax.set_ylabel( 'Surface pressure (bar)', fontsize=12 )
    ax.tick_params( axis='x', labelsize=11 )
    ax.tick_params( axis='y', labelsize=11 )
    ax.set_yscale( 'log' )
    ax.set_xlim( view[ 'xlim' ] )
    ax.set_ylim( view[ 'ylim' ] )
    if view[ 'plain_ticks' ]:
        # Under a decade of pressure holds only one power of ten, so label plain values
        ax.set_yticks( [ 0.5, 1, 2, 5 ], labels=[ '0.5', '1', '2', '5' ] )
        ax.yaxis.set_minor_formatter( plt.NullFormatter() )

def draw_empty( ax, name, label ):
    ax.set_axis_off()
    ax.set_title( name, fontsize=14 )
    ax.text( 0.5, 0.5, label, ha='center', va='center',
             transform=ax.transAxes, fontsize=13, style='italic', color='gray' )

def draw_panel( ax, name, fs, ps, vals, view ):
    z, var = krige( name, fs, ps, vals, view )
    xv, yv = np.meshgrid( view[ 'pres_grid' ], view[ 'flux_grid' ] )
    levels = np.linspace( view[ 'cmin' ], view[ 'cmax' ], cinterval )
    cf = ax.contourf( yv*fluxscale, xv, sigmoid(z), cmap=cm, levels=levels, vmin=view[ 'cmin' ], vmax=view[ 'cmax' ], extend='neither' )
    if view[ 'hatch' ]:
        ax.contourf( yv*fluxscale, xv, warm_edge_sigma( np.sqrt(var) ), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
    ax.scatter( fs*fluxscale, ps, c=vals, cmap=cm, vmin=view[ 'cmin' ], vmax=view[ 'cmax' ], marker='o', s=70, edgecolors=marker_edge )
    if name == labeled_model:
        label_cases( ax, fs, ps )
    setup_panel( ax, f'{name} (n={len(vals)})', view )
    return cf

def draw_block( axs, models, view ):
    cf = None
    for ax, ( name, model ) in zip( axs.flat, models.items() ):
        if isinstance( model, str ):
            draw_empty( ax, name, model )
        else:
            cf = draw_panel( ax, name, *model, view )
    return cf

def add_colorbar( fig, cf, rect, view ):
    cax = fig.add_axes( rect )
    cb = fig.colorbar( cf, cax=cax, extend='neither', ticks=view[ 'cticks' ] )
    cb.ax.tick_params( labelsize=11 )
    cb.ax.get_yaxis().labelpad = 15
    cb.set_label( cbar_label, rotation=270, fontsize=12 )

#--------------------------------------------------------------------
# Panels: rows of four, with the colorbar alongside rather than occupying a
# panel slot. Each block is two rows, so panels keep the same size.

nrows = len( MODELS ) // 4
if len( blocks ) == 1:
    _, models, view = blocks[ 0 ]
    fig, axs = plt.subplots( nrows, 4, figsize=(22, 4.5*nrows), squeeze=False )
    cf = draw_block( axs, models, view )
    fig.subplots_adjust( wspace=0.3, hspace=0.4, right=0.88 )
    add_colorbar( fig, cf, [ 0.905, 0.12, 0.013, 0.76 ], view )
else:
    # Blocks one above another, each under a bold header and with a colorbar
    # of its own spanning its rows
    fig   = plt.figure( figsize=(22, 10*nrows) )
    outer = fig.add_gridspec( len( blocks ), 1, hspace=0.25, right=0.88 )
    above = offset_copy( fig.transFigure, fig=fig, y=32, units='points' )
    for b, ( header, models, view ) in enumerate( blocks ):
        axs = outer[ b ].subgridspec( nrows, 4, wspace=0.3, hspace=0.4 ).subplots( squeeze=False )
        cf = draw_block( axs, models, view )
        top_left, top_right = axs[ 0, 0 ].get_position(), axs[ 0, -1 ].get_position()
        fig.text( ( top_left.x0 + top_right.x1 )/2, top_left.y1, header, transform=above,
                  ha='center', va='bottom', fontsize=18, fontweight='bold' )
        bottom = axs[ -1, 0 ].get_position().y0
        add_colorbar( fig, cf, [ 0.905, bottom, 0.013, top_left.y1 - bottom ], view )

#--------------------------------------------------------------------
# Finalize

fig.savefig( f"{outname}.png", bbox_inches='tight' )
fig.savefig( f"{outname}.eps", bbox_inches='tight' )
#plt.show()
