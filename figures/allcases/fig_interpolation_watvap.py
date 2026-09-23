import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean

from matplotlib.transforms import offset_copy
from matplotlib import patheffects
from pykrige.ok import OrdinaryKriging
from scipy import ndimage
from slide_halves import save_halves

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

# ─── Variable configuration ──────────────────────────────────────────────────
cm              = cmocean.cm.rain
contourmin      = 1.e-3
contourmax      = 1.e3
cinterval       = 40
sigma_threshold = 3.5       # log-units; hatch where kriging σ exceeds this
cbar_label      = 'Average Water Vapor Column (kg m$^{-2}$)'
cbar_ticks      = [ 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3 ]
# ─────────────────────────────────────────────────────────────────────────────

runaway   = 1.e4    # sentinel (kg m⁻²) for runaway/unavailable cases
fluxscale = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Water Vapor Column (kg m⁻²)
# Generic PCM values are the Qmass column reported in samosa_gcm_output_case-N_OHT_off.dat,
# which confirms kg/m^2 as the common unit here.
plasim  = np.array( [ 0.059, 5113.363, 271.733, 7.509, 37.305, 1452.570, 76.876, 0.351, 7.048, 0.007, 3.042, 1258.514, 59.214, 1.955, 0.416, 1111.793 ] )
exocam  = np.array( [ 0.2335, runaway, runaway, 19.2138, runaway, runaway, runaway, 1.7490, 7.9707, 0.0102, 5.8914, 1430.7613, runaway, 4.2041, 0.9534, 1295.6428 ] )
rocke3d = np.array( [ 0.25980374, runaway, runaway, 14.9693165, 31.839989, runaway, 32.45563, 1.5794185, 5.687923, 0.031635746, 4.2059116, 271.75916, 46.070984, 2.964403, 0.52949935, 132.49808 ] )
# PlaHab: no water vapor data (2D model)
pcm       = np.array( [ 0.37484651163423993, 47.86514350558743, 2.1906175203429563, 25.650192288432617, 0.06080825624148188, 5.93210602524276, 0.8447688576786204 ] )
pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 0.41, 8.18, 1735.22, 1.01, 7.37, 0.04, 5.46, 829.15, 2.34, 0.86, 1863.11 ] )
lfric_flux1 = np.array( [ 500, 1200, 1600, 800, 1100, 400, 900, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )

# ExoColumn, cases 1, 4, 8, 9, 10, 11, 14, 15 (kg m^-2). It carries a vertical
# coordinate and prognostic water vapour, so unlike HEXTOR it reports a column.
exocolumn       = np.array( [ 0.00248419, 23.5491, 0.418244, 3.59523, 0.00097159, 0.199347, 0.591193, 0.0102235 ] )
exocolumn_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600 ] ) / fluxscale
exocolumn_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43 ] )

exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]

# Each model's stable samples as ( instellation, pressure, water vapor column ),
# in panel order: by model class, ending with the two one-dimensional models.
# A model that reports no water vapor column is a string, drawn as a labelled
# empty panel.
MODELS = {
    'ExoPlaSim':   ( flux1,           pres1,           plasim         ),
    'ExoCAM':      ( exocam_flux1,    exocam_pres1,    exocam_stable  ),
    'ROCKE-3D':    ( rocke3d_flux1,   rocke3d_pres1,   rocke3d_stable ),
    'Generic PCM': ( pcm_flux1,       pcm_pres1,       pcm            ),
    'LFRic':       ( lfric_flux1,     lfric_pres1,     lfric          ),
    'PlaHab':      'No data\n(2D model)',
    'HEXTOR':      'No data\n(1D model)',
    'ExoColumn':   ( exocolumn_flux1, exocolumn_pres1, exocolumn      ),
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
outname = 'fig_interpolation_watvap' + ( '_stacked' if STACKED else '_common' if COMMON else '' )

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
                 xlim=[ max( flux*fluxscale ) + 50, min( flux*fluxscale ) - 50 ], xticks=[ 2500, 2000, 1500, 1000, 500 ],
                 ylim=[ min( pn2 )*0.9, max( pn2 )*1.1 ],
                 cmin=contourmin, cmax=contourmax, cticks=cbar_ticks )

# A color range fitted to the samples shown, rounded out to whole decades and
# ticked every decade, for a block that has a colorbar of its own
def fitted_colors( models ):
    vals_shown = np.concatenate( [ m[ 2 ] for m in models.values() if not isinstance( m, str ) ] )
    lo = int( np.floor( np.log10( vals_shown.min() ) ) )
    hi = int( np.ceil( np.log10( vals_shown.max() ) ) )
    return dict( cmin=10.0**lo, cmax=10.0**hi, cticks=[ 10.0**k for k in range( lo, hi + 1 ) ] )

def zoomed_view( models ):
    with_data = [ m for m in models.values() if not isinstance( m, str ) ]
    fs_shown  = np.concatenate( [ fs for fs, _, _ in with_data ] )
    ps_shown  = np.concatenate( [ ps for _, ps, _ in with_data ] )
    flux_grid = np.linspace( fs_shown.min() - 0.5, fs_shown.max() + 0.5, 41 )
    pres_grid = np.geomspace( ps_shown.min()*0.9, ps_shown.max()*1.1, 41 )
    return dict( flux_grid=flux_grid, pres_grid=pres_grid, hatch=False, plain_ticks=True,
                 xlim=[ max( flux_grid*fluxscale ), min( flux_grid*fluxscale ) ], xticks=[ 1100, 900, 700, 500 ],
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
# for water vapor: rerun fit_anisotropy.py after any resubmission.
ANISO = {
    'ExoCAM':       10,
    'ROCKE-3D':     7,
    'ExoPlaSim':    3,
    'Generic PCM':  10,
    'LFRic':        10,
    'ExoColumn':    10,
}

# Normalize both axes to [0, 1] for kriging so distance metric is balanced
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

# The column spans more than six orders of magnitude, so it is kriged in log
# space and transformed back for plotting
def krige( name, fs, ps, vals, view ):
    OK = OrdinaryKriging(
        norm_pres( ps ),
        norm_flux( fs ),
        np.log( vals ),
        anisotropy_scaling=ANISO[ name ],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        exact_values=True,
    )
    return OK.execute( "grid", norm_pres( view[ 'pres_grid' ] ), norm_flux( view[ 'flux_grid' ] ) )

marker_edge = 'k'

# Each model's sample points take its marker from Figure 2 (fig_energy_balance.py),
# filled here by the simulated value rather than the model's color. The star is
# drawn larger so that it reads at the same weight as the other shapes, and no
# marker is clipped at the panel edge, where a cut shape would be hard to name.
MARKER = { 'ExoPlaSim': 'o', 'ExoCAM': 's', 'ROCKE-3D': '^', 'Generic PCM': 'D',
           'LFRic': 'v', 'PlaHab': 'P', 'HEXTOR': 'X', 'ExoColumn': '*' }
MARKER_SIZE = { '*': 80 }

# Of the regions where σ exceeds the threshold, hatch only those reaching the
# highest instellation on the grid, as the temperature figure does. Every region
# presently reaches it, so nothing is dropped here; the rule is carried for the
# slivers that appear along the other panel edges, just past the outermost
# samples, once a resubmission steepens a variogram. The regions are 8-connected,
# so a dropped one never shares a grid cell with a kept one and zeroing it leaves
# the kept boundaries where they were.
def warm_edge_sigma( sigma ):
    regions, _ = ndimage.label( sigma > sigma_threshold, structure=np.ones( ( 3, 3 ) ) )
    dropped    = np.setdiff1d( regions, np.append( regions[ -1, : ], 0 ) )
    return np.where( np.isin( regions, dropped ), 0.0, sigma )

# ExoPlaSim is stable at all sixteen cases, so its panels carry the case numbers.
# Labels sit to the right of each marker, except where that would crowd a
# neighbor or run off the panel.
labeled_model = 'ExoPlaSim'
label_left    = { 1, 8, 10, 13, 15 }

def label_cases( ax, fs, ps ):
    for f, p in zip( fs, ps ):
        case = np.where( _at( f, p, flux1, pres1 ) )[ 0 ][ 0 ] + 1
        left = case in label_left
        ax.annotate( str( case ), ( f*fluxscale, p ), xytext=( -6 if left else 6, 0 ), textcoords='offset points',
                     ha='right' if left else 'left', va='center', fontsize=9,
                     path_effects=[ patheffects.withStroke( linewidth=2, foreground='w' ) ] )

def setup_panel( ax, title, view ):
    ax.set_title( title, fontsize=12 )
    ax.tick_params( axis='both', labelsize=10 )
    ax.set_yscale( 'log' )
    ax.set_xlim( view[ 'xlim' ] )
    ax.set_xticks( view[ 'xticks' ] )
    ax.set_ylim( view[ 'ylim' ] )
    ax.set_box_aspect( 1 )
    ax.apply_aspect()
    if view[ 'plain_ticks' ]:
        # Under a decade of pressure holds only one power of ten, so label plain values
        ax.set_yticks( [ 0.5, 1, 2, 5 ], labels=[ '0.5', '1', '2', '5' ] )
        ax.yaxis.set_minor_formatter( plt.NullFormatter() )

# The model name goes inside the empty slot rather than above it, where it
# would collide with the tick labels of the panel overhead
def draw_empty( ax, name, label ):
    ax.set_axis_off()
    ax.set_box_aspect( 1 )
    ax.apply_aspect()
    ax.text( 0.5, 0.56, name, ha='center', va='bottom', transform=ax.transAxes, fontsize=12 )
    ax.text( 0.5, 0.52, label, ha='center', va='top',
             transform=ax.transAxes, fontsize=11, style='italic', color='gray' )

def draw_panel( ax, name, fs, ps, vals, view ):
    z, var = krige( name, fs, ps, vals, view )
    xv, yv = np.meshgrid( view[ 'pres_grid' ], view[ 'flux_grid' ] )
    norm   = mcolors.LogNorm( vmin=view[ 'cmin' ], vmax=view[ 'cmax' ] )
    levels = np.logspace( np.log10( view[ 'cmin' ] ), np.log10( view[ 'cmax' ] ), cinterval )
    cf = ax.contourf( yv*fluxscale, xv, np.exp(z), cmap=cm, levels=levels, norm=norm, extend='both' )
    if view[ 'hatch' ]:
        ax.contourf( yv*fluxscale, xv, warm_edge_sigma( np.sqrt(var) ), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
    ax.scatter( fs*fluxscale, ps, c=vals, cmap=cm, norm=norm, marker=MARKER[ name ],
                s=MARKER_SIZE.get( MARKER[ name ], 45 ), edgecolors=marker_edge, clip_on=False )
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

# Every panel in a block shares one view, so the axes are labeled once per
# block: pressure to the left of it, instellation under its bottom row, and
# tick labels only along its outer edges. A panel keeps its instellation tick
# labels when the one below it is an empty "No data" slot.
def label_block( fig, axs ):
    for ( r, c ), ax in np.ndenumerate( axs ):
        if c > 0:
            ax.tick_params( labelleft=False )
        if r < axs.shape[ 0 ] - 1 and axs[ r + 1, c ].axison:
            ax.tick_params( labelbottom=False )
    top_left, bottom_right = axs[ 0, 0 ].get_position(), axs[ -1, -1 ].get_position()
    fig.text( top_left.x0, ( top_left.y1 + bottom_right.y0 )/2, 'Surface pressure (bar)',
              rotation=90, ha='right', va='center', fontsize=12, fontweight='bold',
              transform=offset_copy( fig.transFigure, fig=fig, x=-38, units='points' ) )
    fig.text( ( top_left.x0 + bottom_right.x1 )/2, bottom_right.y0, 'Instellation (W m$^{-2}$)',
              ha='center', va='top', fontsize=12, fontweight='bold',
              transform=offset_copy( fig.transFigure, fig=fig, y=-25, units='points' ) )

def add_colorbar( fig, cf, rect, view ):
    cax = fig.add_axes( rect )
    cb = fig.colorbar( cf, cax=cax, extend='both', ticks=view[ 'cticks' ] )
    cb.ax.tick_params( labelsize=10 )
    cb.ax.get_yaxis().labelpad = 16
    cb.set_label( cbar_label, rotation=270, fontsize=12 )

#--------------------------------------------------------------------
# Panels: rows of four, with the colorbar alongside rather than occupying a
# panel slot. Each block is two rows, so panels keep the same size.

nrows = len( MODELS ) // 4
if len( blocks ) == 1:
    _, models, view = blocks[ 0 ]
    fig, axs = plt.subplots( nrows, 4, figsize=(12.2, 2.95*nrows), squeeze=False )
    cf = draw_block( axs, models, view )
    fig.subplots_adjust( wspace=0.12, hspace=0.15, right=0.88 )
    label_block( fig, axs )
    add_colorbar( fig, cf, [ 0.905, 0.12, 0.013, 0.76 ], view )
else:
    # Blocks one above another, each under a bold header and with a colorbar
    # of its own spanning its rows
    fig   = plt.figure( figsize=(12.2, 6.87*nrows) )
    outer = fig.add_gridspec( len( blocks ), 1, hspace=0.33, right=0.88 )
    above = offset_copy( fig.transFigure, fig=fig, y=23, units='points' )
    for b, ( header, models, view ) in enumerate( blocks ):
        axs = outer[ b ].subgridspec( nrows, 4, wspace=0.12, hspace=0.15 ).subplots( squeeze=False )
        cf = draw_block( axs, models, view )
        top_left, top_right = axs[ 0, 0 ].get_position(), axs[ 0, -1 ].get_position()
        fig.text( ( top_left.x0 + top_right.x1 )/2, top_left.y1, header, transform=above,
                  ha='center', va='bottom', fontsize=14, fontweight='bold' )
        label_block( fig, axs )
        bottom = axs[ -1, 0 ].get_position().y0
        add_colorbar( fig, cf, [ 0.905, bottom, 0.013, top_left.y1 - bottom ], view )

#--------------------------------------------------------------------
# Finalize

fig.savefig( f"{outname}.png", bbox_inches='tight' )
fig.savefig( f"{outname}.eps", bbox_inches='tight' )

# Each block on its own, for slides
if STACKED:
    save_halves( fig, outname.replace( '_stacked', '_slide' ) )
#plt.show()
