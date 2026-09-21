import sys
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from matplotlib.transforms import offset_copy
from matplotlib import patheffects
from pykrige.ok import OrdinaryKriging
from scipy import ndimage

# ─── Variable configuration ──────────────────────────────────────────────────
# cmocean.ice bottoms out at near-black, and the old floor of 10% put four of
# HEXTOR's nine cases and LFRic's Case 7 below it, so both panels read as dark
# slabs with no structure rather than as the genuinely low albedos they are.
# haline holds a readable lightness across the whole ensemble range, and the
# floor now sits at 0% so nothing real is clipped into the under-color; only
# PlaHab Case 11 (63.7%, see the note above) runs past the top.
cm              = cmocean.cm.haline
contourmin      = 0.0
contourmax      = 45.0
cinterval       = 46      # 1% per level over the range
sigma_threshold = 1.0       # logit-units; hatch where kriging σ exceeds this
cbar_label      = 'Planetary Albedo (%)'
cbar_ticks      = np.arange( 0, 46, 5 )
# ─────────────────────────────────────────────────────────────────────────────

runaway   = 200.0   # sentinel (%) for runaway/unavailable cases
fluxscale = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Planetary albedo (%), from the standardized SAMOSA global output.
#   ROCKE-3D  plan_alb_hemis[2] as reported by the modeling group
#   ExoPlaSim rsut / ( rst + rsut ) from the area-weighted TOA fluxes
#   ExoCAM    FUS / FDS at the topmost interface level, gw-weighted from
#             samosaN.cam.h0.avg.nc
#   others    1 - ASR / ( S / 4 ), using the incident flux fixed by the protocol
# Cross-checks against the primary NetCDF:
#   ROCKE-3D   reported plan_alb 23.08% vs 1 - ASR/(S/4) = 23.08% at Case 1.
#   LFRic      sw_net_toa / lw_up_toa reproduce the .txt global diagnostics
#              exactly, so the derived albedos here are confirmed.
#   ExoCAM     FSNTOA and every clear-sky field are archived as identically
#              zero in the submitted files, but the resolved shortwave flux
#              profiles FDS and FUS are populated, and their topmost interface
#              is the top of the model. Reflected over incident there is the
#              planetary albedo, and it reproduces the summary TOAALB to within
#              0.03 pp in all ten rows of output.txt, so no field the group
#              zeroed is needed after all. The earlier 1 - FSNT/(S/4) read
#              0.3-0.9 pp high because the incident flux the model sees, mean
#              FDS at that level, runs 0.4-0.5% below the protocol S/4. This
#              also puts ExoCAM on the same footing as ExoPlaSim and ROCKE-3D,
#              which report reflected over incident from their own fluxes
#              rather than against the prescribed S/4. The gw-weighted TS
#              reproduces the summary TS exactly for all 11 files, which
#              validates the weighting.
plasim  = np.array( [ 40.77, 25.00, 21.85, 39.99, 31.52, 18.87, 29.82, 40.23, 33.58, 42.23, 38.32, 17.26, 31.10, 36.40, 39.34, 32.58 ] )
exocam  = np.array( [ 26.98, runaway, runaway, 30.74, runaway, runaway, runaway, 20.33, 34.23, 31.44, 22.00, 18.10, runaway, 28.57, 20.02, 16.49 ] )
rocke3d = np.array( [ 23.08, runaway, runaway, 31.56, 39.86, runaway, 44.31, 22.14, 37.63, 21.21, 28.72, 17.96, 40.84, 30.09, 22.26, 12.78 ] )
plahab  = np.array( [ 19.11, runaway, runaway, 33.03, 33.02, runaway, 34.90, 29.62, 30.82, 0.89, 63.70, 35.51, 34.85, 31.12, 19.93, 35.54 ] )
pcm     = np.array( [ 25.57, 14.42, 22.69, 17.72, 28.53, 19.54, 21.31 ] )

pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 26.03, 35.74, 3.44, 33.84, 34.16, 23.99, 30.39, 23.22, 34.41, 28.25, 21.35 ] )
lfric_flux1 = np.array( [ 500, 1200, 1600, 800, 1100, 400, 900, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )

# HEXTOR, cases 1, 4, 8, 9, 10, 11, 14, 15, 16 (see fig_interpolation_temp.py).
# Clear-sky by construction: HEXTOR has no
# clouds, so these are surface-plus-Rayleigh albedos and sit well below the rest
# of the ensemble wherever the surface is ice-free.
hextor       = np.array( [ 19.96, 3.40, 12.92, 3.32, 21.80, 15.64, 6.49, 19.25, 3.79 ] )
hextor_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600, 1400 ] ) / fluxscale
hextor_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43, 10.00 ] )

# ExoColumn, cases 1, 4, 8, 9, 10, 11, 14, 15. Cloud-free, but its fixed surface
# albedo of 0.2736 stands in for the shortwave effect of clouds, so unlike
# HEXTOR its albedo lands inside the range spanned by the GCMs.
exocolumn       = np.array( [ 26.01, 15.51, 23.11, 19.68, 27.26, 23.95, 21.97, 25.53 ] )
exocolumn_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600 ] ) / fluxscale
exocolumn_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43 ] )

exocam_mask  = exocam  != runaway
rocke3d_mask = rocke3d != runaway
plahab_mask  = plahab  != runaway

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

# Each model's stable samples as ( instellation, pressure, albedo ), in panel
# order: by model class, ending with the two one-dimensional models.
MODELS = {
    'ExoPlaSim':   ( flux1,           pres1,           plasim         ),
    'ExoCAM':      ( exocam_flux1,    exocam_pres1,    exocam_stable  ),
    'ROCKE-3D':    ( rocke3d_flux1,   rocke3d_pres1,   rocke3d_stable ),
    'Generic PCM': ( pcm_flux1,       pcm_pres1,       pcm            ),
    'LFRic':       ( lfric_flux1,     lfric_pres1,     lfric          ),
    'PlaHab':      ( plahab_flux1,    plahab_pres1,    plahab_stable  ),
    'HEXTOR':      ( hextor_flux1,    hextor_pres1,    hextor         ),
    'ExoColumn':   ( exocolumn_flux1, exocolumn_pres1, exocolumn      ),
}

# With --common, every model is kriged from only the sample points at which all
# models reached a steady state, so the panels differ in the models and not in
# where each one was sampled. The set is computed rather than listed, so it
# follows the arrays above; it is Cases 1, 4, 8, 9, 10, 14 and 15, the same set
# as for surface temperature. The anisotropy ratios are left at the values
# fitted on each model's full set of cases.
#
# With --stacked, the full figure is drawn above the common-case one, each set
# under its own header and with its own colorbar, the common cases on a color
# range fitted to their samples.
COMMON  = '--common' in sys.argv
STACKED = '--stacked' in sys.argv
outname = 'fig_interpolation_albedo' + ( '_stacked' if STACKED else '_common' if COMMON else '' )

def _at( f, p, fs, ps ):
    return np.isclose( fs, f ) & np.isclose( ps, p )

def restrict_to_common( models ):
    common = np.array( [ all( _at( f, p, fs, ps ).any() for fs, ps, _ in models.values() )
                         for f, p in zip( flux1, pres1 ) ] )
    cases  = ( np.where( common )[ 0 ] + 1 ).tolist()
    print( 'cases stable in every model shown:', cases )
    restricted = {}
    for name, ( fs, ps, vals ) in models.items():
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

# A color range fitted to the samples shown, rounded out to 5% and ticked every
# 5%, for a block that has a colorbar of its own
def fitted_colors( models ):
    vals_shown = np.concatenate( [ vals for _, _, vals in models.values() ] )
    cmin = 5*np.floor( vals_shown.min()/5 )
    cmax = 5*np.ceil( vals_shown.max()/5 )
    return dict( cmin=cmin, cmax=cmax, cticks=np.arange( cmin, cmax + 1, 5 ) )

def zoomed_view( models ):
    fs_shown  = np.concatenate( [ fs for fs, _, _ in models.values() ] )
    ps_shown  = np.concatenate( [ ps for _, ps, _ in models.values() ] )
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
# for albedo: rerun fit_anisotropy.py after any resubmission.
# LFRic resolved no surface at any ratio on its first seven cases and was
# pinned at 1. The dark Case 7 (3.44%) made it resolvable from 1.5 upward, and
# with Cases 8, 10 and 11 (2026-09-10) the LOO minimum moved to 3.
# Variogram family. Linear everywhere, which is what cross-validation supports
# on the ensemble mean, with one exception. ExoCAM albedo is the only fit in
# this figure where linear fails outright: on the seven common cases it fits a
# pure nugget, so the lower panel collapsed to a constant 27.15% and had to be
# stippled, while the upper panel plainly resolved a surface -- the same model
# and variable reading two contradictory ways. The failure is not the
# anisotropy; swept from isotropic to a ratio of 50, linear never lifts those
# seven points above a resolution of 0.064 against the 0.25 floor. It is the
# family. Leave-one-out prefers spherical for ExoCAM in BOTH blocks
# independently (0.288 against 0.341 on the full set, 0.278 against 0.325 on
# the common set), and once the anisotropy is refitted under it the ratio is 15
# -- an interior optimum rather than one pinned at the edge of the search range
# -- giving LOO 0.258 on the full set, 24% better than the linear fit, and
# resolutions of 0.60 and 0.88 in the two blocks. So this is not a patch to get
# contours out of a bad fit; it is the family the data prefer, everywhere
# ExoCAM albedo is kriged.
VARIOGRAM = { 'ExoCAM': 'spherical' }
DEFAULT_VARIOGRAM = 'linear'

ANISO = {
    'ExoCAM':       15,
    'ROCKE-3D':     1.5,
    'ExoPlaSim':    1.5,
    'Generic PCM':  5,
    'PlaHab':       4,
    'LFRic':        3,
    'HEXTOR':       3,
    'ExoColumn':    5,
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

# A variogram whose structure term is zero is a pure nugget: ordinary kriging
# then weights every sample point equally regardless of distance, so the
# interpolated surface collapses to the sample mean and carries no spatial
# information. Those panels are stippled to distinguish that case from a
# genuinely flat but resolved field. variogram_model_parameters[0] is the
# structure term for both families in use here -- the slope for linear, the
# partial sill for spherical -- so the same test reads either.
structure_eps = 1.0e-8

# Albedo is bounded, so it is kriged on the logit and mapped back through the
# inverse logit; the fitted structure term is returned for the stippling test
def krige( name, fs, ps, vals, view ):
    OK = OrdinaryKriging(
        norm_pres( ps ),
        norm_flux( fs ),
        logit( vals ),
        anisotropy_scaling=ANISO[ name ],
        variogram_model=VARIOGRAM.get( name, DEFAULT_VARIOGRAM ),
        verbose=False,
        enable_plotting=False,
        exact_values=True,
    )
    z, var = OK.execute( "grid", norm_pres( view[ 'pres_grid' ] ), norm_flux( view[ 'flux_grid' ] ) )
    return z, var, OK.variogram_model_parameters[ 0 ]

marker_edge = 'k'

# Of the regions where σ exceeds the threshold, hatch only those reaching the
# highest instellation on the grid, as the temperature figure does. The rest is
# one grid cell in the cool, thin corner of the PlaHab panel, just past its
# outermost sample, where σ tops the threshold by 0.02 logit units and hatched it
# reads as a hole in the sampled region. The regions are 8-connected, so a
# dropped one never shares a grid cell with a kept one and zeroing it leaves the
# kept boundaries where they were.
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

# The label names the sample count because the degeneracy belongs to the fit in
# front of the reader, not to the model or the variable. ExoCAM is the case in
# point: its seven common cases resolve nothing, while the ten cases in the
# upper block resolve a surface plainly enough to see. Saying "no resolvable
# spatial structure" unqualified invited the two panels to be read as a
# contradiction.
def flag_degenerate( ax, structure, xv, yv, n ):
    """Stipple a panel whose variogram fit collapsed to a pure nugget."""
    if structure > structure_eps:
        return
    ax.contourf( yv*fluxscale, xv, np.ones_like( xv ), levels=[0.5, 1.5],
                 hatches=['....'], colors='none', alpha=0 )
    ax.text( 0.5, 0.06, f'these {n} cases resolve no spatial structure', transform=ax.transAxes,
             ha='center', va='bottom', fontsize=10, style='italic', color='0.15',
             bbox=dict( facecolor='white', edgecolor='none', alpha=0.75, pad=2.0 ) )

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

def draw_panel( ax, name, fs, ps, vals, view ):
    z, var, structure = krige( name, fs, ps, vals, view )
    fam = VARIOGRAM.get( name, DEFAULT_VARIOGRAM )
    print( f'  {name:<12} n={len(vals):<3} {fam} variogram, structure term {structure:.3g}' )
    xv, yv = np.meshgrid( view[ 'pres_grid' ], view[ 'flux_grid' ] )
    levels = np.linspace( view[ 'cmin' ], view[ 'cmax' ], cinterval )
    cf = ax.contourf( yv*fluxscale, xv, sigmoid(z), cmap=cm, levels=levels, vmin=view[ 'cmin' ], vmax=view[ 'cmax' ], extend='both' )
    if view[ 'hatch' ]:
        ax.contourf( yv*fluxscale, xv, warm_edge_sigma( np.sqrt(var) ), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
    ax.scatter( fs*fluxscale, ps, c=vals, cmap=cm, vmin=view[ 'cmin' ], vmax=view[ 'cmax' ], marker='o', s=70, edgecolors=marker_edge )
    if name == labeled_model:
        label_cases( ax, fs, ps )
    setup_panel( ax, f'{name} (n={len(vals)})', view )
    flag_degenerate( ax, structure, xv, yv, len( vals ) )
    return cf

def add_colorbar( fig, cf, rect, view ):
    cax = fig.add_axes( rect )
    cb = fig.colorbar( cf, cax=cax, extend='both', ticks=view[ 'cticks' ] )
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
    for ax, ( name, ( fs, ps, vals ) ) in zip( axs.flat, models.items() ):
        cf = draw_panel( ax, name, fs, ps, vals, view )
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
        for ax, ( name, ( fs, ps, vals ) ) in zip( axs.flat, models.items() ):
            cf = draw_panel( ax, name, fs, ps, vals, view )
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
