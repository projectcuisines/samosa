import sys
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from matplotlib.transforms import offset_copy
from matplotlib import patheffects
from pykrige.ok import OrdinaryKriging
from scipy import ndimage

# ─── Variable configuration ──────────────────────────────────────────────────
cm              = cmocean.cm.thermal
contourmin      = 175.0
contourmax      = 370.0
cinterval       = 40
sigma_threshold = 45.0      # K; hatch where kriging σ exceeds this
cbar_label      = 'Average Surface Temperature (K)'
cbar_ticks      = np.arange( 200, 370, 50 )
# ─────────────────────────────────────────────────────────────────────────────

runawaytemp = 600.0
fluxscale   = 100

flux = np.arange( 400, 2700, 100 ) / fluxscale
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

# QMC sequence 1 + sequence 2
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] ) / fluxscale
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Average Surface Temperature (K)
plasim  = np.array( [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9, 239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ] )
exocam  = np.array( [ 196.8, runawaytemp, runawaytemp, 260.0, runawaytemp, runawaytemp, runawaytemp, 243.8, 244.8, 194.1, 234.0, 350.9, runawaytemp, 236.8, 211.5, 356.7 ] )
rocke3d = np.array( [ 202.8284, runawaytemp, runawaytemp, 260.1185, 265.88116, runawaytemp, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ] )
plahab  = np.array( [ 196.3, runawaytemp, runawaytemp, 273.2, 281.4, runawaytemp, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ] )

exocam_mask  = exocam  != runawaytemp
rocke3d_mask = rocke3d != runawaytemp
plahab_mask  = plahab  != runawaytemp

exocam_flux1  = flux1[ exocam_mask ];  exocam_pres1  = pres1[ exocam_mask ];  exocam_stable  = exocam[ exocam_mask ]
rocke3d_flux1 = flux1[ rocke3d_mask ]; rocke3d_pres1 = pres1[ rocke3d_mask ]; rocke3d_stable = rocke3d[ rocke3d_mask ]
plahab_flux1  = flux1[ plahab_mask ];  plahab_pres1  = pres1[ plahab_mask ];  plahab_stable  = plahab[ plahab_mask ]

pcm = np.array( [ 210.9195445942203, 286.7294656230531, 246.76730657647218, 266.5987224285321, 210.69131033681012, 246.04296230476365, 217.2519558970929 ] )
pcm_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 600 ] ) / fluxscale
pcm_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 1.44, 0.43 ] )

lfric       = np.array( [ 195.37, 251.48, 400.52, 231.83, 241.35, 197.81, 227.52, 333.20, 228.84, 203.64, 361.70 ] )
lfric_flux1 = np.array( [ 500, 1200, 1600, 800, 1100, 400, 900, 1500, 900, 600, 1400 ] ) / fluxscale
lfric_pres1 = np.array( [ 0.70, 2.34, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 1.44, 0.43, 10.00 ] )

# HEXTOR, cases 1, 4, 8, 9, 10, 11, 14, 15, 16, from its RH 0.8 resubmission
# (2026-09-16). The other seven are runaways beyond the radiative lookup table.
# Case 16 was treated as a runaway at 465.18 K, above the table and ~100 K over
# every other model; it is now 376.07 K, inside the table and 14 K above the
# warmest other model, though its day side is still near the runaway plateau of
# the outgoing longwave and the archive README calls it less certain.
hextor       = np.array( [ 173.10, 292.22, 224.40, 267.86, 152.42, 225.52, 241.46, 188.62, 376.07 ] )
hextor_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600, 1400 ] ) / fluxscale
hextor_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43, 10.00 ] )

# ExoColumn, cases 1, 4, 8, 9, 10, 11, 14, 15. The other eight are incipient
# runaways: no steady state exists at that (S, p), so the RCE loop never closes.
exocolumn       = np.array( [ 206.98, 293.26, 248.49, 269.66, 201.36, 242.60, 251.63, 216.92 ] )
exocolumn_flux1 = np.array( [ 500, 1200, 800, 1100, 400, 900, 900, 600 ] ) / fluxscale
exocolumn_pres1 = np.array( [ 0.70, 2.34, 6.16, 0.70, 4.83, 0.10, 1.44, 0.43 ] )

# Each model's stable samples as ( instellation, pressure, temperature ), in
# panel order: by model class, ending with the two one-dimensional models.
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

# With --gcm, only the four full-complexity GCMs are drawn, dropping ExoPlaSim,
# PlaHab and the two one-dimensional models.
GCM = '--gcm' in sys.argv
if GCM:
    MODELS = { name: m for name, m in MODELS.items() if name in ( 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic' ) }

# With --common, every model is kriged from only the sample points at which all
# models shown reached a steady state, so the panels differ in the models and
# not in where each one was sampled. The set is computed rather than listed, so
# it follows the arrays above; for all eight models, and for the GCMs alone, it
# is Cases 1, 4, 8, 9, 10, 14 and 15. The anisotropy ratios are left at the
# values fitted on each model's full set of cases.
#
# With --stacked, the full figure is drawn above the common-case one, each set
# under its own header and with its own colorbar, the common cases on a color
# range fitted to their samples.
COMMON  = '--common' in sys.argv
STACKED = '--stacked' in sys.argv
outname = 'fig_interpolation_temp' + ( '_gcm' if GCM else '' ) + ( '_stacked' if STACKED else '_common' if COMMON else '' )

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

# A color range fitted to the samples shown, rounded out to 10 K and ticked
# every 25 K, for a block that has a colorbar of its own
def fitted_colors( models ):
    vals_shown = np.concatenate( [ vals for _, _, vals in models.values() ] )
    cmin = 10*np.floor( vals_shown.min()/10 )
    cmax = 10*np.ceil( vals_shown.max()/10 )
    return dict( cmin=cmin, cmax=cmax, cticks=np.arange( 25*np.ceil( cmin/25 ), cmax + 1, 25 ) )

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
# each block has its own colorbar and the common cases get a narrower color
# range; alone, they keep the full one so they compare directly with Figure 3.
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
# for temperature: rerun fit_anisotropy.py after any resubmission.
ANISO = {
    'ExoCAM':       10,
    'ROCKE-3D':     4,
    'ExoPlaSim':    2,
    'PlaHab':       3,
    'LFRic':        15,
    'Generic PCM':  5,
    'HEXTOR':       10,
    'ExoColumn':    7,
}

# Normalize both axes to [0, 1] for kriging so distance metric is balanced
log_pn2  = np.log( pn2 )
lpn2_min, lpn2_max = log_pn2.min(), log_pn2.max()
flux_min, flux_max = flux.min(), flux.max()

def norm_pres( p ):
    return ( np.log( p ) - lpn2_min ) / ( lpn2_max - lpn2_min )

def norm_flux( f ):
    return ( f - flux_min ) / ( flux_max - flux_min )

def krige( name, fs, ps, vals, view ):
    OK = OrdinaryKriging(
        norm_pres( ps ),
        norm_flux( fs ),
        vals,
        anisotropy_scaling=ANISO[ name ],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        exact_values=True,
    )
    return OK.execute( "grid", norm_pres( view[ 'pres_grid' ] ), norm_flux( view[ 'flux_grid' ] ) )

marker_edge = 'k'

# Of the regions where σ exceeds the threshold, hatch only those reaching the
# highest instellation on the grid. The rest are slivers along the other panel
# edges, just past the outermost samples, where σ tops the threshold by at most
# 12 K; they are largest in HEXTOR and LFRic, whose warmest cases sit beside much
# cooler ones and so steepen the variogram, and hatched they read as holes in
# the sampled region. The regions are 8-connected,
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

def draw_panel( ax, name, fs, ps, vals, view ):
    z, var = krige( name, fs, ps, vals, view )
    xv, yv = np.meshgrid( view[ 'pres_grid' ], view[ 'flux_grid' ] )
    levels = np.linspace( view[ 'cmin' ], view[ 'cmax' ], cinterval )
    cf = ax.contourf( yv*fluxscale, xv, z, cmap=cm, levels=levels, extend='both' )
    if view[ 'hatch' ]:
        ax.contourf( yv*fluxscale, xv, warm_edge_sigma( np.sqrt(var) ), levels=[sigma_threshold, 1e9], hatches=['///'], colors='none', alpha=0 )
    ax.scatter( fs*fluxscale, ps, c=vals, cmap=cm, vmin=view[ 'cmin' ], vmax=view[ 'cmax' ], marker='o', s=70, edgecolors=marker_edge )
    if name == labeled_model:
        label_cases( ax, fs, ps )
    setup_panel( ax, f'{name} (n={len(vals)})', view )
    return cf

def add_colorbar( fig, cf, rect, view ):
    cax = fig.add_axes( rect )
    cb = fig.colorbar( cf, cax=cax, extend='both', ticks=view[ 'cticks' ] )
    cb.ax.tick_params( labelsize=11 )
    cb.ax.get_yaxis().labelpad = 15
    cb.set_label( cbar_label, rotation=270, fontsize=12 )

#--------------------------------------------------------------------
# Panels: rows of four, with the colorbar alongside rather than occupying a
# panel slot. Each block is two rows for all eight models or one for the GCMs,
# so panels keep the same size.

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
