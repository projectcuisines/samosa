import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from scipy.stats import qmc

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

#--------------------------------------------------------------------
# Sparse sample grid

flux = np.arange( 400, 2700, 100 )
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

seed1   = 5936744
seed2   = 397676
seq2sol = 1800

grid = np.array( [ [flux[i], pn2[j]] for i in range( len(flux) ) for j in range( len(pn2) ) ] )

sampler1 = qmc.Sobol( d=2, scramble=True, seed=seed1 )
sample1a = sampler1.random_base2( m=3 )
sample1  = np.floor( qmc.scale( sample1a, [0, 0], [len(flux), len(pn2)] ) ).astype( int )
flux1    = flux[ sample1[:,0] ]
pres1    = pn2[  sample1[:,1] ]

sampler2 = qmc.Sobol( d=2, scramble=True, seed=seed2 )
sample2a = sampler2.random_base2( m=3 )
sample2  = np.floor( qmc.scale( sample2a, [0, 0], [np.where( flux == seq2sol )[0][0], len(pn2)] ) ).astype( int )
flux2    = flux[ sample2[:,0] ]
pres2    = pn2[  sample2[:,1] ]

flux_all = np.concatenate( [flux1, flux2] )
pres_all = np.concatenate( [pres1, pres2] )

# Stable (completed) mask for each model — True = stable, False = runaway/unavailable
exocam_mask  = np.array( [True,  False, False, True,  False, False, False, True,
                           True,  True,  True,  True,  False, True,  True,  True ] )
rocke3d_mask = np.array( [True,  False, False, True,  True,  False, True,  True,
                           True,  True,  True,  True,  True,  True,  True,  True ] )
pcm_mask     = np.array( [True,  False, False, True,  False, False, False, True,
                           True,  True,  False, False, False, True,  True,  False] )
plahab_mask  = np.array( [True,  False, False, True,  True,  False, True,  True,
                           True,  True,  True,  True,  True,  True,  True,  True ] )
lfric_mask   = np.array( [True,  False, False, True,  False, False, True,  True,
                           True,  True,  True,  True,  False, True,  True,  True ] )
# HEXTOR: all seven gaps are runaways beyond its radiative lookup table. Case 10
# was a CO2-condensation exclusion until its CO2 was corrected to the protocol's
# 400 ubar partial pressure (2026-09-14), and Case 16 was treated as a runaway
# until the RH 0.8 resubmission (2026-09-16) brought it inside the table.
hextor_mask  = np.array( [True,  False, False, True,  False, False, False, True,
                           True,  True,  True,  False, False, True,  True,  True ] )
# ExoColumn: all eight gaps are incipient runaways with no steady state at that
# (S, p): HEXTOR's runaway set plus Case 16.
exocolumn_mask = np.array( [True,  False, False, True,  False, False, False, True,
                            True,  True,  True,  False, False, True,  True,  False] )
# UM: the nine cases Mei Ting Mak ran, HEXTOR's set; the rest were not run.
um_mask      = np.array( [True,  False, False, True,  False, False, False, True,
                           True,  True,  True,  False, False, True,  True,  True ] )
# CLERO, in both configurations: every case but the two the emulator's
# classifier gives better-than-even odds of a runaway (3 and 13)
clero_mask   = np.array( [True,  True,  False, True,  True,  True,  True,  True,
                           True,  True,  True,  True,  False, True,  True,  True ] )

color_stable  = '#183629'
color_unavail = '#183629'
color_grid    = '#aaaaaa'

# Completed cases take each model's color and marker from Figure 2
# (fig_energy_balance.py); runaway or unavailable cases stay a dark cross.
style = {
    'ExoPlaSim':   dict( color='#ff7f0e', marker='o' ),
    'ExoCAM':      dict( color='#1f77b4', marker='s' ),
    'ROCKE-3D':    dict( color='#2ca02c', marker='^' ),
    'Generic PCM': dict( color='#d62728', marker='D' ),
    'LFRic':       dict( color='#9467bd', marker='v' ),
    'PlaHab':      dict( color='#8c564b', marker='P' ),
    'HEXTOR':      dict( color='#17becf', marker='X' ),
    'ExoColumn':   dict( color='#7f7f7f', marker='*' ),
    'UM':          dict( color='#e377c2', marker='>' ),
    'CLERO-UM':     dict( color='#bcbd22', marker='h' ),
    'CLERO-ExoCAM': dict( color='#e7ba52', marker='p' ),
}

def plot_cases( ax, name, mask ):
    st = style[ name ]
    ax.scatter( flux_all[ mask ], pres_all[ mask ], marker=st[ 'marker' ], s=90 if st[ 'marker' ] == '*' else 45,
                facecolors=st[ 'color' ], edgecolors='k', linewidths=0.6, zorder=3, clip_on=False )
    ax.scatter( flux_all[ ~mask ], pres_all[ ~mask ], color=color_unavail, marker='x', s=40, zorder=3, clip_on=False )

# Three rows of four, ordered by model class: the resolved models, the two
# one-dimensional models, then the two configurations of the CLERO emulator.
MASKS = {
    'ExoPlaSim':    np.ones( len( flux_all ), dtype=bool ),
    'ExoCAM':       exocam_mask,
    'ROCKE-3D':     rocke3d_mask,
    'Generic PCM':  pcm_mask,
    'LFRic':        lfric_mask,
    'UM':           um_mask,
    'PlaHab':       plahab_mask,
    'HEXTOR':       hextor_mask,
    'ExoColumn':    exocolumn_mask,
    'CLERO-UM':     clero_mask,
    'CLERO-ExoCAM': clero_mask,
}
nrows = -( -len( MASKS ) // 4 )
fig, axs = plt.subplots( nrows, 4, figsize=(13.1, 3.45*nrows), squeeze=False )

xlim = [ max( flux ) + 50, min( flux ) - 50 ]
ylim = [ min( pn2 ) * 0.9, max( pn2 ) * 1.1 ]

def setup_panel( ax, title ):
    ax.set_title( title, fontsize=12 )
    ax.tick_params( axis='both', labelsize=10 )
    ax.set_yscale( 'log' )
    ax.set_xlim( xlim )
    ax.set_xticks( [ 2500, 2000, 1500, 1000, 500 ] )
    ax.set_ylim( ylim )
    ax.set_box_aspect( 1 )

for ax, ( name, mask ) in zip( axs.flat, MASKS.items() ):
    plot_cases( ax, name, mask )
    setup_panel( ax, f'{name} (n={mask.sum()})' )
for ax in axs.flat[ len( MASKS ): ]:
    ax.set_axis_off()

#--------------------------------------------------------------------
# ExoPlaSim (all 16 stable) also carries the full grid and the QMC point
# numbers, placed as in Figures 3-6: to the right of each marker, except where
# that would crowd a neighbor or run off the panel

axs[ 0, 0 ].scatter( grid[:,0], grid[:,1], s=1.5, color=color_grid, zorder=0 )
label_left = { 1, 8, 10, 13, 15 }
for case, ( f, p ) in enumerate( zip( flux_all, pres_all ), start=1 ):
    left = case in label_left
    axs[ 0, 0 ].annotate( str( case ), ( f, p ), xytext=( -5 if left else 5, 0 ), textcoords='offset points',
                          ha='right' if left else 'left', va='center', fontsize=9, color=color_stable )

#--------------------------------------------------------------------
# Finalize

fig.subplots_adjust( wspace=0.08, hspace=0.22 )

# Every panel shares one view, so the axes are labeled once: pressure to the
# left, instellation under the bottom row, tick labels along the outer edges
# A panel keeps its instellation tick labels when the slot below it is empty.
for ( r, c ), ax in np.ndenumerate( axs ):
    ax.apply_aspect()
    ax.tick_params( labelleft=( c == 0 ), labelbottom=( r == nrows - 1 or not axs[ r + 1, c ].axison ) )
top_left, bottom_right = axs[ 0, 0 ].get_position(), axs[ -1, -1 ].get_position()
fig.text( top_left.x0, ( top_left.y1 + bottom_right.y0 )/2, 'Surface pressure (bar)',
          rotation=90, ha='right', va='center', fontsize=12, fontweight='bold',
          transform=offset_copy( fig.transFigure, fig=fig, x=-38, units='points' ) )
fig.text( ( top_left.x0 + bottom_right.x1 )/2, bottom_right.y0, 'Instellation (W m$^{-2}$)',
          ha='center', va='top', fontsize=12, fontweight='bold',
          transform=offset_copy( fig.transFigure, fig=fig, y=-25, units='points' ) )

fig.savefig( "fig_tally.png", bbox_inches='tight' )
fig.savefig( "fig_tally.eps", bbox_inches='tight' )
#plt.show()
