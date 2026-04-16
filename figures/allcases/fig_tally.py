import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

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
lfric_mask   = np.array( [True,  False, False, True,  False, False, False, False,
                           True,  False, False, True,  False, True,  True,  False] )

color_stable  = '#183629'
color_unavail = '#183629'
color_grid    = '#aaaaaa'

fig, axd = plt.subplot_mosaic( [[ 'P1', 'P2', 'P3' ],
                                  [ 'P4', 'P5', 'P6' ]],
                                figsize=(18, 9) )

xlim = [ max( flux ) + 50, min( flux ) - 50 ]
ylim = [ min( pn2 ) * 0.9, max( pn2 ) * 1.1 ]

def setup_panel( ax, title ):
    ax.set_title( title, fontsize=14 )
    ax.set_xlabel( 'Instellation (W m$^{-2}$)', fontsize=12 )
    ax.set_ylabel( 'Surface pressure (bar)', fontsize=12 )
    ax.tick_params( axis='x', labelsize=11 )
    ax.tick_params( axis='y', labelsize=11 )
    ax.set_yscale( 'log' )
    ax.set_xlim( xlim )
    ax.set_ylim( ylim )

#--------------------------------------------------------------------
# Panel 1 — ExoPlaSim (all 16 stable; labels identify QMC point numbers)

axd[ 'P1' ].scatter( grid[:,0], grid[:,1], s=4, color=color_grid, zorder=0 )
axd[ 'P1' ].scatter( flux_all, pres_all, color=color_stable, marker='o', s=50 )

axd[ 'P1' ].text( flux1[0]+80, pres1[0],       1,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux1[1]+80, pres1[1],       2,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux1[2]+80, pres1[2],       3,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux1[3]+80, pres1[3],       4,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux1[4]+80, pres1[4],       5,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux1[5]+80, pres1[5],       6,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux1[6]+80, pres1[6],       7,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux1[7]+80, pres1[7],       8,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[0]+80,  pres2[0],      9,  fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[1]+140, pres2[1],      10, fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[2]+140, pres2[2]*1.03, 11, fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[3]+140, pres2[3],      12, fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[4]+140, pres2[4],      13, fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[5]+140, pres2[5],      14, fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[6]+140, pres2[6],      15, fontsize=10, color=color_stable )
axd[ 'P1' ].text( flux2[7]+140, pres2[7]*0.81, 16, fontsize=10, color=color_stable )

setup_panel( axd[ 'P1' ], f'ExoPlaSim (n=16)' )

#--------------------------------------------------------------------
# Panel 2 — ExoCAM

axd[ 'P2' ].scatter( flux_all[  exocam_mask ], pres_all[  exocam_mask ], color=color_stable,  marker='o', s=50 )
axd[ 'P2' ].scatter( flux_all[ ~exocam_mask ], pres_all[ ~exocam_mask ], color=color_unavail, marker='x', s=50 )
setup_panel( axd[ 'P2' ], f'ExoCAM (n={exocam_mask.sum()})' )

#--------------------------------------------------------------------
# Panel 3 — ROCKE-3D

axd[ 'P3' ].scatter( flux_all[  rocke3d_mask ], pres_all[  rocke3d_mask ], color=color_stable,  marker='o', s=50 )
axd[ 'P3' ].scatter( flux_all[ ~rocke3d_mask ], pres_all[ ~rocke3d_mask ], color=color_unavail, marker='x', s=50 )
setup_panel( axd[ 'P3' ], f'ROCKE-3D (n={rocke3d_mask.sum()})' )

#--------------------------------------------------------------------
# Panel 4 — Generic PCM (without OHT)

axd[ 'P4' ].scatter( flux_all[  pcm_mask ], pres_all[  pcm_mask ], color=color_stable,  marker='o', s=50 )
axd[ 'P4' ].scatter( flux_all[ ~pcm_mask ], pres_all[ ~pcm_mask ], color=color_unavail, marker='x', s=50 )
setup_panel( axd[ 'P4' ], f'Generic PCM without OHT (n={pcm_mask.sum()})' )

#--------------------------------------------------------------------
# Panel 5 — LFRic

axd[ 'P5' ].scatter( flux_all[  lfric_mask ], pres_all[  lfric_mask ], color=color_stable,  marker='o', s=50 )
axd[ 'P5' ].scatter( flux_all[ ~lfric_mask ], pres_all[ ~lfric_mask ], color=color_unavail, marker='x', s=50 )
setup_panel( axd[ 'P5' ], f'LFRic (n={lfric_mask.sum()})' )

#--------------------------------------------------------------------
# Panel 6 — PlaHab

axd[ 'P6' ].scatter( flux_all[  plahab_mask ], pres_all[  plahab_mask ], color=color_stable,  marker='o', s=50 )
axd[ 'P6' ].scatter( flux_all[ ~plahab_mask ], pres_all[ ~plahab_mask ], color=color_unavail, marker='x', s=50 )
setup_panel( axd[ 'P6' ], f'PlaHab (n={plahab_mask.sum()})' )

#--------------------------------------------------------------------
# Finalize

fig.subplots_adjust( wspace=0.3, hspace=0.4 )

fig.savefig( "fig_tally.png", bbox_inches='tight' )
fig.savefig( "fig_tally.eps", bbox_inches='tight' )
#plt.show()
