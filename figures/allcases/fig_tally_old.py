import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import qmc


flux = np.arange( 400, 2700, 100 )
pn2  = [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ]

seed1 = 5936744
seed2 = 397676

seq2sol = 1800
#seq2pres = 0.34

grid = np.zeros( [ len( flux )*len( pn2 ), 2 ] )
for i in range( 0, len( flux ) ):
	for j in range( 0, len( pn2 ) ):
		grid[ j+i*len(pn2), 0 ] = flux[i]
		grid[ j+i*len(pn2), 1] = pn2[j]

sampler1 = qmc.Sobol( d=2, scramble=True, seed=seed1 )
lbound1  = [ 0, 0 ]
ubound1  = [ len( flux ), len( pn2 ) ]
sample1a = sampler1.random_base2( m=3 )
sample1  = np.floor( qmc.scale( sample1a, lbound1, ubound1 ) ).astype( int )
disc1    = qmc.discrepancy( sample1a )
print( disc1 )
flux1 = np.zeros( len( sample1 ) )
pres1 = np.zeros( len( sample1  ) )
for i in range( 0, len( sample1 ) ):
	flux1[i] = flux[ sample1[i,0] ]
	pres1[i] = pn2[ sample1[i,1] ]
print( flux1 )
print( pres1 )

sampler2 = qmc.Sobol( d=2, scramble=True, seed=seed2 )
#lbound2  = [ 0, next(i for i, _ in enumerate( pn2 ) if np.isclose(_, seq2pres, 0.01)) ]
lbound2  = [ 0, 0 ]
ubound2  = [ np.where( flux==seq2sol )[0][0], len( pn2 ) ]
sample2a = sampler2.random_base2( m=3 )
sample2  = np.floor( qmc.scale( sample2a, lbound2, ubound2 ) ).astype( int )
disc2    = qmc.discrepancy( sample2a )
print( disc2 )
flux2 = np.zeros( len( sample2 ) )
pres2 = np.zeros( len( sample2  ) )
for i in range( 0, len( sample2 ) ):
	flux2[i] = flux[ sample2[i,0] ]
	pres2[i] = pn2[ sample2[i,1] ]
print( flux2 )
print( pres2 )

numplots = 1
fig, (ax) = plt.subplots( 1, numplots, sharex=True, figsize = (6.0,4.75) )
plt.rc('axes', titlesize=15) 

color1 = '#183629'
color2 = '#208eb7'
color3 = '#3ea275'
color0 = '#aaaaaa'

ax.scatter( grid[:,0], grid[:,1], s=4, color=color0 )
#ax.scatter( flux1, pres1, color=color1 )
ax.tick_params( axis='x', labelsize=12 )
ax.tick_params( axis='y', labelsize=12 )
ax.set( title='SAMOSA Cases' )
ax.set_xlabel( 'Instellation (W m$^2$)', fontsize = 12 )
ax.set_ylabel( 'Surface pressure (atm)', fontsize = 12 )
ax.set_yscale('log')
ax.set_xlim( [ max( flux ) + 50, min( flux ) - 50 ] )
ax.set_ylim( [ min( pn2 )*0.9,  max( pn2 )*1.1 ] )

ax.scatter( flux1[0], pres1[0], color=color1 )
ax.scatter( flux1[1], pres1[1], color=color1 )
ax.scatter( flux1[2], pres1[2], color=color1 )
ax.scatter( flux1[3], pres1[3], color=color2 )
ax.scatter( flux1[4], pres1[4], color=color1 )
ax.scatter( flux1[5], pres1[5], color=color1 )
ax.scatter( flux1[6], pres1[6], color=color1 )
ax.scatter( flux1[7], pres1[7], color=color1 )
ax.text( flux1[0]+80, pres1[0], 1, fontsize=10, color=color1 )
ax.text( flux1[1]+80, pres1[1], 2, fontsize=10, color=color1 )
ax.text( flux1[2]+80, pres1[2], 3, fontsize=10, color=color1 )
ax.text( flux1[3]+80, pres1[3], 4, fontsize=10, color=color2 )
ax.text( flux1[4]+80, pres1[4], 5, fontsize=10, color=color1 )
ax.text( flux1[5]+80, pres1[5], 6, fontsize=10, color=color1 )
ax.text( flux1[6]+80, pres1[6], 7, fontsize=10, color=color1 )
ax.text( flux1[7]+80, pres1[7], 8, fontsize=10, color=color1 )

ax.scatter( flux2[0], pres2[0], color=color1 )
ax.scatter( flux2[1], pres2[1], color=color1 )
ax.scatter( flux2[2], pres2[2], color=color1 )
ax.scatter( flux2[3], pres2[3], color=color1 )
ax.scatter( flux2[4], pres2[4], color=color1 )
ax.scatter( flux2[5], pres2[5], color=color1 )
ax.scatter( flux2[6], pres2[6], color=color1 )
ax.scatter( flux2[7], pres2[7], color=color1 )
ax.text( flux2[0]+80,  pres2[0], 9, fontsize=10, color=color1 )
ax.text( flux2[1]+140, pres2[1], 10, fontsize=10, color=color1 )
ax.text( flux2[2]+140, pres2[2]*1.03, 11, fontsize=10, color=color1 )
ax.text( flux2[3]+140, pres2[3], 12, fontsize=10, color=color1 )
ax.text( flux2[4]+140, pres2[4], 13, fontsize=10, color=color1 )
ax.text( flux2[5]+140, pres2[5], 14, fontsize=10, color=color1 )
ax.text( flux2[6]+140, pres2[6], 15, fontsize=10, color=color1 )
ax.text( flux2[7]+140, pres2[7]*0.81, 16, fontsize=10, color=color1 )



fig.savefig( "fig_tally.png", bbox_inches='tight' )
fig.savefig( "fig_tally.eps", bbox_inches='tight' )
#plt.show()
