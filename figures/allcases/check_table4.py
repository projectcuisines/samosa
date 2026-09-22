"""Check the protocol's case table against the Sobol draws that define it.

The protocol (Haqq-Misra et al. 2022) lists Cases 1-16 in its Table 1 and the
optional Cases 17-64 in its Table 4. This regenerates them the way the
protocol's own script did (samosa_extended_sample.py, archived with the
protocol paper): Sequences 1 and 1b are the first and second eight points of
one scrambled Sobol draw (seed 5936744, m = 4) on the ExoPlaSim grid;
Sequences 2 and 2b the same with seed 397676 over S <= 1800 W/m2; Sequence 3
a fresh 32-point draw (seed 1043337, m = 5) in (S, ln p), rounded as printed.
It compares them with the case table fig_interpolation_temp_1d.py uses.

    cd figures/allcases && python check_table4.py

Under scipy 1.17 every case matches except 39, which the draw puts at
2276 W/m2 against the 2279 printed in Table 4; the paper, and both models,
use the printed value.
"""
import runpy
import numpy as np
import scipy
from scipy.stats import qmc

flux = np.arange( 400, 2700, 100 )
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89, 1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )

def grid_draw( seed, nflux ):
    s = qmc.Sobol( d=2, scramble=True, seed=seed ).random_base2( m=4 )
    i = np.floor( qmc.scale( s, [ 0, 0 ], [ nflux, len( pn2 ) ] ) ).astype( int )
    return [ ( int( flux[ a ] ), float( pn2[ b ] ) ) for a, b in i ]

seq1 = grid_draw( 5936744, len( flux ) )                              # Cases 1-8, then 17-24
seq2 = grid_draw( 397676, int( np.where( flux == 1800 )[ 0 ][ 0 ] ) )  # Cases 9-16, then 25-32
s3   = qmc.scale( qmc.Sobol( d=2, scramble=True, seed=1043337 ).random_base2( m=5 ),
                  [ 400, np.log( 0.10 ) ], [ 2600, np.log( 10.0 ) ] )
seq3 = [ ( int( np.rint( f ) ), float( np.round( np.exp( lp ), 2 ) ) ) for f, lp in s3 ]   # Cases 33-64

drawn = {}
for k in range( 8 ):
    drawn[ 1 + k ], drawn[ 17 + k ] = seq1[ k ], seq1[ 8 + k ]
    drawn[ 9 + k ], drawn[ 25 + k ] = seq2[ k ], seq2[ 8 + k ]
for k in range( 32 ):
    drawn[ 33 + k ] = seq3[ k ]

CASES = runpy.run_path( 'fig_interpolation_temp_1d.py' )[ 'CASES' ]
print( f'scipy {scipy.__version__}' )
bad = [ c for c in range( 1, 65 )
        if abs( drawn[ c ][ 0 ] - CASES[ c ][ 0 ] ) > 0 or abs( drawn[ c ][ 1 ] - CASES[ c ][ 1 ] ) > 1e-9 ]
for c in bad:
    print( f'  Case {c}: table {CASES[ c ]}, draw {drawn[ c ]}' )
print( f'{64 - len( bad )} of 64 cases match the draw' )
