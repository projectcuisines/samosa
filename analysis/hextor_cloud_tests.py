"""HEXTOR with and without its cloud correction, against the resolved models.

Reproduces the HEXTOR paragraph of Section 3.1: HEXTOR's departure from the
median of the six three-dimensional and two-dimensional models at each of its
stable Cases 1-16, as submitted (a constant +45.09 W/m2 added to the OLR of
every belt), without the correction, and with it scaled by S/900 (anchored at
the THAI Hab 1 instellation it was calibrated at). Everything else is the
submitted configuration. Also the greenhouse effect sigma*Ts^4 - OLR of
ExoColumn at Cases 1 and 10, from the archive, which the same paragraph sets
against HEXTOR's isothermal table columns.

    cd analysis && python hextor_cloud_tests.py

Runs HEXTOR's own tools/run_samosa.py three times (16 cases, warm start, a
minute or two each) into temporary directories; nothing in /models/hextor is
written. Needs the Intel runtime, sourced from /opt/intel/oneapi/setvars.sh.
"""
import csv, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
from _paths import FIG_ALL, ARCHIVE
from tables import table

HEXTOR   = '/models/hextor'
RESOLVED = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'PlaHab', 'Generic PCM', 'LFRic' ]
BASE     = ( '--sequence all16 --init warm --transport constant --d0-ref 2.46 --moistdiff --rhmoist 0.8 '
             '--table ./radiation/radiation_N2_CO2_3000K_p_rh0.8.h5 --workers 4' )
CONFIGS  = [ ( 'submitted',                    '--cloudir -45.09' ),
             ( 'no cloud correction',          '--cloudir 0' ),
             ( 'correction scaled with S/900', '--cloudir -45.09 --cloud-scaling instellation' ) ]

def run( args ):
    out = tempfile.mkdtemp( prefix='hextor_cloud_' )
    cmd = ( f'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cd {HEXTOR} && '
            f'python3 tools/run_samosa.py {BASE} {args} --outdir {out}' )
    subprocess.run( [ 'bash', '-c', cmd ], check=True, stdout=subprocess.DEVNULL )
    rows = list( csv.DictReader( open( os.path.join( out, 'samosa_summary.csv' ) ) ) )
    subprocess.run( [ 'rm', '-rf', out ] )
    return { int( r[ 'case' ] ): ( float( r[ 'T_global' ] ), r[ 'state' ] ) for r in rows }

T = table( FIG_ALL, 'fig_interpolation_temp.py', 'runawaytemp' )
med = { c: np.median( [ T[ m ][ c ] for m in RESOLVED if c in T[ m ] ] ) for c in range( 1, 17 ) }
cases = sorted( T[ 'HEXTOR' ] )             # the nine HEXTOR submitted as stable

print( 'HEXTOR minus the median of the six resolved models (K), at its stable Cases 1-16' )
print( f"{'':32s}" + ''.join( f'  C{c:<5d}' for c in cases ) )
for label, args in CONFIGS:
    res = run( args )
    cells = [ ( f'{res[ c ][ 0 ] - med[ c ]:+7.1f}' if res[ c ][ 1 ] == 'equilibrium' else '  runaw' ) for c in cases ]
    print( f'{label:32s}' + ''.join( f' {x}' for x in cells ) )
    if label == 'submitted':
        worst = max( abs( res[ c ][ 0 ] - T[ 'HEXTOR' ][ c ] ) for c in cases )
        print( f'{"  (vs the archived values)":32s} max |difference| {worst:.3f} K' )

SIGMA = 5.670374e-8
print( '\nExoColumn greenhouse effect sigma*Ts^4 - OLR, from the archive' )
for line in open( f'{ARCHIVE}/exocolumn/global_output_ExoColumn_a2736.dat' ):
    if line.startswith( '#' ) or not line.strip():
        continue
    f = line.split()
    if int( f[ 0 ] ) in ( 1, 10 ):
        ts, olr = float( f[ 3 ] ), float( f[ 6 ] )
        print( f'  Case {f[ 0 ]:>2s}: Ts {ts:.2f} K, OLR {olr:.2f} W/m2 -> G = {SIGMA*ts**4 - olr:.1f} W/m2' )
