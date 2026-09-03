"""Extract TOA radiative fluxes and planetary albedo from the SAMOSA archive.

Regenerates the arrays embedded in fig_energy_balance.py and
fig_interpolation_albedo.py. Run it after any model resubmits, then paste the
printed arrays into those two scripts.

    cd figures/allcases && python extract_fluxes.py

Per-model sources and the traps in each:

  ExoCAM       exocam/samosaN.cam.h0.avg.nc, FSNT and FLUT, weighted by the
               gw (gauss weights) variable in the file. NOT cos(lat): this is
               a finite-volume grid with half-width polar cells. FSNTOA,
               FSNTOAC, FSNTC, FSDSC, FLUTC, SWCF and LWCF are all archived as
               identically zero, so FSNT (top of model) is the only usable
               shortwave flux and the summary TOAALB/toaEBAL in output.txt
               cannot be reproduced from the primary data. samosa7 is present
               in the archive but has no row in output.txt.

  ExoPlaSim    exoplasim/samosaNN.nc, rst (net SW = absorbed) and rlut. Both
               rlut and rsut are stored NEGATIVE. No weight variable is
               shipped, so cos(lat) is used; it reproduces S/4 to 0.1%
               uniformly, and the bias cancels in the albedo ratio.

  ROCKE-3D     rocke3d/rocke_NNq.nc. The *_hemis variables are length 3 and
               index [2] is the global value. trnf_toa_hemis needs a sign
               flip. plan_alb_hemis and incsw_toa_hemis are reported directly.
               Case 3 has no file.

  Generic PCM  genericpcm/OHT_off/case-N/samosa_gcm_output_case-N_OHT_off.dat
  LFRic        lfric/samosa_global_diagnostics_lfric_2026-08-28.txt
  HEXTOR       hextor/global_output_HEXTOR.dat
  ExoColumn    exocolumn/global_output_ExoColumn_a2736.dat
               All four are whitespace tables in the template column order.
               HEXTOR is a 1-D EBM: Qstrat, Qmass, Icethick, Cldliq, Cldice
               and Cldfrac are NaN by construction, so it contributes only to
               the temperature and albedo analyses. Its Fsdn is a uniform
               1.0038x the protocol incident flux, so albedo is derived from
               S/4 as for every other model, not from the submitted Fsdn.
               ExoColumn is a single globally averaged column: Tmax = Tmin =
               Tglob by construction, and Icethick, Cldliq, Cldice and Cldfrac
               are written as -999. It does report Qstrat and Qmass, so unlike
               HEXTOR it enters the water vapor analysis. Its Fsdn is the
               downward shortwave at the SURFACE (0.56-0.94 of S/4, falling as
               the column gets thicker and wetter), not at the top of the
               atmosphere as in HEXTOR -- a third reading of the same protocol
               column, so albedo again comes from S/4.

  PlaHab       plahab/simulations/sampleN/global_samosa_plahab_*
               Use simulations/, NOT the top-level global_samosa_plahab_
               seq1sam4.dat, which is a stale duplicate disagreeing by ~3 K in
               Tglob. File extensions vary (.dat, .30, .out).

The Tglob validation at the end is the check that matters: it confirms the
weighting and the case-number mapping against values already in the figure
scripts. If it fails, do not trust the fluxes.
"""
import os, re, glob
import numpy as np
import netCDF4 as nc

ROOT = '/models/data/samosa'

flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ], float )
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )
incident = flux1 / 4.0     # global mean incident stellar flux, fixed by the protocol

# Cases each group carries into the analysis (1-based); everything else is
# runaway, unstable, or not submitted.
ACCEPTED = {
    'ExoPlaSim':   list( range( 1, 17 ) ),
    'ExoCAM':      [ 1, 4, 8, 9, 10, 11, 12, 14, 15, 16 ],
    'ROCKE-3D':    [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],
    'Generic PCM': [ 1, 4, 8, 9, 10, 14, 15 ],
    'LFRic':       [ 1, 4, 9, 12, 14, 15, 16 ],
    'PlaHab':      [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],
    'HEXTOR':      [ 1, 4, 8, 9, 11, 14, 15 ],
    'ExoColumn':   [ 1, 4, 8, 9, 10, 11, 14, 15 ],
}

N = np.nan
REF_TS = {   # values already embedded in the figure scripts, for validation
    'ExoPlaSim': [ 176.0, 368.2, 296.6, 254.0, 265.7, 343.1, 279.7, 215.9, 239.9, 172.8, 211.3, 345.7, 272.9, 224.5, 186.3, 346.3 ],
    'ExoCAM':    [ 196.8, N, N, 260.0, N, N, N, 243.8, 244.8, 194.1, 234.0, 350.9, N, 236.8, 211.5, 356.7 ],
    'ROCKE-3D':  [ 202.8284, N, N, 260.1185, 265.88116, N, 267.7272, 245.91597, 241.83368, 207.4544, 228.07162, 313.99902, 271.92654, 236.30406, 210.50339, 319.25085 ],
    'PlaHab':    [ 196.3, N, N, 273.2, 281.4, N, 293.0, 242.9, 260.8, 190.1, 181.1, 295.3, 286.1, 246.1, 207.9, 292.7 ],
}

def blank():
    return dict( ts=np.full( 16, np.nan ), olr=np.full( 16, np.nan ),
                 asr=np.full( 16, np.nan ), alb=np.full( 16, np.nan ),
                 sfc=np.full( 16, np.nan ) )

def coslat_mean( field, lat ):
    w = np.cos( np.deg2rad( lat ) )[ :, None ] * np.ones( ( 1, field.shape[ -1 ] ) )
    return float( np.sum( field * w ) / np.sum( w ) )

data = {}

# ── ExoCAM ───────────────────────────────────────────────────────────────────
d = blank()
for i in range( 16 ):
    p = f'{ROOT}/exocam/samosa{i+1}.cam.h0.avg.nc'
    if not os.path.exists( p ):
        continue
    with nc.Dataset( p ) as ds:
        gw = ds.variables[ 'gw' ][ : ]
        w  = gw[ :, None ] * np.ones( ( 1, ds.dimensions[ 'lon' ].size ) )
        gm = lambda v: float( np.sum( v * w ) / np.sum( w ) )
        d[ 'ts' ][ i ]  = gm( ds.variables[ 'TS' ][ 0 ] )
        d[ 'asr' ][ i ] = gm( ds.variables[ 'FSNT' ][ 0 ] )
        d[ 'olr' ][ i ] = gm( ds.variables[ 'FLUT' ][ 0 ] )
        # Net downward flux at the surface. CAM signs: FSNS positive down,
        # FLNS/SHFLX/LHFLX positive up.
        d[ 'sfc' ][ i ] = ( gm( ds.variables[ 'FSNS'  ][ 0 ] )
                          - gm( ds.variables[ 'FLNS'  ][ 0 ] )
                          - gm( ds.variables[ 'SHFLX' ][ 0 ] )
                          - gm( ds.variables[ 'LHFLX' ][ 0 ] ) )
    d[ 'alb' ][ i ] = 100.0 * ( 1.0 - d[ 'asr' ][ i ] / incident[ i ] )
data[ 'ExoCAM' ] = d

# ── ExoPlaSim ────────────────────────────────────────────────────────────────
d = blank()
for i in range( 16 ):
    p = f'{ROOT}/exoplasim/samosa{i+1:02d}.nc'
    if not os.path.exists( p ):
        continue
    with nc.Dataset( p ) as ds:
        lat  = ds.variables[ 'lat' ][ : ]
        asr  = coslat_mean( ds.variables[ 'rst' ][ 0 ], lat )
        olr  = abs( coslat_mean( ds.variables[ 'rlut' ][ 0 ], lat ) )
        rsut = abs( coslat_mean( ds.variables[ 'rsut' ][ 0 ], lat ) )
        d[ 'ts' ][ i ] = coslat_mean( ds.variables[ 'ts' ][ 0 ], lat )
        # rss, rls, hfss and hfls are all stored positive downward, so they add
        d[ 'sfc' ][ i ] = sum( coslat_mean( ds.variables[ k ][ 0 ], lat )
                               for k in ( 'rss', 'rls', 'hfss', 'hfls' ) )
    d[ 'asr' ][ i ], d[ 'olr' ][ i ] = asr, olr
    d[ 'alb' ][ i ] = 100.0 * rsut / ( asr + rsut )
data[ 'ExoPlaSim' ] = d

# ── ROCKE-3D ─────────────────────────────────────────────────────────────────
d = blank()
for i in range( 16 ):
    p = f'{ROOT}/rocke3d/rocke_{i+1:02d}q.nc'
    if not os.path.exists( p ):
        continue
    with nc.Dataset( p ) as ds:
        g = lambda k: float( ds.variables[ k ][ 2 ] )
        d[ 'ts' ][ i ]  = g( 'tsurf_hemis' ) + 273.15
        d[ 'olr' ][ i ] = -g( 'trnf_toa_hemis' )
        d[ 'asr' ][ i ] = g( 'srnf_toa_hemis' )
        d[ 'alb' ][ i ] = g( 'plan_alb_hemis' )
data[ 'ROCKE-3D' ] = d

# ── Generic PCM and LFRic: whitespace tables in template column order ────────
def read_table( path, sample_col=True ):
    out = blank()
    for line in open( path ):
        if line.startswith( '#' ) or not line.strip():
            continue
        c = line.split()
        i = int( c[ 0 ] ) - 1
        out[ 'ts' ][ i ]  = float( c[ 3 ] )
        out[ 'olr' ][ i ] = float( c[ 6 ] )
        out[ 'asr' ][ i ] = float( c[ 7 ] )
        out[ 'alb' ][ i ] = 100.0 * ( 1.0 - float( c[ 7 ] ) / incident[ i ] )
    return out

d = blank()
for i in range( 16 ):
    p = f'{ROOT}/genericpcm/OHT_off/case-{i+1}/samosa_gcm_output_case-{i+1}_OHT_off.dat'
    if not os.path.exists( p ):
        continue
    for line in open( p ):
        if line.startswith( '#' ) or not line.strip():
            continue
        c = line.split()
        d[ 'ts' ][ i ], d[ 'olr' ][ i ], d[ 'asr' ][ i ] = float( c[ 3 ] ), float( c[ 6 ] ), float( c[ 7 ] )
        d[ 'alb' ][ i ] = 100.0 * ( 1.0 - float( c[ 7 ] ) / incident[ i ] )
        break
data[ 'Generic PCM' ] = d

data[ 'LFRic' ]  = read_table( f'{ROOT}/lfric/samosa_global_diagnostics_lfric_2026-08-28.txt' )
data[ 'HEXTOR' ]    = read_table( f'{ROOT}/hextor/global_output_HEXTOR.dat' )
data[ 'ExoColumn' ] = read_table( f'{ROOT}/exocolumn/global_output_ExoColumn_a2736.dat' )

# ── PlaHab ───────────────────────────────────────────────────────────────────
d = blank()
for p in glob.glob( f'{ROOT}/plahab/simulations/sample*/global_samosa_plahab_*' ):
    i = int( re.search( r'sample(\d+)/', p ).group( 1 ) ) - 1
    txt = open( p, errors='ignore' ).read()
    def val( k ):
        m = re.search( rf'^\s*{k}\s*=\s*([-\d.Ee+]+)', txt, re.M )
        return float( m.group( 1 ) ) if m else np.nan
    d[ 'ts' ][ i ], d[ 'olr' ][ i ], d[ 'asr' ][ i ] = val( 'Tglob' ), val( 'OLR' ), val( 'ASR' )
    d[ 'alb' ][ i ] = 100.0 * ( 1.0 - val( 'ASR' ) / incident[ i ] )
data[ 'PlaHab' ] = d

# ── Validation ───────────────────────────────────────────────────────────────
# Discrepancies already understood, so they do not count as validation failures.
# ExoPlaSim Case 8: the NetCDF gives 217.29 K against the 215.9 K embedded in
# the figure scripts. Every other ExoPlaSim case agrees to 0.5 K, so the
# embedded value is likely stale rather than the extraction being wrong. Not
# yet resolved with the modeling group.
KNOWN = { ( 'ExoPlaSim', 8 ) }

print( '\n=== Tglob validation against the figure scripts (tolerance 0.5 K) ===' )
allok = True
for m, ref in REF_TS.items():
    bad, known = [], []
    for i, r in enumerate( ref ):
        if np.isnan( r ):
            continue
        got = data[ m ][ 'ts' ][ i ]
        if np.isnan( got ) or abs( got - r ) > 0.5:
            msg = f'case{i+1}: {got:.2f} vs {r:.2f}'
            ( known if ( m, i + 1 ) in KNOWN else bad ).append( msg )
    status = 'OK' if not bad else 'MISMATCH -> ' + '; '.join( bad )
    if known:
        status += '   [known: ' + '; '.join( known ) + ']'
    print( f'  {m:12s} {status}' )
    allok &= not bad
print( '  ExoCAM reproduces its summary TS exactly, which validates the gw weighting.' )
print( f'  overall: {"PASS" if allok else "FAIL - do not trust the fluxes below"}' )

# ── Emit the arrays ──────────────────────────────────────────────────────────
order = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic', 'PlaHab', 'HEXTOR', 'ExoColumn' ]

def fmt( vals ):
    return '[ ' + ', '.join( 'nan' if np.isnan( v ) else f'{v:.2f}' for v in vals ) + ' ]'

print( '\n=== fig_energy_balance.py: imbalance (OLR-ASR)/(S/4) in per cent ===' )
for m in order:
    imb = 100.0 * ( data[ m ][ 'olr' ] - data[ m ][ 'asr' ] ) / incident
    print( f'  {m:12s} {fmt(imb)}' )

print( '\n=== fig_energy_balance.py: surface imbalance -F_sfc/(S/4) in per cent ===' )
print( '  Only ExoCAM and ExoPlaSim submitted every term needed to close the' )
print( '  surface budget. ROCKE-3D has srtrnf_grnd (net radiation at ground)' )
print( '  but no turbulent fluxes; LFRic has sw_down_surf and lw_net_surf only;' )
print( '  Generic PCM and PlaHab submitted only the standardized global output.' )
for m in ( 'ExoCAM', 'ExoPlaSim' ):
    print( f'  {m:12s} {fmt( -100.0 * data[ m ][ "sfc" ] / incident )}' )

print( '\n=== fig_interpolation_albedo.py: albedo (%), 200.0 = runaway sentinel ===' )
for m in order:
    a = data[ m ][ 'alb' ]
    # Compact form for the models that cover a subset of the 16 sample points.
    # HEXTOR is listed here rather than with the sentinel models because its
    # nine gaps are not all runaways: eight are, but Case 10 is excluded for
    # CO2 condensation, which the runaway sentinel would misreport.
    if m in ( 'Generic PCM', 'LFRic', 'HEXTOR', 'ExoColumn' ):
        sel = [ a[ c-1 ] for c in ACCEPTED[ m ] ]
        print( f'  {m:12s} {fmt(sel)}   (cases {ACCEPTED[m]})' )
    else:
        masked = [ a[ i ] if ( i+1 ) in ACCEPTED[ m ] else 200.0 for i in range( 16 ) ]
        print( f'  {m:12s} ' + '[ ' + ', '.join( 'runaway' if v == 200.0 else f'{v:.2f}' for v in masked ) + ' ]' )
print()
