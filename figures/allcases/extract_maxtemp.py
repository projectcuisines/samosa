"""Extract the maximum time-mean surface temperature from the SAMOSA archive.

A global mean below 273.16 K does not mean the surface is frozen everywhere: a
synchronously rotating planet can hold an unfrozen substellar region while the
global mean is well below freezing. This script extracts, for every accepted
case, the maximum of the time-mean surface temperature field, which is the
diagnostic that separates a completely frozen surface from a partly frozen one.

Regenerates the arrays embedded in fig_summary.py. Run it after any model
resubmits, then paste the printed arrays into that script.

    cd figures/allcases && python extract_maxtemp.py

Per-model sources:

  ExoCAM       exocam/samosaN.cam.h0.avg.nc, TS, time axis length 1. Global
               mean weighted by the gw (gauss weights) variable, NOT cos(lat):
               this is a finite-volume grid with half-width polar cells.
  ExoPlaSim    exoplasim/samosaNN.nc, ts, time axis length 1, cos(lat).
  ROCKE-3D     rocke3d/rocke_NNq.nc, tsurf, already a time mean, in CELSIUS
               so 273.16 is added. Case 3 has no file.
  Generic PCM  genericpcm/OHT_off/case-N/SAMOSA_output_file_*.nc,
               surface_temperature.
  LFRic        lfric/lfric_samosa_caseNN.nc, grid_surface_temperature. The
               coordinate is lat, not latitude as in the Generic PCM files.
  HEXTOR       hextor/global_output_HEXTOR.dat. No spatial field is submitted:
               HEXTOR is a 1-D EBM on 18 belts in the tidally locked
               coordinate, so Tmax and Tmin are read straight from the
               Tmax/Tmin columns rather than reduced from a map. They are
               substellar and antistellar belt means, which is the closest
               available analogue to the GCM field extrema.
  ExoColumn    exocolumn/global_output_ExoColumn_a2736.dat. A single globally
               averaged column with no horizontal dimension, so Tmax = Tmin =
               Tglob by construction and the frozen / partly frozen / ice-free
               classification below can only ever return the two extremes for
               it. That is a real property of the model, not a gap in the
               submission, but it means its state counts are not comparable
               with the resolved models'.
  PlaHab       plahab/simulations/sampleN/model_samosa_plahab_*.out. Four
               36x20 maps are concatenated in one file; block 0 is tsurf,
               verified against the standalone caseN_tsurf.out for samples 1
               and 16. Column 0 is latitude, columns 1: are the 20 longitude
               bins. Sample 4 breaks the naming convention of the others. The latitude grid is non-uniform, so the global mean uses
               the dx weights listed in the LATITUDE GRID DATA header.

The Tglob validation is the check that matters: it confirms the weighting and
the case-number mapping against the global means already embedded in the
figure scripts. If a row fails, do not trust that model's maximum.
"""
import glob
import numpy as np
import netCDF4 as nc

ROOT = '/models/data/samosa'

ACCEPTED = {
    'ExoPlaSim':   list( range( 1, 17 ) ),
    'ExoCAM':      [ 1, 4, 8, 9, 10, 11, 12, 14, 15, 16 ],
    'ROCKE-3D':    [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],
    'PlaHab':      [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],
    'Generic PCM': [ 1, 4, 8, 9, 10, 14, 15 ],
    'LFRic':       [ 1, 4, 9, 12, 14, 15, 16 ],
    'HEXTOR':      [ 1, 4, 8, 9, 11, 14, 15 ],
    'ExoColumn':   [ 1, 4, 8, 9, 10, 11, 14, 15 ],
}

# Global means already in fig_summary.py, used only to validate the mapping
REF = {
 'ExoPlaSim': { 1:176.0, 2:368.2, 3:296.6, 4:254.0, 5:265.7, 6:343.1, 7:279.7, 8:215.9,
                9:239.9, 10:172.8, 11:211.3, 12:345.7, 13:272.9, 14:224.5, 15:186.3, 16:346.3 },
 'ExoCAM':    { 1:196.8, 4:260.0, 8:243.8, 9:244.8, 10:194.1, 11:234.0, 12:350.9,
                14:236.8, 15:211.5, 16:356.7 },
 'ROCKE-3D':  { 1:202.83, 4:260.12, 5:265.88, 7:267.73, 8:245.92, 9:241.83, 10:207.45,
                11:228.07, 12:313.99, 13:271.93, 14:236.30, 15:210.50, 16:319.25 },
 'PlaHab':    { 1:196.3, 4:273.2, 5:281.4, 7:293.0, 8:242.9, 9:260.8, 10:190.1,
                11:181.1, 12:295.3, 13:286.1, 14:246.1, 15:207.9, 16:292.7 },
 'Generic PCM': { 1:210.92, 4:286.73, 8:246.77, 9:266.60, 10:210.69, 14:246.04, 15:217.25 },
 'LFRic':     { 1:195.37, 4:251.48, 9:241.35, 12:333.20, 14:228.84, 15:203.64, 16:361.70 },
 'HEXTOR':    { 1:173.92, 4:312.24, 8:225.08, 9:277.17, 11:228.72, 14:242.30, 15:189.11 },
 'ExoColumn': { 1:206.98, 4:293.26, 8:248.49, 9:269.66, 10:201.36, 11:242.60, 14:251.63, 15:216.92 },
}

T_FREEZE = 273.16


def _mean_max( field, wlat ):
    """Area-weighted global mean, spatial minimum and maximum of a (lat, lon) field."""
    field = np.asarray( field, float )
    w     = np.broadcast_to( wlat[ :, None ], field.shape )
    good  = np.isfinite( field ) & ( field > 1.0 )
    gmean = ( field[ good ] * w[ good ] ).sum() / w[ good ].sum()
    return ( gmean, np.nanmax( field[ good ] ), np.nanmin( field[ good ] ) )


def read_exocam( case ):
    with nc.Dataset( f'{ROOT}/exocam/samosa{case}.cam.h0.avg.nc' ) as d:
        ts = np.asarray( d.variables[ 'TS' ] )[ 0 ]
        gw = np.asarray( d.variables[ 'gw' ] )
    return _mean_max( ts, gw )


def read_exoplasim( case ):
    with nc.Dataset( f'{ROOT}/exoplasim/samosa{case:02d}.nc' ) as d:
        ts  = np.asarray( d.variables[ 'ts' ] )[ 0 ]
        lat = np.asarray( d.variables[ 'lat' ] )
    return _mean_max( ts, np.cos( np.radians( lat ) ) )


def read_rocke3d( case ):
    with nc.Dataset( f'{ROOT}/rocke3d/rocke_{case:02d}q.nc' ) as d:
        ts  = np.asarray( d.variables[ 'tsurf' ] ) + T_FREEZE
        lat = np.asarray( d.variables[ 'lat' ] )
    ts = np.where( ts > 1e10, np.nan, ts )
    return _mean_max( ts, np.cos( np.radians( lat ) ) )


def read_pcm( case ):
    path = glob.glob( f'{ROOT}/genericpcm/OHT_off/case-{case}/SAMOSA_output_file_*.nc' )[ 0 ]
    with nc.Dataset( path ) as d:
        ts  = np.asarray( d.variables[ 'surface_temperature' ] )
        lat = np.asarray( d.variables[ 'latitude' ] )
    return _mean_max( ts, np.cos( np.radians( lat ) ) )


def read_lfric( case ):
    with nc.Dataset( f'{ROOT}/lfric/lfric_samosa_case{case:02d}.nc' ) as d:
        ts  = np.asarray( d.variables[ 'grid_surface_temperature' ] )
        lat = np.asarray( d.variables[ 'lat' ] )
    return _mean_max( ts, np.cos( np.radians( lat ) ) )


def read_plahab( case ):
    # sample4 is named model_plahab_seq1samp4.out, not model_samosa_plahab_*
    path = glob.glob( f'{ROOT}/plahab/simulations/sample{case}/model_*plahab*.out' )[ 0 ]
    lines = open( path ).readlines()

    # dx column of the LATITUDE GRID DATA header gives the equal-area weights
    dx = []
    for line in lines:
        f = line.split()
        if len( f ) == 5 and f[ 0 ].isdigit():
            dx.append( float( f[ 2 ] ) )

    rows = [ l for l in lines if len( l.split() ) == 21 ]
    blk  = np.array( [ [ float( x ) for x in r.split() ] for r in rows[ :36 ] ] )
    ts   = blk[ :, 1: ]
    w    = np.array( dx[ 1:37 ] ) if len( dx ) >= 37 else np.ones( 36 )
    return _mean_max( ts, w )


def read_exocolumn( case ):
    """ExoColumn is one column: Tglob, Tmax and Tmin are the same number."""
    for line in open( f'{ROOT}/exocolumn/global_output_ExoColumn_a2736.dat' ):
        if line.startswith( '#' ) or not line.strip():
            continue
        c = line.split()
        if int( c[ 0 ] ) == case:
            return ( float( c[ 3 ] ), float( c[ 4 ] ), float( c[ 5 ] ) )
    raise KeyError( f'case {case} not in the ExoColumn output file' )


def read_hextor( case ):
    """HEXTOR reports Tglob, Tmax and Tmin directly; there is no field to reduce."""
    for line in open( f'{ROOT}/hextor/global_output_HEXTOR.dat' ):
        if line.startswith( '#' ) or not line.strip():
            continue
        c = line.split()
        if int( c[ 0 ] ) == case:
            return ( float( c[ 3 ] ), float( c[ 4 ] ), float( c[ 5 ] ) )
    raise KeyError( f'case {case} not in the HEXTOR output file' )


READERS = { 'ExoPlaSim': read_exoplasim, 'ExoCAM': read_exocam, 'ROCKE-3D': read_rocke3d,
            'PlaHab': read_plahab, 'Generic PCM': read_pcm, 'LFRic': read_lfric,
            'HEXTOR': read_hextor, 'ExoColumn': read_exocolumn }

results = {}
print( f"{'model':<12} {'case':>4} {'Tglob':>8} {'ref':>8} {'d':>7} {'Tmin':>8} {'Tmax':>8}  state" )
for model, cases in ACCEPTED.items():
    results[ model ] = {}
    for case in cases:
        try:
            gmean, tmax, tmin = READERS[ model ]( case )
        except Exception as exc:
            print( f"{model:<12} {case:>4}   FAILED: {exc}" )
            continue
        results[ model ][ case ] = ( tmax, tmin )
        ref   = REF[ model ].get( case, np.nan )
        state = ( 'completely ice-free' if tmin >= T_FREEZE else
                  'completely frozen'   if tmax <  T_FREEZE else
                  'partly frozen' )
        flag  = '' if abs( gmean - ref ) < 1.5 else '   <-- MISMATCH'
        print( f"{model:<12} {case:>4} {gmean:8.2f} {ref:8.2f} {gmean-ref:7.2f} "
               f"{tmin:8.2f} {tmax:8.2f}  {state}{flag}" )

print( "\n\n# ── Maximum time-mean surface temperature (K) ──" )
print( "# Paste into fig_summary.py. Runaway/unsubmitted cases carry runawaytemp." )
order = [ ( 'ExoPlaSim', 'tmax_plasim' ), ( 'ExoCAM', 'tmax_exocam' ),
          ( 'ROCKE-3D', 'tmax_rocke3d' ), ( 'PlaHab', 'tmax_plahab' ) ]
for model, name in order:
    for idx, tag in ( ( 0, 'tmax' ), ( 1, 'tmin' ) ):
        vals = [ f"{results[model][c][idx]:.1f}" if c in results[ model ] else 'runawaytemp'
                 for c in range( 1, 17 ) ]
        print( f"{name.replace('tmax',tag):<14}= np.array( [ " + ", ".join( vals ) + " ] )" )
for model, name in [ ( 'Generic PCM', 'tmax_pcm' ), ( 'LFRic', 'tmax_lfric' ),
                     ( 'HEXTOR', 'tmax_hextor' ), ( 'ExoColumn', 'tmax_exocolumn' ) ]:
    for idx, tag in ( ( 0, 'tmax' ), ( 1, 'tmin' ) ):
        vals = [ f"{results[model][c][idx]:.1f}" for c in ACCEPTED[ model ] ]
        print( f"{name.replace('tmax',tag):<14}= np.array( [ " + ", ".join( vals ) + " ] )" )

print( "\n\n# ── State counts per model ──" )
print( f"{'model':<12} {'cases':>6} {'ice-free':>9} {'frozen':>8} {'partly':>7}" )
for model in ACCEPTED:
    tf = [ v for v in results[ model ].values() ]
    free = sum( 1 for tmax, tmin in tf if tmin >= T_FREEZE )
    froz = sum( 1 for tmax, tmin in tf if tmax <  T_FREEZE )
    print( f"{model:<12} {len(tf):>6} {free:>9} {froz:>8} {len(tf)-free-froz:>7}" )
