"""Extract the dynamical regime diagnostics from the SAMOSA archive.

Regenerates the arrays embedded in fig_regimes.py. Run it after any model
resubmits, then paste the printed arrays into that script.

    cd figures/allcases && python extract_regimes.py

Four numbers per model per case:

  lamR    the non-dimensional equatorial Rossby deformation radius, lambda_R/a
  LR      the non-dimensional Rhines length, L_R/a
  jet     'SJ' for a single equatorial jet, 'DJ' for two midlatitude jets
  ratio   the day-night to equator-pole surface temperature contrast ratio,
          (T_day - T_night) / (T_equator - T_pole), the ordinate of the right
          panel of Figure 6 of the same paper. T_day and T_night are the
          area-weighted hemispheric means either side of the substellar
          meridian; following the 2018 paper T_equator is the maximum surface
          temperature beneath the substellar region and T_pole the colder of
          the two polar row means. The substellar longitude is found from the
          meridional mean rather than assumed, because it differs by model.
          This diagnostic needs only a surface temperature field, so PlaHab
          joins the ensemble here even though it submits no wind.

following Haqq-Misra et al. (2018), ApJ 852, 67, but with the equations as
corrected by its two errata. The article's printed Equations (8) and (9) are
BOTH wrong and neither can be used as written:

  Erratum 1 (2020, ApJ 896, 133; doi:10.3847/1538-4357/ab9a4b) restores the
  Gill (1982) shallow-water form, lambda_R^2 = c/(2*beta) with c^2 = g*H.
  Erratum 2 (2020, ApJ 900, 96; doi:10.3847/1538-4357/abb19a) corrects beta in
  both the article and the first erratum to beta = 2*Omega/a.

So the definitions used here are

  lambda_R = sqrt( sqrt(g*H) / (2*beta) ),   H = R*Tbar_s / (m_air*g)
  L_R      = pi * sqrt( U / beta )
  beta     = 2*Omega/a

with m_air = 0.028 kg/mol, Omega set by the 15-day synchronous rotation the
protocol fixes for every case, and a the Earth radius. Both errata state that
the published figures were computed correctly; only the printed equations were
wrong. As a check, these give lambda_R/a = 1 near a 5-day rotation period at
Tbar_s = 250 K, which is the boundary the original text quotes.

Two deliberate departures from the 2018 paper, both forced by this ensemble:

  U is the area-weighted RMS surface wind over the WHOLE planet, not over the
  day hemisphere as in 2018. The substellar longitude differs between models in
  the submitted files (ROCKE-3D and ExoCAM place it at native lon=180, the rest
  at lon=0), so a dayside average needs a per-model rotation that a global
  average does not. It is not a cosmetic choice: ExoCAM is the only model whose
  dayside is calmer than its global mean, and switching moves its Case 1 from
  L_R/a = 1.02 to 0.59, across the regime boundary.

  The jet structure is classified at a FIXED sigma = 0.30, not at the
  tropopause. These atmospheres carry no ozone and so have no stratospheric
  inversion: the coldest level in the global mean profile is frequently the
  model lid itself, which makes a cold-point tropopause an artifact of where
  each group chose to stop. sigma = 0.30 is inside the domain of all five
  models. The classification is sensitive to this choice -- of the 66 model-case
  combinations in the archive, including the ones the paper rejects, only 30
  keep the same label across sigma = 0.5, 0.3, 0.15, the cold point and an
  upper-tropospheric mean -- so the level has to be stated wherever the figure
  is discussed, and the margin printed below says how close each call was.

Per-model sources and the traps in each:

  ExoCAM       exocam/samosaN.cam.h0.avg.nc, U and V on hybrid levels; pressure
               from hyam*P0 + hybm*PS. Surface temperature is weighted by the
               gw (gauss weights) variable, NOT cos(lat), as in
               extract_fluxes.py. The staggered wind grid ships no matching
               weight variable, so the wind RMS uses cos(lat); the difference is
               below 0.1% in the Tglob check. samosa7 is in the archive but is a
               runaway and has no accepted row.

  ExoPlaSim    exoplasim/samosaNN.nc, ua and va on sigma levels; ps is in hPa
               and is converted. Only 10 levels, so the sigma = 0.30 wind is
               interpolated in log sigma rather than taken from the nearest
               level -- with this few levels the nearest-level value can sit a
               third of a scale height away.

  ROCKE-3D     rocke3d/rocke_NNq.nc, ub and vb on the staggered lat2/lon2 B
               grid, plm in hPa running surface to TOA. Temperatures are in
               CELSIUS and need +273.16. The lat2 = -90 row of ub and vb is
               entirely masked, a B-grid artifact; the area mean is therefore
               NaN-aware, which costs nothing because cos(-90) = 0. tsurf is
               weighted by axyp, which reproduces the published Tglob to 0.01 K
               against 0.04 K for cos(lat). Cases 2, 3 and 6 have no accepted
               row; 2 and 6 still have files.

  LFRic        lfric/lfric_samosa_caseNN.nc. Winds are on 41 half_levels while
               pressure and temperature are on 42 full_levels, so pressure is
               interpolated column by column from height_wth onto height_w3,
               logarithmically in pressure. 8 of 16 cases.

  Generic PCM  genericpcm/OHT_off/case-N/SAMOSA_output_file_*.nc, u_wind_speed
               and v_wind_speed with atmospheric_pressure as a full 3-D field.
               Only 29 of the 40 archived levels actually carry data: levels
               5-7, 11-12, 20, 29-30 and 37-39 are entirely fill in every case,
               in the winds and the pressure alike. They are the same 29 levels
               in every case, so this is a thinned archive rather than damage,
               but the empty levels have to be dropped before anything is
               interpolated in sigma or the profile comes back as NaN. Only 7
               of 16 cases are accepted.

  PlaHab       plahab/simulations/sampleN/caseN_tsurf.out, a whitespace table
               whose first column is latitude and whose remaining 20 columns
               are the longitudes hard-coded in fig_compare_temp_select.py.
               13 of 16 cases. No wind is submitted, so PlaHab appears in the
               contrast ratio only. Its Case 1 night side sits on an exact
               150.000 K floor over 47% of the surface, so the ratio for that
               case is an upper bound, not a resolved value.

  HEXTOR and ExoColumn are 1-D and submit neither wind nor a two-dimensional
  surface temperature, so they cannot appear at all. HEXTOR resolves the
  tidally locked coordinate and so does have a day-night contrast, but it is
  symmetric about the substellar meridian by construction and has no
  equator-to-pole contrast to divide by.

The Tglob validation at the end is the check that matters: it confirms the
weighting and the case-number mapping against the values already embedded in
fig_interpolation_temp.py. If it fails, do not trust the regime numbers.
"""
import warnings
warnings.filterwarnings( 'ignore', category=DeprecationWarning )

import numpy as np
import netCDF4

A     = 6.371e6                    # planetary radius, m
G     = 9.81                       # m s^-2, as in Haqq-Misra et al. (2018)
RGAS  = 8.314                      # J mol^-1 K^-1
MAIR  = 0.028                      # kg mol^-1, as in Haqq-Misra et al. (2018)
OMEGA = 2.0 * np.pi / ( 15.0 * 86400.0 )   # 15-day synchronous rotation
BETA  = 2.0 * OMEGA / A                    # s^-1 m^-1, per Erratum 2
SIGMA_JET = 0.30                   # level at which the jet structure is judged

# Cases each model has an accepted row for, matching fig_interpolation_temp.py
ACCEPTED = {
    'ExoCAM':      [ 1, 4, 8, 9, 10, 11, 12, 14, 15, 16 ],
    'ExoPlaSim':   list( range( 1, 17 ) ),
    'ROCKE-3D':    [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],
    'LFRic':       [ 1, 4, 7, 9, 12, 14, 15, 16 ],
    'Generic PCM': [ 1, 4, 8, 9, 10, 14, 15 ],
    'PlaHab':      [ 1, 4, 16 ],
}

# Published global mean surface temperatures, from fig_interpolation_temp.py,
# indexed by case number. Used only to validate the case-number mapping.
TGLOB = {
    'ExoCAM':      { 1: 196.8, 4: 260.0, 8: 243.8, 9: 244.8, 10: 194.1, 11: 234.0,
                     12: 350.9, 14: 236.8, 15: 211.5, 16: 356.7 },
    'ExoPlaSim':   { 1: 176.0, 2: 368.2, 3: 296.6, 4: 254.0, 5: 265.7, 6: 343.1,
                     7: 279.7, 8: 215.9, 9: 239.9, 10: 172.8, 11: 211.3, 12: 345.7,
                     13: 272.9, 14: 224.5, 15: 186.3, 16: 346.3 },
    'ROCKE-3D':    { 1: 202.8284, 4: 260.1185, 5: 265.88116, 7: 267.7272,
                     8: 245.91597, 9: 241.83368, 10: 207.4544, 11: 228.07162,
                     12: 313.99902, 13: 271.92654, 14: 236.30406, 15: 210.50339,
                     16: 319.25085 },
    'LFRic':       { 1: 195.37, 4: 251.48, 7: 400.52, 9: 241.35, 12: 333.20,
                     14: 228.84, 15: 203.64, 16: 361.70 },
    'Generic PCM': { 1: 210.9195445942203, 4: 286.7294656230531,
                     8: 246.76730657647218, 9: 266.5987224285321,
                     10: 210.69131033681012, 14: 246.04296230476365,
                     15: 217.2519558970929 },
    'PlaHab':      { 1: 196.3, 4: 273.2, 5: 281.4, 7: 293.0, 8: 242.9, 9: 260.8,
                     10: 190.1, 11: 181.1, 12: 295.3, 13: 286.1, 14: 246.1,
                     15: 207.9, 16: 292.7 },
}

_d = '/models/data/samosa'

# The PlaHab tables carry latitude in column 0 and these 20 longitudes in the
# rest, as in fig_compare_temp_select.py.
LON_PLAHAB = np.array( [ -171., -153., -135., -117., -99., -81., -63., -45., -27., -9.,
                            9.,   27.,   45.,   63.,  81.,  99., 117., 135., 153., 171. ] )


def area_mean( x, w ):
    """Area-weighted mean over the last axis, ignoring masked-out points."""
    x = np.asarray( x, dtype=float )
    w = np.broadcast_to( np.asarray( w, dtype=float ), x.shape ).copy()
    w[ ~np.isfinite( x ) ] = 0.0
    return np.nansum( np.nan_to_num( x ) * w, axis=-1 ) / w.sum( axis=-1 )


def _get( ds, name ):
    """Read a variable as a plain float array with fill values as NaN."""
    return np.ma.filled( ds.variables[ name ][:], np.nan ).astype( float )


def read_exocam( case ):
    with netCDF4.Dataset( f'{_d}/exocam/samosa{case}.cam.h0.avg.nc' ) as ds:
        u   = np.squeeze( _get( ds, 'U'  ) )
        v   = np.squeeze( _get( ds, 'V'  ) )
        ps  = np.squeeze( _get( ds, 'PS' ) )
        ts  = np.squeeze( _get( ds, 'TS' ) )
        hyam, hybm = _get( ds, 'hyam' ), _get( ds, 'hybm' )
        p0  = float( ds.variables[ 'P0' ][:] )
        lat, lon, gw = _get( ds, 'lat' ), _get( ds, 'lon' ), _get( ds, 'gw' )
    p = hyam[ :, None, None ] * p0 + hybm[ :, None, None ] * ps[ None, :, : ]
    return dict( lat=lat, lon=lon, p=p, u=u, v=v, ts=ts, wts=gw,
                 surface_is_last=True )


def read_plasim( case ):
    with netCDF4.Dataset( f'{_d}/exoplasim/samosa{case:02d}.nc' ) as ds:
        u   = np.squeeze( _get( ds, 'ua' ) )
        v   = np.squeeze( _get( ds, 'va' ) )
        ps  = np.squeeze( _get( ds, 'ps' ) )
        ts  = np.squeeze( _get( ds, 'ts' ) )
        lev, lat, lon = _get( ds, 'lev' ), _get( ds, 'lat' ), _get( ds, 'lon' )
    if np.nanmax( ps ) < 1.0e4:            # hPa in the archive, Pa here
        ps = ps * 100.0
    p = lev[ :, None, None ] * ps[ None, :, : ]
    return dict( lat=lat, lon=lon, p=p, u=u, v=v, ts=ts, wts=None,
                 surface_is_last=True )


def read_rocke3d( case ):
    with netCDF4.Dataset( f'{_d}/rocke3d/rocke_{case:02d}q.nc' ) as ds:
        u, v = _get( ds, 'ub' ), _get( ds, 'vb' )
        ts   = _get( ds, 'tsurf' )
        plm  = _get( ds, 'plm' )
        lat2, lon2 = _get( ds, 'lat2' ), _get( ds, 'lon2' )
        lat,  lon  = _get( ds, 'lat'  ), _get( ds, 'lon'  )
        axyp = _get( ds, 'axyp' )
    p = np.broadcast_to( ( plm * 100.0 )[ :, None, None ], u.shape )
    # Winds are on the staggered B grid, tsurf on the primary grid
    return dict( lat=lat2, lon=lon2, p=p, u=u, v=v, ts=ts + 273.16, wts=axyp,
                 ts_lat=lat, ts_lon=lon, surface_is_last=False )


def read_lfric( case ):
    with netCDF4.Dataset( f'{_d}/lfric/lfric_samosa_case{case:02d}.nc' ) as ds:
        u, v = _get( ds, 'u_in_w3' ), _get( ds, 'v_in_w3' )
        pf   = _get( ds, 'pressure_in_wth' )
        hw, hf = _get( ds, 'height_w3' ), _get( ds, 'height_wth' )
        ts   = _get( ds, 'grid_surface_temperature' )
        lat, lon = _get( ds, 'lat' ), _get( ds, 'lon' )
    # Winds live on the 41 half levels, pressure on the 42 full levels
    p = np.empty_like( u )
    for j in range( u.shape[1] ):
        for i in range( u.shape[2] ):
            p[ :, j, i ] = np.interp( hw[ :, j, i ], hf[ :, j, i ],
                                      np.log( pf[ :, j, i ] ) )
    return dict( lat=lat, lon=lon, p=np.exp( p ), u=u, v=v, ts=ts, wts=None,
                 surface_is_last=False )


def read_pcm( case ):
    path = ( f'{_d}/genericpcm/OHT_off/case-{case}/'
             f'SAMOSA_output_file_Generic_PCM_case-{case}_OHT_off.nc' )
    with netCDF4.Dataset( path ) as ds:
        u, v = _get( ds, 'u_wind_speed' ), _get( ds, 'v_wind_speed' )
        p    = _get( ds, 'atmospheric_pressure' )
        ts   = _get( ds, 'surface_temperature' )
        lat, lon = _get( ds, 'latitude' ), _get( ds, 'longitude' )
    # Only 29 of the 40 archived levels carry data; the rest are entirely fill
    live = np.array( [ k for k in range( p.shape[0] )
                       if np.isfinite( p[ k ] ).any() and np.isfinite( u[ k ] ).any() ] )
    return dict( lat=lat, lon=lon, p=p[ live ], u=u[ live ], v=v[ live ], ts=ts,
                 wts=None, surface_is_last=False )


def read_plahab( case ):
    table = np.loadtxt( f'{_d}/plahab/simulations/sample{case}/case{case}_tsurf.out' )
    return dict( lat=table[ :, 0 ], lon=LON_PLAHAB, p=None, u=None, v=None,
                 ts=table[ :, 1: ], wts=None, surface_is_last=False )


READERS = { 'ExoCAM':      read_exocam,
            'ExoPlaSim':   read_plasim,
            'ROCKE-3D':    read_rocke3d,
            'LFRic':       read_lfric,
            'Generic PCM': read_pcm,
            'PlaHab':      read_plahab }


def surface_contrasts( ts, lat, lon ):
    """Day-night and equator-pole surface temperature contrasts.

    Which longitude is substellar is not assumed, because the archive is not
    consistent: ROCKE-3D and ExoCAM put it at native lon=180 and the rest at
    lon=0, so a hard-coded convention would silently invert the hemispheres for
    half the ensemble. It is not read off the temperature maximum either. The
    warmest meridian is displaced downwind of the substellar point by up to 30
    degrees in ROCKE-3D, and splitting the hemispheres there would rotate the
    terminator by that much and mix day into night. The protocol places the
    substellar point on a grid convention, so the temperature maximum is used
    only to choose between the two conventions and is then snapped; the
    displacement is returned separately as the hot spot offset, positive
    eastward.
    """
    cw   = np.cos( np.radians( lat ) )
    zonal_of_lon = area_mean( np.moveaxis( ts, 0, -1 ), cw )   # (lon,)
    lon_hot = lon[ int( np.nanargmax( zonal_of_lon ) ) ]

    lon_sub = 0.0 if abs( ( ( lon_hot + 180.0 ) % 360.0 ) - 180.0 ) <= 90.0 else 180.0
    hotspot = ( ( lon_hot - lon_sub + 180.0 ) % 360.0 ) - 180.0

    offset = np.abs( ( ( lon - lon_sub + 180.0 ) % 360.0 ) - 180.0 )
    day    = offset <= 90.0
    t_day   = area_mean( np.nanmean( ts[ :, day  ], axis=-1 ), cw )
    t_night = area_mean( np.nanmean( ts[ :, ~day ], axis=-1 ), cw )

    # Following the 2018 paper: the substellar maximum against the colder pole
    t_eq   = np.nanmax( ts )
    t_pole = min( np.nanmean( ts[ 0 ] ), np.nanmean( ts[ -1 ] ) )
    return t_day, t_night, t_eq, t_pole, hotspot


def diagnose( D ):
    lat = D[ 'lat' ]
    cw  = np.cos( np.radians( lat ) )

    # Surface temperature may live on its own grid (ROCKE-3D staggers the winds)
    ts_lat = D.get( 'ts_lat', lat )
    ts_lon = D.get( 'ts_lon', D[ 'lon' ] )
    ts_cw  = np.cos( np.radians( ts_lat ) )

    # Global mean surface temperature, on each model's own area weights
    if D[ 'wts' ] is not None and D[ 'wts' ].ndim == 2:       # ROCKE-3D axyp
        w = D[ 'wts' ]
        tglob = float( np.nansum( D[ 'ts' ] * w ) / np.sum( w ) )
    else:
        weights = ts_cw if D[ 'wts' ] is None else D[ 'wts' ]
        tglob = float( area_mean( np.nanmean( D[ 'ts' ], axis=-1 ), weights ) )

    t_day, t_night, t_eq, t_pole, hotspot = surface_contrasts( D[ 'ts' ], ts_lat, ts_lon )
    ratio = ( t_day - t_night ) / ( t_eq - t_pole )

    out = dict( tglob=tglob, t_day=t_day, t_night=t_night, t_eq=t_eq,
                t_pole=t_pole, ratio=ratio, hotspot=hotspot )

    if D[ 'u' ] is None:                      # PlaHab: surface contrasts only
        out.update( u_rms=np.nan, lam_r=np.nan, l_r=np.nan, jet='--',
                    margin=np.nan )
        return out

    ksfc = -1 if D[ 'surface_is_last' ] else 0

    # Rossby deformation radius from the scale height of the global mean state
    scale_height = RGAS * tglob / ( MAIR * G )
    lam_r = np.sqrt( np.sqrt( G * scale_height ) / ( 2.0 * BETA ) ) / A

    # Rhines length from the RMS surface wind
    speed2 = D[ 'u' ][ ksfc ] ** 2 + D[ 'v' ][ ksfc ] ** 2
    u_rms  = float( np.sqrt( area_mean( np.nanmean( speed2, axis=-1 ), cw ) ) )
    l_r    = np.pi * np.sqrt( u_rms / BETA ) / A

    # Zonal mean zonal wind interpolated in log sigma to the classification level
    ubar  = np.nanmean( D[ 'u' ], axis=-1 )
    sigma = area_mean( np.nanmean( D[ 'p' ], axis=-1 ), cw )
    sigma = sigma / sigma[ ksfc ]
    if not ( sigma.min() < SIGMA_JET < sigma.max() ):
        raise ValueError( f'sigma = {SIGMA_JET} outside the model domain' )
    order = np.argsort( np.log( sigma ) )
    u_jet = np.array( [ np.interp( np.log( SIGMA_JET ), np.log( sigma )[ order ],
                                   ubar[ order, j ] ) for j in range( len( lat ) ) ] )

    # Single equatorial jet if the equatorial wind beats the midlatitude maximum
    u_eq   = np.nanmean( u_jet[ np.abs( lat ) <= 10 ] )
    u_mid  = np.nanmax(  u_jet[ np.abs( lat ) >= 25 ] )
    margin = float( u_eq - u_mid )
    out.update( u_rms=u_rms, lam_r=lam_r, l_r=l_r,
                jet=( 'SJ' if margin >= 0.0 else 'DJ' ), margin=margin )
    return out


if __name__ == '__main__':
    results = {}
    print( f"{'model':12s} {'case':>4s} {'Tglob':>8s} {'pub':>8s} {'dT':>6s} "
           f"{'hotspt':>6s} {'Urms':>6s} {'lamR/a':>7s} {'LR/a':>6s} {'jet':>4s} "
           f"{'margin':>7s} {'dTdn':>6s} {'ratio':>6s}" )
    worst, worst_where = 0.0, ''
    for name, reader in READERS.items():
        rows = []
        for case in ACCEPTED[ name ]:
            r   = diagnose( reader( case ) )
            pub = TGLOB[ name ][ case ]
            if abs( r[ 'tglob' ] - pub ) > worst:
                worst, worst_where = abs( r[ 'tglob' ] - pub ), f'{name} case {case}'
            rows.append( ( case, r ) )
            print( f"{name:12s} {case:4d} {r['tglob']:8.2f} {pub:8.2f} "
                   f"{r['tglob'] - pub:+6.2f} {r['hotspot']:+6.0f} {r['u_rms']:6.2f} "
                   f"{r['lam_r']:7.3f} {r['l_r']:6.3f} {r['jet']:>4s} "
                   f"{r['margin']:+7.1f} {r['t_day'] - r['t_night']:6.1f} "
                   f"{r['ratio']:6.3f}" )
        results[ name ] = rows

    print( f'\nlargest Tglob discrepancy: {worst:.2f} K at {worst_where} '
           f'({"PASS" if worst < 1.5 else "FAIL -- do not trust the regime numbers"})' )

    print( '\n=== paste into fig_regimes.py ===\n' )
    key = { 'ExoCAM': 'exocam', 'ExoPlaSim': 'plasim', 'ROCKE-3D': 'rocke3d',
            'LFRic': 'lfric', 'Generic PCM': 'pcm', 'PlaHab': 'plahab' }
    fmt = lambda seq, f: '[ ' + ', '.join( f % x for x in seq ) + ' ]'
    for name, rows in results.items():
        k = key[ name ]
        print( f"# {name}" )
        print( f"{k}_case  = np.array( {fmt([ c for c, _ in rows ], '%d')} )" )
        if name != 'PlaHab':
            print( f"{k}_lamr  = np.array( {fmt([ r['lam_r'] for _, r in rows ], '%.3f')} )" )
            print( f"{k}_lr    = np.array( {fmt([ r['l_r'  ] for _, r in rows ], '%.3f')} )" )
            print( f"{k}_jet   = {[ r['jet'] for _, r in rows ]}" )
        print( f"{k}_ratio = np.array( {fmt([ r['ratio'] for _, r in rows ], '%.3f')} )" )
        print()
