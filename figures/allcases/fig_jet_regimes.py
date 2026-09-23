#
# Jet regime diagnostics, after the top two panels of Figure 1 of
# Sergeev et al. (2022), applied to the SAMOSA ensemble.
#
# Panel (a)  latitude of the tropospheric jet against the maximum equatorial
#            zonal wind. Their abscissa is the wind at 300 hPa; surface
#            pressure spans two orders of magnitude here, so the counterpart of
#            their fixed pressure is the fixed sigma = 0.30 used throughout.
# Panel (b)  minimum surface temperature against the ratio of the day-night to
#            equator-pole temperature difference.
#
# Arrays are produced by extract_regimes.py; rerun it after any resubmission.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

# ─── Regime diagnostics, from extract_regimes.py ───────────────────────────
# ExoCAM
exocam_case  = np.array( [ 1, 4, 8, 9, 10, 11, 12, 14, 15, 16 ] )
exocam_jet   = ['SJ', 'DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ', 'DJ']
exocam_jetlat= np.array( [ 2.0, 42.0, 66.0, 2.0, 70.0, 2.0, 66.0, 34.0, 2.0, 58.0 ] )
exocam_umax  = np.array( [ 24.8, 17.9, 11.4, 21.3, 13.3, 27.9, 3.4, 22.7, 28.9, 17.7 ] )
exocam_tsmin = np.array( [ 147.4, 211.1, 218.2, 200.4, 166.5, 195.2, 348.2, 183.2, 157.9, 353.8 ] )
exocam_ratio = np.array( [ 0.572, 0.507, 0.463, 0.590, 0.461, 0.638, 0.424, 0.553, 0.531, 0.374 ] )

# ExoPlaSim
plasim_case  = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
plasim_jet   = ['DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ']
plasim_jetlat= np.array( [ 58.1, 30.5, 2.8, 52.6, 2.8, 47.1, 2.8, 63.7, 2.8, 69.2, 2.8, 58.1, 2.8, 52.6, 41.5, 2.8 ] )
plasim_umax  = np.array( [ 9.1, 8.8, 27.8, 16.9, 24.0, 16.2, 22.4, 6.6, 23.7, 4.8, 21.2, 13.5, 23.8, 16.5, 20.0, 5.4 ] )
plasim_tsmin = np.array( [ 124.5, 350.8, 279.3, 201.2, 222.0, 334.1, 249.1, 159.6, 182.5, 132.7, 136.7, 336.7, 234.0, 158.4, 124.0, 330.9 ] )
plasim_ratio = np.array( [ 0.655, 0.199, 0.512, 0.609, 0.443, 0.460, 0.545, 0.454, 0.665, 0.685, 0.632, 0.408, 0.418, 0.541, 0.651, 0.283 ] )

# ROCKE-3D
rocke3d_case  = np.array( [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
rocke3d_jet   = ['SJ', 'DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'DJ', 'SJ', 'DJ']
rocke3d_jetlat= np.array( [ 0.0, 40.0, 20.0, 4.0, 68.0, 4.0, 68.0, 4.0, 64.0, 24.0, 40.0, 4.0, 68.0 ] )
rocke3d_umax  = np.array( [ 22.0, 20.1, 22.3, 13.5, 10.4, 20.6, 9.5, 29.9, 46.8, 23.8, 24.5, 31.2, 53.2 ] )
rocke3d_tsmin = np.array( [ 143.8, 208.5, 239.8, 246.2, 221.3, 202.7, 175.8, 187.7, 272.3, 243.4, 187.8, 155.2, 293.5 ] )
rocke3d_ratio = np.array( [ 0.454, 0.497, 0.546, 0.408, 0.525, 0.529, 0.477, 0.573, 0.198, 0.458, 0.519, 0.464, 0.246 ] )

# LFRic
lfric_case  = np.array( [ 1, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16 ] )
lfric_jet   = ['SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ', 'SJ', 'DJ', 'SJ', 'SJ', 'DJ']
lfric_jetlat= np.array( [ 1.0, 43.0, 1.0, 39.0, 1.0, 1.0, 5.0, 61.0, 1.0, 1.0, 69.0 ] )
lfric_umax  = np.array( [ 25.2, 28.2, 21.4, 21.9, 23.9, 18.0, 48.7, 8.8, 26.5, 28.4, -9.4 ] )
lfric_tsmin = np.array( [ 137.2, 204.5, 389.8, 199.9, 180.4, 165.8, 168.0, 330.4, 170.1, 143.9, 356.1 ] )
lfric_ratio = np.array( [ 0.704, 0.522, 0.441, 0.726, 0.614, 0.817, 0.717, 0.812, 0.645, 0.705, 0.430 ] )

# Generic PCM
pcm_case  = np.array( [ 1, 4, 8, 9, 10, 14, 15 ] )
pcm_jet   = ['SJ', 'DJ', 'DJ', 'DJ', 'DJ', 'DJ', 'SJ']
pcm_jetlat= np.array( [ 3.9, 58.7, 62.6, 50.9, 62.6, 43.0, 0.0 ] )
pcm_umax  = np.array( [ 25.0, 30.0, 9.9, 30.8, 7.3, 22.8, 29.1 ] )
pcm_tsmin = np.array( [ 170.0, 257.0, 218.3, 217.2, 187.0, 189.0, 170.2 ] )
pcm_ratio = np.array( [ 0.520, 0.430, 0.492, 0.438, 0.355, 0.502, 0.548 ] )

# PlaHab
plahab_case  = np.array( [ 1, 4, 16 ] )
plahab_tsmin = np.array( [ 150.0, 247.9, 286.8 ] )
plahab_ratio = np.array( [ 0.615, 0.709, 0.557 ] )

# PlaHab is not shown. It submits no wind, so it has no abscissa in panel (a)
# and no regime label in either, and its minimum surface temperature is not
# usable at Case 1, where the model sits on an exact 150.000 K floor over 47%
# of the surface. Its three contrast ratios are 0.615, 0.709 and 0.557 at Cases
# 1, 4 and 16, against minima of 150.0 (floored), 247.9 and 286.8 K.

# ─── Model style: colors and markers as in Figure 2 (fig_energy_balance.py) and
# Figure 15, with the models in that order ─────────────────────────────────
style  = { 'ExoPlaSim':   '#ff7f0e',
           'ExoCAM':      '#1f77b4',
           'ROCKE-3D':    '#2ca02c',
           'Generic PCM': '#d62728',
           'LFRic':       '#9467bd' }
marker = { 'ExoPlaSim':   'o',
           'ExoCAM':      's',
           'ROCKE-3D':    '^',
           'Generic PCM': 'D',
           'LFRic':       'v' }

models = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic' ]

data = {
    'ExoCAM':      dict( case=exocam_case,  jet=exocam_jet,  jetlat=exocam_jetlat,
                         umax=exocam_umax,  tsmin=exocam_tsmin,  ratio=exocam_ratio ),
    'ExoPlaSim':   dict( case=plasim_case,  jet=plasim_jet,  jetlat=plasim_jetlat,
                         umax=plasim_umax,  tsmin=plasim_tsmin,  ratio=plasim_ratio ),
    'ROCKE-3D':    dict( case=rocke3d_case, jet=rocke3d_jet, jetlat=rocke3d_jetlat,
                         umax=rocke3d_umax, tsmin=rocke3d_tsmin, ratio=rocke3d_ratio ),
    'LFRic':       dict( case=lfric_case,   jet=lfric_jet,   jetlat=lfric_jetlat,
                         umax=lfric_umax,   tsmin=lfric_tsmin,   ratio=lfric_ratio ),
    'Generic PCM': dict( case=pcm_case,     jet=pcm_jet,     jetlat=pcm_jetlat,
                         umax=pcm_umax,     tsmin=pcm_tsmin,     ratio=pcm_ratio ),
}

T_FREEZE = 273.16
c_label  = '0.35'

# Surface pressure of each sample point (bar), replicated from fig_regimes.py
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16,
                    0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )


def min_ellipse( P, tol=1e-4 ):
    """Smallest ellipse (x-c)' A (x-c) <= 1 holding the points P (Khachiyan)."""
    n, d = P.shape
    Q = np.vstack( [ P.T, np.ones( n ) ] )
    u = np.full( n, 1.0 / n )
    for _ in range( 20000 ):
        X = Q @ np.diag( u ) @ Q.T
        M = np.einsum( 'ij,ji->i', Q.T @ np.linalg.inv( X ), Q )
        j = np.argmax( M )
        if M[j] < ( d + 1.0 ) * ( 1.0 + tol ):
            break
        step = ( M[j] - d - 1.0 ) / ( ( d + 1.0 ) * ( M[j] - 1.0 ) )
        u = ( 1.0 - step ) * u
        u[j] += step
    ctr = P.T @ u
    A = np.linalg.inv( P.T @ np.diag( u ) @ P - np.outer( ctr, ctr ) ) / d
    r = np.einsum( 'ij,jk,ik->i', P - ctr, A, P - ctr )
    return ctr, A / max( r.max(), 1.0 )


def jet_of( name, case ):
    d = data[ name ]
    return d[ 'jet' ][ list( d[ 'case' ] ).index( case ) ]


def oval( ax, keep, xkey, ykey, label, side=( 0.0, 1.0 ) ):
    """Dashed oval around the points picked by keep( name, case ), labeled at
    the point of the oval furthest in the direction side ( dx, dy ). Fitted on the page in inches, as in Figure 16, so call it only once
    the axis limits are fixed."""
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed( fig.dpi_scale_trans.inverted() )
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    sx, sy = bb.width / ( x1 - x0 ), bb.height / ( y1 - y0 )
    xs, ys = [], []
    for name in models:
        d = data[ name ]
        for c, x, y in zip( d[ 'case' ], d[ xkey ], d[ ykey ] ):
            if keep( name, c ):
                xs.append( x * sx ); ys.append( y * sy )
    ring = np.linspace( 0.0, 2.0 * np.pi, 16, endpoint=False )
    P = np.column_stack( [ ( np.array( xs )[:, None] + 0.095 * np.cos( ring ) ).ravel(),
                           ( np.array( ys )[:, None] + 0.095 * np.sin( ring ) ).ravel() ] )
    ctr, A = min_ellipse( P )
    w, V = np.linalg.eigh( A )
    t = np.linspace( 0.0, 2.0 * np.pi, 200 )
    E = ctr[:, None] + V @ ( np.diag( 1.0 / np.sqrt( w ) ) @ np.vstack( [ np.cos( t ), np.sin( t ) ] ) )
    ax.plot( E[0] / sx, E[1] / sy, color='0.4', ls='--', lw=0.9, zorder=2 )
    dx, dy = side
    k = np.argmax( dx * E[0] + dy * E[1] )
    ax.text( ( E[0, k] + 0.04 * dx ) / sx, ( E[1, k] + 0.04 * dy ) / sy, label,
             fontsize=9, color=c_label, zorder=6,
             ha='left' if dx > 0.3 else 'right' if dx < -0.3 else 'center',
             va='bottom' if dy > 0.3 else 'top' if dy < -0.3 else 'center' )

fig, axes = plt.subplots( 1, 2, figsize=(11.2, 4.6), layout='constrained' )
fig.get_layout_engine().set( wspace=0.08 )   # a little air between the panels


def draw( ax, xkey, ykey ):
    """Scatter every model, filled for a single jet and open for a double."""
    for name in models:
        d = data[ name ]
        for x, y, j in zip( d[ xkey ], d[ ykey ], d[ 'jet' ] ):
            single = ( j == 'SJ' )
            ax.scatter( x, y, s=45, marker=marker[ name ],
                        facecolor=style[ name ] if single else 'none',
                        edgecolors='k' if single else style[ name ],
                        linewidths=0.7 if single else 1.2, zorder=5 )


#--------------------------------------------------------------------
# Panel (a) — jet latitude against equatorial wind

ax = axes[0]
ax.axhspan( -6, 20, color='#eef3f8', zorder=0 )
draw( ax, 'umax', 'jetlat' )

ax.set_ylim( -6, 78 )
ax.set_xlabel( 'Maximum zonal wind within $10\\degree$ of the equator\n'
               'at $\\sigma = 0.30$ (m s$^{-1}$)', fontsize=12 )
ax.set_ylabel( 'Latitude of the tropospheric jet ($\\degree$)', fontsize=12 )
ax.set_title( '(a) Jet latitude against equatorial wind', fontsize=12 )
ax.set_xlim( ax.get_xlim() )
# The single jets of the thin atmospheres; the regime is set by the surface pressure
oval( ax, lambda n, c: pres1[ c - 1 ] < 1.0 and jet_of( n, c ) == 'SJ', 'umax', 'jetlat',
      '$p_s < 1$ bar', side=( 0.0, 1.0 ) )

#--------------------------------------------------------------------
# Panel (b) — minimum surface temperature against the contrast ratio

ax = axes[1]
ax.axhline( T_FREEZE, color='k', ls='--', lw=1.0, zorder=1 )
draw( ax, 'ratio', 'tsmin' )

ax.set_xlabel( 'Ratio of the day-night to\nequator-pole temperature difference',
               fontsize=12 )
ax.set_ylabel( 'Minimum surface temperature (K)', fontsize=12 )
# At the right-hand end of the line, clear of ROCKE-3D at 272 K on the left
ax.text( 0.98, T_FREEZE + 6, '273.16 K', fontsize=9, color=c_label, ha='right',
         transform=ax.get_yaxis_transform() )
ax.set_title( '(b) Minimum surface temperature against contrast ratio', fontsize=12 )
for ax in axes:
    ax.tick_params( axis='both', labelsize=10 )

# One legend row above the panels, for the model colors and the jet symbols,
# on the layout of Figure 15
fig.legend( handles=[ Line2D( [], [], ls='', marker=marker[ m ], mfc=style[ m ], mec='k', ms=7, label=m )
                      for m in models ] +
                    [ Line2D( [], [], ls='', marker='o', mfc='0.55', mec='k', ms=7,
                              label='filled: single (equatorial) jet' ),
                      Line2D( [], [], ls='', marker='o', mfc='none', mec='0.4',
                              mew=1.2, ms=7, label='open: double (midlatitude) jet' ) ],
            loc='outside upper center', ncol=7, fontsize=10,
            frameon=False, columnspacing=1.2, handletextpad=0.3 )
fig.savefig( 'fig_jet_regimes.png', bbox_inches='tight' )
fig.savefig( 'fig_jet_regimes.eps', bbox_inches='tight' )

#--------------------------------------------------------------------
# Do the two regimes separate in these planes, as they do in Sergeev et al.?

print( '=== regime separation ===' )
for xkey, label in ( ( 'umax', 'equatorial wind (m/s)' ),
                     ( 'jetlat', 'jet latitude (deg)' ),
                     ( 'ratio', 'contrast ratio' ),
                     ( 'tsmin', 'minimum Ts (K)' ) ):
    sj = np.concatenate( [ data[ m ][ xkey ][ np.array( data[ m ][ 'jet' ] ) == 'SJ' ]
                           for m in models ] )
    dj = np.concatenate( [ data[ m ][ xkey ][ np.array( data[ m ][ 'jet' ] ) == 'DJ' ]
                           for m in models ] )
    lo, hi = max( sj.min(), dj.min() ), min( sj.max(), dj.max() )
    span   = max( sj.max(), dj.max() ) - min( sj.min(), dj.min() )
    print( f'  {label:24s} SJ {sj.min():7.1f}-{sj.max():6.1f} med {np.median(sj):6.1f}'
           f'   DJ {dj.min():7.1f}-{dj.max():6.1f} med {np.median(dj):6.1f}'
           f'   overlap {100.0 * max(0.0, hi - lo) / span:3.0f}% of the range' )

#--------------------------------------------------------------------
# The surface pressure sets the jet state (the oval in panel a)

print( '\n=== jet state against surface pressure ===' )
C = np.concatenate( [ data[ m ][ 'case' ] for m in models ] )
J = np.concatenate( [ data[ m ][ 'jet'  ] for m in models ] )
P = pres1[ C - 1 ]
thin = P < 1.0
print( f'  below 1 bar: {np.sum(J[thin] == "SJ")} of {thin.sum()} single jets;'
       f' at or above: {np.sum(J[~thin] == "SJ")} of {(~thin).sum()}' )
print( f'  "single jet below 1 bar" is right for {np.sum(thin == (J == "SJ"))} of {len(J)}' )
print( '  exceptions: ' + ', '.join( f'{m} {c} ({pres1[c-1]:.2f} bar, {j})'
       for m in models for c, j in zip( data[ m ][ 'case' ], data[ m ][ 'jet' ] )
       if ( pres1[ c - 1 ] < 1.0 ) != ( j == 'SJ' ) ) )
