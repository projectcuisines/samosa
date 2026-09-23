#
# Dynamical regimes across the parameter space
#
# Panel (a)  the non-dimensional Rhines length against the non-dimensional
#            Rossby deformation radius, after Figure 4 of Haqq-Misra et al.
#            (2018), with the equations as corrected by its two errata
# Panel (b)  the night-side static energy flux convergence at each sample
#            point, after Figure 2 of the same 2018 paper
#
# The jet latitude and regime label are plotted in fig_jet_regimes.py; they
# stay here for the numbers printed at the end.
#
# Arrays are produced by extract_regimes.py; rerun it after any resubmission.
#
import numpy as np
import matplotlib.pyplot as plt

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

SHOW_TRANSPORT = True     # set False to drop panel (b)
TRANSPORT_X    = 'flux'   # abscissa of (b): 'flux' (S/S0), 'contrast' or 'case'
CASE_OVALS     = ( TRANSPORT_X != 'contrast' )   # dashed oval and number around each case
                          # in (b); on the contrast axis the cases overlap

# ─── QMC sample points, replicated from fig_interpolation_temp.py ────────────
flux = np.arange( 400, 2700, 100 )
pn2  = np.array( [ 0.10, 0.13, 0.16, 0.21, 0.26, 0.34, 0.43, 0.55, 0.70, 0.89,
                   1.13, 1.44, 1.83, 2.34, 2.98, 3.79, 4.83, 6.16, 7.85, 10.0 ] )
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800,
                    1100, 400, 900, 1500, 1600, 900, 600, 1400 ] )
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16,
                    0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# ─── Regime diagnostics, from extract_regimes.py ─────────────────────────────
# ExoCAM
exocam_case  = np.array( [ 1, 4, 8, 9, 10, 11, 12, 14, 15, 16 ] )
exocam_lamr  = np.array( [ 1.399, 1.500, 1.476, 1.477, 1.394, 1.461, 1.616, 1.465, 1.424, 1.623 ] )
exocam_lr    = np.array( [ 1.017, 0.834, 0.783, 1.034, 0.626, 1.454, 0.637, 0.944, 1.135, 0.664 ] )
exocam_jet   = ['SJ', 'DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ', 'DJ']
exocam_jetlat= np.array( [ 2.0, 42.0, 66.0, 2.0, 70.0, 2.0, 66.0, 34.0, 2.0, 58.0 ] )
exocam_conv  = np.array( [ 57.1, 196.9, 140.7, 167.3, 55.7, 152.4, 310.8, 136.8, 84.9, 296.1 ] )
exocam_dn    = np.array( [ 0.323, 0.164, 0.136, 0.191, 0.156, 0.205, 0.011, 0.228, 0.283, 0.009 ] )
exocam_ratio = np.array( [ 0.572, 0.507, 0.463, 0.590, 0.461, 0.638, 0.424, 0.553, 0.531, 0.374 ] )

# ExoPlaSim
plasim_case  = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
plasim_lamr  = np.array( [ 1.360, 1.636, 1.550, 1.490, 1.508, 1.607, 1.527, 1.434, 1.469, 1.353, 1.424, 1.610, 1.518, 1.445, 1.380, 1.611 ] )
plasim_lr    = np.array( [ 1.021, 0.709, 1.418, 1.081, 1.438, 0.878, 1.220, 0.848, 1.272, 0.695, 1.591, 0.841, 1.417, 1.116, 1.235, 0.662 ] )
plasim_jet   = ['DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ']
plasim_jetlat= np.array( [ 58.1, 30.5, 2.8, 52.6, 2.8, 47.1, 2.8, 63.7, 2.8, 69.2, 2.8, 58.1, 2.8, 52.6, 41.5, 2.8 ] )
plasim_conv  = np.array( [ 27.6, 361.0, 464.2, 174.3, 245.8, 430.4, 280.7, 94.7, 153.7, 29.6, 95.3, 313.8, 266.2, 113.1, 36.6, 249.5 ] )
plasim_dn    = np.array( [ 0.429, 0.016, 0.061, 0.164, 0.105, 0.023, 0.122, 0.220, 0.240, 0.334, 0.297, 0.024, 0.087, 0.262, 0.443, 0.024 ] )
plasim_ratio = np.array( [ 0.655, 0.199, 0.512, 0.609, 0.443, 0.460, 0.545, 0.454, 0.665, 0.685, 0.632, 0.408, 0.418, 0.541, 0.651, 0.283 ] )

# ROCKE-3D
rocke3d_case  = np.array( [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
rocke3d_lamr  = np.array( [ 1.409, 1.500, 1.508, 1.511, 1.479, 1.473, 1.417, 1.451, 1.572, 1.517, 1.464, 1.423, 1.579 ] )
rocke3d_lr    = np.array( [ 0.979, 0.854, 1.318, 1.073, 0.949, 1.075, 0.900, 1.429, 0.767, 1.293, 0.927, 1.078, 0.650 ] )
rocke3d_jet   = ['SJ', 'DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'DJ', 'SJ', 'DJ']
rocke3d_jetlat= np.array( [ 0.0, 40.0, 20.0, 4.0, 68.0, 4.0, 68.0, 4.0, 64.0, 24.0, 40.0, 4.0, 68.0 ] )
rocke3d_conv  = np.array( [ 56.4, 196.3, 227.1, 234.2, 138.0, 157.2, 65.6, 130.1, 295.2, 240.9, 130.0, 74.1, 303.5 ] )
rocke3d_dn    = np.array( [ 0.268, 0.148, 0.079, 0.070, 0.140, 0.177, 0.162, 0.192, 0.041, 0.067, 0.209, 0.254, 0.040 ] )
rocke3d_ratio = np.array( [ 0.454, 0.497, 0.546, 0.408, 0.525, 0.529, 0.477, 0.573, 0.198, 0.458, 0.519, 0.464, 0.246 ] )

# LFRic
lfric_case  = np.array( [ 1, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16 ] )
lfric_lamr  = np.array( [ 1.396, 1.487, 1.671, 1.457, 1.472, 1.401, 1.450, 1.596, 1.453, 1.411, 1.629 ] )
lfric_lr    = np.array( [ 0.803, 0.647, 0.396, 0.560, 0.784, 0.563, 1.107, 0.338, 0.737, 0.899, 0.290 ] )
lfric_jet   = ['SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ', 'SJ', 'DJ', 'SJ', 'SJ', 'DJ']
lfric_jetlat= np.array( [ 1.0, 43.0, 1.0, 39.0, 1.0, 1.0, 5.0, 61.0, 1.0, 1.0, 69.0 ] )
lfric_conv  = np.array( [ 53.0, 177.3, 447.4, 112.8, 157.4, 56.5, 136.9, 299.6, 116.5, 67.4, 288.9 ] )
lfric_dn    = np.array( [ 0.403, 0.171, 0.055, 0.184, 0.232, 0.238, 0.239, 0.014, 0.270, 0.377, 0.017 ] )
lfric_ratio = np.array( [ 0.704, 0.522, 0.441, 0.726, 0.614, 0.817, 0.717, 0.812, 0.645, 0.705, 0.430 ] )

# Generic PCM
pcm_case  = np.array( [ 1, 4, 8, 9, 10, 14, 15 ] )
pcm_lamr  = np.array( [ 1.423, 1.537, 1.480, 1.509, 1.422, 1.479, 1.433 ] )
pcm_lr    = np.array( [ 0.798, 0.640, 0.603, 0.771, 0.541, 0.756, 0.905 ] )
pcm_jet   = ['SJ', 'DJ', 'DJ', 'DJ', 'DJ', 'DJ', 'SJ']
pcm_jetlat= np.array( [ 3.9, 58.7, 62.6, 50.9, 62.6, 43.0, 0.0 ] )
pcm_conv  = np.array( [ 75.4, 261.3, 142.2, 223.6, 73.2, 155.4, 90.9 ] )
pcm_dn    = np.array( [ 0.272, 0.097, 0.143, 0.147, 0.161, 0.214, 0.257 ] )
pcm_ratio = np.array( [ 0.520, 0.430, 0.492, 0.438, 0.355, 0.502, 0.548 ] )

# PlaHab, HEXTOR and ExoColumn cannot appear in any of the three panels. Panel
# (a) needs a wind field and panels (b) and (c) need a jet and a flux map, and
# none of the three submits them. extract_regimes.py does report a contrast
# ratio for the three PlaHab cases carrying a surface temperature field (0.615,
# 0.709 and 0.557 for Cases 1, 4 and 16), usable in the text.

# ─── Model style, following fig_summary.py ───────────────────────────────────
# Colors and markers as in Figure 2 (fig_energy_balance.py), so every model reads
# the same across the paper; the models are also listed in that order.
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

wind_models = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic' ]

data = {
    'ExoCAM':      dict( case=exocam_case,  lamr=exocam_lamr,  lr=exocam_lr,
                         jet=exocam_jet,  jetlat=exocam_jetlat,  conv=exocam_conv,
                         ratio=exocam_ratio, dn=exocam_dn ),
    'ExoPlaSim':   dict( case=plasim_case,  lamr=plasim_lamr,  lr=plasim_lr,
                         jet=plasim_jet,  jetlat=plasim_jetlat,  conv=plasim_conv,
                         ratio=plasim_ratio, dn=plasim_dn ),
    'ROCKE-3D':    dict( case=rocke3d_case, lamr=rocke3d_lamr, lr=rocke3d_lr,
                         jet=rocke3d_jet, jetlat=rocke3d_jetlat, conv=rocke3d_conv,
                         ratio=rocke3d_ratio, dn=rocke3d_dn ),
    'LFRic':       dict( case=lfric_case,   lamr=lfric_lamr,   lr=lfric_lr,
                         jet=lfric_jet,   jetlat=lfric_jetlat,   conv=lfric_conv,
                         ratio=lfric_ratio, dn=lfric_dn ),
    'Generic PCM': dict( case=pcm_case,     lamr=pcm_lamr,     lr=pcm_lr,
                         jet=pcm_jet,     jetlat=pcm_jetlat,     conv=pcm_conv,
                         ratio=pcm_ratio, dn=pcm_dn ),
}

S0 = 1361.0              # present-day solar constant, W m-2

# Three pairs of cases share an instellation, and 7/13 and 11/14 overlap in
# convergence too, so on the S/S0 axis each pair is nudged apart, in S/S0. The
# tall Case 7 goes outermost, and Case 14 into the gap towards 1100 W m-2.
PAIR_SHIFT = { 7: +0.024, 13: -0.016, 12: +0.016, 5: -0.018, 14: +0.050, 11: -0.010 }

def fan( name ):
    """Small horizontal offset so co-located models stay distinguishable."""
    return ( wind_models.index( name ) - 2.0 ) * 0.14

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
    # Stretch just enough that every point is inside
    r = np.einsum( 'ij,jk,ik->i', P - ctr, A, P - ctr )
    A = A / max( r.max(), 1.0 )
    return ctr, A

c_slow   = '#eef3f8'
c_rhines = '#faf3ec'
c_label  = '0.35'

npanel = 2 if SHOW_TRANSPORT else 1
fig, axes = plt.subplots( 1, npanel, figsize=( 5.6 * npanel, 4.6 ), layout='constrained',
                         width_ratios=( [ 1.0, 1.4 ] if SHOW_TRANSPORT else None ) )
axes = np.atleast_1d( axes )
fig.get_layout_engine().set( wspace=0.08 )   # a little air between the panels

#--------------------------------------------------------------------
# Panel (a) — Rhines length against Rossby deformation radius
#
# Every point lies at lambda_R/a > 1, so no case in any model is a rapid
# rotator; the axis is extended past 1 to show that the boundary is empty
# rather than cropped away. The protocol fixes the rotation period at 15 days
# for every case, so lambda_R/a varies only through the scale height and is
# a monotone function of the global mean surface temperature. The Rhines
# length carries the discrimination.

ax = axes[0]
ax.axhspan( 1.0, 2.0, color=c_slow,   zorder=0 )
ax.axhspan( 0.0, 1.0, color=c_rhines, zorder=0 )
ax.axhline( 1.0, color='k', ls='--', lw=1.0, zorder=1 )
ax.axvline( 1.0, color='k', ls='--', lw=1.0, zorder=1 )

for name in wind_models:
    d = data[ name ]
    ax.scatter( d[ 'lamr' ], d[ 'lr' ], s=45, marker=marker[ name ], color=style[ name ], edgecolors='k',
                linewidths=0.7, label=name, zorder=5 )

ax.set_xlim( 0.92, 1.78 )
ax.set_ylim( 0.15, 1.80 )
ax.set_xlabel( 'Non-dimensional Rossby\ndeformation radius, $\\lambda_R/a$', fontsize=12 )
ax.set_ylabel( 'Non-dimensional Rhines\nlength, $L_R/a$', fontsize=12 )
ax.text( 0.96, 0.45, 'rapid rotators', fontsize=9, style='italic',
         color=c_label, ha='center', va='center', rotation=90 )
ax.text( 1.75, 1.73, 'slow rotators',   fontsize=10, style='italic', color=c_label, ha='right' )
ax.text( 1.75, 0.20, 'Rhines rotators', fontsize=10, style='italic', color=c_label, ha='right' )
ax.set_title( 'Circulation regime', fontsize=12 )

#--------------------------------------------------------------------
# Panel (b) — night-side energy transport
#
# Against the scaled day-night surface temperature contrast (T_day -
# T_night) / T_eq, as in the lower-right panel of Figure 2 of Haqq-Misra et al.
# (2018), where the axes are the other way round; or against the case number,
# with the models fanned out within each case.

if SHOW_TRANSPORT:
    ax = axes[1]
    by_case = ( TRANSPORT_X == 'case' )
    by_flux = ( TRANSPORT_X == 'flux' )
    def xpos( name ):
        d = data[ name ]
        if by_case:
            return d[ 'case' ] + fan( name )
        if by_flux:
            shift = np.array( [ PAIR_SHIFT.get( c, 0.0 ) for c in d[ 'case' ] ] )
            return flux1[ d[ 'case' ] - 1 ] / S0 + shift + 0.07 * fan( name )
        return d[ 'dn' ]
    xlim = { 'case': ( 0.2, 16.8 ), 'flux': ( 1.80, 0.24 ), 'contrast': ( 0.0, 0.47 ) }[ TRANSPORT_X ]
    ylim = ( 0.0, 520.0 )

    if CASE_OVALS:
        # A dashed oval around the models at each shared case, labeled with the
        # case number, so each cluster reads as one sample point; the cases
        # only ExoPlaSim reports carry the number alone. Each oval is the
        # smallest ellipse holding the markers, found on the page in inches so
        # it keeps its shape whatever the axis ranges; it may tilt with the
        # cluster.
        ax.set_xlim( xlim )
        ax.set_ylim( ylim )
        fig.canvas.draw()
        bb = ax.get_window_extent().transformed( fig.dpi_scale_trans.inverted() )
        sx = bb.width  / abs( xlim[1] - xlim[0] )    # inches per abscissa unit
        sy = bb.height / ( ylim[1] - ylim[0] )    # inches per W m-2
        rm = 0.085                                # marker radius plus a gap, in
        ring = np.linspace( 0.0, 2.0 * np.pi, 16, endpoint=False )
        for c in range( 1, 17 ):
            here = [ n for n in wind_models if c in data[ n ][ 'case' ] ]
            x = np.array( [ xpos( n )[ data[ n ][ 'case' ] == c ][0] for n in here ] ) * sx
            y = np.array( [ data[ n ][ 'conv' ][ data[ n ][ 'case' ] == c ][0] for n in here ] ) * sy
            if len( here ) == 1:
                # ExoPlaSim alone: the number only, where an oval's top would be
                ax.text( x[0] / sx, ( y[0] + rm ) / sy + 3.0, str( c ), fontsize=9,
                         color=c_label, ha='center', va='bottom', zorder=4,
                         bbox=dict( fc='w', ec='none', pad=0.5 ) )
                continue
            P = np.column_stack( [ ( x[:, None] + rm * np.cos( ring ) ).ravel(),
                                   ( y[:, None] + rm * np.sin( ring ) ).ravel() ] )
            ctr, A = min_ellipse( P )
            w, V = np.linalg.eigh( A )
            t = np.linspace( 0.0, 2.0 * np.pi, 200 )
            E = ctr[:, None] + V @ ( np.diag( 1.0 / np.sqrt( w ) )
                                     @ np.vstack( [ np.cos( t ), np.sin( t ) ] ) )
            ax.plot( E[0] / sx, E[1] / sy, color='0.5', ls='--', lw=0.9, zorder=2 )
            # Below the oval where the space above is taken (Case 13 on S/S0)
            below = by_flux and c in ( 13, )
            k = np.argmin( E[1] ) if below else np.argmax( E[1] )
            ax.text( E[0, k] / sx, E[1, k] / sy + ( -3.0 if below else 3.0 ), str( c ),
                     fontsize=9, color=c_label, ha='center',
                     va='top' if below else 'bottom', zorder=4,
                     bbox=dict( fc='w', ec='none', pad=0.5 ) )

    for name in wind_models:
        ax.scatter( xpos( name ), data[ name ][ 'conv' ], s=45, marker=marker[ name ],
                    color=style[ name ], edgecolors='k', linewidths=0.7,
                    label=name, zorder=5 )

    ax.set_xlim( xlim )
    ax.set_ylim( ylim )
    if by_case:
        ax.set_xticks( [ 1, 4, 7, 10, 13, 16 ] )
        ax.set_xticks( range( 1, 17 ), minor=True )
        ax.set_xlabel( 'Case', fontsize=12 )
    elif by_flux:
        ax.set_xlabel( 'Relative instellation, $S/S_0$', fontsize=12 )
    else:
        ax.set_xlabel( 'Day-night temperature contrast,\n'
                       '$(T_{\\mathrm{day}} - T_{\\mathrm{night}})/T_{\\mathrm{eq}}$', fontsize=12 )
    ax.set_ylabel( 'Night-side static energy flux\nconvergence (W m$^{-2}$)', fontsize=12 )
    ax.set_title( 'Night-side energy transport', fontsize=12 )

for ax in axes:
    ax.tick_params( axis='both', labelsize=10 )

# One legend row above the panels; inside them it hid points at this size
model_handles, _ = axes[0].get_legend_handles_labels()
fig.legend( handles=model_handles, loc='outside upper center', ncol=5, fontsize=10,
            frameon=False, columnspacing=1.2, handletextpad=0.3 )
fig.savefig( 'fig_regimes.png', bbox_inches='tight' )
fig.savefig( 'fig_regimes.eps', bbox_inches='tight' )

#--------------------------------------------------------------------
# Numbers quoted in the text

print( '=== regime spread across models, per case ===' )
straddle = 0
for c in range( 1, 17 ):
    vals = { n: data[ n ][ 'lr' ][ np.where( data[ n ][ 'case' ] == c )[0][0] ]
             for n in wind_models if c in data[ n ][ 'case' ] }
    if len( vals ) < 3:
        continue
    v = np.array( list( vals.values() ) )
    hit = v.min() < 1.0 < v.max()
    straddle += hit
    print( f'  case {c:2d}  n={len(v)}  L_R/a {v.min():.2f}-{v.max():.2f}'
           f'{"   straddles the boundary" if hit else ""}' )
print( f'  {straddle} of the multi-model cases straddle L_R/a = 1' )

agree = total = 0
for n in wind_models:
    d = data[ n ]
    for c, l, j in zip( d[ 'case' ], d[ 'lr' ], d[ 'jet' ] ):
        total += 1
        agree += ( l < 1.0 ) == ( j == 'DJ' )
print( f'\n=== the Rhines criterion predicts the jet structure in '
       f'{agree} of {total} cases ({100.0 * agree / total:.0f}%) ===' )

print( '\n=== contrast ratio against the Rhines length ===' )
all_lr, all_ratio = [], []
for n in wind_models:
    lr, ratio = data[ n ][ 'lr' ], data[ n ][ 'ratio' ]
    all_lr.append( lr ); all_ratio.append( ratio )
    print( f'  {n:12s} n={len(lr):2d}  r = {np.corrcoef(lr, ratio)[0,1]:+.2f}'
           f'   ratio {ratio.min():.2f}-{ratio.max():.2f}' )
pooled = np.corrcoef( np.concatenate( all_lr ), np.concatenate( all_ratio ) )[0,1]
print( f'  {"pooled":12s} n={len(np.concatenate(all_lr)):2d}  r = {pooled:+.2f}' )

print( '\n=== night-side convergence against the day-night contrast ===' )
all_dn, all_cv = [], []
for n in wind_models:
    dn, cv = data[ n ][ 'dn' ], data[ n ][ 'conv' ]
    all_dn.append( dn ); all_cv.append( cv )
    print( f'  {n:12s} n={len(dn):2d}  r = {np.corrcoef(dn, cv)[0,1]:+.2f}'
           f'   contrast {dn.min():.3f}-{dn.max():.3f}' )
all_dn, all_cv = np.concatenate( all_dn ), np.concatenate( all_cv )
print( f'  {"pooled":12s} n={len(all_dn):2d}  r = {np.corrcoef(all_dn, all_cv)[0,1]:+.2f}'
       f'   contrast {all_dn.min():.3f}-{all_dn.max():.3f}' )
