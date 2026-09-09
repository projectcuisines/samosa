#
# Dynamical regimes across the parameter space
#
# Panel (a)  the non-dimensional Rhines length against the non-dimensional
#            Rossby deformation radius, after Figure 4 of Haqq-Misra et al.
#            (2018), with the equations as corrected by its two errata
# Panel (b)  the upper-tropospheric jet structure at each sample point, after
#            Figure 3 of Mak et al. (2024)
# Panel (c)  the day-night to equator-pole surface temperature contrast ratio
#            against the Rhines length, after Figure 6 (right) of the same 2018
#            paper
#
# Arrays are produced by extract_regimes.py; rerun it after any resubmission.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SHOW_TRANSPORT = True     # set False to drop panel (c)

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
exocam_ratio = np.array( [ 0.572, 0.507, 0.463, 0.590, 0.461, 0.638, 0.424, 0.553, 0.531, 0.374 ] )

# ExoPlaSim
plasim_case  = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
plasim_lamr  = np.array( [ 1.360, 1.636, 1.550, 1.490, 1.508, 1.607, 1.527, 1.434, 1.469, 1.353, 1.424, 1.610, 1.518, 1.445, 1.380, 1.611 ] )
plasim_lr    = np.array( [ 1.021, 0.709, 1.418, 1.081, 1.438, 0.878, 1.220, 0.848, 1.272, 0.695, 1.591, 0.841, 1.417, 1.116, 1.235, 0.662 ] )
plasim_jet   = ['DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ']
plasim_ratio = np.array( [ 0.655, 0.199, 0.512, 0.609, 0.443, 0.460, 0.545, 0.454, 0.665, 0.685, 0.632, 0.408, 0.418, 0.541, 0.651, 0.283 ] )

# ROCKE-3D
rocke3d_case  = np.array( [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
rocke3d_lamr  = np.array( [ 1.409, 1.500, 1.508, 1.511, 1.479, 1.473, 1.417, 1.451, 1.572, 1.517, 1.464, 1.423, 1.579 ] )
rocke3d_lr    = np.array( [ 0.979, 0.854, 1.318, 1.073, 0.949, 1.075, 0.900, 1.429, 0.767, 1.293, 0.927, 1.078, 0.650 ] )
rocke3d_jet   = ['SJ', 'DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'DJ', 'SJ', 'DJ']
rocke3d_ratio = np.array( [ 0.454, 0.497, 0.546, 0.408, 0.525, 0.529, 0.477, 0.573, 0.198, 0.458, 0.519, 0.464, 0.246 ] )

# LFRic
lfric_case  = np.array( [ 1, 4, 7, 9, 12, 14, 15, 16 ] )
lfric_lamr  = np.array( [ 1.396, 1.487, 1.671, 1.472, 1.596, 1.453, 1.411, 1.629 ] )
lfric_lr    = np.array( [ 0.803, 0.647, 0.396, 0.784, 0.338, 0.737, 0.899, 0.290 ] )
lfric_jet   = ['SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ', 'SJ', 'DJ']
lfric_ratio = np.array( [ 0.704, 0.522, 0.441, 0.614, 0.812, 0.645, 0.705, 0.430 ] )

# Generic PCM
pcm_case  = np.array( [ 1, 4, 8, 9, 10, 14, 15 ] )
pcm_lamr  = np.array( [ 1.423, 1.537, 1.480, 1.509, 1.422, 1.479, 1.433 ] )
pcm_lr    = np.array( [ 0.798, 0.640, 0.603, 0.771, 0.541, 0.756, 0.905 ] )
pcm_jet   = ['DJ', 'DJ', 'DJ', 'DJ', 'DJ', 'DJ', 'DJ']
pcm_ratio = np.array( [ 0.520, 0.430, 0.492, 0.438, 0.355, 0.502, 0.548 ] )

# PlaHab, HEXTOR and ExoColumn cannot appear in any of the three panels. Both
# abscissae here are derived from the wind field, and none of the three submits
# one. extract_regimes.py does report a contrast ratio for the three PlaHab
# cases that carry a surface temperature field (0.615, 0.709 and 0.557 for
# Cases 1, 4 and 16), which is usable in the text but has no abscissa to sit on.

# ─── Model style, following fig_summary.py ───────────────────────────────────
style = { 'ExoPlaSim':   '#ff7f0e',
          'ExoCAM':      '#1f77b4',
          'ROCKE-3D':    '#2ca02c',
          'Generic PCM': '#d62728',
          'LFRic':       '#9467bd' }

wind_models = [ 'ExoCAM', 'ExoPlaSim', 'ROCKE-3D', 'LFRic', 'Generic PCM' ]

data = {
    'ExoCAM':      ( exocam_case,  exocam_lamr,  exocam_lr,  exocam_jet,  exocam_ratio  ),
    'ExoPlaSim':   ( plasim_case,  plasim_lamr,  plasim_lr,  plasim_jet,  plasim_ratio  ),
    'ROCKE-3D':    ( rocke3d_case, rocke3d_lamr, rocke3d_lr, rocke3d_jet, rocke3d_ratio ),
    'LFRic':       ( lfric_case,   lfric_lamr,   lfric_lr,   lfric_jet,   lfric_ratio   ),
    'Generic PCM': ( pcm_case,     pcm_lamr,     pcm_lr,     pcm_jet,     pcm_ratio     ),
}

c_slow   = '#eef3f8'
c_rhines = '#faf3ec'
c_label  = '0.35'

npanel = 3 if SHOW_TRANSPORT else 2
fig, axes = plt.subplots( 1, npanel, figsize=( 7.2 * npanel, 6.4 ) )

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
    case, lamr, lr, jet, _ = data[ name ]
    ax.scatter( lamr, lr, s=95, color=style[ name ], edgecolors='k',
                linewidths=0.7, label=name, zorder=5 )

ax.set_xlim( 0.92, 1.78 )
ax.set_ylim( 0.15, 1.80 )
ax.set_xlabel( 'Non-dimensional Rossby deformation radius, $\\lambda_R/a$', fontsize=12 )
ax.set_ylabel( 'Non-dimensional Rhines length, $L_R/a$', fontsize=12 )
ax.text( 0.945, 0.30, 'rapid\nrotators', fontsize=10, style='italic',
         color=c_label, ha='center', va='center', rotation=90 )
ax.text( 1.75, 1.73, 'slow rotators',   fontsize=11, style='italic', color=c_label, ha='right' )
ax.text( 1.75, 0.20, 'Rhines rotators', fontsize=11, style='italic', color=c_label, ha='right' )
ax.legend( loc='upper left', fontsize=10, framealpha=1 )
ax.set_title( '(a) Circulation regime', fontsize=13 )

#--------------------------------------------------------------------
# Panel (b) — jet structure across the parameter space
#
# Axes match fig_tally.py: instellation decreasing to the right, pressure
# logarithmic. Models are fanned out vertically within each sample point so
# that agreement and disagreement can both be read off directly.

ax = axes[1]
grid_f, grid_p = np.meshgrid( flux, pn2 )
ax.scatter( grid_f.ravel(), grid_p.ravel(), s=3, color='#cccccc', zorder=0 )

fan = { name: 10.0 ** ( ( i - 2.0 ) * 0.052 ) for i, name in enumerate( wind_models ) }
for name in wind_models:
    case, _, _, jet, _ = data[ name ]
    for c, j in zip( case, jet ):
        single = ( j == 'SJ' )
        ax.scatter( flux1[ c - 1 ], pres1[ c - 1 ] * fan[ name ],
                    s=105, marker='o' if single else 's',
                    facecolor=style[ name ] if single else 'none',
                    edgecolors='k' if single else style[ name ],
                    linewidths=0.7 if single else 1.7, zorder=5 )

# The fan spans +/- 0.104 decades, so a label at 1.45x clears it in every case
for c in range( 1, 17 ):
    ax.text( flux1[ c - 1 ], pres1[ c - 1 ] * 1.45, c, fontsize=9, color=c_label,
             ha='center', va='bottom' )

ax.set_yscale( 'log' )
ax.set_xlim( max( flux ) + 50, min( flux ) - 50 )
ax.set_ylim( min( pn2 ) * 0.62, max( pn2 ) * 2.4 )
ax.set_xlabel( 'Instellation (W m$^{-2}$)', fontsize=12 )
ax.set_ylabel( 'Surface pressure (bar)', fontsize=12 )
ax.legend( handles=[ Line2D( [], [], ls='', marker='o', mfc='0.55', mec='k', ms=9,
                             label='single (equatorial) jet' ),
                     Line2D( [], [], ls='', marker='s', mfc='none', mec='0.4',
                             mew=1.7, ms=9, label='double (midlatitude) jet' ) ],
           loc='upper left', fontsize=10, framealpha=1 )
ax.set_title( '(b) Jet structure at $\\sigma = 0.30$', fontsize=13 )

#--------------------------------------------------------------------
# Panel (c) — day-night against equator-pole heat transport

if SHOW_TRANSPORT:
    ax = axes[2]
    ax.axvspan( 1.0, 2.0, color=c_slow,   zorder=0 )
    ax.axvspan( 0.0, 1.0, color=c_rhines, zorder=0 )
    ax.axvline( 1.0, color='k', ls='--', lw=1.0, zorder=1 )

    # Pooled across the ensemble the relation almost vanishes (r = 0.25), but
    # that is an artifact of pooling: each model tracks it internally and the
    # models sit at different offsets, so the trend lines are drawn per model.
    for name in wind_models:
        case, _, lr, _, ratio = data[ name ]
        ax.scatter( lr, ratio, s=95, color=style[ name ], edgecolors='k',
                    linewidths=0.7, label=name, zorder=5 )
        slope, icept = np.polyfit( lr, ratio, 1 )
        xs = np.array( [ lr.min(), lr.max() ] )
        ax.plot( xs, slope * xs + icept, color=style[ name ], lw=1.4,
                 alpha=0.75, zorder=3 )

    ax.set_xlim( 0.15, 1.80 )
    ax.set_xlabel( 'Non-dimensional Rhines length, $L_R/a$', fontsize=12 )
    ax.set_ylabel( '$(T_{day} - T_{night}) / (T_{equator} - T_{pole})$', fontsize=12 )
    ax.text( 1.75, 0.93, 'slow rotators',   fontsize=11, style='italic',
             color=c_label, ha='right', transform=ax.get_xaxis_transform() )
    ax.text( 0.18, 0.93, 'Rhines rotators', fontsize=11, style='italic',
             color=c_label, ha='left',  transform=ax.get_xaxis_transform() )
    ax.set_title( '(c) Day-night against equator-pole transport', fontsize=13 )

fig.tight_layout()
fig.savefig( 'fig_regimes.png', bbox_inches='tight' )
fig.savefig( 'fig_regimes.eps', bbox_inches='tight' )

#--------------------------------------------------------------------
# Numbers quoted in the text

print( '=== regime spread across models, per case ===' )
straddle = 0
for c in range( 1, 17 ):
    vals = { n: data[ n ][ 2 ][ np.where( data[ n ][ 0 ] == c )[0][0] ]
             for n in wind_models if c in data[ n ][ 0 ] }
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
    case, _, lr, jet, _ = data[ n ]
    for c, l, j in zip( case, lr, jet ):
        total += 1
        agree += ( l < 1.0 ) == ( j == 'DJ' )
print( f'\n=== the Rhines criterion predicts the jet structure in '
       f'{agree} of {total} cases ({100.0 * agree / total:.0f}%) ===' )

print( '\n=== contrast ratio against the Rhines length ===' )
all_lr, all_ratio = [], []
for n in wind_models:
    _, _, lr, _, ratio = data[ n ]
    all_lr.append( lr ); all_ratio.append( ratio )
    print( f'  {n:12s} n={len(lr):2d}  r = {np.corrcoef(lr, ratio)[0,1]:+.2f}'
           f'   ratio {ratio.min():.2f}-{ratio.max():.2f}' )
pooled = np.corrcoef( np.concatenate( all_lr ), np.concatenate( all_ratio ) )[0,1]
print( f'  {"pooled":12s} n={len(np.concatenate(all_lr)):2d}  r = {pooled:+.2f}' )
