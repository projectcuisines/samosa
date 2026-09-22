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

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

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
exocam_jetlat= np.array( [ 2.0, 42.0, 66.0, 2.0, 70.0, 2.0, 66.0, 34.0, 2.0, 58.0 ] )
exocam_conv  = np.array( [ 57.1, 196.9, 140.7, 167.3, 55.7, 152.4, 310.8, 136.8, 84.9, 296.1 ] )
exocam_ratio = np.array( [ 0.572, 0.507, 0.463, 0.590, 0.461, 0.638, 0.424, 0.553, 0.531, 0.374 ] )

# ExoPlaSim
plasim_case  = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
plasim_lamr  = np.array( [ 1.360, 1.636, 1.550, 1.490, 1.508, 1.607, 1.527, 1.434, 1.469, 1.353, 1.424, 1.610, 1.518, 1.445, 1.380, 1.611 ] )
plasim_lr    = np.array( [ 1.021, 0.709, 1.418, 1.081, 1.438, 0.878, 1.220, 0.848, 1.272, 0.695, 1.591, 0.841, 1.417, 1.116, 1.235, 0.662 ] )
plasim_jet   = ['DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ']
plasim_jetlat= np.array( [ 58.1, 30.5, 2.8, 52.6, 2.8, 47.1, 2.8, 63.7, 2.8, 69.2, 2.8, 58.1, 2.8, 52.6, 41.5, 2.8 ] )
plasim_conv  = np.array( [ 27.6, 361.0, 464.2, 174.3, 245.8, 430.4, 280.7, 94.7, 153.7, 29.6, 95.3, 313.8, 266.2, 113.1, 36.6, 249.5 ] )
plasim_ratio = np.array( [ 0.655, 0.199, 0.512, 0.609, 0.443, 0.460, 0.545, 0.454, 0.665, 0.685, 0.632, 0.408, 0.418, 0.541, 0.651, 0.283 ] )

# ROCKE-3D
rocke3d_case  = np.array( [ 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] )
rocke3d_lamr  = np.array( [ 1.409, 1.500, 1.508, 1.511, 1.479, 1.473, 1.417, 1.451, 1.572, 1.517, 1.464, 1.423, 1.579 ] )
rocke3d_lr    = np.array( [ 0.979, 0.854, 1.318, 1.073, 0.949, 1.075, 0.900, 1.429, 0.767, 1.293, 0.927, 1.078, 0.650 ] )
rocke3d_jet   = ['SJ', 'DJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'DJ', 'SJ', 'DJ']
rocke3d_jetlat= np.array( [ 0.0, 40.0, 20.0, 4.0, 68.0, 4.0, 68.0, 4.0, 64.0, 24.0, 40.0, 4.0, 68.0 ] )
rocke3d_conv  = np.array( [ 56.4, 196.3, 227.1, 234.2, 138.0, 157.2, 65.6, 130.1, 295.2, 240.9, 130.0, 74.1, 303.5 ] )
rocke3d_ratio = np.array( [ 0.454, 0.497, 0.546, 0.408, 0.525, 0.529, 0.477, 0.573, 0.198, 0.458, 0.519, 0.464, 0.246 ] )

# LFRic
lfric_case  = np.array( [ 1, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16 ] )
lfric_lamr  = np.array( [ 1.396, 1.487, 1.671, 1.457, 1.472, 1.401, 1.450, 1.596, 1.453, 1.411, 1.629 ] )
lfric_lr    = np.array( [ 0.803, 0.647, 0.396, 0.560, 0.784, 0.563, 1.107, 0.338, 0.737, 0.899, 0.290 ] )
lfric_jet   = ['SJ', 'DJ', 'SJ', 'DJ', 'DJ', 'SJ', 'SJ', 'DJ', 'SJ', 'SJ', 'DJ']
lfric_jetlat= np.array( [ 1.0, 43.0, 1.0, 39.0, 1.0, 1.0, 5.0, 61.0, 1.0, 1.0, 69.0 ] )
lfric_conv  = np.array( [ 53.0, 177.3, 447.4, 112.8, 157.4, 56.5, 136.9, 299.6, 116.5, 67.4, 288.9 ] )
lfric_ratio = np.array( [ 0.704, 0.522, 0.441, 0.726, 0.614, 0.817, 0.717, 0.812, 0.645, 0.705, 0.430 ] )

# Generic PCM
pcm_case  = np.array( [ 1, 4, 8, 9, 10, 14, 15 ] )
pcm_lamr  = np.array( [ 1.423, 1.537, 1.480, 1.509, 1.422, 1.479, 1.433 ] )
pcm_lr    = np.array( [ 0.798, 0.640, 0.603, 0.771, 0.541, 0.756, 0.905 ] )
pcm_jet   = ['SJ', 'DJ', 'DJ', 'DJ', 'DJ', 'DJ', 'SJ']
pcm_jetlat= np.array( [ 3.9, 58.7, 62.6, 50.9, 62.6, 43.0, 0.0 ] )
pcm_conv  = np.array( [ 75.4, 261.3, 142.2, 223.6, 73.2, 155.4, 90.9 ] )
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
                         ratio=exocam_ratio ),
    'ExoPlaSim':   dict( case=plasim_case,  lamr=plasim_lamr,  lr=plasim_lr,
                         jet=plasim_jet,  jetlat=plasim_jetlat,  conv=plasim_conv,
                         ratio=plasim_ratio ),
    'ROCKE-3D':    dict( case=rocke3d_case, lamr=rocke3d_lamr, lr=rocke3d_lr,
                         jet=rocke3d_jet, jetlat=rocke3d_jetlat, conv=rocke3d_conv,
                         ratio=rocke3d_ratio ),
    'LFRic':       dict( case=lfric_case,   lamr=lfric_lamr,   lr=lfric_lr,
                         jet=lfric_jet,   jetlat=lfric_jetlat,   conv=lfric_conv,
                         ratio=lfric_ratio ),
    'Generic PCM': dict( case=pcm_case,     lamr=pcm_lamr,     lr=pcm_lr,
                         jet=pcm_jet,     jetlat=pcm_jetlat,     conv=pcm_conv,
                         ratio=pcm_ratio ),
}

def fan( name ):
    """Small horizontal offset so co-located models stay distinguishable."""
    return ( wind_models.index( name ) - 2.0 ) * 0.14

c_slow   = '#eef3f8'
c_rhines = '#faf3ec'
c_label  = '0.35'

npanel = 3 if SHOW_TRANSPORT else 2
fig, axes = plt.subplots( 1, npanel, figsize=( 3.75 * npanel, 4.2 ), layout='constrained' )

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
ax.set_title( '(a) Circulation regime', fontsize=12 )

#--------------------------------------------------------------------
# Panel (b) — jet structure across the parameter space
#
# Axes match fig_tally.py: instellation decreasing to the right, pressure
# logarithmic. Models are fanned out vertically within each sample point so
# that agreement and disagreement can both be read off directly.

ax = axes[1]
ax.axhspan( 0.0, 20.0, color='#eef3f8', zorder=0 )
for c in range( 1, 17 ):
    ax.axvline( c, color='#e8e8e8', lw=0.8, zorder=0 )

for name in wind_models:
    d = data[ name ]
    for c, jl, j in zip( d[ 'case' ], d[ 'jetlat' ], d[ 'jet' ] ):
        single = ( j == 'SJ' )
        # The marker names the model, as everywhere else; the fill gives the jet
        # regime, filled for a single equatorial jet and open for a double one.
        ax.scatter( c + fan( name ), jl, s=45,
                    marker=marker[ name ],
                    facecolor=style[ name ] if single else 'none',
                    edgecolors='k' if single else style[ name ],
                    linewidths=0.7 if single else 1.2, zorder=5 )

ax.set_xlim( 0.4, 16.6 )
ax.set_ylim( -4, 78 )
ax.set_xticks( [ 1, 4, 7, 10, 13, 16 ] )
ax.set_xticks( range( 1, 17 ), minor=True )
ax.set_xlabel( 'Case', fontsize=12 )
ax.set_ylabel( 'Latitude of the\ntropospheric jet ($\\degree$)', fontsize=12 )
ax.text( 16.3, 8, 'equatorial jet', fontsize=9, style='italic', color=c_label, ha='right' )
jet_handles = [ Line2D( [], [], ls='', marker='o', mfc='0.55', mec='k', ms=7,
                        label='filled: single (equatorial) jet' ),
                Line2D( [], [], ls='', marker='o', mfc='none', mec='0.4',
                        mew=1.2, ms=7, label='open: double (midlatitude) jet' ) ]
ax.set_title( '(b) Jet structure at $\\sigma = 0.30$', fontsize=12 )

#--------------------------------------------------------------------
# Panel (c) — day-night against equator-pole heat transport

if SHOW_TRANSPORT:
    ax = axes[2]
    for c in range( 1, 17 ):
        ax.axvline( c, color='#e8e8e8', lw=0.8, zorder=0 )

    for name in wind_models:
        d = data[ name ]
        ax.scatter( d[ 'case' ] + fan( name ), d[ 'conv' ], s=45, marker=marker[ name ],
                    color=style[ name ], edgecolors='k', linewidths=0.7,
                    label=name, zorder=5 )

    ax.set_xlim( 0.4, 16.6 )
    ax.set_xticks( [ 1, 4, 7, 10, 13, 16 ] )
    ax.set_xticks( range( 1, 17 ), minor=True )
    ax.set_xlabel( 'Case', fontsize=12 )
    ax.set_ylabel( 'Night-side static energy flux\nconvergence (W m$^{-2}$)', fontsize=12 )
    ax.set_title( '(c) Night-side energy transport', fontsize=12 )

for ax in axes:
    ax.tick_params( axis='both', labelsize=10 )

# One legend row above the panels, for the model colors of all three and the
# jet symbols of (b); inside the panels it hid points at this size
model_handles, _ = axes[0].get_legend_handles_labels()
fig.legend( handles=model_handles + jet_handles, loc='outside upper center', ncol=7, fontsize=10,
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
