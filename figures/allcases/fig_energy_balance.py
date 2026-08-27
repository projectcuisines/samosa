import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── Top-of-atmosphere energy balance for all SAMOSA cases ───────────────────
#
# Plotted quantity is the residual TOA radiative imbalance
#
#     ( OLR - ASR ) / ( S / 4 )  x 100 %
#
# expressed as a percentage of the global mean incident stellar flux S/4, which
# is fixed by the experiment design and therefore identical across models. A
# positive value means the planet is losing energy (still cooling); a negative
# value means it is gaining energy (still warming).
#
# OLR and ASR are the standardized SAMOSA global output quantities:
#   ExoCAM       samosaN.cam.h0.avg.nc, gw-weighted FSNT and FLUT. FSNTOA and
#                every clear-sky field are archived as identically zero in the
#                submitted files, so FSNT (top of model) is the only usable
#                shortwave flux and the summary TOAALB/toaEBAL cannot be
#                reproduced from the primary output. We therefore derive
#                ExoCAM from the NetCDF as we do every other model rather than
#                mixing sources. The gw-weighted TS reproduces the summary TS
#                exactly for all 11 files, which validates the weighting.
#                This route also recovers Case 7, which has no summary row.
#   ExoPlaSim    exoplasim/samosaNN.nc, area-weighted rst / rlut
#   ROCKE-3D     rocke3d/rocke_NNq.nc, -trnf_toa_hemis[2] / srnf_toa_hemis[2]
#   Generic PCM  genericpcm/OHT_off/case-N/samosa_gcm_output_case-N_OHT_off.dat
#   LFRic        lfric/samosa_global_diagnostics_lfric_2025-08-22.txt
#   PlaHab       plahab/simulations/sampleN/global_samosa_plahab_*
#                (the simulations/ copies are authoritative; the top-level
#                 seq1sam4 file is a stale duplicate that disagrees by ~3 K)
# ─────────────────────────────────────────────────────────────────────────────

nan = np.nan
cases = np.arange( 1, 17 )

# QMC sample points, for the axis annotation
flux1 = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] )
pres1 = np.array( [ 0.70, 7.85, 0.21, 2.34, 0.16, 1.83, 0.55, 6.16, 0.70, 4.83, 0.10, 2.98, 0.16, 1.44, 0.43, 10.0 ] )

# Residual TOA imbalance, per cent of incident flux (nan = no data submitted)
imbalance = {
    'ExoPlaSim':   np.array( [  0.46,   0.85,   0.03,  -0.04,   0.04,   0.26,   0.08,   0.52,  -0.04,   0.23,  -0.01,   0.25,   0.03,   0.07,   0.07,   0.36 ] ),
    'ExoCAM':      np.array( [  2.00,    nan,    nan,   0.31,    nan,    nan,  -5.23,   0.68,   0.38,   2.19,   1.35,   0.97,    nan,   0.55,   1.30,   0.20 ] ),
    'ROCKE-3D':    np.array( [  0.53,  -8.94,    nan,   0.11,   0.00, -16.21,   0.09,   0.41,   0.01,   6.32,   0.07,  -3.12,  -0.06,  -0.10,   0.00,  -0.10 ] ),
    'Generic PCM': np.array( [ 16.81, -23.95, -36.01,   1.83, -24.56, -31.78, -20.63,   2.18,   3.27,  19.66,  13.26, -12.10, -28.24,   2.41,   7.70, -10.70 ] ),
    'LFRic':       np.array( [  0.66,    nan,    nan,  -0.37,    nan,    nan,    nan,    nan,  -0.12,    nan,    nan,  -0.11,    nan,  -0.33,   0.98,    nan ] ),
    'PlaHab':      np.array( [ -0.19,    nan,    nan,   0.76,  -0.54,    nan,  -0.57,   2.04,   1.06,  -9.65,  -0.25,  -0.50,  -0.36,   2.79,  -0.07,  -0.42 ] ),
}

# True where the case is carried into the analysis of Figures 2-5; False where
# the group submitted output but classified the run as runaway or unstable.
# Only the True cases are plotted: this figure is a convergence check on the
# runs that actually enter the analysis, so a rejected run has no bearing on
# whether the results shown elsewhere are equilibrated. The imbalances of the
# rejected runs are quoted in the text where they bear on a group's own
# classification.
accepted = {
    'ExoPlaSim':   np.array( [ True ] * 16 ),
    'ExoCAM':      np.array( [ True, False, False, True, False, False, False, True, True, True, True, True, False, True, True, True ] ),
    'ROCKE-3D':    np.array( [ True, False, False, True, True, False, True, True, True, True, True, True, True, True, True, True ] ),
    'Generic PCM': np.array( [ True, False, False, True, False, False, False, True, True, True, False, False, False, True, True, False ] ),
    'LFRic':       np.array( [ True, False, False, True, False, False, False, False, True, False, False, True, False, True, True, False ] ),
    'PlaHab':      np.array( [ True, False, False, True, True, False, True, True, True, True, True, True, True, True, True, True ] ),
}

# Colors follow the selectcases figures so the models read consistently
style = {
    'ExoPlaSim':   dict( color='#ff7f0e', marker='o' ),
    'ExoCAM':      dict( color='#1f77b4', marker='s' ),
    'ROCKE-3D':    dict( color='#2ca02c', marker='^' ),
    'Generic PCM': dict( color='#d62728', marker='D' ),
    'LFRic':       dict( color='#9467bd', marker='v' ),
    'PlaHab':      dict( color='#8c564b', marker='P' ),
}

models   = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic', 'PlaHab' ]
tol      = 1.0      # per cent; band within which a run is taken as equilibrated
linthresh = 1.0     # per cent; linear/log crossover of the symlog axis

fig, ax = plt.subplots( figsize=( 13, 7.0 ) )

# Shade alternate cases, so it is unambiguous which markers belong to which
# case. This replaces the separator lines, which are redundant once the
# columns are shaded, and the +/-1% band, which competed with the shading.
for i in cases[ 1::2 ]:
    ax.axvspan( i - 0.5, i + 0.5, color='#eef1f4', lw=0, zorder=0 )

# Zero line, and the equilibrium tolerance now marked by rules rather than fill
ax.axhline( 0.0, color='0.45', lw=0.9, zorder=2 )
for sgn in ( -1, 1 ):
    ax.axhline( sgn * tol, color='0.72', lw=0.8, ls=( 0, ( 4, 3 ) ), zorder=1 )

offsets = np.linspace( -0.30, 0.30, len( models ) )

for off, name in zip( offsets, models ):
    y  = imbalance[ name ]
    ok = accepted[ name ]
    st = style[ name ]
    x  = cases + off

    good = ~np.isnan( y ) & ok

    ax.scatter( x[ good ], y[ good ], marker=st[ 'marker' ], s=58,
                facecolors=st[ 'color' ], edgecolors='k', linewidths=0.6, zorder=4 )

ax.set_yscale( 'symlog', linthresh=linthresh, linscale=1.1 )
ax.set_yticks( [ -10, -3, -1, 0, 1, 3, 10, 20 ] )
ax.set_yticklabels( [ '-10', '-3', '-1', '0', '1', '3', '10', '20' ] )
ax.set_ylim( -14, 26 )
ax.set_xlim( 0.4, 16.6 )

ax.set_xticks( cases )
ax.set_xticklabels( [ f'{c}\n{f:.0f}\n{p:.2f}' for c, f, p in zip( cases, flux1, pres1 ) ], fontsize=9 )
ax.set_xlabel( 'Case / instellation (W m$^{-2}$) / N$_2$ surface pressure (bar)', fontsize=12, labelpad=8 )
ax.set_ylabel( 'TOA imbalance, (OLR $-$ ASR) / (S/4)  (%)', fontsize=12 )
ax.tick_params( axis='y', labelsize=11 )

ax.text( 16.45, tol, f'$\\pm${tol:.0f}%', ha='right', va='bottom', fontsize=9, color='0.55' )
ax.text( 0.55, 20.5, 'still cooling  $\\uparrow$', fontsize=10, color='0.4', style='italic' )
ax.text( 0.55, -11.7, 'still warming  $\\downarrow$', fontsize=10, color='0.4', style='italic' )

# One marker shape and color per model, and nothing else encoded in the symbol
model_handles = [ Line2D( [0], [0], marker=style[ m ][ 'marker' ], color='none',
                          markerfacecolor=style[ m ][ 'color' ], markeredgecolor='k',
                          markeredgewidth=0.6, markersize=9, label=m ) for m in models ]

ax.legend( handles=model_handles, loc='lower left', ncol=6, fontsize=11,
           bbox_to_anchor=( 0.0, 1.005 ), frameon=False, columnspacing=1.8,
           handletextpad=0.4 )

fig.tight_layout()
fig.savefig( "fig_energy_balance.png", bbox_inches='tight', dpi=150 )
fig.savefig( "fig_energy_balance.eps", bbox_inches='tight' )
