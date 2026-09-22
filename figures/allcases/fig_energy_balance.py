import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Axis labels in bold, and set a little clear of the tick labels
plt.rcParams[ 'axes.labelweight' ] = 'bold'
plt.rcParams[ 'axes.labelpad' ]    = 8

# ─── Top-of-atmosphere energy balance for all SAMOSA cases ───────────────────
#
# Plotted quantity is the magnitude of the residual TOA radiative imbalance
#
#     | ASR - OLR | / ( S / 4 )  x 100 %
#
# expressed as a percentage of the global mean incident stellar flux S/4, which
# is fixed by the experiment design and therefore identical across models. The
# figure is a convergence check, and how far a run sits from balance is what
# that check turns on, so the magnitude is plotted and the sign suppressed: a
# case is equilibrated if it lies below the 1% rule, whatever the direction of
# its residual. The signed values are retained in the dict below, since the
# direction still matters where the text attributes a warm bias to a run that
# had not finished cooling.
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
#   LFRic        lfric/samosa_global_diagnostics_lfric_2026-09-10.txt
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
    'LFRic':       np.array( [  0.66,    nan,    nan,  -0.37,    nan,    nan,  29.09,  -1.28,  -0.12,  -1.65,   2.12,  -0.11,    nan,  -0.33,   0.98,   0.16 ] ),
    'PlaHab':      np.array( [ -0.19,    nan,    nan,   0.76,  -0.54,    nan,  -0.57,   2.04,   1.06,  -9.65,  -0.48,  -0.50,  -0.36,   2.79,  -0.07,  -0.42 ] ),
    'HEXTOR':      np.array( [ -0.50,    nan,    nan,  -0.74,    nan,    nan,    nan,  -0.64,  -0.72,  -0.51,  -0.62,    nan,    nan,  -0.66,  -0.50,  -0.70 ] ),
    'ExoColumn':   np.array( [ -0.05,    nan,    nan,  -0.29,    nan,    nan,    nan,   0.06,  -0.20,   0.11,  -0.04,    nan,    nan,  -0.21,  -0.15,    nan ] ),
}

# True where the case is carried into the analysis of Figures 2-5; False where
# the group submitted output but classified the run as runaway or unstable.
# Only the True cases are plotted: this figure is a convergence check on the
# runs that actually enter the analysis, so a rejected run has no bearing on
# whether the results shown elsewhere are equilibrated. The imbalances of the
# rejected runs are quoted in the text where they bear on a group's own
# classification.
#
# LFRic Cases 8, 10 and 11 (submitted 2026-09-10) are the first LFRic cases
# other than Case 7 to sit above the 1% rule, at -1.28, -1.65 and +2.12%. That
# is the same band as ExoCAM Cases 1 and 10 and PlaHab Cases 8 and 14, all
# carried. On a Planck-only estimate the residual is worth at most ~2 K of
# further drift in the global mean (Case 11, 4.8 W/m^2 over 4*sigma*T^3 at
# 227.5 K), far inside the 30-60 K inter-model spread at those points. They are
# accepted without further query.
#
# LFRic Case 7 (submitted 2026-09-04) is accepted, and is by a wide margin the
# largest imbalance carried anywhere in this figure: +29.09%, against -1.65 to
# +2.12% over every other LFRic case. It was queried with the LFRic group
# rather than assumed to be a spin-up artifact, and Sergeev confirmed the
# imbalance is persistent rather than decaying, attributing it to a cloud layer
# that becomes stuck at the top of the model domain. It is therefore a property
# of the model configuration at this point in parameter space, not an
# unequilibrated run, and it is carried on the same footing as every other
# submitted case. The accompanying diagnostics are consistent with that
# reading and equally extreme: planetary albedo 3.4% against 21-36% elsewhere
# in the model, stratospheric specific humidity 3.04e-01 kg/kg, over 500 times
# the next wettest LFRic case (5.44e-04 at Case 11, whose model top is only
# 1.4 hPa over a 0.1 bar surface), and an OLR of 502.6 W/m^2 over a 1735 kg/m^2
# water column, far above the ~276-288 W/m^2 at which LFRic's own Cases 12 and
# 16 settle. Its 400.52 K global mean is 107.5 K above the warmest other model
# at this sample point (PlaHab 293.0 K; ExoPlaSim 279.7, ROCKE-3D 267.7), so it
# is an outlier in the ensemble comparison as well and should be read as one.
accepted = {
    'ExoPlaSim':   np.array( [ True ] * 16 ),
    'ExoCAM':      np.array( [ True, False, False, True, False, False, False, True, True, True, True, True, False, True, True, True ] ),
    'ROCKE-3D':    np.array( [ True, False, False, True, True, False, True, True, True, True, True, True, True, True, True, True ] ),
    'Generic PCM': np.array( [ True, False, False, True, False, False, False, True, True, True, False, False, False, True, True, False ] ),
    'LFRic':       np.array( [ True, False, False, True, False, False, True, True, True, True, True, True, False, True, True, True ] ),
    'PlaHab':      np.array( [ True, False, False, True, True, False, True, True, True, True, True, True, True, True, True, True ] ),
    'HEXTOR':      np.array( [ True, False, False, True, False, False, False, True, True, True, True, False, False, True, True, True ] ),
    'ExoColumn':   np.array( [ True, False, False, True, False, False, False, True, True, True, True, False, False, True, True, False ] ),
}

# Colors follow the selectcases figures so the models read consistently
style = {
    'ExoPlaSim':   dict( color='#ff7f0e', marker='o' ),
    'ExoCAM':      dict( color='#1f77b4', marker='s' ),
    'ROCKE-3D':    dict( color='#2ca02c', marker='^' ),
    'Generic PCM': dict( color='#d62728', marker='D' ),
    'LFRic':       dict( color='#9467bd', marker='v' ),
    'PlaHab':      dict( color='#8c564b', marker='P' ),
    'HEXTOR':      dict( color='#17becf', marker='X' ),
    'ExoColumn':   dict( color='#7f7f7f', marker='*' ),
}

# Regime of each case, on the consensus rule of fig_summary.py but evaluated at
# the sample point from the models that actually ran it, rather than from the
# kriged field. 'runaway' where at least half of the four full-coverage models
# (ExoPlaSim, ExoCAM, ROCKE-3D, PlaHab) fail; otherwise 'frozen' if every model
# with data there puts the global mean below 273.16 K, 'warm' if every model
# puts it above, and 'mixed' if they disagree on the sign.
# Case 9 was 'mixed' while HEXTOR put it at 278.2 K, and is 'frozen' again since
# HEXTOR's RH 0.8 resubmission (2026-09-16) brought it to 267.9 K. Re-run the
# sample-point cross-check against fig_summary.py after any resubmission.
regime = [ 'frozen', 'runaway', 'runaway', 'mixed', 'mixed', 'runaway', 'mixed', 'frozen',
           'frozen', 'frozen', 'frozen', 'warm', 'mixed', 'frozen', 'frozen', 'warm' ]

# Colors are taken from fig_summary.py so the two figures read as one scheme
regime_color = { 'frozen':  '#d6e6f4',     # every model below 273.16 K
                 'warm':    '#dcefdb',     # every model above 273.16 K
                 'mixed':   '#ffffff',     # models disagree on the sign
                 'runaway': '#f2d6d8' }    # majority of full-coverage models runaway
regime_label = { 'frozen':  'All below 273 K',
                 'mixed':   'Models disagree',
                 'warm':    'All above 273 K',
                 'runaway': 'Majority runaway' }

models   = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic', 'PlaHab', 'HEXTOR', 'ExoColumn' ]
tol      = 1.0      # per cent; below which a run is taken as equilibrated
# Plotting the magnitude frees the half of the axis that used to carry the
# negative residuals, so the linear/log crossover drops from 1.0 to 0.1 and the
# well-converged cluster, which is most of the ensemble, is resolved rather than
# compressed against the zero line. Exact zeros still plot, which a pure log
# axis would not allow.
linthresh = 0.1     # per cent; linear/log crossover of the symlog axis

fig, ax = plt.subplots( figsize=( 13, 4.6 ) )

# Shade each case by its climate regime, in the colors of Figure 14, so the
# convergence diagnostic can be read against the regime it belongs to.
for i, r in zip( cases, regime ):
    ax.axvspan( i - 0.5, i + 0.5, color=regime_color[ r ], lw=0, zorder=0 )

# Regime color alone does not separate the cases, since neighbours sharing a
# regime merge into one block, so the boundaries are ruled explicitly.
for i in range( 1, 16 ):
    ax.axvline( i + 0.5, color='0.74', lw=0.8, zorder=0.5 )

# Perfect balance is the foot of the axis, and the tolerance is a single rule
# rather than a symmetric pair, so a converged run is one that plots below it.
ax.axhline( 0.0, color='0.45', lw=0.9, zorder=2 )
ax.axhline( tol, color='0.72', lw=0.8, ls=( 0, ( 4, 3 ) ), zorder=1 )

offsets = np.linspace( -0.30, 0.30, len( models ) )

for off, name in zip( offsets, models ):
    ok = accepted[ name ]
    st = style[ name ]
    x  = cases + off

    y = np.abs( imbalance[ name ] )
    good = ~np.isnan( y ) & ok
    ax.scatter( x[ good ], y[ good ], marker=st[ 'marker' ], s=58,
                facecolors=st[ 'color' ], edgecolors='k', linewidths=0.6, zorder=4 )

ax.set_yscale( 'symlog', linthresh=linthresh, linscale=1.1 )
ax.set_yticks( [ 0, 0.1, 0.3, 1, 3, 10, 30 ] )
ax.set_yticklabels( [ '0', '0.1', '0.3', '1', '3', '10', '30' ] )
# The upper limit has to clear LFRic Case 7 at 29.09%, the largest imbalance
# carried anywhere in the ensemble; at the previous limit of 26 it was silently
# clipped off the top of the axis. The lower limit sits just below zero so that
# the exactly balanced ROCKE-3D cases are not cut in half by the spine.
ax.set_ylim( -0.02, 38 )
ax.set_ylabel( 'TOA imbalance, |ASR $-$ OLR| / (S/4)  (%)', fontsize=12 )

ax.set_xlim( 0.4, 16.6 )
ax.set_xticks( cases )
ax.set_xticklabels( [ f'{c}\n{f:.0f}\n{p:.2f}' for c, f, p in zip( cases, flux1, pres1 ) ], fontsize=9 )
ax.set_xlabel( 'Case / instellation (W m$^{-2}$) / N$_2$ surface pressure (bar)', fontsize=12, labelpad=8 )

ax.tick_params( axis='y', labelsize=11 )
ax.text( 16.45, tol, f'{tol:.0f}%', ha='right', va='bottom', fontsize=9, color='0.55' )

# One marker shape and color per model, and nothing else encoded in the symbol
model_handles = [ Line2D( [0], [0], marker=style[ m ][ 'marker' ], color='none',
                          markerfacecolor=style[ m ][ 'color' ], markeredgecolor='k',
                          markeredgewidth=0.6, markersize=9, label=m ) for m in models ]

regime_handles = [ Patch( facecolor=regime_color[ r ], edgecolor='0.7', linewidth=0.6,
                          label=regime_label[ r ] )
                   for r in ( 'frozen', 'mixed', 'warm', 'runaway' ) ]

fig.legend( handles=model_handles, loc='lower left', ncol=8, fontsize=11,
            bbox_to_anchor=( 0.0, 1.085 ), bbox_transform=ax.transAxes,
            frameon=False, columnspacing=1.1, handletextpad=0.3 )
fig.legend( handles=regime_handles, loc='lower left', ncol=4, fontsize=10,
            bbox_to_anchor=( 0.0, 1.005 ), bbox_transform=ax.transAxes,
            frameon=False, columnspacing=1.8, handletextpad=0.6, handlelength=1.6 )

fig.savefig( "fig_energy_balance.png", bbox_inches='tight', dpi=150 )
fig.savefig( "fig_energy_balance.eps", bbox_inches='tight' )
