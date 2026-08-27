import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

# Residual surface energy imbalance, per cent of incident flux, on the same
# sign convention as the TOA panel: positive = the surface is losing energy.
#
# Net downward surface flux is net SW + net LW - sensible - latent. Only two
# groups submitted every term needed to close it:
#   ExoCAM     FSNS - FLNS - SHFLX - LHFLX, gw-weighted. CAM signs are FSNS
#              positive down and FLNS/SHFLX/LHFLX positive up.
#   ExoPlaSim  rss + rls + hfss + hfls, cos(lat)-weighted. All four are stored
#              positive downward already, so they add.
#
# The other four cannot supply this panel:
#   ROCKE-3D     has srtrnf_grnd (net radiation at ground) but no sensible or
#                latent heat flux, so the budget cannot be closed. Plotting net
#                radiation alone would show a spurious 24-65 W/m2 imbalance.
#   LFRic        has sw_down_surf and lw_net_surf only: no upward SW at the
#                surface (nor a surface albedo) and no turbulent fluxes.
#   Generic PCM  submitted only the standardized global output, whose Fsdn and
#   PlaHab       Fnet columns are insufficient to close a surface budget.
surface = {
    'ExoCAM':      np.array( [   2.00,    nan,    nan,   0.24,    nan,    nan,  -3.62,   0.67,   0.36,   2.11,   0.28,   0.06,    nan,   0.59,   1.20,   0.06 ] ),
    'ExoPlaSim':   np.array( [  -0.00,   0.39,  -0.03,  -0.13,  -1.54,  -0.04,  -0.38,  -0.40,  -0.11,   0.01,  -0.63,   0.06,  -2.01,  -0.16,  -0.73,  -0.25 ] ),
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

# Regime of each case, on the consensus rule of fig_summary.py but evaluated at
# the sample point from the models that actually ran it, rather than from the
# kriged field. 'runaway' where at least half of the four full-coverage models
# (ExoPlaSim, ExoCAM, ROCKE-3D, PlaHab) fail; otherwise 'frozen' if every model
# with data there puts the global mean below 273.16 K, 'warm' if every model
# puts it above, and 'mixed' if they disagree on the sign.
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

models   = [ 'ExoPlaSim', 'ExoCAM', 'ROCKE-3D', 'Generic PCM', 'LFRic', 'PlaHab' ]
tol      = 1.0      # per cent; band within which a run is taken as equilibrated
linthresh = 1.0     # per cent; linear/log crossover of the symlog axis

fig, ( ax, axs ) = plt.subplots( 2, 1, figsize=( 13, 8.6 ), sharex=True,
                                 gridspec_kw=dict( height_ratios=[ 2.0, 1.0 ], hspace=0.08 ) )

for a in ( ax, axs ):
    # Shade each case by its climate regime, in the colors of Figure 14, so the
    # convergence diagnostic can be read against the regime it belongs to.
    for i, r in zip( cases, regime ):
        a.axvspan( i - 0.5, i + 0.5, color=regime_color[ r ], lw=0, zorder=0 )

    # Regime color alone no longer separates the cases, since neighbours sharing
    # a regime merge into one block, so the boundaries are ruled explicitly.
    for i in range( 1, 16 ):
        a.axvline( i + 0.5, color='0.74', lw=0.8, zorder=0.5 )

    # Zero line, and the tolerance marked by rules rather than fill
    a.axhline( 0.0, color='0.45', lw=0.9, zorder=2 )
    for sgn in ( -1, 1 ):
        a.axhline( sgn * tol, color='0.72', lw=0.8, ls=( 0, ( 4, 3 ) ), zorder=1 )

offsets = np.linspace( -0.30, 0.30, len( models ) )

for off, name in zip( offsets, models ):
    ok = accepted[ name ]
    st = style[ name ]
    x  = cases + off

    y = imbalance[ name ]
    good = ~np.isnan( y ) & ok
    ax.scatter( x[ good ], y[ good ], marker=st[ 'marker' ], s=58,
                facecolors=st[ 'color' ], edgecolors='k', linewidths=0.6, zorder=4 )

    # Same lane, same symbol, so a model sits in the same place in both panels
    if name in surface:
        ys = surface[ name ]
        goods = ~np.isnan( ys ) & ok
        axs.scatter( x[ goods ], ys[ goods ], marker=st[ 'marker' ], s=58,
                     facecolors=st[ 'color' ], edgecolors='k', linewidths=0.6, zorder=4 )

ax.set_yscale( 'symlog', linthresh=linthresh, linscale=1.1 )
ax.set_yticks( [ -10, -3, -1, 0, 1, 3, 10, 20 ] )
ax.set_yticklabels( [ '-10', '-3', '-1', '0', '1', '3', '10', '20' ] )
ax.set_ylim( -14, 26 )
ax.set_ylabel( 'TOA imbalance\n(OLR $-$ ASR) / (S/4)  (%)', fontsize=11.5 )

# The surface residuals span only about +/-4%, so a linear axis is clearer here
# than the symlog the TOA panel needs, and it keeps the +/-1% rules comparable.
axs.set_yticks( [ -2, -1, 0, 1, 2 ] )
axs.set_ylim( -3.5, 2.7 )
axs.set_ylabel( 'Surface imbalance\n$-F_\\mathrm{sfc}$ / (S/4)  (%)', fontsize=11.5 )

axs.set_xlim( 0.4, 16.6 )
axs.set_xticks( cases )
axs.set_xticklabels( [ f'{c}\n{f:.0f}\n{p:.2f}' for c, f, p in zip( cases, flux1, pres1 ) ], fontsize=9 )
axs.set_xlabel( 'Case / instellation (W m$^{-2}$) / N$_2$ surface pressure (bar)', fontsize=12, labelpad=8 )

for a in ( ax, axs ):
    a.tick_params( axis='y', labelsize=11 )
    a.text( 16.45, tol, f'$\\pm${tol:.0f}%', ha='right', va='bottom', fontsize=9, color='0.55' )

# Say why four of the six models are absent below, so their absence does not
# read as a result
axs.text( 0.55, -3.35, 'ExoCAM and ExoPlaSim only: no other group submitted '
                       'every term needed to close the surface budget',
          fontsize=9.5, color='0.35', style='italic', va='bottom' )

# One marker shape and color per model, and nothing else encoded in the symbol
model_handles = [ Line2D( [0], [0], marker=style[ m ][ 'marker' ], color='none',
                          markerfacecolor=style[ m ][ 'color' ], markeredgecolor='k',
                          markeredgewidth=0.6, markersize=9, label=m ) for m in models ]

regime_handles = [ Patch( facecolor=regime_color[ r ], edgecolor='0.7', linewidth=0.6,
                          label=regime_label[ r ] )
                   for r in ( 'frozen', 'mixed', 'warm', 'runaway' ) ]

fig.legend( handles=model_handles, loc='lower left', ncol=6, fontsize=11,
            bbox_to_anchor=( 0.0, 1.055 ), bbox_transform=ax.transAxes,
            frameon=False, columnspacing=1.8, handletextpad=0.4 )
fig.legend( handles=regime_handles, loc='lower left', ncol=4, fontsize=10,
            bbox_to_anchor=( 0.0, 1.005 ), bbox_transform=ax.transAxes,
            frameon=False, columnspacing=1.8, handletextpad=0.6, handlelength=1.6 )

fig.savefig( "fig_energy_balance.png", bbox_inches='tight', dpi=150 )
fig.savefig( "fig_energy_balance.eps", bbox_inches='tight' )
