import sys, os, runpy, contextlib, io, warnings, numpy as np, matplotlib
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
from _paths import FIG_ALL, scratch_copy
src = scratch_copy(sys.argv[1] if len(sys.argv) > 1 else FIG_ALL); os.chdir(src)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    g = runpy.run_path(f'{src}/fig_summary.py')
full = g['panels'][0]                                   # the all-cases panel
B, W, C, C3 = (np.asarray(full[k]) for k in ('band_blue', 'band_warm', 'contested_all', 'contested_3d'))
fl = np.asarray(full['flux_grid']) * g['fluxscale']; pr = np.asarray(full['pres_grid'])   # C is (flux, pressure)
# invariant 1: the three regions partition the plane
tot = B.astype(int) + W.astype(int) + C.astype(int)
print('partition: overlaps', int((tot > 1).sum()), 'uncovered', int((tot == 0).sum()))
# invariant 2: no gap in any pressure row of the contested band
gaps = 0
for j in range(C.shape[1]):
    idx = np.where(C[:, j])[0]
    if len(idx) and (idx[-1] - idx[0] + 1) != len(idx): gaps += 1
print('pressure rows with a gap in the contested band:', gaps)
# invariant 3: the 3-D-only contested area lies inside the total
print('white (3-D contested) subset of contested:', bool(np.all(~C3 | C)))
# invariant 4: the shading at every sample point matches fig_energy_balance.py's regime list
F1 = np.array([500,1900,2400,1200,1500,2100,1600,800,1100,400,900,1500,1600,900,600,1400], float)
P1 = np.array([0.70,7.85,0.21,2.34,0.16,1.83,0.55,6.16,0.70,4.83,0.10,2.98,0.16,1.44,0.43,10.0])
fr = np.asarray(full['frac_run'])
src_eb = open('fig_energy_balance.py').read()
regime = eval(src_eb[src_eb.index('regime = [') + len('regime = '): src_eb.index(']', src_eb.index('regime = [')) + 1])
bad = []
for c in range(16):
    i = int(np.argmin(np.abs(fl - F1[c]))); j = int(np.argmin(np.abs(pr - P1[c])))
    assert abs(fl[i] - F1[c]) < 1e-6 and abs(pr[j] - P1[c]) < 1e-9, ('case not on a grid node', c+1, fl[i], pr[j])
    r = 'frozen' if B[i, j] else 'warm' if W[i, j] else 'mixed'
    if fr[i, j] > 0.5: r = 'runaway'
    if r != regime[c]: bad.append((c+1, r, regime[c]))
print('sample-point regime mismatches vs fig_energy_balance:', bad or 'none')
