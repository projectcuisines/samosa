import sys, os, runpy, contextlib, io, warnings, numpy as np, matplotlib
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
from _paths import FIG_ALL, scratch_copy
d = sys.argv[1] if len(sys.argv) > 1 else FIG_ALL; os.chdir(scratch_copy(d))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    g = runpy.run_path(f'{d}/fig_spread.py')
F = g['flux']*g['fluxscale']; P = g['pn2']
FF, PP = np.meshgrid(F, P, indexing='ij')          # grid is (flux, pn2)
spread, VARS = g['spread'], g['VARS']
ts, wv, cf = VARS
std_ts, std_wv, std_cf = (g['blocks'][0]['std'][k] for k in ('ts', 'wv', 'cf'))
assert std_ts.shape == FF.shape, (std_ts.shape, FF.shape)
def loc(a, f=np.nanargmax):
    i = np.unravel_index(f(a), a.shape); return f'{a[i]:.4f} at ({FF[i]:.0f}, {PP[i]:.2f})'
print('T  median %.4f  min %s  max %s' % (np.median(std_ts), loc(std_ts, np.nanargmin), loc(std_ts)))
print('   at (1900,10) %.2f ; max flux>=1800 & p>=5: %s' % (std_ts[(FF==1900)&(PP==10.0)][0], loc(np.where((FF>=1800)&(PP>=5), std_ts, np.nan))))
print('   max excluding flux in [1400,1800]: %s' % loc(np.where((FF<1400)|(FF>1800), std_ts, np.nan)))
print('   max flux>=1800: %s' % loc(np.where(FF>=1800, std_ts, np.nan)))
print('CF median %.8f max %s' % (np.median(std_cf), loc(std_cf)))
print('WV median %.4f max %s' % (np.median(std_wv), loc(std_wv)))
# Subsets re-call spread() on the subset, so weighted_std's variance floor is the
# subset's own, which is how the manuscript's six-model numbers were defined.
TSM = ts['models']; names = list(TSM)
six   = spread(ts, {n: TSM[n] for n in names[:6]})
sevn  = spread(ts, {n: TSM[n] for n in names[:7]})
eight = std_ts
band = (FF >= 400) & (FF <= 1200)      # the band both one-dimensional models sampled
for nm, a in (('six', six), ('+HEXTOR', sevn), ('+ExoColumn', eight)):
    print(f'   T {nm:11s} median {np.median(a):.4f}  in-band {np.median(a[band]):.4f}  out {np.median(a[~band]):.4f}')
noHC16 = dict(TSM); p, f, v = TSM['HEXTOR']; keep = ~(np.isclose(f*g['fluxscale'], 1400) & np.isclose(p, 10.0))
noHC16['HEXTOR'] = (p[keep], f[keep], v[keep])
print('   T all eight without HEXTOR Case 16: median %.4f' % np.median(spread(ts, noHC16)))
wv5 = spread(wv, {n: m for n, m in wv['models'].items() if n != 'ExoColumn'})
print('   WV without ExoColumn median %.4f, with %.4f' % (np.median(wv5), np.median(std_wv)))
