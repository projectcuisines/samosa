import os, runpy, contextlib, io, numpy as np, warnings
from _paths import FIG_ALL, scratch_copy
os.chdir(scratch_copy(FIG_ALL))
warnings.filterwarnings('ignore')
src = open('fit_anisotropy.py').read()
g = {}; exec(src[:src.index('fitted = {}')], g)
SC, loo, resol, MINR, CAP = g['SCALINGS'], g['loo_rmse'], g['resolution'], g['MIN_RESOLUTION'], g['CAP']
with contextlib.redirect_stdout(io.StringIO()):
    gt = runpy.run_path('crossval_variogram.py'); ga = runpy.run_path('crossval_variogram_albedo.py')
fs = 100
cases = np.array([1,4,7,8,9,10,11,12,14,15,16])
sets = {'orig7 (no 7, no new)': [1,4,9,12,14,15,16], 'with7 (8 cases)': [1,4,7,9,12,14,15,16],
        'new, no 7 (10)': [1,4,8,9,10,11,12,14,15,16], 'all 11': list(cases)}
def pick(rm, rs):
    ok = [i for i in range(len(SC)) if rs[i] >= MINR and np.isfinite(rm[i])]
    if not ok: return None
    b = min(ok, key=lambda i: rm[i]); tol = [i for i in ok if rm[i] <= rm[b]*1.02]
    return CAP if b == len(SC)-1 else SC[min(tol)]
for var, gg, key, tf in (('T', gt, 'lfric', lambda v: v), ('albedo', ga, 'lfric', gg_logit := None)):
    pass
logit = ga['logit']
for var, gg, tf in (('temperature', gt, lambda v: v), ('albedo', ga, logit)):
    vals_all = np.asarray(gg['lfric']); pr_all = np.asarray(gg['lfric_pres1']); fl_all = np.asarray(gg['lfric_flux1'])
    assert len(vals_all) == 11
    for name, cs in sets.items():
        m = np.isin(cases, cs)
        pr, fl, v = pr_all[m], fl_all[m], tf(vals_all[m])
        rm = [loo(gg['norm_pres'], gg['norm_flux'], pr, fl, v, s) for s in SC]
        rs = [resol(gg['norm_pres'], gg['norm_flux'], pr, fl, v, s, (gg['pn2'], gg['flux'])) for s in SC]
        p = pick(rm, rs)
        cell = ' '.join(f'{x:6.2f}{"" if r>=MINR else "~"}' for x, r in zip(rm, rs))
        at15 = rm[SC.index(15)]
        print(f'{var:11s} {name:22s} pick={p}  LOO@pick={rm[SC.index(p)] if p else float("nan"):.2f}  LOO@15={at15:.2f}\n      {cell}')
