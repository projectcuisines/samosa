import numpy as np, re, sys, os, io, contextlib, runpy
from _paths import FIG_ALL, scratch_copy
# Parses the table extract_regimes.py prints. Pass a saved copy of that output,
# or leave it off and the extractor is run here (it reads the archive).
if len(sys.argv) > 1:
    text = open(sys.argv[1]).read()
else:
    work = scratch_copy(FIG_ALL); os.chdir(work); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(os.path.join(work, 'extract_regimes.py'), run_name='__main__')
    text = buf.getvalue()
rows = []
for line in text.splitlines():
    m = re.match(r'^(ExoCAM|ExoPlaSim|ROCKE-3D|LFRic|Generic PCM)\s+(\d+)\s+(.*)$', line)
    if not m: continue
    f = m.group(3).split()
    rows.append(dict(model=m.group(1), case=int(m.group(2)), urms=float(f[4]), lamr=float(f[5]), lr=float(f[6]),
                     jet=f[7], jetlat=float(f[9]), umax=float(f[10]), tsmin=float(f[11]), conv=float(f[12]), ratio=float(f[13])))
NEW = {('LFRic', 8), ('LFRic', 10), ('LFRic', 11)}
for tag, R in (('BEFORE', [r for r in rows if (r['model'], r['case']) not in NEW]), ('AFTER', rows)):
    print('=====', tag, len(R), 'model-cases')
    bc = {}
    for r in R: bc.setdefault(r['case'], []).append(r)
    multi = [c for c in sorted(bc) if len(bc[c]) >= 3]
    two = [c for c in sorted(bc) if len(bc[c]) >= 2]
    rat = {c: max(r['urms'] for r in bc[c]) / min(r['urms'] for r in bc[c]) for c in two}
    rat3 = {c: rat[c] for c in multi}
    print(' Urms max/min: median(>=2) %.2f median(>=3) %.2f max %.2f at %s' % (np.median(list(rat.values())), np.median(list(rat3.values())), max(rat.values()), max(rat, key=rat.get)))
    shared = [c for c in sorted(bc) if len(bc[c]) == 5]
    print(' all-five cases', shared)
    for c in shared:
        w = max(bc[c], key=lambda r: r['urms']); k = min(bc[c], key=lambda r: r['urms'])
        print(f'   c{c}: windiest {w["model"]} {w["urms"]}, calmest {k["model"]} {k["urms"]}')
    wind = {c: (max(bc[c], key=lambda r: r['urms'])['model']) for c in two}
    print(' windiest at >=2 cases:', {m: sum(1 for v in wind.values() if v == m) for m in set(wind.values())}, ' non-ExoPlaSim at', {c: v for c, v in wind.items() if v != 'ExoPlaSim'})
    calm = {c: (min(bc[c], key=lambda r: r['urms'])['model']) for c in two}
    print(' calmest at >=2 cases:', {m: sum(1 for v in calm.values() if v == m) for m in set(calm.values())}, {c: v for c, v in calm.items() if v not in ('LFRic', 'Generic PCM')})
    unan = [c for c in multi if len({r['jet'] for r in bc[c]}) == 1]
    print(' jet unanimous at %d of %d multi (>=3) cases; disagreements %s' % (len(unan), len(multi), [c for c in multi if c not in unan]))
    for c in multi:
        if c not in unan: print(f'     c{c}: ' + ', '.join(f'{r["model"]}:{r["jet"]}' for r in bc[c]))
    hit = sum(1 for r in R if (r['lr'] < 1) == (r['jet'] == 'DJ'))
    print(' Rhines hit %d of %d (%.0f%%), fails %d' % (hit, len(R), 100 * hit / len(R), len(R) - hit))
    fails = sorted((r['case'], r['model']) for r in R if (r['lr'] < 1) != (r['jet'] == 'DJ'))
    print('   fails at', fails)
    conv = {c: (max(r['conv'] for r in bc[c]) - min(r['conv'] for r in bc[c])) / np.mean([r['conv'] for r in bc[c]]) for c in two}
    conv3 = {c: conv[c] for c in multi}
    print(' conv spread/mean: median(>=2) %.0f%% median(>=3) %.0f%%' % (100 * np.median(list(conv.values())), 100 * np.median(list(conv3.values()))), {c: round(100 * v) for c, v in conv.items()})
    for c in (1, 10): print(f'   conv c{c}:', sorted(round(r['conv'], 1) for r in bc[c]))
    jl = sorted((r['jetlat'], r['model'], r['case'], r['jet']) for r in R)
    print(' jetlat values in (3.9, 50):', [(x, m, c, j) for x, m, c, j in jl if 3.9 < x < 50])
    sj = [r['umax'] for r in R if r['jet'] == 'SJ']
    print(' SJ umax: n=%d, below 20: %s, above 31.2: %s' % (len(sj), sorted(x for x in sj if x < 20), sorted(x for x in sj if x > 31.2)))
    lrv = sorted((r['lr'], r['model'], r['case']) for r in R); lam = [r['lamr'] for r in R]
    print(' L_R/a %.2f (%s %d) to %.2f (%s %d); lamR/a %.2f-%.2f' % (lrv[0][0], lrv[0][1], lrv[0][2], lrv[-1][0], lrv[-1][1], lrv[-1][2], min(lam), max(lam)))
