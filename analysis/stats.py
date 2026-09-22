import sys, numpy as np
from tables import *
from _paths import FIG_ALL
d = sys.argv[1] if len(sys.argv) > 1 else FIG_ALL
t = all_tables(d)
def rng(dd): v = np.array(list(dd.values())); return v.max()-v.min()
def summ(bc, nmin, f=rng, label=''):
    cs = [c for c in sorted(bc) if len(bc[c]) >= nmin]
    r = {c: f(bc[c]) for c in cs}
    v = np.array(list(r.values()))
    return f"{label} n={len(cs)} cases={cs} range {v.min():.2f}-{v.max():.2f} median {np.median(v):.2f}", r
# --- temperature
bcT = by_case(t['T'])
s, rT = summ(bcT, 3, label='T all'); print(s)
print('   per case:', {c: round(x,1) for c,x in rT.items()})
bcT3 = by_case(t['T'], [m for m in t['T'] if m not in ONE_D])
s, rT3 = summ(bcT3, 3, label='T no1D(>=3 of non-1D)'); print(s)
print('   per case:', {c: round(x,1) for c,x in rT3.items()})
cs_all3 = [c for c in sorted(bcT) if len(bcT[c])>=3]
v = [rng({m:x for m,x in bcT[c].items() if m not in ONE_D}) for c in cs_all3 if len({m for m in bcT[c] if m not in ONE_D})>=2]
print('   T no1D over the same >=3 cases: median %.2f' % np.median(v))
print('   Case 7 others:', {m: x for m,x in bcT[7].items()}, 'excl LFRic range %.1f' % rng({m:x for m,x in bcT[7].items() if m!='LFRic'}))
print('   Case 10:', {m: round(x,2) for m,x in bcT[10].items()})
# HEXTOR departure from median
for c in sorted(t['T']['HEXTOR']):
    others = [x for m,x in bcT[c].items() if m!='HEXTOR']; allm = list(bcT[c].values())
    print(f'   HEXTOR c{c}: vs median(others) {t["T"]["HEXTOR"][c]-np.median(others):+.1f}  vs median(all) {t["T"]["HEXTOR"][c]-np.median(allm):+.1f}')
# --- water vapor (dex)
bcW = by_case(t['W'])
dex = lambda dd: np.log10(max(dd.values())/min(dd.values()))
for nmin in (5,):
    s, rW = summ(bcW, nmin, f=dex, label=f'WV >= {nmin} WV-models'); print(s); print('   ', {c: round(x,2) for c,x in rW.items()})
    csT5 = [c for c in sorted(bcT) if len(bcT[c])>=5]
    r = {c: dex(bcW[c]) for c in csT5 if c in bcW and len(bcW[c])>=2}
    v=np.array(list(r.values())); print(f'   WV at T>=5 cases: {v.min():.2f}-{v.max():.2f} median {np.median(v):.2f}', {c: round(x,2) for c,x in r.items()})
    r2 = {c: dex({m:x for m,x in bcW[c].items() if m!='ExoColumn'}) for c in csT5 if c in bcW}
    v=np.array(list(r2.values())); print(f'   WV no ExoColumn at T>=5 cases: {v.min():.2f}-{v.max():.2f} median {np.median(v):.2f}', {c: round(x,2) for c,x in r2.items()})
    bcW2 = by_case(t['W'], [m for m in t['W'] if m!='ExoColumn'])
    s,_ = summ(bcW2, nmin, f=dex, label=f'   WV no ExoColumn >= {nmin} WV-models'); print(s)
# --- cloud
bcC = by_case(t['C'])
s, rC = summ(bcC, 3, label='CF >=3 CF-models'); print(s); print('   ', {c: round(x,1) for c,x in rC.items()})
for c in (8,10,11): print(f'   CF case {c}:', {m: round(x,1) for m,x in bcC[c].items()})
# --- albedo
bcA = by_case(t['A'])
s, rA = summ(bcA, 3, label='Alb >=3'); print(s); print('   ', {c: round(x,1) for c,x in rA.items()})
bcA6 = by_case(t['A'], [m for m in t['A'] if m not in ONE_D])
s, rA6 = summ(bcA6, 3, label='Alb 6 resolved >=3'); print(s)
bright = {}; dark = {}
for c in sorted(bcA):
    if len(bcA[c]) < 2: continue
    b = max(bcA[c], key=bcA[c].get); k = min(bcA[c], key=bcA[c].get)
    bright[b] = bright.get(b,0)+1; dark[k] = dark.get(k,0)+1
print('   brightest counts', bright); print('   darkest counts', dark)
bright = {}; dark = {}
for c in sorted(bcA6):
    if len(bcA6[c]) < 2: continue
    b = max(bcA6[c], key=bcA6[c].get); k = min(bcA6[c], key=bcA6[c].get)
    bright[b] = bright.get(b,0)+1; dark[k] = dark.get(k,0)+1
print('   6-resolved brightest', bright, 'darkest', dark)
print('   HEXTOR darkest at', sum(1 for c in t['A']['HEXTOR'] if min(bcA[c], key=bcA[c].get)=='HEXTOR'), 'of', len(t['A']['HEXTOR']))
print('   Case 4 albedo range %.1f pp -> %.0f W/m2' % (rng(bcA[4]), rng(bcA[4])/100*1200/4))
