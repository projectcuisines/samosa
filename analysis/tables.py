"""Per-case value tables for every model, loaded from the arrays embedded in a
directory of figure scripts. Only the data block of each script is executed."""
import numpy as np

F1 = np.array([500,1900,2400,1200,1500,2100,1600,800,1100,400,900,1500,1600,900,600,1400], float)
P1 = np.array([0.70,7.85,0.21,2.34,0.16,1.83,0.55,6.16,0.70,4.83,0.10,2.98,0.16,1.44,0.43,10.0])
FULL = {'ExoPlaSim': 'plasim', 'ExoCAM': 'exocam', 'ROCKE-3D': 'rocke3d', 'PlaHab': 'plahab'}
PART = {'Generic PCM': 'pcm', 'LFRic': 'lfric', 'HEXTOR': 'hextor', 'ExoColumn': 'exocolumn'}
ONE_D = ('HEXTOR', 'ExoColumn')

def load(path, stop='ANISO = {'):
    # Executes the script only up to its ANISO table, i.e. its data block,
    # without kriging or drawing anything.
    src = open(path).read()
    g = {}
    exec(compile(src[:src.index(stop)], path, 'exec'), g)
    return g

def table(d, script, sentinel_name):
    g = load(f'{d}/{script}')
    sent = g[sentinel_name]; fs = g['fluxscale']
    out = {}
    for m, k in FULL.items():
        if k not in g: continue
        a = np.asarray(g[k], float)
        out[m] = {c+1: a[c] for c in range(16) if a[c] != sent}
    for m, k in PART.items():
        if k not in g: continue
        vals, fl, pr = g[k], g[f'{k}_flux1']*fs, g[f'{k}_pres1']
        dd = {}
        for v, f, p in zip(vals, fl, pr):
            c = [i+1 for i in range(16) if np.isclose(F1[i], f) and np.isclose(P1[i], p)]
            assert len(c) == 1, (m, f, p); dd[c[0]] = float(v)
        out[m] = dd
    return out

def all_tables(d):
    return dict(T=table(d, 'fig_interpolation_temp.py', 'runawaytemp'),
                W=table(d, 'fig_interpolation_watvap.py', 'runaway'),
                C=table(d, 'fig_interpolation_clouds.py', 'runaway'),
                A=table(d, 'fig_interpolation_albedo.py', 'runaway'))

def by_case(t, models=None):
    res = {}
    for m, dd in t.items():
        if models is not None and m not in models: continue
        for c, v in dd.items(): res.setdefault(c, {})[m] = v
    return res
