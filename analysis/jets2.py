import sys, numpy as np
from recon import MODELS, diagnose

def at_sigma(ubar, sig, target):
    k = int(np.nanargmin(np.abs(sig - target)))
    return ubar[k], sig[k]

def classify(u, lat):
    """Equatorial (single) vs midlatitude (double) upper-level jet."""
    u = np.asarray(u, float)
    if not np.isfinite(u).any(): return '--', np.nan
    ueq  = np.nanmean(u[np.abs(lat) <= 10])
    imid = np.abs(lat) >= 25
    umid = np.nanmax(u[imid]) if imid.any() else np.nan
    return ('SJ' if ueq >= umid else 'DJ'), ueq - umid

print(f"{'model':12s} {'c':>3s}   " + "  ".join(f"{s:>10s}" for s in ['sig=0.5','sig=0.3','sig=0.15','tpause','uppertrop']))
for m,(rd,cases) in MODELS.items():
    for c in cases:
        try: D = diagnose(rd(c))
        except Exception as e:
            print(f'{m:12s} {c:3d} FAIL {e}'); continue
        lat, ubar, sig, ktp = D['lat'], D['ubar'], D['sig'], D['ktp']
        cells=[]
        for tgt in (0.5, 0.3, 0.15):
            u,_ = at_sigma(ubar, sig, tgt); lab,d = classify(u, lat)
            cells.append(f"{lab}{d:+6.1f}")
        lab,d = classify(ubar[ktp], lat); cells.append(f"{lab}{d:+6.1f}")
        band = (sig >= sig[ktp]) & (sig <= 3*sig[ktp])
        lab,d = classify(np.nanmean(ubar[band], axis=0), lat); cells.append(f"{lab}{d:+6.1f}")
        print(f"{m:12s} {c:3d}   " + "  ".join(f"{x:>10s}" for x in cells))
