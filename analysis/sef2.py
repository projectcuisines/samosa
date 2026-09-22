import os
"""Direct terminator static-energy flux with a reference-s mass correction."""
import sys, warnings, numpy as np, importlib.util
warnings.filterwarnings('ignore')
S = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('sef', os.path.join(S, 'sef.py'))
sef = importlib.util.module_from_spec(spec); spec.loader.exec_module(sef)
er = sef.er; A, G, CP, LV = er.A, er.G, er.CP, er.LV

def terminator(D):
    lat, lon = D['lat'], D['lon']
    ts_lat = D.get('ts_lat', lat); ts_lon = D.get('ts_lon', lon)
    cw = np.cos(np.radians(ts_lat))
    zof = er.area_mean(np.moveaxis(D['ts'], 0, -1), cw)
    lon_hot = ts_lon[int(np.nanargmax(zof))]
    lon_sub = 0.0 if abs(((lon_hot + 180.0) % 360.0) - 180.0) <= 90.0 else 180.0
    p, t, q = D['p'], D['t'], D['q']
    if p.ndim == 1: p = np.broadcast_to(p[:, None, None], t.shape)
    z = D['z'] if D['z'] is not None else sef.hydrostatic_z(p, t)
    s = CP * t + G * z + LV * q
    dphi = np.abs(np.gradient(np.radians(ts_lat)))
    E, M, SM, MM = 0.0, 0.0, 0.0, 0.0
    for sign, term in ((+1, lon_sub + 90.0), (-1, lon_sub - 90.0)):
        u_m = sef.at_lon(D['u'], lon, term); s_m = sef.at_lon(s, ts_lon, term); p_m = sef.at_lon(p, ts_lon, term)
        if not np.array_equal(lat, ts_lat):
            u_m = np.array([np.interp(ts_lat, lat, np.nan_to_num(u_m[k])) for k in range(u_m.shape[0])])
        for j in range(len(ts_lat)):
            pp, uu, ss = p_m[:, j], u_m[:, j], s_m[:, j]
            ok = np.isfinite(pp) & np.isfinite(uu) & np.isfinite(ss)
            if ok.sum() < 2: continue
            o = np.argsort(pp[ok]); P_, U_, S_ = pp[ok][o], uu[ok][o], ss[ok][o]
            w = A * dphi[j]
            E += sign * w * np.trapezoid(U_ * S_, P_) / G     # W
            M += sign * w * np.trapezoid(U_, P_) / G          # kg/s
            SM += w * np.trapezoid(S_, P_) / G; MM += w * np.trapezoid(np.ones_like(P_), P_) / G
    sref = SM / MM                                             # mass-weighted mean s on the contour
    area = 2.0 * np.pi * A ** 2
    off = np.abs(((ts_lon - lon_sub + 180.0) % 360.0) - 180.0); night = off > 90.0
    rad = er.area_mean(np.nanmean((D['olr'] - D['asr'])[:, night], axis=-1), cw)
    return E / area, (E - sref * M) / area, M, rad

only = sys.argv[1:] and [(m.split(':')[0], int(m.split(':')[1])) for m in sys.argv[1:]]
print(f"{'model':12s} {'case':>4s} {'raw':>8s} {'corr':>8s} {'rad':>8s} {'corr/rad':>8s} {'massflux':>10s}")
for name, rd in er.READERS.items():
    if name == 'PlaHab': continue
    for c in er.ACCEPTED[name]:
        if only and (name, c) not in only: continue
        raw, corr, M, rad = terminator(rd(c))
        flag = '' if abs(corr / rad - 1) <= 0.2 else '  <--'
        print(f'{name:12s} {c:4d} {raw:8.1f} {corr:8.1f} {rad:8.1f} {corr/rad:8.2f} {M:10.2e}{flag}')
