"""Night-side static energy flux convergence, prototyped before folding in."""
import warnings, numpy as np, importlib.util
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIG_ALL
spec = importlib.util.spec_from_file_location('er', os.path.join(FIG_ALL, 'extract_regimes.py'))
er = importlib.util.module_from_spec(spec); spec.loader.exec_module(er)
A, G, CP, LV, RDRY = er.A, er.G, er.CP, er.LV, er.RDRY


def at_lon(field, lon, target):
    """Interpolate the last axis (longitude) to a target longitude, with wrap."""
    lon = np.asarray(lon, float)
    order = np.argsort(lon % 360.0)
    lw = (lon % 360.0)[order]
    f  = np.take(field, order, axis=-1)
    lw = np.concatenate([lw - 360.0, lw, lw + 360.0])
    f  = np.concatenate([f, f, f], axis=-1)
    t  = target % 360.0
    k  = np.searchsorted(lw, t) - 1
    w  = (t - lw[k]) / (lw[k+1] - lw[k])
    return f[..., k] * (1 - w) + f[..., k+1] * w


def hydrostatic_z(p, t):
    """Geopotential height from the surface up, for models shipping no z."""
    z = np.zeros_like(t)
    order = np.argsort(-p, axis=0)          # surface (high p) first
    ps, ts = np.take_along_axis(p, order, 0), np.take_along_axis(t, order, 0)
    zs = np.zeros_like(ts)
    for k in range(1, ps.shape[0]):
        tbar = 0.5 * (ts[k] + ts[k-1])
        zs[k] = zs[k-1] + RDRY * tbar / G * np.log(ps[k-1] / ps[k])
    np.put_along_axis(z, order, zs, 0)
    return z


def night_convergence(D):
    lat, lon = D['lat'], D['lon']
    ts_lat = D.get('ts_lat', lat); ts_lon = D.get('ts_lon', lon)
    cw = np.cos(np.radians(ts_lat))

    # Substellar meridian, snapped to the grid convention as in surface_contrasts
    zof = er.area_mean(np.moveaxis(D['ts'], 0, -1), cw)
    lon_hot = ts_lon[int(np.nanargmax(zof))]
    lon_sub = 0.0 if abs(((lon_hot + 180.0) % 360.0) - 180.0) <= 90.0 else 180.0

    p, t, q = D['p'], D['t'], D['q']
    if p.ndim == 1:
        p = np.broadcast_to(p[:, None, None], t.shape)
    z = D['z'] if D['z'] is not None else hydrostatic_z(p, t)
    s = CP * t + G * z + LV * q                        # moist static energy

    flux = []
    for term in (lon_sub + 90.0, lon_sub - 90.0):
        u_m = at_lon(D['u'], lon,    term)             # (lev, lat) on the wind grid
        s_m = at_lon(s,      ts_lon, term)             # (lev, lat) on the scalar grid
        p_m = at_lon(p,      ts_lon, term)
        if not np.array_equal(lat, ts_lat):            # ROCKE-3D B grid
            u_m = np.array([np.interp(ts_lat, lat, np.nan_to_num(u_m[k]))
                            for k in range(u_m.shape[0])])
        col = []
        for j in range(len(ts_lat)):
            pp, uu, ss = p_m[:, j], u_m[:, j], s_m[:, j]
            ok = np.isfinite(pp) & np.isfinite(uu) & np.isfinite(ss)
            if ok.sum() < 2:
                col.append(0.0); continue
            o = np.argsort(pp[ok])
            col.append(np.trapezoid((uu[ok] * ss[ok])[o], pp[ok][o]) / G)
        flux.append(np.array(col))                      # W per metre of boundary

    dphi = np.abs(np.gradient(np.radians(ts_lat)))
    total = A * np.sum((flux[0] - flux[1]) * dphi)      # W into the night side
    dyn = total / (2.0 * np.pi * A ** 2)                # W m^-2 of night hemisphere

    # Steady-state check: what the night side must import to balance its own
    # radiative loss. Independent of the dynamics above.
    off = np.abs(((ts_lon - lon_sub + 180.0) % 360.0) - 180.0)
    night = off > 90.0
    rad = er.area_mean(np.nanmean((D['olr'] - D['asr'])[:, night], axis=-1), cw)
    return dyn, rad


if __name__ == '__main__':
    print(f"{'model':12s} {'case':>4s} {'dyn':>8s} {'rad':>8s} {'ratio':>6s}")
    for name, rd in er.READERS.items():
        if name == 'PlaHab': continue
        for c in er.ACCEPTED[name]:
            dyn, rad = night_convergence(rd(c))
            print(f'{name:12s} {c:4d} {dyn:8.1f} {rad:8.1f} {dyn/rad:6.2f}')
