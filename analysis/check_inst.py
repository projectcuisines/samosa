"""Check every SAMOSA submission's instellation and surface pressure against the protocol."""
import glob, re, os, numpy as np, netCDF4 as nc, warnings
warnings.filterwarnings('ignore')
D = '/models/data/samosa'
S_P = [500,1900,2400,1200,1500,2100,1600,800,1100,400,900,1500,1600,900,600,1400]
P_P = [0.70,7.85,0.21,2.34,0.16,1.83,0.55,6.16,0.70,4.83,0.10,2.98,0.16,1.44,0.43,10.0]
rows = []
def add(model, case, src, S=None, p=None, note=''):
    rows.append((model, case, src, S, p, note))

def wmean(x, w):
    x = np.ma.filled(np.ma.asarray(x, dtype=float), np.nan); ok = np.isfinite(x)
    return np.sum(np.where(ok, x, 0)*w)/np.sum(w*ok)

def dat_header(f):
    txt = open(f).read()
    out = {}
    for k in ('Inst','Pres'):
        m = re.search(rf'^\s*#?\s*{k}\s*=\s*([0-9.eE+-]+)', txt, re.M)
        if m: out[k] = float(m.group(1))
    return out

def dat_table(f):
    """Rows of a SAMOSA column table, keyed by sample number."""
    hdr = None; out = {}
    for l in open(f):
        s = l.strip()
        if s.startswith('#') and 'Sample' in s and 'Inst' in s: hdr = s.lstrip('#').split(); continue
        if hdr and s and not s.startswith('#'):
            v = s.split()
            try: out[int(float(v[0]))] = dict(zip(hdr, [float(x) if re.match(r'^[-+0-9.eEnaN]+$', x) else x for x in v]))
            except ValueError: pass
    return out

# ExoCAM: sol_tsi, 4 * gw-mean FDS at the model top, gw-mean PS
for f in sorted(glob.glob(f'{D}/exocam/samosa*.cam.h0.avg.nc')):
    n = int(re.search(r'samosa(\d+)\.cam', f).group(1)); d = nc.Dataset(f)
    W = d['gw'][:][:,None]*np.ones(len(d['lon']))
    add('ExoCAM', n, 'sol_tsi', float(d['sol_tsi'][:].mean()))
    add('ExoCAM', n, '4*<FDS top>', 4*wmean(np.squeeze(d['FDS'][:])[0], W), wmean(np.squeeze(d['PS'][:]), W)/1e5)

# ExoPlaSim: 4 * <rst + rsut>, <ps>; Gaussian weights
for f in sorted(glob.glob(f'{D}/exoplasim/samosa??.nc')):
    n = int(re.search(r'samosa(\d+)\.nc', f).group(1)); d = nc.Dataset(f)
    lat = d['lat'][:]; x, gw = np.polynomial.legendre.leggauss(len(lat))
    W = gw[np.argsort(np.arcsin(x))][:,None] if np.allclose(np.sort(np.rad2deg(np.arcsin(x))), np.sort(lat), atol=0.05) else np.cos(np.deg2rad(lat))[:,None]
    W = W*np.ones(len(d['lon']))
    inc = (d['rst'][:] + d['rsut'][:]).mean(0)
    add('ExoPlaSim', n, '4*<rst+rsut>', 4*wmean(inc, W), wmean(d['ps'][:].mean(0), W)/1e3)

# ROCKE-3D: incsw_toa (hemis global and axyp-weighted), prsurf
for f in sorted(glob.glob(f'{D}/rocke3d/rocke_??q.nc')):
    n = int(re.search(r'rocke_(\d+)q', f).group(1)); d = nc.Dataset(f)
    A = d['axyp'][:]
    add('ROCKE-3D', n, '4*<incsw_toa>', 4*wmean(d['incsw_toa'][:], A), wmean(d['prsurf'][:], A)/1e3,
        f"hemis[2]*4={4*float(d['incsw_toa_hemis'][2]):.2f}")

# Generic PCM: .dat header, 4 * <incoming_stellar_radiation>, lowest-level pressure
for dd in sorted(glob.glob(f'{D}/genericpcm/OHT_off/case-*')):
    n = int(dd.split('-')[-1])
    dats = glob.glob(dd+'/*.dat'); ncs = glob.glob(dd+'/*.nc')
    if dats:
        h = dat_header(dats[0]); t = dat_table(dats[0])
        add('Generic PCM', n, '.dat header', h.get('Inst'), h.get('Pres'), f"table Sample={list(t)}")
    if ncs:
        d = nc.Dataset(ncs[0]); A = d['surface_area'][:]
        pa = np.ma.filled(d['atmospheric_pressure'][:].astype(float), np.nan)
        k = np.nanargmax(np.nanmean(pa, axis=(1,2)))   # surface-most level
        add('Generic PCM', n, '4*<incoming_stellar>', 4*wmean(d['incoming_stellar_radiation'][:], A), wmean(pa[k], A)/1e5)

# LFRic: .txt table, title attribute, lowest-level pressure; no incident SW archived
t = dat_table(f'{D}/lfric/samosa_global_diagnostics_lfric_2026-09-10.txt')
for n, r in sorted(t.items()): add('LFRic', n, '.txt table', r.get('Inst'), r.get('Pres'))
for f in sorted(glob.glob(f'{D}/lfric/lfric_samosa_case??.nc')):
    n = int(re.search(r'case(\d+)', f).group(1)); d = nc.Dataset(f)
    W = np.cos(np.deg2rad(d['lat'][:]))[:,None]*np.ones(len(d['lon']))
    p0 = d['pressure_in_wth'][:][0]
    snet_max = float(np.max(d['sw_net_toa'][:]))
    add('LFRic', n, 'nc: p(level0), max sw_net_toa', None, wmean(p0, W)/1e5,
        f"title='{d.title}'  max sw_net_toa={snet_max:.1f} ({snet_max/S_P[n-1]:.3f} of S_prot)")

# HEXTOR: namelist solarcon/pg0 for all 16, .dat table, belt incident flux
for f in sorted(glob.glob('/models/hextor/samosa/case_??_warm/input.nml')):
    n = int(re.search(r'case_(\d+)_', f).group(1)); txt = open(f).read()
    sc = float(re.search(r'^\s*solarcon\s*=\s*([0-9.]+)', txt, re.M).group(1))
    pg = float(re.search(r'^\s*pg0\s*=\s*([0-9.]+)', txt, re.M).group(1))
    cold = open(f.replace('_warm','_cold')).read()
    sc2 = float(re.search(r'^\s*solarcon\s*=\s*([0-9.]+)', cold, re.M).group(1))
    add('HEXTOR', n, 'input.nml solarcon/pg0', sc, pg, '' if sc2 == sc else f'COLD solarcon {sc2}')
for n, r in sorted(dat_table(f'{D}/hextor/global_output_HEXTOR.dat').items()):
    add('HEXTOR', n, '.dat table', r['Inst'], r['Pres'], f"Fsdn*4={4*r['Fsdn']:.2f}")
for f in sorted(glob.glob(f'{D}/hextor/zonal_output_HEXTOR_case??.dat')):
    n = int(re.search(r'case(\d+)', f).group(1))
    if n > 16:
        continue      # the archive also holds Cases 17-64 since 2026-09-22
    z = np.array([[float(x) for x in l.split()] for l in open(f) if l.strip() and not l.startswith('#')])
    th, alb, asr = z[:,0], z[:,4], z[:,6]; w = np.sin(np.deg2rad(th))
    inc = np.where(alb < 1, asr/(1-alb), 0.0)
    add('HEXTOR', n, 'belts 4*<ASR/(1-alb)>', 4*np.sum(inc*w)/np.sum(w))

# ExoColumn: .dat table, run-dir namelist, TOA SWDN and ps from the output file
for n, r in sorted(dat_table(f'{D}/exocolumn/global_output_ExoColumn_a2736.dat').items()):
    add('ExoColumn', n, '.dat table', r['Inst'], r['Pres'])
for dd in sorted(glob.glob('/hugespace/local/research/exocolumn_samosa/cases/a2736/case??')):
    n = int(dd[-2:])
    if n > 16:
        continue      # Cases 17-64 are checked by figures/allcases/fig_interpolation_temp_1d.py
    txt = open(dd+'/exocol_config.nml').read()
    ms = float(re.search(r'msdist\s*=\s*([0-9.eE+-]+)', txt).group(1))
    ps = float(re.search(r'^\s*ps\s*=\s*([0-9.eE+-]+)', txt, re.M).group(1))
    add('ExoColumn', n, 'nml 1360/msdist, ps', 1360/ms, ps/1e5)
    o = dd+'/iofiles/exocol_out.nc'
    if os.path.exists(o):
        d = nc.Dataset(o)
        add('ExoColumn', n, 'out 4*SWDN[top], ps', 4*float(d['SWDN'][0]), float(np.squeeze(d['ps'][:]))/1e5)

# PlaHab, for completeness
for dd in sorted(glob.glob(f'{D}/plahab/simulations/sample*')):
    n = int(dd.split('sample')[-1]); f = glob.glob(dd+'/global_*')[0]
    g = {l.split('=')[0].strip(): l.split('=')[1].strip() for l in open(f) if '=' in l}
    add('PlaHab', n, '.dat', float(g['Inst']), float(g['Pres']))

print(f"{'model':12s} {'case':>4} {'source':30s} {'S_prot':>6} {'S_model':>9} {'dS%':>7}   {'p_prot':>6} {'p_model':>8} {'p/p_prot':>8}  note")
for m, n, src, S, p, note in rows:
    sp, pp = S_P[n-1], P_P[n-1]
    flag = ''
    if S is not None and abs(S/sp-1) > 0.01: flag += ' <<S'
    if p is not None and not (0.97 < p/pp < 1.15): flag += ' <<p'
    Ss = f'{S:9.2f} {100*(S/sp-1):7.2f}' if S is not None else f'{"-":>9} {"":>7}'
    ps = f'{p:8.4f} {p/pp:8.4f}' if p is not None else f'{"-":>8} {"":>8}'
    print(f"{m:12s} {n:4d} {src:30s} {sp:6d} {Ss}   {pp:6.2f} {ps}  {note}{flag}")
