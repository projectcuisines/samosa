import os
import glob, re, numpy as np, netCDF4 as nc, warnings
warnings.filterwarnings('ignore')
import ast
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_inst.py')).read()
_tree = ast.parse(_src)
# the constants and helper functions of check_inst.py, without running its checks
exec(compile(ast.Module([n for n in _tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                        or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in ('D', 'S_P', 'P_P', 'rows') for t in n.targets))],
                        type_ignores=[]), 'check_inst.py', 'exec'))
def gw_for(lat):
    x, w = np.polynomial.legendre.leggauss(len(lat)); la = np.rad2deg(np.arcsin(x))
    return w[np.argsort(la)] if np.allclose(np.sort(la), np.sort(lat), atol=0.05) else np.cos(np.deg2rad(lat))
print("model      case  S_prot   4*<inc>    d%    max(inc)/S   | p_prot  p_mod   excess[Pa]  vapor[kg/m2]  implied g")
# ExoPlaSim
for f in sorted(glob.glob('/models/data/samosa/exoplasim/samosa??.nc')):
    n = int(re.search(r'samosa(\d+)', f).group(1)); d = nc.Dataset(f)
    lat = d['lat'][:]; W = gw_for(lat)[:,None]*np.ones(len(d['lon']))
    inc = (d['rst'][:] - d['rsut'][:]).mean(0); ps = d['ps'][:].mean(0)*100; pw = d['prw'][:].mean(0)
    exc = wmean(ps,W) - (P_P[n-1]*1e5+40); v = wmean(pw, W)
    print(f"ExoPlaSim  {n:4d} {S_P[n-1]:6d} {4*wmean(inc,W):9.2f} {100*(4*wmean(inc,W)/S_P[n-1]-1):6.2f}   {inc.max()/S_P[n-1]:8.4f}     | {P_P[n-1]:5.2f} {wmean(ps,W)/1e5:7.4f} {exc:10.0f} {v:12.1f} {exc/v if v>1 else np.nan:9.2f}")
# ExoCAM
for f in sorted(glob.glob('/models/data/samosa/exocam/samosa*.cam.h0.avg.nc'), key=lambda s:int(re.search(r'samosa(\d+)',s).group(1))):
    n = int(re.search(r'samosa(\d+)\.cam', f).group(1)); d = nc.Dataset(f)
    W = d['gw'][:][:,None]*np.ones(len(d['lon'])); top = np.squeeze(d['FDS'][:])[0]
    ps = np.squeeze(d['PS'][:]); tmq = np.squeeze(d['TMQ'][:]); exc = wmean(ps,W) - (P_P[n-1]*1e5+40); v = wmean(tmq,W)
    print(f"ExoCAM     {n:4d} {S_P[n-1]:6d} {4*wmean(top,W):9.2f} {100*(4*wmean(top,W)/S_P[n-1]-1):6.2f}   {top.max()/S_P[n-1]:8.4f}     | {P_P[n-1]:5.2f} {wmean(ps,W)/1e5:7.4f} {exc:10.0f} {v:12.1f} {exc/v if v>1 else np.nan:9.2f}")
# Generic PCM
for n in range(1,17):
    fs = glob.glob(f'/models/data/samosa/genericpcm/OHT_off/case-{n}/*.nc')
    if not fs: continue
    d = nc.Dataset(fs[0]); A = d['surface_area'][:]; inc = d['incoming_stellar_radiation'][:]
    lat = d['latitude'][:]
    pa = np.ma.filled(d['atmospheric_pressure'][:].astype(float), np.nan); k = np.nanargmax(np.nanmean(pa,(1,2)))
    wv = d['water_vapor_column'][:]; exc = wmean(pa[k],A) - (P_P[n-1]*1e5+40); v = wmean(wv,A)
    sa = A.sum(); sph = 4*np.pi*np.average(np.sqrt(sa/(4*np.pi)))**2
    print(f"GenericPCM {n:4d} {S_P[n-1]:6d} {4*wmean(inc,A):9.2f} {100*(4*wmean(inc,A)/S_P[n-1]-1):6.2f}   {np.max(inc)/S_P[n-1]:8.4f}     | {P_P[n-1]:5.2f} {wmean(pa[k],A)/1e5:7.4f} {exc:10.0f} {v:12.1f} {exc/v if v>1 else np.nan:9.2f}")
# ROCKE-3D
for f in sorted(glob.glob('/models/data/samosa/rocke3d/rocke_??q.nc')):
    n = int(re.search(r'rocke_(\d+)q', f).group(1)); d = nc.Dataset(f)
    print(f"ROCKE-3D   {n:4d} {S_P[n-1]:6d} {4*wmean(d['incsw_toa'][:], d['axyp'][:]):9.2f} {'':6s}   {np.max(d['incsw_toa'][:])/S_P[n-1]:8.4f}")
# LFRic pressure excess vs vapor
t = dat_table('/models/data/samosa/lfric/samosa_global_diagnostics_lfric_2026-09-10.txt')
for f in sorted(glob.glob('/models/data/samosa/lfric/lfric_samosa_case??.nc')):
    n = int(re.search(r'case(\d+)', f).group(1)); d = nc.Dataset(f)
    W = np.cos(np.deg2rad(d['lat'][:]))[:,None]*np.ones(len(d['lon'])); p0 = wmean(d['pressure_in_wth'][:][0], W)
    v = wmean(d['tot_col_m_v'][:], W); exc = p0 - (P_P[n-1]*1e5+40)
    print(f"LFRic      {n:4d} {S_P[n-1]:6d} {'-':>9} {'':6s}   {'-':>8}     | {P_P[n-1]:5.2f} {p0/1e5:7.4f} {exc:10.0f} {v:12.1f} {exc/v if v>1 else np.nan:9.2f}   txt Qmass={t[n].get('Qmass')}")
# ExoColumn
t = dat_table('/models/data/samosa/exocolumn/global_output_ExoColumn_a2736.dat')
for n in sorted(t):
    d = nc.Dataset(f'/hugespace/local/research/exocolumn_samosa/cases/a2736/case{n:02d}/iofiles/exocol_out.nc')
    ps = float(np.squeeze(d['ps'][:])); exc = ps - (P_P[n-1]*1e5+40); v = t[n]['Qmass']
    print(f"ExoColumn  {n:4d} {S_P[n-1]:6d} {4*float(d['SWDN'][0]):9.2f} {100*(4*float(d['SWDN'][0])/S_P[n-1]-1):6.2f}   {'-':>8}     | {P_P[n-1]:5.2f} {ps/1e5:7.4f} {exc:10.0f} {v:12.4f} {exc/v if v>1 else np.nan:9.2f}   dat Tglob={t[n]['Tglob']} out ts={float(np.squeeze(d['ts'][:])):.2f}")
