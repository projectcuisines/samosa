import numpy as np, netCDF4 as nc
ROOT='/models/data/samosa/lfric'
flux1 = np.array([500,1900,2400,1200,1500,2100,1600,800,1100,400,900,1500,1600,900,600,1400],float)
rows={}
for line in open(f'{ROOT}/samosa_global_diagnostics_lfric_2026-09-10.txt'):
    if line.startswith('#') or not line.strip(): continue
    c=line.split(); rows[int(c[0])]=c
cols='Sample Inst Pres Tglob Tmax Tmin OLR ASR Fsdn Fnet Qstrat Qmass Ocnfrac Icethick Cldliq Cldice Cldfrac'.split()
print(f"{'case':>4} " + ' '.join(f'{k:>16}' for k in ['Tglob','Tmax','Tmin','OLR','ASR','Fsdn','Fnet','Qmass','Cldliq','Cldice','Cldfrac']))
for case in (1,4,7,8,9,10,11,12,14,15,16):
    d=nc.Dataset(f'{ROOT}/lfric_samosa_case{case:02d}.nc')
    lat=d['lat'][:]; w=np.cos(np.deg2rad(lat))[:,None]*np.ones((1,d.dimensions['lon'].size))
    gm=lambda k: float(np.sum(d[k][:]*w)/np.sum(w))
    ts=d['grid_surface_temperature'][:]
    got=dict(Tglob=gm('grid_surface_temperature'),Tmax=float(ts.max()),Tmin=float(ts.min()),
             OLR=gm('lw_up_toa'),ASR=gm('sw_net_toa'),Fsdn=gm('sw_down_surf'),Fnet=gm('lw_net_surf'),
             Qmass=gm('tot_col_m_v'),Cldliq=gm('tot_col_m_cl'),Cldice=gm('tot_col_m_cf'),Cldfrac=gm('cloud_amount_maxrnd'))
    r=rows[case]
    out=[]
    for k,v in got.items():
        ref=float(r[cols.index(k)])
        out.append(f'{v:8.3g}/{ref:<7.3g}' + ('*' if abs(v-ref)>max(0.006,0.005*abs(ref)) else ' '))
    print(f'{case:>4} '+' '.join(out))
