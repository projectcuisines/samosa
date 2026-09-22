import warnings, numpy as np, netCDF4
warnings.filterwarnings('ignore')

A     = 6.371e6
G     = 9.81
RGAS  = 8.314
MAIR  = 0.028
OMEGA = 2*np.pi/(15*86400.0)
BETA  = 2*OMEGA/A           # s^-1 m^-1

def aw(x, lat, axis=-1):
    x = np.asarray(x, float)
    w = np.cos(np.radians(lat))
    w = np.broadcast_to(w, x.shape).copy()
    w[~np.isfinite(x)] = 0.0
    return np.nansum(np.nan_to_num(x)*w, axis=axis)/w.sum(axis=axis)

# ---------- readers: return dict(lat, lon, p3d[Pa], u3d, v3d, T3d, ts2d) ----------
def rd_exocam(c):
    with netCDF4.Dataset(f'/models/data/samosa/exocam/samosa{c}.cam.h0.avg.nc') as d:
        U=np.squeeze(d['U'][:]); V=np.squeeze(d['V'][:]); T=np.squeeze(d['T'][:])
        PS=np.squeeze(d['PS'][:]); TS=np.squeeze(d['TS'][:])
        hyam=d['hyam'][:]; hybm=d['hybm'][:]; P0=float(d['P0'][:])
        lat=d['lat'][:]; lon=d['lon'][:]
    p3d = hyam[:,None,None]*P0 + hybm[:,None,None]*PS[None,:,:]
    return dict(lat=np.array(lat),lon=np.array(lon),p=p3d,u=U,v=V,T=T,ts=TS,top='first')

def rd_plasim(c):
    with netCDF4.Dataset(f'/models/data/samosa/exoplasim/samosa{c:02d}.nc') as d:
        ua=np.squeeze(d['ua'][:]); va=np.squeeze(d['va'][:]); ta=np.squeeze(d['ta'][:])
        ps=np.squeeze(d['ps'][:]); ts=np.squeeze(d['ts'][:])
        lev=d['lev'][:]; lat=d['lat'][:]; lon=d['lon'][:]
    psu = np.array(ps)
    if np.nanmax(psu) < 1.0e4: psu = psu*100.0      # hPa -> Pa
    p3d = np.array(lev)[:,None,None]*psu[None,:,:]
    return dict(lat=np.array(lat),lon=np.array(lon),p=p3d,u=ua,v=va,T=ta,ts=np.array(ts),top='first')

def rd_rocke(c):
    with netCDF4.Dataset(f'/models/data/samosa/rocke3d/rocke_{c:02d}q.nc') as d:
        ub=np.ma.filled(d['ub'][:],np.nan).astype(float)
        vb=np.ma.filled(d['vb'][:],np.nan).astype(float)
        tb=np.ma.filled(d['temp'][:],np.nan).astype(float)
        plm=np.array(d['plm'][:]); lat=np.array(d['lat2'][:]); lon=np.array(d['lon2'][:])
        ts=np.ma.filled(d['tsurf'][:],np.nan).astype(float)
    p3d = np.repeat(np.repeat((plm*100.0)[:,None,None],ub.shape[1],1),ub.shape[2],2)
    return dict(lat=lat,lon=lon,p=p3d,u=ub,v=vb,T=tb+273.16,ts=ts+273.16,top='last')

def rd_lfric(c):
    with netCDF4.Dataset(f'/models/data/samosa/lfric/lfric_samosa_case{c:02d}.nc') as d:
        u=np.array(d['u_in_w3'][:]); v=np.array(d['v_in_w3'][:])
        pf=np.array(d['pressure_in_wth'][:]); Tf=np.array(d['temperature'][:])
        hw=np.array(d['height_w3'][:]); hf=np.array(d['height_wth'][:])
        lat=np.array(d['lat'][:]); lon=np.array(d['lon'][:]); ts=np.array(d['grid_surface_temperature'][:])
    # interpolate full-level p,T onto the 41 half levels column by column
    nz,ny,nx = u.shape
    p=np.empty_like(u); T=np.empty_like(u)
    for j in range(ny):
        for i in range(nx):
            p[:,j,i]=np.interp(hw[:,j,i],hf[:,j,i],np.log(pf[:,j,i]))
            T[:,j,i]=np.interp(hw[:,j,i],hf[:,j,i],Tf[:,j,i])
    return dict(lat=lat,lon=lon,p=np.exp(p),u=u,v=v,T=T,ts=ts,top='last')

def rd_pcm(c):
    p=f'/models/data/samosa/genericpcm/OHT_off/case-{c}/SAMOSA_output_file_Generic_PCM_case-{c}_OHT_off.nc'
    with netCDF4.Dataset(p) as d:
        u=d['u_wind_speed'][:].data; v=d['v_wind_speed'][:].data
        T=d['atmospheric_temperature'][:].data; P=d['atmospheric_pressure'][:].data
        lat=np.array(d['latitude'][:]); lon=np.array(d['longitude'][:])
        ts=d['surface_temperature'][:].data
    return dict(lat=lat,lon=lon,p=P,u=u,v=v,T=T,ts=ts,top='last')

MODELS = {
 'ExoCAM':      (rd_exocam, [1,4,7,8,9,10,11,12,14,15,16]),
 'ExoPlaSim':   (rd_plasim, list(range(1,17))),
 'ROCKE-3D':    (rd_rocke,  [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16]),
 'LFRic':       (rd_lfric,  [1,4,7,8,9,10,11,12,14,15,16]),
 'Generic PCM': (rd_pcm,    list(range(1,17))),
}

def diagnose(D):
    lat,lon,p,u,v,T,ts = D['lat'],D['lon'],D['p'],D['u'],D['v'],D['T'],D['ts']
    ksfc = 0 if D['top']=='last' else -1          # index of lowest model level
    tsbar = aw(np.nanmean(ts,axis=-1),lat)
    # surface wind speed, area-weighted rms (global and dayside)
    spd2 = u[ksfc]**2 + v[ksfc]**2
    Urms = np.sqrt(aw(np.nanmean(spd2,axis=-1),lat))
    lonw = np.array(lon) % 360.0
    day  = (lonw<=90)|(lonw>=270)                 # substellar at lon=0
    Uday = np.sqrt(aw(np.nanmean(spd2[:,day],axis=-1),lat))
    H    = RGAS*tsbar/(MAIR*G)
    lamR = np.sqrt(np.sqrt(G*H)/(2*BETA))/A
    LR   = np.pi*np.sqrt(Urms/BETA)/A
    LRd  = np.pi*np.sqrt(Uday/BETA)/A
    # zonal mean u, and tropopause (cold point of the global-mean T profile)
    ubar = np.nanmean(u,axis=-1)                              # (lev,lat)
    pbar = aw(np.nanmean(p,axis=-1),lat)                      # (lev,)
    Tbar = aw(np.nanmean(T,axis=-1),lat)
    psfc = pbar[ksfc]
    sig  = pbar/psfc
    ok   = (sig>0.02)&(sig<0.6)
    kct  = np.arange(len(Tbar))[ok][np.nanargmin(Tbar[ok])]
    return dict(ts=tsbar,U=Urms,Uday=Uday,lamR=lamR,LR=LR,LRd=LRd,
                ubar=ubar,sig=sig,pbar=pbar,psfc=psfc,ktp=kct,ptp=pbar[kct],lat=lat)

if __name__=='__main__':
    print(f"{'model':12s} {'c':>3s} {'Ts':>7s} {'Urms':>6s} {'Uday':>6s} {'lamR/a':>7s} {'LR/a':>6s} {'LRday':>6s} {'psfc_hPa':>9s} {'ptp_hPa':>8s}")
    out={}
    for m,(rd,cases) in MODELS.items():
        for c in cases:
            try: D=diagnose(rd(c))
            except Exception as e:
                print(f'{m:12s} {c:3d}  FAIL {type(e).__name__}: {e}'); continue
            out[(m,c)]=D
            print(f"{m:12s} {c:3d} {D['ts']:7.1f} {D['U']:6.2f} {D['Uday']:6.2f} {D['lamR']:7.2f} {D['LR']:6.2f} {D['LRd']:6.2f} {D['psfc']/100:9.1f} {D['ptp']/100:8.2f}")
    # (the scratch original also np.save'd `out`; nothing reads it)
