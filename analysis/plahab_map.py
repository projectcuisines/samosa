import glob, numpy as np, sys
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plahab_asr.py')).read().split("print(f\"{'case'")[0])
lon = np.array([-171.,-153.,-135.,-117.,-99.,-81.,-63.,-45.,-27.,-9.,9.,27.,45.,63.,81.,99.,117.,135.,153.,171.])
for n in map(int, sys.argv[1:]):
    d = f'{root}/sample{n}'; log = sorted(glob.glob(d+'/model_*.out'))[0]
    L = open(log).read().splitlines(); asr = block(L, 'NET ABSORBED STELLAR FLUX')
    lat = asr[:,0]; A = asr[:,1:]
    Srun = [float(l.split('=')[1].split()[0]) for l in L if 'planet average insolation' in l][0]*4
    I = Srun*np.clip(np.cos(np.deg2rad(lat))[:,None]*np.cos(np.deg2rad(lon))[None,:],0,None)
    w = np.cos(np.deg2rad(lat))[:,None]*np.ones_like(A)
    print(f"Case {n}  S_run={Srun:.0f}")
    print("  equator row lon:", lon[5:15])
    k = np.argmin(abs(lat-2))
    print("  ASR  :", np.round(A[k,5:15],1)); print("  I    :", np.round(I[k,5:15],1))
    day = I>0
    print(f"  nightside ASR max {A[~day].max():.2f};  mean I (cos-weighted) {(I*w).sum()/w.sum():.2f};  mean ASR {(A*w).sum()/w.sum():.2f}; insol-weighted albedo from map {100*(1-(A*w).sum()/(I*w).sum()):.1f}%")
