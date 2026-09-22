import glob, re, numpy as np
root = '/models/data/samosa/plahab/simulations'
# protocol instellation per case
S = {1:500,4:1200,5:1500,7:1600,8:800,9:1100,10:400,11:900,12:1500,13:1600,14:900,15:600,16:1400}
def block(lines, title):
    i = next(k for k,l in enumerate(lines) if l.strip() == title)
    rows = []
    for l in lines[i+1:]:
        p = l.split()
        if len(p) == 21:
            try: rows.append([float(x) for x in p])
            except ValueError: break
        elif rows: break
    return np.array(rows)
def glob1(f, key):
    for l in open(f):
        if l.strip().startswith(key):
            return float(l.split('=')[1])
print(f"{'case':>4} {'S_prot':>6} {'Inst':>7} {'S/4run':>7} {'ASRmap':>7} {'ASRdat':>7} {'OLRmap':>7} {'OLRdat':>7} {'alb_log':>7} {'1-ASRmap/(Srun/4)':>8} {'fig(1-ASRdat/(Sprot/4))':>8}")
for n in sorted(S):
    d = f'{root}/sample{n}'
    log = sorted(glob.glob(d+'/model_*.out'))[0]
    dat = [f for f in glob.glob(d+'/global_*') ][0]
    L = open(log).read().splitlines()
    asr = block(L, 'NET ABSORBED STELLAR FLUX'); olr = block(L, 'OUTGOING LONGWAVE RADIATION')
    lat = np.deg2rad(asr[:,0]); w = np.cos(lat)
    # 36 bins of 5 deg, uniform lon bins -> weight by cos(lat)
    mean = lambda a: (a[:,1:].mean(1)*w).sum()/w.sum()
    ins = alb = None
    for l in L:
        if 'planet average insolation' in l: ins = float(l.split('=')[1].split()[0])
        if 'planet average albedo' in l: alb = float(l.split('=')[1])
    print(f"{n:4d} {S[n]:6d} {glob1(dat,'Inst'):7.1f} {ins:7.2f} {mean(asr):7.2f} {glob1(dat,'ASR'):7.2f} {mean(olr):7.2f} {glob1(dat,'OLR'):7.2f} {100*alb:7.1f} {100*(1-mean(asr)/ins):8.2f} {100*(1-glob1(dat,'ASR')/(S[n]/4)):8.2f}")
