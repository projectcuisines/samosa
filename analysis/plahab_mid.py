import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plahab_asr.py')).read().split("print(f\"{'case'")[0])
print(f"{'case':>4} {'T':>6} {'alb_log':>7} {'ASR_log':>8} {'ASRdat':>8} {'OLR':>8} {'2OLR-ASR_log':>12} {'resid':>6} {'imb_log%':>8} {'imb_dat%':>8}")
for n in sorted(S):
    d = f'{root}/sample{n}'
    log = sorted(glob.glob(d+'/model_*.out'))[0]; dat = glob.glob(d+'/global_*')[0]
    L = open(log).read().splitlines()
    ins = [float(l.split('=')[1].split()[0]) for l in L if 'planet average insolation' in l][0]
    alb = [float(l.split('=')[1]) for l in L if 'planet average albedo' in l][0]
    T = glob1(dat,'Tglob'); A = glob1(dat,'ASR'); O = glob1(dat,'OLR'); Al = ins*(1-alb)
    print(f"{n:4d} {T:6.1f} {100*alb:7.1f} {Al:8.2f} {A:8.2f} {O:8.2f} {2*O-Al:12.2f} {A-(2*O-Al):6.2f} {100*(Al-O)/Al:8.2f} {100*(A-O)/A:8.2f}")
