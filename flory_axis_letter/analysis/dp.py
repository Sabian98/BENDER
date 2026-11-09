import sys, numpy as np, pandas as pd
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
def reg(r):
    if r.FCR<0.25: return 'R1 weak'
    if r.FCR<=0.35: return 'R2 Janus'
    if abs(r.NCPR)<=0.35: return 'R3 strong PA'
    return 'R4 polyanion' if r.NCPR<0 else 'R5 polycation'
d['reg']=d.apply(reg,axis=1)
print('occupancy % by taxon'); print((pd.crosstab(d.kingdom,d.reg,normalize='index')*100).round(1).to_string())
for col,lab in [('nu_orig','CAL'),('nu','RECH')]:
    print(f'\nmedian nu by region x taxon ({lab})'); print(d.pivot_table(index='kingdom',columns='reg',values=col,aggfunc='median').round(3).to_string())
# same descriptor box, different nu? within R3 strong polyampholytes: nu by taxon, and within narrow FCR/NCPR/SCD bins
d['fb']=pd.cut(d.FCR,[0,.2,.25,.3,.35,.45,.6,1]); d['nb']=pd.cut(d.NCPR.abs(),[-.01,.05,.1,.2,.35,1])
g=d.groupby(['fb','nb','kingdom'],observed=True).nu_orig.agg(['median','size']).reset_index()
g=g[g['size']>=30]
w=g.pivot_table(index=['fb','nb'],columns='kingdom',values='median').dropna(thresh=4)
w['spread']=w.max(axis=1)-w.min(axis=1)
print('\nCAL median nu in matched FCR x |NCPR| boxes (>=30 per taxon)'); print(w.round(3).to_string())
