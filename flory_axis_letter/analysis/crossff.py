import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['logN']=np.log(d.length); d['aN']=d.NCPR.abs()
F=['FCR','NCPR','SCD','SHD','f_aro','f_P','f_G','hydropathy_avg','f_K','f_R','logN']; d=d.dropna(subset=F)
g=pd.read_csv(sys.argv[2]); g=g[g.nu_r2_mpipi>=0.99][['UniProt_ID','nu_mpipi']]
G=d.merge(g,on='UniProt_ID'); print('GG overlap with both-ok set',len(G))
print('-- Mpipi-GG two-branch check')
for lo,hi in [(0,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.35),(0.35,1)]:
    s=G[(G.aN>lo)&(G.aN<=hi)] if lo>0 else G[G.aN<=hi]
    r=spearmanr(s.FCR,s.nu_mpipi); r2=spearmanr(s.FCR,s.nu_orig)
    print(f'  |NCPR| {lo}-{hi}: n={len(s)} GG rho={r.correlation:+.2f} (p={r.pvalue:.1e}) | CAL same set {r2.correlation:+.2f}')
c=G[G.nu_mpipi<0.5]; e=G[G.nu_mpipi>0.63]; print(f'  GG nu<0.5 n={len(c)} FCR {c.FCR.mean():.3f} |NCPR| {c.aN.mean():.3f}; nu>0.63 n={len(e)} FCR {e.FCR.mean():.3f} |NCPR| {e.aN.mean():.3f}')
print('-- cross-force-field transfer of the map (5-fold, train on X, test on Y for held-out sequences)')
kf=KFold(5,shuffle=True,random_state=0)
for a,b in [('nu_orig','nu'),('nu','nu_orig')]:
    pa=np.zeros(len(d))
    for tr,te in kf.split(d):
        m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,random_state=0).fit(d.iloc[tr][F],d.iloc[tr][a]); pa[te]=m.predict(d.iloc[te][F])
    print(f'  train {a} -> predict {b}: r^2={pearsonr(pa,d[b])[0]**2:.3f} (own-FF CV for {b} given in letter); r^2 vs {a} itself {pearsonr(pa,d[a])[0]**2:.3f}')
    d['p_'+a]=pa
# GG: CAL-trained map predicts GG nu?
m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,random_state=0).fit(d[~d.UniProt_ID.isin(G.UniProt_ID)][F],d[~d.UniProt_ID.isin(G.UniProt_ID)]['nu_orig'])
pg=m.predict(G[F]); print(f'  CAL-trained map (GG sequences held out) -> GG nu r^2={pearsonr(pg,G.nu_mpipi)[0]**2:.3f}; -> CAL nu same seqs {pearsonr(pg,G.nu_orig)[0]**2:.3f}; raw CAL vs GG nu r^2={pearsonr(G.nu_orig,G.nu_mpipi)[0]**2:.3f}')
# taxon means across FFs
L=d.groupby('kingdom')[['nu_orig','nu']].mean(); LG=G.groupby('kingdom')[['nu_orig','nu_mpipi']].mean()
print('-- taxon mean nu'); print(L.join(LG.rename(columns={'nu_orig':'cal_GGsubset'})).round(4).to_string())
print('  Spearman of taxon means CAL vs RECH',round(spearmanr(L.nu_orig,L.nu).correlation,2),' CAL vs GG (subset)',round(spearmanr(LG.nu_orig,LG.nu_mpipi).correlation,2))
