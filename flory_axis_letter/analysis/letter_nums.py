import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, chi2_contingency
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['logN']=np.log(d.length); d['aN']=d.NCPR.abs()
F=['FCR','NCPR','SCD','SHD','f_aro','f_P','f_G','hydropathy_avg','f_K','f_R','logN']; d=d.dropna(subset=F)
def reg(r):
    if r.FCR<0.25: return 'R1'
    if r.FCR<=0.35: return 'R2'
    if r.aN<=0.35: return 'R3'
    return 'R4' if r.NCPR<0 else 'R5'
d['reg']=d.apply(reg,axis=1)
ct=pd.crosstab(d.kingdom,d.reg); chi=chi2_contingency(ct); print('occupancy chi2',round(chi[0]),'dof',chi[2],'p',chi[1])
print('pooled occupancy %',(d.reg.value_counts(normalize=True)*100).round(1).to_dict())
for col,lab in [('nu_orig','CAL'),('nu','RECH')]:
    print(f'\n=== {lab}  nu SD {d[col].std():.4f}')
    print(' median nu by region', d.groupby('reg')[col].median().round(3).to_dict(), ' n',d.reg.value_counts().to_dict())
    for lo,hi in [(0,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.35),(0.35,1)]:
        s=d[(d.aN>lo)&(d.aN<=hi)] if lo>0 else d[d.aN<=hi]
        r=spearmanr(s.FCR,s[col]); print(f'  |NCPR| {lo}-{hi}: n={len(s)} rho(FCR,nu)={r.correlation:+.2f} p={r.pvalue:.1e}')
    # FCR minimum location per taxon
    b=np.arange(0.38,0.74,0.02); mins=[]
    for k,g in d.groupby('kingdom'):
        m=g.groupby(pd.cut(g[col],b),observed=True).FCR.agg(['mean','size']); m=m[m['size']>=20]
        mins.append((k,round(m['mean'].idxmin().mid,2),round(m['mean'].min(),3)))
    print('  FCR minimum (nu, FCR):',mins)
    # compact high-FCR chains: |NCPR|
    c=d[(d[col]<0.5)]; print(f'  nu<0.5: n={len(c)} mean FCR {c.FCR.mean():.3f} mean |NCPR| {c.aN.mean():.3f}; nu>0.63: n={(d[col]>0.63).sum()} mean FCR {d[d[col]>0.63].FCR.mean():.3f} |NCPR| {d[d[col]>0.63].aN.mean():.3f}')
    # occupancy explains taxon means
    p=cross_val_predict(HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,random_state=0),d[F],d[col],cv=KFold(5,shuffle=True,random_state=0))
    d['p']=p; g=d.groupby('kingdom').agg(act=(col,'mean'),pred=('p','mean'))
    dev_a=g.act-g.act.mean(); dev_r=g.act-g.pred
    print('  taxon means actual vs predicted:', g.round(4).to_dict('index'))
    print(f'  between-taxon variance in mean nu explained by occupancy: {1-np.sum(dev_r**2)/np.sum(dev_a**2):.3f}; range actual means {g.act.max()-g.act.min():.4f}')
