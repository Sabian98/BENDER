import sys, numpy as np, pandas as pd
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['aN']=d.NCPR.abs()
bins=np.arange(0.38,0.76,0.02); rng=np.random.default_rng(0)
def hinge(x,y):
    b=pd.cut(x,bins); m=pd.DataFrame({'y':y,'b':b}).groupby('b',observed=True).y.agg(['mean','size']); m=m[m['size']>=10]
    xs=np.array([i.mid for i in m.index]); ys=m['mean'].values; best=(None,-1)
    for c in np.arange(0.46,0.66,0.005):
        X=np.c_[np.ones_like(xs),np.maximum(xs-c,0)]; beta,res,*_=np.linalg.lstsq(X,ys,rcond=None)
        r2=1-np.sum((X@beta-ys)**2)/np.sum((ys-ys.mean())**2)
        if r2>best[1]: best=(c,r2)
    return best[0]
def boot(g,col,var,n=200):
    v=[hinge(s[col].values,s[var].values) for s in (g.sample(len(g),replace=True,random_state=int(rng.integers(1e9))) for _ in range(n))]
    return np.percentile(v,[50,2.5,97.5])
for col,lab in [('nu_orig','CAL'),('nu','RECH')]:
    print(f'=== {lab}: hinge (onset of rise) along nu, median [95% CI], 200 bootstraps')
    for var in ['SCD','aN']:
        out=[]
        for k,g in list(d.groupby('kingdom'))+[('Pooled',d)]:
            m,lo,hi=boot(g,col,var); out.append(f'{k} {m:.3f} [{lo:.3f},{hi:.3f}]')
        print(f'  {var}: '+'; '.join(out))
    # FCR minimum and f_K/f_R crossover along nu
    mins=[];xs=[]
    for k,g in d.groupby('kingdom'):
        m=g.groupby(pd.cut(g[col],bins),observed=True)[['FCR','f_K','f_R']].mean(); n=g.groupby(pd.cut(g[col],bins),observed=True).size(); m=m[n>=20]
        mins.append(f'{k} {m.FCR.idxmin().mid:.2f}')
        diff=(m.f_K-m.f_R).values; mid=np.array([i.mid for i in m.index]); idx=np.where((diff[:-1]<0)&(diff[1:]>=0))[0]
        xs.append(f'{k} {mid[idx[0]]+0.01 if len(idx) else float("nan"):.2f}')
    print('  FCR minimum:', ', '.join(mins)); print('  fK>fR crossover:', ', '.join(xs))
