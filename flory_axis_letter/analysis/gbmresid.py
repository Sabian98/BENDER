import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import kruskal, ttest_1samp
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['logN']=np.log(d.length)
F=['FCR','NCPR','SCD','SHD','f_aro','f_P','f_G','hydropathy_avg','f_K','f_R','logN']
AA=[c for c in d.columns if c.startswith('f_') and len(c)==3]
d=d.dropna(subset=F)
for feats,fl in [(F,'11 descriptors'),(sorted(set(F+AA)),'descriptors + 20 aa fractions')]:
  for col,lab in [('nu_orig','CAL'),('nu','RECH')]:
    p=cross_val_predict(HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,random_state=0),d[feats],d[col],cv=KFold(5,shuffle=True,random_state=0))
    r=d[col]-p; d['r']=r
    g=d.groupby('kingdom').r.agg(['mean','sem','size'])
    print(f'== {lab}, {fl}: pooled CV R2 {1-np.var(r)/np.var(d[col]):.3f}; Kruskal p={kruskal(*[x.r for _,x in d.groupby("kingdom")]).pvalue:.1e}')
    print('   ', ', '.join(f'{k} {m:+.4f}±{s:.4f}' for k,(m,s,n) in g.iterrows()))
