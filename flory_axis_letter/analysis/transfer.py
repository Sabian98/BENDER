import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import pearsonr
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['logN']=np.log(d.length)
F=['FCR','NCPR','SCD','SHD','f_aro','f_P','f_G','hydropathy_avg','f_K','f_R','logN']
d=d.dropna(subset=F)
print('n',len(d))
K=sorted(d.kingdom.unique())
for col,lab in [('nu_orig','CALVADOS-2'),('nu','Recharged')]:
    for mname,M in [('linear',LinearRegression),('GBM',lambda: HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=0))]:
        m=M() if mname=='linear' else M()
        # within-taxon 5-fold CV
        within={}
        for k in K:
            g=d[d.kingdom==k]; p=cross_val_predict(M() if mname=='linear' else M(),g[F],g[col],cv=KFold(5,shuffle=True,random_state=0)); within[k]=pearsonr(p,g[col])[0]**2
        # train Mammals -> test others; leave-one-taxon-out
        rows=[]
        for k in K:
            tr=d[d.kingdom!=k]; te=d[d.kingdom==k]
            mm=(M() if mname=='linear' else M()).fit(tr[F],tr[col]); p=mm.predict(te[F])
            mt=(M() if mname=='linear' else M()).fit(d[d.kingdom=='Mammals'][F],d[d.kingdom=='Mammals'][col]); pm=mt.predict(te[F])
            rows.append((k,round(within[k],3),round(pearsonr(p,te[col])[0]**2,3),round(pearsonr(pm,te[col])[0]**2,3),round(np.mean(p-te[col]),4)))
        print(f'\n== {lab} {mname}  (R2: within-taxon CV | leave-taxon-out | trained on Mammals only | LOLO bias)')
        for r in rows: print('  ',r)
