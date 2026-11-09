import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['logN']=np.log(d.length)
F=['FCR','NCPR','SCD','SHD','f_aro','f_P','f_G','hydropathy_avg','f_K','f_R','logN']
d=d.dropna(subset=F); rng=np.random.default_rng(0)
nM=(d.kingdom=='Mammals').sum(); print('Mammals n',nM)
for col,lab in [('nu_orig','CAL'),('nu','RECH')]:
  for mn,M in [('linear',lambda:LinearRegression()),('GBM',lambda:HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=0))]:
    print(f'== {lab} {mn}: R2 on target | Mammals-trained | size-matched mix of other non-mammal taxa (mean of 5)')
    for k in sorted(set(d.kingdom)-{'Mammals'}):
        te=d[d.kingdom==k]; mam=d[d.kingdom=='Mammals']; pool=d[(d.kingdom!=k)&(d.kingdom!='Mammals')]
        r_m=pearsonr(M().fit(mam[F],mam[col]).predict(te[F]),te[col])[0]**2
        r_o=np.mean([pearsonr(M().fit(s[F],s[col]).predict(te[F]),te[col])[0]**2 for s in [pool.sample(nM,random_state=i) for i in range(5)]])
        print(f'   {k:9s} {r_m:.3f} | {r_o:.3f} | diff {r_o-r_m:+.3f}')
