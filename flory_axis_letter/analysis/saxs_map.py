"""Sequence-only map of nu_eff trained on BENDER, against SAXS for the 63 constructs.
Usage: python saxs_map.py <bio.csv> <data/flory_axis_letter/saxs_eval_63.csv> <outdir>   (writes <outdir>/saxs63_with_map.csv)"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from desc import desc
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
R0=0.1963
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['rg_orig','rg'])
d['logN']=np.log(d.length)
# BENDER rg units: check
print('BENDER rg_orig median',d.rg_orig.median(),' rg median',d.rg.median())
sc=10 if d.rg_orig.median()>10 else 1   # convert A to nm if needed
d['nueff_cal']=np.log(d.rg_orig/sc/R0)/np.log(d.length); d['nueff_rech']=np.log(d.rg/sc/R0)/np.log(d.length)
F=['FCR','NCPR','SCD','f_aro','f_P','f_G','hydropathy_avg','f_K','f_R','logN']
x=pd.read_csv(sys.argv[2]); X=pd.DataFrame([desc(s) for s in x.sequence]); x=pd.concat([x.reset_index(drop=True),X],axis=1)
x['nueff_saxs']=np.log(x.saxs_rg_nm/R0)/np.log(x.n_res)
x['nueff_cal_sim']=np.log(x.cal_rg_nm/R0)/np.log(x.n_res); x['nueff_rech_sim']=np.log(x.rech_rg_nm/R0)/np.log(x.n_res)
print('SAXS constructs',len(x),' check nu_saxs vs recomputed r=',round(pearsonr(x.nu_saxs,x.nueff_saxs)[0],4))
ov=x.calvados2_overlap.astype(str); print('calvados2_overlap values',ov.value_counts().to_dict())
def mae(a,b): return np.mean(np.abs(a-b))
for tgt,lab in [('nueff_cal','CAL-trained map'),('nueff_rech','Recharged-trained map')]:
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,random_state=0).fit(d[F],d[tgt]); x['pred_'+tgt]=m.predict(x[F])
    for sub,mask in [('all',np.ones(len(x),bool)),('not in CAL training',~ov.str.contains('train'))]:
        y=x[mask]
        print(f'{lab:22s} {sub:20s} n={mask.sum():2d}  map->SAXS r={pearsonr(y["pred_"+tgt],y.nueff_saxs)[0]:.2f} MAE={mae(y["pred_"+tgt],y.nueff_saxs):.3f} | direct sim r={pearsonr(y["nueff_cal_sim" if "cal" in tgt else "nueff_rech_sim"],y.nueff_saxs)[0]:.2f} MAE={mae(y["nueff_cal_sim" if "cal" in tgt else "nueff_rech_sim"],y.nueff_saxs):.3f}')
x['aN']=x.NCPR.abs()
print('\nSAXS branch check: rho(FCR, nu_saxs) by |NCPR|')
for lo,hi in [(0,0.1),(0.1,1)]:
    s=x[(x.aN>lo)&(x.aN<=hi)] if lo>0 else x[x.aN<=hi]
    r=spearmanr(s.FCR,s.nueff_saxs); print(f'  |NCPR| {lo}-{hi}: n={len(s)} rho={r.correlation:+.2f} p={r.pvalue:.2f}')
print('  |NCPR| distribution:',x.aN.describe().round(3).to_dict())
x.to_csv(sys.argv[3]+'/saxs63_with_map.csv',index=False)
