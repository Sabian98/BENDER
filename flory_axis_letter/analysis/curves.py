import sys, numpy as np, pandas as pd
from scipy.stats import pearsonr
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['aNCPR']=d.NCPR.abs()
g=pd.read_csv(sys.argv[2]); g=g[g.nu_r2_mpipi>=0.99][['UniProt_ID','nu_mpipi']]
D=['FCR','NCPR','aNCPR','SCD','SHD','f_aro','hydropathy_avg','f_K','f_R','f_P','f_G']
def curve(x,col,q=10):
    b=pd.qcut(x[col].rank(method='first'),q,labels=False); return x.groupby(b)[D].mean()
rows=[]
for k,x in d.groupby('kingdom'):
    a=curve(x,'nu_orig'); b=curve(x,'nu')
    rows.append({'taxon':k,**{v:pearsonr(a[v],b[v])[0] for v in D}})
R=pd.DataFrame(rows).set_index('taxon')
a=curve(d,'nu_orig'); b=curve(d,'nu'); R.loc['Pooled']={v:pearsonr(a[v],b[v])[0] for v in D}
print('Pearson r between CALVADOS-2 and Recharged decile curves (descriptor mean per nu decile)')
print(R.round(2).to_string()); print('median over taxon x descriptor:',round(np.nanmedian(R.drop('Pooled').values),3),' min',round(np.nanmin(R.drop('Pooled').values),2))
G=d.merge(g,on='UniProt_ID'); a=curve(G,'nu_orig',8); c=curve(G,'nu_mpipi',8); b=curve(G,'nu',8)
print(f'\nMpipi-GG subset n={len(G)} (8 quantile bins, pooled):')
print(pd.DataFrame({'CAL_vs_GG':{v:pearsonr(a[v],c[v])[0] for v in D},'RECH_vs_GG':{v:pearsonr(b[v],c[v])[0] for v in D}}).round(2).T.to_string())
# FCR minimum location agreement in quantile terms
print('\nFCR minimum decile (CAL, RECH) per taxon:')
for k,x in d.groupby('kingdom'):
    a=curve(x,'nu_orig'); b=curve(x,'nu'); print(' ',k,int(a.FCR.idxmin()),int(b.FCR.idxmin()))
