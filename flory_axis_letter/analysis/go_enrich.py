import sys, re, pandas as pd, numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
src, ann_f, out = sys.argv[1:4]
d=pd.read_csv(src); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
a=pd.read_csv(ann_f,sep='\t').rename(columns={'Entry':'UniProt_ID'})
def terms(r):
    t=set()
    for col,pre in [('Gene Ontology (biological process)','BP'),('Gene Ontology (molecular function)','MF'),('Gene Ontology (cellular component)','CC')]:
        if isinstance(r[col],str):
            t|={f'{pre}: '+re.sub(r' \[GO:\d+\]','',x.strip()) for x in r[col].split(';') if x.strip()}
    if isinstance(r['Keywords'],str):
        t|={'KW: '+x.strip() for x in r['Keywords'].split(';') if x.strip() not in ('Reference proteome','Proteomics identification','3D-structure','Direct protein sequencing','Alternative splicing','Phosphoprotein','Acetylation')}
    return t
a['terms']=a.apply(terms,axis=1)
d=d.merge(a[['UniProt_ID','Protein names','terms']],on='UniProt_ID',how='left')
d['annotated']=d.terms.apply(lambda s: isinstance(s,set) and len(s)>0)
print('sequences',len(d),'with any GO/keyword',d.annotated.sum())
res=[]
for ff,col in [('CALVADOS-2','nu_orig'),('Mpipi-Recharged','nu')]:
    E=d[col]>0.63
    groups={'expanded_basic':E&(d.NCPR>0.1),'expanded_acidic':E&(d.NCPR<-0.1)}
    bg=d[d.annotated]
    for g,mask in groups.items():
        sel=d[mask&d.annotated]
        print(f'{ff} {g}: n={mask.sum()} annotated={len(sel)}', sel.kingdom.value_counts().to_dict())
        cnt={}
        for s in sel.terms:
            for t in s: cnt[t]=cnt.get(t,0)+1
        bgc={}
        for s in bg.terms:
            for t in s: bgc[t]=bgc.get(t,0)+1
        rows=[]
        for t,k in cnt.items():
            if k<5: continue
            K=bgc[t]; n=len(sel); N=len(bg)
            OR,p=fisher_exact([[k,n-k],[K-k,N-n-(K-k)]],alternative='greater')
            rows.append(dict(ff=ff,group=g,term=t,k=k,n=n,frac_group=k/n,frac_bg=K/N,odds=OR,p=p))
        r=pd.DataFrame(rows)
        if len(r): r['q']=multipletests(r.p,method='fdr_bh')[1]; res.append(r)
R=pd.concat(res).sort_values(['ff','group','p']); R.to_csv(out,sep='\t',index=False)
for (ff,g),x in R.groupby(['ff','group']):
    print(f'\n== {ff} {g}'); print(x[x.q<0.05].head(18)[['term','k','n','frac_group','frac_bg','odds','q']].round(3).to_string(index=False))
