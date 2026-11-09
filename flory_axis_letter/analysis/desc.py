import numpy as np
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
def desc(s):
    N=len(s); q=np.array([1 if c in 'KR' else -1 if c in 'DE' else 0 for c in s],float)
    i,j=np.triu_indices(N,1); sep=(j-i).astype(float)
    scd=np.sum(q[i]*q[j]*np.sqrt(sep))/N
    h=np.array([(KD[c]+4.5)/9.0 if c in KD else 0.5 for c in s])
    shd=np.sum((h[i]+h[j])/sep)/N
    f=lambda a: sum(s.count(x) for x in a)/N
    return dict(FCR=f('KRDE'),NCPR=f('KR')-f('DE'),SCD=scd,SHD=shd,f_aro=f('FWY'),f_P=f('P'),f_G=f('G'),
                hydropathy_avg=np.mean([KD.get(c,0) for c in s]),f_K=f('K'),f_R=f('R'),logN=np.log(N))
