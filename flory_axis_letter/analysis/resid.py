import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
d=pd.read_csv(sys.argv[1]); d=d[d.nu_ok_cal.astype(bool)&d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig','nu'])
d['logN']=np.log(d.length)
base='FCR + I(NCPR**2) + SCD + SHD + f_aro + f_P + f_G + hydropathy_avg + f_K + f_R + logN'
for col,lab in [('nu_orig','CAL'),('nu','RECH')]:
    m0=smf.ols(f'{col} ~ {base}',d).fit(); m1=smf.ols(f'{col} ~ {base} + C(kingdom, Treatment("Mammals"))',d).fit()
    print(f'== {lab}: R2 descriptors {m0.rsquared:.3f}, + taxon {m1.rsquared:.3f}; F-test p={m1.compare_f_test(m0)[1]:.1e}')
    for k,v in m1.params.items():
        if 'kingdom' in k: print('   ',k.split('T.')[1].rstrip(']'), f'{v:+.4f}', f'(SE {m1.bse[k]:.4f})')
