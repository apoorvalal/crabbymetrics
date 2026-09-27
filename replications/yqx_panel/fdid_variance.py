"""Compose fdid's implemented OLS2 variance using CM's HC3 and a population correction."""
from pathlib import Path
import numpy as np
import pandas as pd
import crabbymetrics as cm
root=Path(__file__).resolve().parent
d=pd.read_csv(root/'.cache/mortality.csv');wide=d.pivot(index='countyid',columns='year',values='mortality')
a=d[d.year.eq(1957)].set_index('countyid').loc[wide.index]
x=a[['avggrain','nograin','urban','dis_bj','dis_pc','rice','minority','edu','lnpop']].to_numpy(float)
x=(x-x.mean(0))/x.std(0,ddof=1);g=(a.pczupu>0).to_numpy(float)
y=(wide.loc[:,1958:1961].mean(axis=1)-wide[1957]).to_numpy(float)
design=np.column_stack([g,x,g[:,None]*x]);m=cm.OLS();m.fit(design,y);s=m.summary(vcov='hc3')
gamma=np.asarray(s['coef'])[1+x.shape[1]:];extra=float(gamma@np.cov(x,rowvar=False)@gamma/len(y))
reference=pd.read_csv(root/'results/fdid-reference.csv').query("method=='ols2'").iloc[0]
se=float(np.sqrt(s['coef_se'][0]**2+extra))
pd.DataFrame([{'estimate':s['coef'][0],'hc3_se':s['coef_se'][0],'population_variance_addition':extra,'corrected_se':se,'reference_se':reference['Std.Error'],'error':abs(se-reference['Std.Error'])}]).to_csv(root/'results/fdid-ols2-composed-variance.csv',index=False)
assert abs(se-reference['Std.Error'])<1e-8
