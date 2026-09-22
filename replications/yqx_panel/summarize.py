"""Create auditable comparisons and figures; never count saved R results as new fits."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';CACHE=ROOT/'.cache'
FIG=ROOT.parents[1]/'docs/ablations/data/yqx-panel';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})

def save(name):
    plt.tight_layout();plt.savefig(FIG/f'{name}.svg',bbox_inches='tight');plt.close()

c=pd.read_csv(OUT/'linear-crabby.csv');r=pd.read_csv(OUT/'linear-reference.csv')
v=c.merge(r,on=['model','term'],suffixes=('_cm','_r'))
v['coef_error']=(v.estimate-v.reference).abs();v['se_aligned_error']=(v.se-v.aligned_se).abs()
v.to_csv(OUT/'linear-comparison.csv',index=False)
a=pd.read_csv(OUT/'archive-crabby.csv').merge(pd.read_csv(OUT/'archive-reference.csv'),on='name',how='left',suffixes=('_cm','_r'))
a['coef_error']=(a.estimate_cm-a.estimate_r).abs();a['se_aligned_error']=(a.se_cm-a.aligned_se).abs()
p=pd.read_csv(OUT/'reanalysis-published-summary.csv')
lookup={s.strip():row for s,(_,row) in zip(p.Folder_Name,p.iterrows())}
for i,row in a.iterrows():
    name=row['name'].rsplit('_',1)[0]
    if name=='Grumbach_Hill_2022':name='Grumbach_Hill'
    paper=lookup.get(name)
    if paper is not None:
        a.loc[i,'archived_twfe']=paper.twfe
        a.loc[i,'archived_point_error']=abs(row.estimate_cm-paper.twfe)
a.to_csv(OUT/'archive-comparison.csv',index=False)

es=c[c.model.eq('divorce_event')].copy();es['event']=es.term.str.replace('event_m','-',regex=False).str.replace('event_p','',regex=False).astype(int)
source=pd.read_csv(OUT/'divorce-source-event.csv');source['event']=source.term.str.split('::').str[1].astype(int)
ev=es.merge(source,on='event',suffixes=('_cm','_r'));ev.to_csv(OUT/'divorce-event-comparison.csv',index=False)
assert v.coef_error.max()<1e-8
assert (ev.estimate_cm-ev.estimate_r).abs().max()<1e-8
assert a[a.name.ne('Schafer_2022_AJPS')].coef_error.max()<1e-8
assert (OUT/'linear-errors.json').read_text().strip()=='[]'
summary={
 'linear_specifications':int(c.model.nunique()),'linear_coefficients':len(v),'linear_max_coef_error':v.coef_error.max(),
 'archive_studies':len(a),'archive_completed':int(a.status.eq('completed').sum()),
 'archive_source_specs':int(a.scope.eq('source point specification').sum()),
 'archive_max_coef_error_excluding_schafer':a[a.name.ne('Schafer_2022_AJPS')].coef_error.max(),
 'archive_schafer_coef_error':float(a.loc[a.name.eq('Schafer_2022_AJPS'),'coef_error'].iloc[0]),
 'archived_summary_ratio_mean':float((p.fect/p.reported).mean()),
 'archived_summary_ratio_median':float((p.fect/p.reported).median()),
 'archived_summary_numeric_matches_1e6':int((a.archived_point_error<1e-6).sum()),
 'scope':'Partial replication and API audit, NOT all empirical results reproduced.'}
if (OUT/'schafer-reference-tolerance.csv').exists():
    tight=pd.read_csv(OUT/'schafer-reference-tolerance.csv').iloc[-1].coef
    point=a.loc[a.name.eq('Schafer_2022_AJPS'),'estimate_cm'].iloc[0]
    summary['schafer_error_tight_R']=abs(point-tight)
    assert abs(point-tight)<1e-8
(OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

# Central numerical finding: the standalone SCM optimizer misses the same-objective QP.
sc=pd.read_csv(OUT/'prop99-scm-path.csv'); aug=pd.read_csv(OUT/'augsynth-paths.csv')
plt.figure(figsize=(8,4))
plt.plot(sc.year,sc.gap,label='CrabbyMetrics SCM',color='#ce5c35')
for method,label,color in [('scm','R outcome-only SCM','#245c89'),('ridge','R augmented SCM','#658b42'),('ridge_covariates','R augmented + covariates','#785f93')]:
    q=aug[aug.method.eq(method)];plt.plot(q.Time,q.Estimate,label=label,color=color,alpha=.9)
plt.axvline(1988.5,ls='--',color='gray');plt.axhline(0,color='gray',lw=.6)
plt.ylabel('California minus counterfactual (packs/person)');plt.xlabel('Year');plt.legend(fontsize=8);save('prop99')

# Divorce endpoints remain in estimation, omitted only in the display.
q=ev[ev.event.between(-8,15)]
plt.figure(figsize=(8,3.8));plt.errorbar(q.event,q.estimate_cm,1.96*q.se_cm,fmt='o-',ms=3,label='CM (all-FE df)',color='#245c89')
plt.plot(q.event,q.estimate_r,'x',label='Lecture R point estimates',color='#ce5c35');plt.axhline(0,color='gray',lw=.6)
plt.axvline(-.5,color='gray',ls='--');plt.xlabel('Years relative to unilateral divorce');plt.ylabel('Female suicides per million');plt.legend(fontsize=8);save('divorce')

plt.figure(figsize=(8,4))
for method,color in [('unadjusted','#777777'),('adjusted','#245c89'),('interacted','#ce5c35')]:
    q=c[c.model.str.match(f'famine_{method}_\\d+')].copy();q['year']=q.model.str[-4:].astype(int);plt.plot(q.year,q.estimate,'o-',ms=3,label=f'CM {method}',color=color)
fd=pd.read_csv(OUT/'fdid-dynamic-reference.csv');q=fd[fd.method.eq('aipw')];plt.plot(q.year,q.Estimate,'s--',ms=3,label='R causal-forest AIPW (reference)',color='#658b42')
plt.axvspan(1957.5,1961.5,color='#ddd',alpha=.4);plt.axhline(0,color='gray',lw=.6);plt.ylabel('Mortality difference relative to 1957');plt.xlabel('Year');plt.legend(fontsize=8);save('famine')

fig,ax=plt.subplots(1,2,figsize=(9,3.6))
ck=pd.read_csv(CACHE/'ck.csv');counts=[]
for post in [0,1]:
 for nj in [0,1]:
    q=ck[ck.postperiod.eq(post)&ck.nj.eq(nj)].dropna(subset=['wage_st']);idx=np.clip(np.rint((q.wage_st-4.25)/.1).astype(int),0,13);bins=np.bincount(idx,minlength=14);pct=100*bins/bins.sum()
    ax[post].bar(np.arange(14)+(nj-.5)*.38,pct,width=.36,label=['Pennsylvania','New Jersey'][nj],color=['#999','#245c89'][nj])
    counts.extend({'post':post,'nj':nj,'wage_bin':4.25+k*.1,'n':int(n),'pct':float(pc)} for k,(n,pc) in enumerate(zip(bins,pct)))
for i in [0,1]:ax[i].set_title(['February 1992','November 1992'][i]);ax[i].set_xticks(np.arange(0,14,2),[f'{4.25+k*.1:.2f}' for k in range(0,14,2)]);ax[i].set_xlabel('Starting wage ($)')
ax[0].set_ylabel('Percent of stores');ax[1].legend(fontsize=8);save('ck-wages');pd.DataFrame(counts).to_csv(OUT/'ck-wage-bins.csv',index=False)

plt.figure(figsize=(7,4));q=a[a.status.eq('completed')].copy();q=q.sort_values('coef_error')
plt.semilogx(np.maximum(q.coef_error,1e-16),np.arange(len(q)),'.',color='#245c89');plt.xlabel('Absolute CM–R-default coefficient difference');plt.ylabel('47 source-derived regression cases (sorted)');plt.axvline(1e-8,color='gray',ls='--');plt.annotate('Schäfer: tightening R tolerance\nrestores agreement (< 2e-10)',xy=(q.coef_error.max(),46),xytext=(1e-11,36),arrowprops={'arrowstyle':'->'},fontsize=8);save('archive-errors')

# This reconstructs a figure from published outputs, not fresh causal estimates.
plt.figure(figsize=(5.6,4.4));plt.scatter(p.reported/p.reported_se,p.fect/p.reported_se,s=19,color='#245c89',alpha=.7);plt.plot([-14,14],[-14,14],ls='--',color='gray');plt.xlim(-14,14);plt.ylim(-14,14);plt.xlabel('Archived reported estimate / reported SE');plt.ylabel('Archived imputation estimate / reported SE');plt.title('49 saved results — graphical reconstruction only',fontsize=10);save('archive-saved')
