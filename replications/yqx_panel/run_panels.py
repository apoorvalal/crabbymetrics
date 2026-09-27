"""Empirical panel attempts. Composed workarounds are not native estimator APIs."""
from pathlib import Path
import json
import time

import crabbymetrics as cm
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
CACHE, OUT = ROOT / '.cache', ROOT / 'results'
rows, failures = [], []


def read(name):
    return pd.read_csv(CACHE / (name + '.csv'))


def record(model, estimate, **kwargs):
    rows.append(dict(model=model, estimate=float(estimate), **kwargs))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'panels-crabby.csv', index=False)


def smoking():
    wide = read('smoking').pivot(index='year', columns='state', values='cigsale').sort_index()
    donors = wide.drop(columns='California')
    pre = wide.index < 1989
    x, y = donors.to_numpy(float), wide.California.to_numpy(float)
    m = cm.SyntheticControl(max_iterations=500)
    m.fit(x[pre], y[pre])
    pred = m.predict(x)
    record('prop99_scm', (y-pred)[~pre].mean(), pre_rmse=m.summary()['pre_rmse'], status='outcome-only SCM; not canonical predictor-V SCM')
    pd.DataFrame({'year':wide.index,'actual':y,'counterfactual':pred,'gap':y-pred}).to_csv(OUT/'prop99-scm-path.csv',index=False)
    # Independent constrained solver on exactly the same squared-error objective.
    xx, yy = x[pre], y[pre]
    loss = lambda w: np.mean((xx@w-yy)**2)
    jac = lambda w: 2*xx.T@(xx@w-yy)/len(yy)
    qp = minimize(loss,np.ones(x.shape[1])/x.shape[1],jac=jac,bounds=[(0,1)]*x.shape[1],constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],method='SLSQP',options={'maxiter':10000,'ftol':1e-11})
    record('prop99_scm_scipy_audit',(y-x@qp.x)[~pre].mean(),pre_rmse=loss(qp.x)**.5,status=str(qp.message),objective_gap=loss(m.summary()['weights'])-loss(qp.x))
    w = np.zeros((wide.shape[1],len(wide))); w[-1,~pre] = 1
    panel = np.column_stack([x,y]).T.copy()
    sd = cm.SyntheticDID(); sd.fit(panel,w); s = sd.summary()
    record('prop99_sdid',s['att'],pre_rmse=s['pre_rmse'],placebo_se=sd.se(method='placebo',replications=100,seed=20260921),status='native; compare R scalar ATT, not displayed gap average')
    pd.DataFrame({'state':donors.columns,'scm':m.summary()['weights'],'sdid':np.asarray(s['unit_weights'])[0,:len(donors.columns)]}).to_csv(OUT/'prop99-unit-weights.csv',index=False)
    for iterations in [50, 5000]:
        long = cm.SyntheticControl(max_iterations=iterations); long.fit(xx, yy)
        ss = long.summary()
        record(f'prop99_scm_iterations_{iterations}',(y-long.predict(x))[~pre].mean(),pre_rmse=ss['pre_rmse'],reported_converged=ss['converged'],objective_gap=loss(ss['weights'])-loss(qp.x),status='solver-budget diagnostic')


def imputation(keys=('gs2020_fe','turnout_fe','hh2019_fe')):
    # Match fect's retained sample and observation mask; no treated outcome enters training.
    for key in keys:
        d = read(key+'-grid'); obs = d.observed.eq(1) & d.y.notna(); train = obs & d.d.eq(0); test = obs & d.d.eq(1)
        xcols = [c for c in d if c.startswith('x')]
        effects = [pd.get_dummies(d[c],drop_first=True,dtype=float) for c in ['unit','time']]
        x = np.column_stack([d[xcols].to_numpy(float), *[e.to_numpy(float) for e in effects]])
        tic = time.monotonic(); m = cm.OLS(); m.fit(x[train],d.y[train].to_numpy(float)); pred = m.predict(x[test])
        record(key+'_explicit_ols',np.mean(d.y[test]-pred),n_train=int(train.sum()),n_treated=int(test.sum()),max_counterfactual_error=float(np.max(np.abs(pred-d.ref_counterfactual[test]))),elapsed_seconds=time.monotonic()-tic,status='explicit FE dummies composed around cm.OLS; no imputation inference')


def famine():
    d = read('mortality'); wide=d.pivot(index='countyid',columns='year',values='mortality').sort_index()
    a=d[d.year==1957].set_index('countyid').loc[wide.index]
    names=['avggrain','nograin','urban','dis_bj','dis_pc','rice','minority','edu','lnpop']
    x=a[names].to_numpy(float); x=(x-x.mean(0))/x.std(0,ddof=1)
    y=(wide.loc[:,1958:1961].mean(axis=1)-wide[1957]).to_numpy(float); g=(a.pczupu>0).to_numpy(float)
    m=cm.AIPW(seed=20260921); m.fit(y,g,x); s=m.summary()
    record('famine_aipw_ridge',s['ate'],se=s['se'],n_train=len(y),status='related ATE: ridge/logit cross-fit, NOT fdid causal forest')


def unsupported():
    # Actual finite panel with reversals, and actual incomplete panel, kept distinct.
    d=read('gs2020'); y=d.pivot(index='district_final',columns='cycle',values='general_sharetotal_A_all'); w=d.pivot(index='district_final',columns='cycle',values='cand_A_all')
    complete=y.notna().all(axis=1)&w.notna().all(axis=1)
    for name, yy, ww in [('gs2020_missing',y,w),('gs2020_complete_reversals',y[complete],w[complete])]:
        for klass in ['MatrixCompletion','SyntheticDID']:
            try:
                m=getattr(cm,klass)(); m.fit(yy.to_numpy(float),ww.to_numpy(float)); status='accepted'
            except Exception as exc:
                status=f'{type(exc).__name__}: {exc}'
            failures.append(dict(dataset=name,api=klass,n_units=len(yy),status=status))
    try:
        cm.InteractiveFixedEffects(rank=2).fit(y[complete].to_numpy(float).T,w[complete].to_numpy(float).T)
    except Exception as exc:
        failures.append(dict(dataset='gs2020_complete_reversals',api='InteractiveFixedEffects.fit(Y,W)',status=f'{type(exc).__name__}: {exc}'))
    (OUT/'capability-probes.json').write_text(json.dumps(failures,indent=2)+'\n')


if __name__=='__main__':
    import sys
    if '--modern-only' in sys.argv:
        rows = pd.read_csv(OUT/'panels-crabby.csv').to_dict('records')
        rows = [r for r in rows if not r['model'].startswith('modern_')]
        imputation(['modern_Bischof_Wagner_2019_AJPS_full','modern_Grumbach_Sahn_2020_APSR_full'])
        raise SystemExit(0)
    for fn in [smoking, imputation, famine, unsupported]:
        try: fn()
        except Exception as exc:
            failures.append(dict(family=fn.__name__,status=f'{type(exc).__name__}: {exc}'))
            print(failures[-1],flush=True)
    (OUT/'capability-probes.json').write_text(json.dumps(failures,indent=2)+'\n')
