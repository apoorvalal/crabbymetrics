"""Rerun native code against frozen independent R references, without changing the audit.

Run with the development extension installed and the acquired .cache inputs.
Writes post-fix evidence in results/fixes/. R is not required for this replay.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import time
import numpy as np
import pandas as pd
import crabbymetrics as cm

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.cache'
OUT = ROOT / 'results' / 'fixes'
OUT.mkdir(exist_ok=True)


def linear():
    spec = importlib.util.spec_from_file_location('lecture_linear',ROOT/'run_linear.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT = OUT
    module.main()
    assert not module.ERRORS, module.ERRORS
    c = pd.read_csv(OUT/'linear-crabby.csv')
    r = pd.read_csv(ROOT/'results/linear-reference.csv')
    v = c.merge(r,on=['model','term'])
    assert len(v)==102 and v.model.nunique()==59
    err = float((v.estimate-v.reference).abs().max())
    assert err<1e-8
    v.to_csv(OUT/'linear-comparison.csv',index=False)
    return {'specifications':59,'coefficients':len(v),'max_coef_error':err}


def archive():
    refs = pd.read_csv(ROOT/'results/archive-reference.csv').set_index('name')
    tight = pd.read_csv(ROOT/'results/schafer-reference-tolerance.csv')
    rows = []
    for j in json.loads((ROOT/'results/archive-specifications.json').read_text()):
        if j['status']!='prepared':continue
        d = pd.read_csv(CACHE/f"archive-{j['name']}.csv")
        cols = j['regressors'];cols=[cols] if isinstance(cols,str) else cols
        effects = j['effects'];effects=[effects] if isinstance(effects,str) else effects
        m=cm.FixedEffectsOLS()
        m.fit(d[cols].to_numpy(float),d[effects].to_numpy(np.uint32),d.y.to_numpy(float))
        s=m.summary(vcov='cluster',clusters=d.cluster.to_numpy(np.int64))
        beta=float(s['coef'][cols.index(j['treatment'])]);reference=refs.loc[j['name'],'estimate']
        if j['name']=='Schafer_2022_AJPS':
            reference=float(tight.iloc[-1]['coef'])
        error=abs(beta-reference)
        assert error<1e-8,(j['name'],error)
        rows.append(dict(name=j['name'],scope=j['scope'],estimate=beta,reference=reference,error=error))
        print(j['name'],error,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'archive-comparison.csv',index=False)
    return {'datasets':len(rows),'max_coef_error':max(r['error'] for r in rows)}


def panels():
    rows=[]
    d=pd.read_csv(CACHE/'smoking.csv').pivot(index='year',columns='state',values='cigsale').sort_index()
    x,y=d.drop(columns='California').to_numpy(),d.California.to_numpy();pre=d.index<1989
    m=cm.SyntheticControl();m.fit(x[pre],y[pre]);s=m.summary()
    att=float(np.mean((y-m.predict(x))[~pre]));assert abs(att+19.5136298827203)<1e-5
    assert abs(s['pre_rmse']-1.65640020695525)<1e-7
    rows.append(dict(model='prop99_scm',att=att,pre_rmse=s['pre_rmse'],iterations=s['iterations'],scaled_kkt_residual=s['scaled_kkt_residual']))
    for key in ['gs2020_fe','turnout_fe','hh2019_fe','modern_Bischof_Wagner_2019_AJPS_full','modern_Grumbach_Sahn_2020_APSR_full']:
        d=pd.read_csv(CACHE/(key+'-grid.csv'))
        wide=lambda c:d.pivot(index='unit',columns='time',values=c).to_numpy(float)
        y,treatment=wide('y'),wide('d');obs=wide('observed').astype(bool)&np.isfinite(y)
        cols=[c for c in d if c.startswith('x')];x=np.stack([wide(c) for c in cols],axis=2) if cols else None
        m=cm.FEImputation();start=time.monotonic();m.fit(y,treatment,x,observed=obs);elapsed=time.monotonic()-start
        s=m.summary();mask=s['evaluation_mask'];ref=wide('ref_counterfactual')
        error=float(np.max(np.abs(m.predict()[mask]-ref[mask])));assert error<1e-6
        rows.append(dict(model=key,att=s['att'],reference_att=float(np.mean((y-ref)[mask])),max_surface_error=error,n_train=s['n_train'],n_treated=s['n_treated'],seconds=elapsed))
    pd.DataFrame(rows).to_csv(OUT/'panels.csv',index=False)
    return rows


if __name__=='__main__':
    result={'scope':'Native post-fix replay of previously saved independent R references; no new R estimation.'}
    result['linear']=linear()
    result['panels']=panels()
    result['archive']=archive()
    result['source_base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    result['implementation_sha256']={str(p.relative_to(ROOT.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT.parents[1]/'src/estimators'/name for name in ['synthetic.rs','linear.rs','imputation.rs']]}
    result['extension_sha256']=hashlib.sha256(Path(cm.crabbymetrics.__file__).read_bytes()).hexdigest()
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
