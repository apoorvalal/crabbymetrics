"""Exercise public CrabbyMetrics APIs on lecture specifications, without changing estimators."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import subprocess
import time

import crabbymetrics as cm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
ROWS, JOBS, ERRORS = [], [], []


def read(name):
    return pd.read_csv(CACHE / f"{name}.csv")


def regression(name, data, outcome, regressors, effects=(), cluster=None, formula=None, keep=None):
    """Materialize exactly the sample/design used on both sides of each comparison."""
    cols = list(dict.fromkeys([outcome, *regressors, *effects, *([cluster] if cluster else [])]))
    d = data.dropna(subset=cols).copy()
    x = d[list(regressors)].to_numpy(dtype=float)
    y = d[outcome].to_numpy(dtype=float)
    cluster_codes = None if cluster is None else pd.factorize(d[cluster], sort=True)[0].astype(np.int64)
    model = cm.FixedEffectsOLS() if effects else cm.OLS()
    tic = time.monotonic()
    try:
        if effects:
            fe = np.column_stack([pd.factorize(d[c], sort=True)[0] for c in effects]).astype(np.uint32)
            model.fit(x, fe, y)
        else:
            model.fit(x, y)
        summary = model.summary(vcov="cluster", clusters=cluster_codes) if cluster else model.summary(vcov="hc1")
        b, se = np.asarray(summary["coef"]), np.asarray(summary["coef_se"])
        # OLS appends an intercept; FixedEffectsOLS reports slopes only.
        for j, term in enumerate(regressors):
            if keep is None or term in keep:
                ROWS.append(dict(model=name, term=term, estimate=float(b[j]), se=float(se[j]), n=len(d),
                                 clusters=None if cluster is None else int(d[cluster].nunique()),
                                 residual_df=float(summary.get("residual_df", len(d)-len(b))),
                                 elapsed_seconds=time.monotonic()-tic))
        if formula is None:
            formula = outcome + " ~ " + " + ".join(regressors)
            if effects:
                formula += " | " + " + ".join(effects)
        r_cols = list(dict.fromkeys([c for c in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", formula) if c in d] + ([cluster] if cluster else [])))
        d[r_cols].to_csv(CACHE / f"design-{name}.csv", index=False)
        JOBS.append(dict(name=name, formula=formula, cluster=cluster, keep=list(keep or regressors)))
        print(name, len(d), [(r['term'], round(r['estimate'],6)) for r in ROWS if r['model']==name][:6], flush=True)
        return model, summary, d
    except Exception as exc:
        ERRORS.append(dict(model=name, exception=type(exc).__name__, message=str(exc)))
        print(name, type(exc).__name__, str(exc), flush=True)
        return None


def swiss():
    d = read("swiss").sort_values(["muniID", "year"])
    d["time"] = d.year - d.year.min()
    regression("swiss_pooled", d, "nat_rate", ["repdem"], cluster="muniID")
    regression("swiss_unit_fe", d, "nat_rate", ["repdem"], ["muniID"], "muniID")
    regression("swiss_twfe", d, "nat_rate", ["repdem"], ["muniID", "year"], "muniID")
    regression("swiss_common_trend", d, "nat_rate", ["repdem", "time"], ["muniID"], "muniID")
    # Explicit slope columns are a harness workaround, not a varying-slopes API.
    units = pd.get_dummies(d.muniID, dtype=float).iloc[:, 1:]
    linear = units.mul(d.time, axis=0)
    linear.columns = [f"trend_{i}" for i in range(linear.shape[1])]
    d = pd.concat([d, linear], axis=1)
    regression("swiss_unit_trends", d, "nat_rate", ["repdem", *linear.columns], ["muniID", "year"], "muniID",
               formula="nat_rate ~ repdem | muniID + year + muniID[time]", keep=["repdem"])
    quadratic = units.mul(d.time**2, axis=0)
    quadratic.columns = [f"quadratic_{i}" for i in range(quadratic.shape[1])]
    d = pd.concat([d, quadratic], axis=1)
    d["time2"] = d.time**2
    regression("swiss_quadratic_trends", d, "nat_rate", ["repdem", *linear.columns, *quadratic.columns],
               ["muniID", "year"], "muniID", formula="nat_rate ~ repdem | muniID + year + muniID[time,time2]", keep=["repdem"])
    for k in range(1, 4):
        d[f"lag{k}"] = d.groupby("muniID").repdem.shift(k)
    d["lead1"] = d.groupby("muniID").repdem.shift(-1)
    regression("swiss_distributed_lags", d, "nat_rate", ["repdem", "lag1", "lag2", "lag3"], ["muniID", "year"], "muniID")
    regression("swiss_leads_lags", d, "nat_rate", ["lead1", "repdem", "lag1", "lag2", "lag3"], ["muniID", "year"], "muniID")
    s = read("swiss_full").dropna(subset=["nat_rate_ord", "institution_dummyB", "ue_rate"]).copy()
    s["repdem"] = 1 - s.institution_dummyB
    s["svp"] = s.svpconstzero
    s["repdem_svp"] = s.repdem * s.svp
    regression("swiss_svp_linear", s, "nat_rate_ord", ["repdem", "repdem_svp", "ue_rate"], ["bfs", "year"], "bfs")
    cuts = s.svp.quantile([.25,.5,.75]).to_numpy()
    s["bin"] = pd.cut(s.svp, [-np.inf,*cuts,np.inf], labels=[1,2,3,4]).astype(float)
    for k in range(1,5): s[f"repdem_q{k}"] = s.repdem * (s.bin == k)
    regression("swiss_svp_bins", s, "nat_rate_ord", [*[f"repdem_q{k}" for k in range(1,5)],"ue_rate"], ["bfs","year"], "bfs")


def fatalities():
    d = read("fatalities")
    d["fatal_rate"] = 10000*d.fatal/d["pop"]
    for name, fe in [("pooled",[]),("unit_fe",["state"]),("twfe",["state","year"])]:
        regression("fatalities_"+name, d, "fatal_rate", ["beertax"], fe, "state")


def card_krueger():
    d = read("ck")
    d["did"] = d.nj*d.postperiod
    regression("ck_pooled",d,"emptot",["postperiod","nj","did"],cluster="ID")
    regression("ck_fe",d,"emptot",["postperiod","did"],["ID"],"ID")
    wide = d.pivot(index="ID",columns="postperiod",values="emptot").dropna()
    z = pd.DataFrame({"ID":wide.index,"delta":wide[1]-wide[0]}).reset_index(drop=True)
    z["nj"] = z.ID.map(d.groupby("ID").nj.first())
    regression("ck_first_difference",z,"delta",["nj"])
    means=d.groupby(["nj","postperiod"]).emptot.agg(['mean','count','std']).reset_index()
    means.to_csv(OUT/'ck-cell-means.csv',index=False)


def divorce():
    d=read("divorce").query('sex == 2 and 1964 <= year <= 1996').copy()
    d["asmrs"]=1e6*d.suiciderate_jag
    d["post"]=d.unilateral
    regression("divorce_twfe",d,"asmrs",["post"],["st","year"],"st")
    d=d.query('divyear > 1964').copy()
    d["rel_b"]=(d.year-d.divyear).clip(-9,16)
    # Include BOTH endpoint bins in estimation. The lecture only hides them
    # in its plot; dropping the -9 regressor changes every event coefficient.
    terms=[]
    for k in range(-9,17):
        if k == -1: continue
        term=f"event_{'m' if k<0 else 'p'}{abs(k)}"; terms.append(term); d[term]=(d.rel_b==k).astype(float)
    regression("divorce_event",d,"asmrs",terms,["st","year"],"st")


def famine():
    d=read('mortality')
    y=d.pivot(index='countyid',columns='year',values='mortality')
    z=d.drop_duplicates('countyid').set_index('countyid').loc[y.index].copy()
    z['any_gen']=(z.pczupu>0).astype(float)
    xs=['avggrain','nograin','urban','dis_bj','dis_pc','rice','minority','edu','lnpop']
    for c in xs: z[c]=z[c]-z[c].mean()
    for c in xs: z['g_'+c]=z[c]*z.any_gen
    z['event_delta']=y[[1958,1959,1960,1961]].mean(axis=1)-y[1957]
    for label,cols in [('unadjusted',['any_gen']),('adjusted',['any_gen',*xs]),('interacted',['any_gen',*xs,*['g_'+c for c in xs]])]:
        regression('famine_'+label,z,'event_delta',cols,keep=['any_gen'])
        for t in y.columns:
            if t == 1957: continue
            z['delta']=y[t]-y[1957]
            regression(f'famine_{label}_{t}',z,'delta',cols,keep=['any_gen'])


def other_panels():
    d=read('gs2020')
    regression('gs2020_twfe',d,'general_sharetotal_A_all',['cand_A_all'],['district_final','cycle'],'district_final')
    d=read('turnout')
    regression('turnout_twfe',d,'turnout',['policy_edr','policy_mail_in','policy_motor'],['abb','year'],'abb')


def main():
    for fn in [fatalities,swiss,card_krueger,divorce,famine,other_panels]:
        try: fn()
        except Exception as exc:
            ERRORS.append(dict(model=fn.__name__,exception=type(exc).__name__,message=str(exc)))
    pd.DataFrame(ROWS).to_csv(OUT/'linear-crabby.csv',index=False)
    (CACHE/'linear-jobs.json').write_text(json.dumps(JOBS,indent=2))
    (OUT/'linear-errors.json').write_text(json.dumps(ERRORS,indent=2)+'\n')
    extension=Path(cm.crabbymetrics.__file__)
    provenance=dict(version=importlib.metadata.version('crabbymetrics'),
                    source_commit=subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip(),
                    extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                    numpy=np.__version__,pandas=pd.__version__)
    (OUT/'runtime.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__': main()
