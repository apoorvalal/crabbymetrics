"""Run the source-derived TWFE baseline through the public NumPy API."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import crabbymetrics as cm

ROOT=Path(__file__).resolve().parent
jobs=json.loads((ROOT/'results/archive-specifications.json').read_text())
rows=[]
for j in jobs:
    row={'name':j['name'],'status':j['status'],'scope':j.get('scope','not run')}
    if j['status']=='prepared':
        try:
            d=pd.read_csv(ROOT/'.cache'/f"archive-{j['name']}.csv")
            regressors=j['regressors']; regressors=[regressors] if isinstance(regressors,str) else regressors
            effects=j['effects']; effects=[effects] if isinstance(effects,str) else effects
            tic=time.monotonic();m=cm.FixedEffectsOLS()
            m.fit(d[regressors].to_numpy(float),d[effects].to_numpy(np.uint32),d.y.to_numpy(float))
            s=m.summary(vcov='cluster',clusters=d.cluster.to_numpy(np.int64));k=regressors.index(j['treatment'])
            row.update(status='completed',estimate=float(s['coef'][k]),se=float(s['coef_se'][k]),n=len(d),seconds=time.monotonic()-tic)
        except Exception as exc:row['status']=f'{type(exc).__name__}: {exc}'
    print(row,flush=True);rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results/archive-crabby.csv',index=False)
