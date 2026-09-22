"""Probe FE-order sensitivity on the actual high-dimensional three-FE case."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import crabbymetrics as cm
root=Path(__file__).resolve().parent
d=pd.read_csv(root/'.cache/archive-Schafer_2022_AJPS.csv');rows=[]
for order in [['fe1','fe2','fe3'],['fe3','fe2','fe1'],['fe2','fe1','fe3']]:
    tic=time.monotonic();m=cm.FixedEffectsOLS()
    m.fit(d[['treatment_alt']].to_numpy(float),d[order].to_numpy(np.uint32),d.y.to_numpy(float))
    rows.append(dict(order=','.join(order),coef=float(m.summary()['coef'][0]),seconds=time.monotonic()-tic))
pd.DataFrame(rows).to_csv(root/'results/schafer-order-probe.csv',index=False)
