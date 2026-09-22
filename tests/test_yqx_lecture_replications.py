"""Optional real-data regressions against the pinned R lecture audit.

Run acquire/reference scripts in replications/yqx_panel first. These tests skip
when the ignored data cache is absent, so CI never downloads third-party data.
Offline original R fixtures are tested in test_yqx_panel_fixes.py.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pytest
import crabbymetrics as cm

ROOT = Path(__file__).resolve().parents[1] / 'replications' / 'yqx_panel'
CACHE = Path(os.environ.get('CRABBY_LECTURE_CACHE', ROOT / '.cache'))


def read(name):
    path = CACHE / (name + '.csv')
    if not path.exists():
        pytest.skip('Acquire lecture inputs first: '+str(path))
    return pd.read_csv(path)


@pytest.mark.parametrize('iterations', [500, 5000])
def test_prop99_scm_R_augsynth_optimum(iterations):
    d = read('smoking').pivot(index='year',columns='state',values='cigsale').sort_index()
    x, y = d.drop(columns='California').to_numpy(), d.California.to_numpy()
    pre = d.index < 1989
    m = cm.SyntheticControl(max_iterations=iterations)
    m.fit(x[pre],y[pre])
    assert m.summary()['pre_rmse'] == pytest.approx(1.65640020695525,abs=1e-7)
    assert np.mean((y-m.predict(x))[~pre]) == pytest.approx(-19.5136298827203,abs=1e-5)
    assert m.summary()['scaled_kkt_residual'] < 1e-8


@pytest.mark.parametrize('key', ['gs2020_fe','turnout_fe','hh2019_fe','modern_Bischof_Wagner_2019_AJPS_full','modern_Grumbach_Sahn_2020_APSR_full'])
def test_native_imputation_matches_R_fect(key):
    d = read(key+'-grid')
    wide = lambda column: d.pivot(index='unit',columns='time',values=column).to_numpy(float)
    y, treatment = wide('y'),wide('d')
    observed = wide('observed').astype(bool) & np.isfinite(y)
    columns = [c for c in d if c.startswith('x')]
    x = np.stack([wide(c) for c in columns],axis=2) if columns else None
    reference = wide('ref_counterfactual')
    m = cm.FEImputation()
    m.fit(y,treatment,x,observed=observed)
    treated = observed & (treatment==1)
    np.testing.assert_allclose(m.predict()[treated],reference[treated],atol=1e-6,rtol=0)
    assert m.summary()['att'] == pytest.approx(np.mean((y-reference)[treated]),abs=1e-6)
    assert m.summary()['n_treated'] == treated.sum()


def test_fatalities_three_R_covariance_conventions():
    d = read('design-fatalities_twfe')
    fe = np.column_stack([pd.factorize(d[c],sort=True)[0] for c in ['state','year']]).astype(np.uint32)
    clusters = fe[:,0].astype(np.int64)
    m = cm.FixedEffectsOLS()
    m.fit(d[['beertax']].to_numpy(),fe,d.fatal_rate.to_numpy())
    cases = [({},.385786721792343),({'fe_df':'non_nested'},.35707834554809),
             ({'fe_df':'none','ssc':'hc1','cluster_correction':False},.350149554143009)]
    for kwargs, expected in cases:
        s = m.summary(vcov='cluster',clusters=clusters,**kwargs)
        assert s['coef'][0] == pytest.approx(-.639979985706868,abs=1e-9)
        assert s['coef_se'][0] == pytest.approx(expected,abs=1e-9)
