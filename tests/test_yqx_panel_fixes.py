"""Offline checks against original fixtures generated independently in base R.

No R installation, network, or lecture microdata is needed by this test module.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import crabbymetrics as cm

FIXTURES = Path(__file__).parent / 'fixtures' / 'yqx_panel'


def panel():
    d = pd.read_csv(FIXTURES / 'panel.csv')
    shape = (12, 8)
    y = d.y.to_numpy().reshape(shape)
    treatment = d.treatment.to_numpy(float).reshape(shape)
    x = d[['x1', 'x2']].to_numpy().reshape((*shape, 2))
    observed = d.observed.to_numpy(bool).reshape(shape)
    w = d.weight.to_numpy().reshape(shape)
    return d, y, treatment, x, observed, w


def test_scm_matches_exhaustive_R_simplex_solution():
    d = pd.read_csv(FIXTURES / 'scm.csv')
    x, y = d.iloc[:, 2:].to_numpy(), d.treated.to_numpy()
    reference = pd.read_csv(FIXTURES / 'scm-reference.csv').iloc[0]
    m = cm.SyntheticControl()
    m.fit(x[:18], y[:18])
    s = m.summary()
    np.testing.assert_allclose(s['weights'], pd.read_csv(FIXTURES / 'scm-weights.csv').weight, atol=1e-8)
    assert s['pre_rmse'] == pytest.approx(reference.pre_rmse, abs=1e-8)
    assert np.mean(y[18:] - m.predict(x)[18:]) == pytest.approx(reference.att, abs=1e-7)
    assert np.count_nonzero(s['weights'] == 0) > 0
    assert s['converged'] and s['stop_reason'] == 'kkt_optimality'
    assert s['scaled_kkt_residual'] < 1e-8
    assert s['feasibility_error'] < 1e-12
    assert s['objective'] == pytest.approx(.5*s['pre_rmse']**2)
    assert 0 < s['iterations'] <= 500
    # Reordering and redundant donors can change a nonunique weight vector,
    # but must not change the fitted path or objective.
    for z in (x[:, ::-1], np.column_stack([x, x[:, 1]])):
        other = cm.SyntheticControl()
        other.fit(z[:18], y[:18])
        np.testing.assert_allclose(other.predict(z), m.predict(x), atol=1e-6)
    shifted = cm.SyntheticControl()
    shifted.fit(x[:18] + 1e7, y[:18] + 1e7)
    np.testing.assert_allclose(shifted.predict(x+1e7)-1e7, m.predict(x), atol=1e-6)


def test_scm_boundary_single_donor_and_failed_refit():
    m = cm.SyntheticControl(max_iterations=1)
    m.fit(np.ones((8,1)), np.arange(8.0))
    np.testing.assert_equal(m.summary()['weights'], [1])
    d = pd.read_csv(FIXTURES / 'scm.csv')
    with pytest.raises(ValueError, match='did not converge'):
        m.fit(d.iloc[:18,2:].to_numpy(), d.treated.to_numpy()[:18])
    with pytest.raises(ValueError, match='not fitted'):
        m.summary()
    good = cm.SyntheticControl()
    good.fit(np.column_stack([np.ones(5),np.full(5,2.0)]), np.full(5,3.0))
    np.testing.assert_equal(good.summary()['weights'], [0,1])
    with pytest.raises(ValueError, match='finite'):
        good.predict(np.array([[np.nan,1.0]]))


@pytest.mark.parametrize('weighted', [False, True])
def test_imputation_matches_R_lm_surfaces_and_aggregation(weighted):
    d, y, treatment, x, obs, w = panel()
    m = cm.FEImputation()
    m.fit(y, treatment, x, observed=obs, sample_weight=w if weighted else None)
    s = m.summary()
    ref = pd.read_csv(FIXTURES / 'imputation-reference.csv').set_index('weighted').loc[weighted]
    assert s['att'] == pytest.approx(ref.att, abs=1e-8)
    np.testing.assert_allclose(s['coef'], ref[['beta1','beta2']].to_numpy(float), atol=1e-8)
    expected = d['cf_weighted' if weighted else 'cf'].to_numpy().reshape(y.shape)
    np.testing.assert_allclose(m.predict()[obs], expected[obs], atol=1e-8)
    assert np.isnan(m.predict()[~obs]).all()
    assert np.array_equal(s['training_mask'], obs & (treatment==0))
    assert np.array_equal(s['evaluation_mask'], obs & (treatment==1))
    assert s['n_reversing_units'] > 0
    assert s['inference'] == 'not_implemented'
    mask = s['evaluation_mask']
    weights = w if weighted else np.ones_like(w)
    for j in range(y.shape[1]):
        if mask[:,j].any():
            assert s['att_by_time'][j] == pytest.approx(np.average((y-expected)[mask[:,j],j],weights=weights[mask[:,j],j]),abs=1e-8)
        else:
            assert np.isnan(s['att_by_time'][j])
    changed = y.copy(); changed[mask] += 10_000; changed[~obs] = np.inf
    m.fit(changed, treatment, x, observed=obs, sample_weight=w if weighted else None)
    np.testing.assert_allclose(m.predict()[obs], expected[obs], atol=1e-8)
    assert m.summary()['att'] == pytest.approx(ref.att+10_000,abs=1e-8)


def test_imputation_fe_only_missing_cells_zero_weights_and_validation():
    _, y, d, _, obs, w = panel()
    y = y.copy(); y[~obs] = np.nan
    m = cm.FEImputation()
    m.fit(y,d)
    assert m.summary()['coef'].size == 0
    w = w.copy(); w[0,1] = 0
    m.fit(y,d, sample_weight=w)
    assert np.isnan(m.predict()[0,1])
    assert m.summary()['n_zero_weight'] == 1
    invalid = d.copy(); invalid[0,1] = .5
    with pytest.raises(ValueError,match='binary'):
        m.fit(y,invalid)
    with pytest.raises(ValueError,match='not fitted'):
        m.predict()
    for bad in (-1, np.nan):
        weights = np.ones_like(y); weights[0,0] = bad
        with pytest.raises(ValueError,match='nonnegative'):
            m.fit(y,d,sample_weight=weights)
    with pytest.raises(ValueError,match='shape'):
        m.fit(y,d,observed=np.ones((2,3),bool))
    with pytest.raises(ValueError,match='untreated and treated'):
        m.fit(y,np.zeros_like(d))
    always = d.copy(); always[0,:] = 1
    with pytest.raises(ValueError,match='unseen'):
        m.fit(y,always)


def test_imputation_refuses_disconnected_counterfactuals():
    y = np.arange(16.0).reshape(4,4)
    d = np.ones((4,4)); d[:2,:2] = 0; d[2:,2:] = 0
    with pytest.raises(ValueError,match='disconnected'):
        cm.FEImputation().fit(y,d)


def test_fe_covariance_matches_R_full_design_all_policies():
    d, *_ = panel(); d = d[d.observed==1]
    x = d[['x1','x2']].to_numpy()
    fe = d[['unit','time']].to_numpy(np.uint32)
    clusters = d.unit.to_numpy(np.int64)
    model = cm.FixedEffectsOLS(tolerance=1e-11)
    model.fit(x,fe,d.y.to_numpy())
    for r in pd.read_csv(FIXTURES / 'fe-reference.csv').itertuples():
        s = model.summary(vcov='cluster',clusters=clusters,fe_df=r.fe_df,ssc=r.ssc,cluster_correction=r.cluster_correction)
        np.testing.assert_allclose(s['coef'],[r.beta1,r.beta2],atol=1e-9)
        np.testing.assert_allclose(s['vcov'],[[r.v11,r.v12],[r.v12,r.v22]],atol=1e-10)
        result = model.wald_test(np.eye(2),vcov='cluster',clusters=clusters,fe_df=r.fe_df,ssc=r.ssc,cluster_correction=r.cluster_correction)
        assert np.isfinite(result['statistic'])
    assert model.summary()['converged']
    assert np.max(model.summary()['absorption_residual_norms']) < 1e-8
    # No huge max-ID allocation; level prediction includes the fitted FE.
    shifted = fe.copy(); shifted[:,0] += 1_000_000_000
    other = cm.FixedEffectsOLS(tolerance=1e-11)
    other.fit(x,shifted,d.y.to_numpy())
    np.testing.assert_allclose(other.predict(x,shifted),model.predict(x,fe),atol=1e-9)
    assert np.linalg.norm(d.y.to_numpy()-model.predict(x,fe)) < np.linalg.norm(d.y.to_numpy()-x@model.summary()['coef'])
    with pytest.raises(ValueError,match='unseen'):
        other.predict(x,fe)
    with pytest.raises(ValueError,match='fe_df'):
        model.summary(fe_df='invented')


def test_fe_only_level_prediction_and_disconnected_rejection():
    fe = np.array([[0,0],[0,1],[1,0],[1,1],[2,2],[2,3],[3,2],[3,3]],dtype=np.uint32)
    y = 1 + fe[:,0]*.3 + fe[:,1]*.8
    x = np.empty((len(fe),0))
    m = cm.FixedEffectsOLS(tolerance=1e-11)
    m.fit(x,fe,y)
    np.testing.assert_allclose(m.predict(x,fe),y,atol=1e-9)
    assert m.summary()['coef'].size==0
    with pytest.raises(ValueError,match='disconnected'):
        m.predict(np.empty((1,0)),np.array([[0,2]],np.uint32))


@pytest.mark.parametrize('kind', [cm.FixedEffectsOLS, cm.FEImputation])
@pytest.mark.parametrize('kwargs', [{'tolerance':0},{'tolerance':float('nan')},{'max_iterations':0}])
def test_invalid_solver_settings(kind,kwargs):
    with pytest.raises(ValueError): kind(**kwargs)


def test_fe_solver_budget_and_covariance_options_are_not_silently_ignored():
    d, y, treatment, x, obs, _ = panel()
    design = d[['x1','x2']].to_numpy()
    fe = d[['unit','time']].to_numpy(np.uint32)
    with pytest.raises(ValueError,match='did not converge'):
        cm.FixedEffectsOLS(max_iterations=1).fit(design,fe,d.y.to_numpy())
    with pytest.raises(ValueError,match='did not converge'):
        cm.FEImputation(max_iterations=1).fit(y,treatment,x,observed=obs)
    m = cm.FixedEffectsOLS();m.fit(design,fe,d.y.to_numpy())
    with pytest.raises(ValueError,match='requires cluster'):
        m.summary(fe_df='non_nested',clusters=d.unit.to_numpy(np.int64))
    with pytest.raises(ValueError,match='only to cluster'):
        m.summary(ssc='hc1')


def test_imputation_rejects_covariates_unidentified_in_untreated_sample():
    n,t = 6,5
    y = np.arange(n*t,dtype=float).reshape(n,t)
    d = np.zeros_like(y); d[3:,3:] = 1
    # Unit-constant in training but changing in treated cells: a dangerous
    # counterfactual extrapolation even though every FE level is observed.
    x = np.arange(n,dtype=float)[:,None,None]*np.ones((n,t,1))
    x[d==1] += 3
    with pytest.raises(ValueError,match='not identified'):
        cm.FEImputation().fit(y,d,x)
    rng = np.random.default_rng(219)
    x = rng.normal(size=(n,t,1)); x = np.concatenate([x,x],axis=2)
    with pytest.raises(ValueError):
        cm.FEImputation().fit(y,d,x)

    # The general FE predictor must also reject an unidentified slope.
    train = d.ravel()==0
    fe = np.column_stack(np.unravel_index(np.arange(n*t),(n,t))).astype(np.uint32)
    raw = fe[:,[0]].astype(float)
    m = cm.FixedEffectsOLS(); m.fit(raw[train],fe[train],y.ravel()[train])
    with pytest.raises(ValueError,match='not identified'):
        m.predict(raw,fe)
