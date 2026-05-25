import numpy as np
import crabbymetrics as cm
from lifelines import CoxPHFitter, ExponentialFitter, WeibullAFTFitter, WeibullFitter
import pandas as pd


def simulate_exponential(n=600, beta=np.array([0.6, -0.35]), base=0.08, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(beta)))
    rate = base * np.exp(x @ beta)
    event_time = rng.exponential(1.0 / rate)
    censor = rng.exponential(20.0, size=n)
    time = np.minimum(event_time, censor)
    event = (event_time <= censor).astype(float)
    return x, time, event, beta


def simulate_weibull_ph(n=1500, beta=np.array([0.45, -0.30]), shape=1.4, scale_hazard=0.025, seed=20260525):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(beta)))
    u = rng.uniform(size=n)
    event_time = ((-np.log(u)) / (scale_hazard * np.exp(x @ beta))) ** (1.0 / shape)
    censor = rng.exponential(60.0, size=n)
    time = np.minimum(event_time, censor)
    event = (event_time <= censor).astype(float)
    return x, time, event, beta, shape, scale_hazard


def lifelines_frame(x, time, event):
    return pd.DataFrame(
        {
            "T": time,
            "E": event.astype(int),
            **{f"x{j + 1}": x[:, j] for j in range(x.shape[1])},
        }
    )


def test_exponential_ph_recovers_simulated_coefficients():
    x, time, event, beta = simulate_exponential()
    model = cm.ExponentialPH()
    model.fit(x, time, event)
    out = model.summary()
    assert np.allclose(out["coef"], beta, atol=0.18)
    assert out["baseline_hazard"] > 0
    assert out["vcov"].shape == (3, 3)


def test_weibull_ph_recovers_shape_and_coefficients():
    rng = np.random.default_rng(456)
    n = 800
    beta = np.array([0.45, -0.25])
    shape = 1.6
    scale_hazard = 0.03
    x = rng.normal(size=(n, 2))
    u = rng.uniform(size=n)
    event_time = ((-np.log(u)) / (scale_hazard * np.exp(x @ beta))) ** (1.0 / shape)
    censor = rng.exponential(35.0, size=n)
    time = np.minimum(event_time, censor)
    event = (event_time <= censor).astype(float)
    model = cm.WeibullPH()
    model.fit(x, time, event)
    out = model.summary()
    assert np.allclose(out["coef"], beta, atol=0.18)
    assert abs(out["shape"] - shape) < 0.25


def test_cox_ph_and_andersen_gill_track_direction():
    x, time, event, beta = simulate_exponential(n=500, beta=np.array([0.7]), seed=789)
    cox = cm.CoxPH()
    cox.fit(x, time, event)
    assert abs(cox.summary()["coef"][0] - beta[0]) < 0.25

    # Split each subject into two counting-process intervals. The event can only
    # occur in the final interval for the observed time, preserving the Cox fit.
    start = np.concatenate([np.zeros_like(time), time / 2.0])
    stop = np.concatenate([time / 2.0, time])
    x_long = np.vstack([x, x])
    event_long = np.concatenate([np.zeros_like(event), event])
    ag = cm.AndersenGill()
    ag.fit(x_long, start, stop, event_long)
    assert abs(ag.summary()["coef"][0] - cox.summary()["coef"][0]) < 1e-6


def test_cox_ph_matches_lifelines_coxph_fitter_on_same_data():
    x, time, event, _, _, _ = simulate_weibull_ph()

    crab = cm.CoxPH()
    crab.fit(x, time, event)

    lf = CoxPHFitter().fit(lifelines_frame(x, time, event), duration_col="T", event_col="E")

    assert np.allclose(crab.summary()["coef"], lf.params_.to_numpy(), atol=2e-6)


def test_exponential_ph_matches_lifelines_exponential_fitter_without_covariates():
    # Mirrors lifelines' ExponentialFitter check: for right-censored exponential
    # data, lifelines reports lambda_ = total observed time / events. Our
    # ExponentialPH reports the reciprocal baseline hazard.
    time = np.array([10.0, 10.0, 10.0, 10.0])
    event = np.array([1.0, 1.0, 1.0, 0.0])
    x = np.empty((len(time), 0))

    crab = cm.ExponentialPH()
    crab.fit(x, time, event)
    lf = ExponentialFitter().fit(time, event)

    assert np.allclose(1.0 / crab.summary()["baseline_hazard"], lf.lambda_, rtol=5e-6)
    assert np.allclose(crab.summary()["baseline_hazard"], event.sum() / time.sum(), rtol=1e-10)


def test_weibull_ph_matches_lifelines_weibull_aft_after_ph_mapping():
    # lifelines' WeibullAFT parameterization is S(t|x)=exp(-(t/lambda(x))^rho).
    # crabbymetrics uses H(t|x)=lambda_h * t^rho * exp(x beta_ph), so
    # beta_ph = -rho * beta_aft and log(lambda_h) = -rho * intercept_aft.
    x, time, event, _, _, _ = simulate_weibull_ph()

    crab = cm.WeibullPH()
    crab.fit(x, time, event)

    lf = WeibullAFTFitter().fit(lifelines_frame(x, time, event), duration_col="T", event_col="E")
    rho = np.exp(lf.params_.loc[("rho_", "Intercept")])
    aft_lambda = lf.params_.loc["lambda_"]
    lf_beta_ph = -rho * aft_lambda[["x1", "x2"]].to_numpy()
    lf_log_scale_hazard = -rho * aft_lambda["Intercept"]

    out = crab.summary()
    assert np.allclose(out["coef"], lf_beta_ph, atol=8e-6)
    assert np.allclose(out["shape"], rho, atol=2e-6)
    assert np.allclose(out["log_scale_hazard"], lf_log_scale_hazard, atol=2e-6)


def test_known_exponential_dgp_params_from_lifelines_style_check():
    # Lifelines tests use factor * Exponential(1) with independent censoring and
    # check the fitted exponential scale against factor. In crabbymetrics the
    # corresponding PH baseline hazard is 1 / factor.
    rng = np.random.default_rng(12345)
    n = 20_000
    factor = 5.0
    event_time = factor * rng.exponential(1.0, size=n)
    censor_time = factor * rng.exponential(1.0, size=n)
    time = np.minimum(event_time, censor_time)
    event = (event_time < censor_time).astype(float)

    crab = cm.ExponentialPH()
    crab.fit(np.empty((n, 0)), time, event)

    assert abs(crab.summary()["baseline_hazard"] - 1.0 / factor) < 0.01


def test_known_weibull_dgp_params_from_lifelines_style_check():
    # Mirrors lifelines' WeibullFitter DGP: T = lambda * np.random.weibull(rho).
    # This implies H(t)=(t/lambda)^rho, so crabbymetrics' scale hazard is
    # lambda^{-rho} and its reported shape is rho.
    rng = np.random.default_rng(45678)
    n = 12_000
    rho = 5.0
    lambda_lifelines = 0.5
    time = lambda_lifelines * rng.weibull(rho, size=n)
    event = np.ones(n)

    crab = cm.WeibullPH()
    crab.fit(np.empty((n, 0)), time, event)

    out = crab.summary()
    crab_lambda_lifelines = np.exp(out["log_scale_hazard"]) ** (-1.0 / out["shape"])
    lf = WeibullFitter().fit(time, event)

    assert abs(out["shape"] - rho) < 0.06
    assert abs(crab_lambda_lifelines / lambda_lifelines - 1.0) < 0.02
    assert np.allclose(out["shape"], lf.rho_, atol=1e-4)
    assert np.allclose(crab_lambda_lifelines, lf.lambda_, atol=1e-4)



def test_exponential_prediction_surfaces_are_consistent():
    x, time, event, _, _, _ = simulate_weibull_ph(n=500, seed=20260528)
    model = cm.ExponentialPH()
    model.fit(x, time, event)

    idx = slice(0, 37)
    eta = model.predict_lin(x[idx])
    hazard = model.predict_hazard(x[idx])
    cumhaz = model.predict_cumulative_hazard(x[idx], time[idx])
    survival = model.predict(x[idx], time[idx])

    np.testing.assert_allclose(hazard, np.exp(eta), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(cumhaz, hazard * time[idx], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(survival, np.exp(-cumhaz), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.predict_log_hazard(x[idx]), eta, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.survival(x[idx], time[idx]), survival, atol=1e-10, rtol=1e-10)



def test_weibull_prediction_surfaces_are_consistent():
    x, time, event, _, shape, scale_hazard = simulate_weibull_ph(n=700, seed=20260529)
    model = cm.WeibullPH()
    model.fit(x, time, event)

    idx = slice(0, 53)
    eta = model.predict_lin(x[idx], time[idx])
    hazard = model.predict_hazard(x[idx], time[idx])
    cumhaz = model.predict_cumulative_hazard(x[idx], time[idx])
    survival = model.predict(x[idx], time[idx])
    expected_cumhaz = scale_hazard * np.power(time[idx], shape) * np.exp(x[idx] @ model.summary()["coef"])

    np.testing.assert_allclose(hazard, np.exp(eta), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(survival, np.exp(-cumhaz), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.predict_log_hazard(x[idx], time[idx]), eta, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.survival(x[idx], time[idx]), survival, atol=1e-10, rtol=1e-10)
    assert np.corrcoef(cumhaz, expected_cumhaz)[0, 1] > 0.98



def test_cox_and_andersen_gill_prediction_surfaces_match_relative_risk_contract():
    x, time, event, _, _, _ = simulate_weibull_ph(n=600, seed=20260530)

    cox = cm.CoxPH()
    cox.fit(x, time, event)
    eta = cox.predict_lin(x[:41])
    rr = cox.predict(x[:41])
    np.testing.assert_allclose(rr, np.exp(eta), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(cox.predict_log_hazard_ratio(x[:41]), eta, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(cox.predict_relative_risk(x[:41]), rr, atol=1e-10, rtol=1e-10)

    start = np.concatenate([np.zeros_like(time), time / 2.0])
    stop = np.concatenate([time / 2.0, time])
    x_long = np.vstack([x, x])
    event_long = np.concatenate([np.zeros_like(event), event])

    ag = cm.AndersenGill()
    ag.fit(x_long, start, stop, event_long)
    eta_ag = ag.predict_lin(x[:41])
    rr_ag = ag.predict(x[:41])
    np.testing.assert_allclose(rr_ag, np.exp(eta_ag), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(ag.predict_log_hazard_ratio(x[:41]), eta_ag, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(ag.predict_relative_risk(x[:41]), rr_ag, atol=1e-10, rtol=1e-10)
