import numpy as np
import crabbymetrics as cm


def simulate_exponential(n=600, beta=np.array([0.6, -0.35]), base=0.08, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(beta)))
    rate = base * np.exp(x @ beta)
    event_time = rng.exponential(1.0 / rate)
    censor = rng.exponential(20.0, size=n)
    time = np.minimum(event_time, censor)
    event = (event_time <= censor).astype(float)
    return x, time, event, beta


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
