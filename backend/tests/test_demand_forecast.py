"""Demand forecast: no leakage, coherent reconciliation, exact baselines and capacity plans on small cases,
and the committed outputs."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from scgame.ml import demand_forecast as df
from scgame.ml import demand_series as ds
from scgame.ml import forecast_baselines as fb
from scgame.ml import forecast_models as fm
from scgame.ml import forecast_uncertainty as fu

OUT = Path(__file__).resolve().parents[2] / "data" / "models" / "demand-forecast"


def seasonal(n_series=3, months=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    base = 100 * (1 + 0.3 * np.sin(2 * np.pi * t / 12))
    return np.vstack([base * (i + 1) + rng.normal(0, 1, months) for i in range(n_series)])


def test_training_windows_never_read_past_the_fitting_window():
    y = seasonal(months=60)
    y[:, 36:] = np.nan  # anything after month 36 would poison the windows
    win = fm.training_windows(y, np.zeros_like(y), 36)
    assert win.origin.max() == 24
    assert np.isfinite(win.target).all() and np.isfinite(win.lags).all()


def test_recursion_sees_nothing_after_the_origin():
    y = seasonal(months=60)
    seen = []

    def predict(win):
        seen.append(win.lags.copy())
        return win.lags  # next year repeats the last one (scaled)

    a, _ = fm.recursive_forecast(predict, y, np.zeros_like(y), 36, 2)
    y2 = y.copy()
    y2[:, 36:] = 1e9
    b, _ = fm.recursive_forecast(predict, y2, np.zeros_like(y), 36, 2)
    assert np.allclose(a, b)
    assert np.allclose(a[:, :12], y[:, 24:36])


def test_seasonal_naive_repeats_the_last_year_and_short_series_use_their_mean():
    y = seasonal(months=36)
    assert np.allclose(fb.seasonal_naive(y, 36, 24), np.hstack([y[:, 24:36], y[:, 24:36]]))
    short = np.zeros((1, 36))
    short[0, 30:] = 6
    assert np.allclose(fb.seasonal_naive(short, 36, 12), 6)


def test_ets_forecast_from_the_end_matches_statsmodels():
    y = seasonal(n_series=1, months=36)[0]
    fit = fb._fit_one((y, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {"trend": "add" if fit["has_trend"] else None, "damped_trend": fit["has_trend"], "seasonal": "add",
              "seasonal_periods": 12, "initialization_method": "estimated"}
        ref = ExponentialSmoothing(np.log1p(y), **kw).fit().forecast(12)
    ours, used = fb.ets_forecast([fit], y[None], 36, 12)
    assert used[0]
    assert np.allclose(ours[0], np.expm1(ref + fit["sigma2"] / 2))


def tiny_hierarchy():
    # Total, two regions, one category: rows Total, R1, R2, Cat, R1×Cat, R2×Cat; bottom = the last two.
    S = np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], dtype=float)
    return S, np.array([4, 5])


def test_mint_and_bottom_up_forecasts_add_up():
    S, bottom = tiny_hierarchy()
    rng = np.random.default_rng(1)
    W = fu.shrunk_covariance(rng.normal(size=(50, 6)))
    base = rng.uniform(50, 150, size=(6, 12))
    for P in (fu.mint_matrix(S, W), fu.bottom_up_matrix(S, bottom)):
        rec = fu.reconcile(P, base)
        assert np.allclose(rec[0], rec[4] + rec[5])
        assert np.allclose(rec[1], rec[4]) and np.allclose(rec[3], rec[0])
        assert np.allclose(fu.reconcile(P, S @ rec[[4, 5]]), S @ rec[[4, 5]])  # coherent forecasts stay put


def test_middle_out_children_sum_to_their_parent():
    pairs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    parent = np.array([0, 0, 1])
    target = np.array([[8.0, 12.0], [10.0, 7.0]])
    out = fu.middle_out(pairs, parent, target, 2)
    assert np.allclose(out[:2].sum(0), target[0])
    assert np.allclose(out[2], [10.0, 0.0])  # nothing to scale where the children forecast nothing


def test_paths_carry_level_errors_over_the_years():
    blocks = fu.ErrorBlocks(level=np.array([[0.1], [0.1]]), noise=np.zeros((2, 1, 12)))
    draws = np.zeros((1, 3), dtype=int)
    p = fu.sample_paths(np.full((1, 36), 99.0), blocks, draws, k=1.0, origin=36)
    assert np.allclose(p[0, 0, 0], 100 * np.exp(0.1) - 1)
    assert np.allclose(p[0, 0, 24], 100 * np.exp(0.3) - 1)


def test_error_blocks_are_centred_and_noise_follows_the_calendar():
    actual = np.ones((1, 3, 12)) * 9
    actual[0, 0, 0] = 19  # a high first month at origin 13 (month 14, a February)
    actual[0, 2, 0] = 19  # and at origin 25
    forecast = np.ones((1, 3, 12)) * 9
    forecast[0, 1] = 4  # the second origin's year runs 1.0 above in log terms
    b = fu.error_blocks(actual, forecast, origins=[13, 24, 25])
    assert np.allclose(b.level.sum(axis=0), 0)  # centred over origins
    assert np.allclose(b.noise.sum(axis=0), 0)  # each calendar month centred over origins
    assert b.noise[0, 0].argmax() == 1  # stored as February
    draws = np.zeros((1, 1), dtype=int)
    p = fu.sample_paths(np.full((1, 12), 9.0), b, draws, k=1.0, origin=36)  # month 37 is a January
    assert p[0, 0].argmax() == 1


def test_weighted_quantile_loss_equals_wape_when_quantiles_coincide():
    a = np.array([[10.0, 20.0, 30.0]])
    q = np.stack([np.array([[12.0, 18.0, 33.0]])] * 3)
    s = fu.scores(a, q)
    assert s["wql"] == pytest.approx(s["wape"])


def test_capacity_plan_matches_brute_force():
    rng = np.random.default_rng(3)
    d = rng.gamma(5, 20, size=(200, 48))
    for rho in (0.2, 0.75, 3.0):
        K = df.capacity_plan(d, rho)
        grid = np.linspace(d.min(), d.max(), 4001)
        cost = [df.capacity_cost(k, d, rho, 1.0).mean() for k in grid]
        assert df.capacity_cost(K, d, rho, 1.0).mean() <= min(cost) + 1e-6 * min(cost)


def test_level_weights_give_each_level_the_same_total():
    meta = pd.DataFrame({"level": ["Total", "Region", "Region", "SKU", "SKU", "SKU"]})
    series = np.arange(6)
    w = ds.level_weights(meta, series, np.array([100.0, 60, 40, 1, 2, 7]), np.zeros(6, dtype=int))
    assert w[0] == pytest.approx(1) and w[1:3].sum() == pytest.approx(1) and w[3:].sum() == pytest.approx(1)
    assert w[1] / w[2] == pytest.approx(1.5)


@pytest.mark.skipif(not (OUT / "overview.json").exists(), reason="run scgame.ml.demand_forecast first")
def test_outputs():
    o = json.loads((OUT / "overview.json").read_text())
    assert o["chosen"] in df.METHODS
    val = o["validation"]
    # The method was chosen on the validation set alone.
    assert o["chosen"] == min(df.METHODS, key=lambda n: val[n]["best"]["score"])
    for n in df.METHODS:
        best = val[n]["best"]
        assert best["width_upper"] == min(val[n]["grid_upper"], key=lambda g: g["score"])["width"]
        assert best["width_lower"] == min(val[n]["grid_lower"], key=lambda g: g["score"])["width"]
        assert len(o["test"][n]["table"]["Total"]) == 5 and len(o["test"][n]["table"]["SKU"]) == 2
    assert o["facts"]["replications"] == 200
    gt = o["capacity"]["curves"]["Ground truth"]["expected"]
    for plan, c in o["capacity"]["curves"].items():
        assert len(c["expected"]) == len(o["capacity"]["rho"])
        if plan != "Ground truth":  # nothing beats planning on the true process by more than sampling noise
            assert np.all(np.array(c["expected"]) >= np.array(gt) * 0.99)


def test_shrinkage_is_light_when_errors_share_a_strong_common_factor():
    rng = np.random.default_rng(5)
    common = rng.normal(size=(300, 1))
    e = common + 0.3 * rng.normal(size=(300, 6))
    cov = fu.shrunk_covariance(e)
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert corr[0, 1] > 0.8  # the true correlation (about 0.92) survives; full shrinkage would give 0
