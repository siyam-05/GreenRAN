"""Sanity checks. Run with:  python -m pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from greenran.anomaly import GaussianDetector  # noqa: E402
from greenran.baselines import AlwaysOn, Clairvoyant, Threshold  # noqa: E402
from greenran.config import MACRO, MICRO  # noqa: E402
from greenran.dqn import MLP, featurize  # noqa: E402
from greenran.env import HetNetEnv, physics, run_policy  # noqa: E402
from greenran.kmeans import kmeans  # noqa: E402
from greenran.pipeline import make_config  # noqa: E402
from greenran.traffic import FLASH, generate  # noqa: E402


def test_earth_power_model_matches_published_numbers():
    assert MACRO.active(1.0) == 6 * (130 + 4.7 * 20)        # 1344 W at full load
    assert MACRO.active(0.0) == 780
    assert MICRO.active(0.0) == 112
    assert MICRO.sleep() == 78


def test_traffic_is_conserved():
    cfg = make_config()
    d = np.array([0.2, 0.5, 1.4] + [0.1] * 9)
    all_on = np.ones(12, dtype=np.int8)
    _, _, unserved, load = physics(d, 1.0, all_on, cfg)
    assert np.isclose(load, 1.0 + 0.4)                       # only the overflow of the 1.4 cell
    assert unserved == 0.0
    _, _, unserved, load = physics(d, 1.0, np.zeros(12, dtype=np.int8), cfg)
    assert np.isclose(load, 1.0 + d.sum())
    assert np.isclose(unserved, max(load - cfg.macro_capacity, 0.0))


def test_same_seed_gives_same_traffic():
    cfg = make_config()
    a, b = generate(cfg, 3, seed=5), generate(cfg, 3, seed=5)
    assert np.array_equal(a["demand"], b["demand"])


def test_controller_cannot_see_the_future():
    cfg = make_config()
    data = generate(cfg, 2, seed=1)
    env = HetNetEnv(data, cfg)
    before = featurize(env.reset(), cfg)
    data["demand"][cfg.n_lags:] += 5.0                       # change everything from "now" on
    data["macro"][cfg.n_lags:] += 5.0
    after = featurize(HetNetEnv(data, cfg).reset(), cfg)
    assert np.array_equal(before, after)


def test_dwell_time_is_respected():
    cfg = make_config(min_dwell=4)
    data = generate(cfg, 2, seed=2)
    env = HetNetEnv(data, cfg)
    env.reset()
    history = []
    flip = 0
    while not env.done:
        flip = 1 - flip                                      # ask for a switch at every step
        _, info = env.step(np.full(cfg.n_cells, flip))
        history.append(info["status"][0])
    runs = np.diff(np.flatnonzero(np.diff(history)))         # lengths of constant runs
    assert runs.min() >= cfg.min_dwell


def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    net = MLP((5, 8, 8, 2), rng)
    x, y = rng.normal(size=(6, 5)), rng.normal(size=(6, 2))

    def loss():
        return 0.5 * ((net.forward(x) - y) ** 2).sum()

    out = net.forward(x, keep=True)
    gW, gb = net.backward(out - y)
    for params, grads in ((net.W, gW), (net.b, gb)):
        for p, g in zip(params, grads):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + 1e-5
            up = loss()
            p[idx] = old - 1e-5
            down = loss()
            p[idx] = old
            assert np.isclose((up - down) / 2e-5, g[idx], rtol=1e-4, atol=1e-6)


def test_kmeans_finds_three_obvious_groups():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [5, 5], [0, 5]])
    X = np.vstack([c + 0.3 * rng.normal(size=(40, 2)) for c in centres])
    _, labels, _ = kmeans(X, 3, seed=0)
    for g in range(3):
        assert len(set(labels[g * 40:(g + 1) * 40])) == 1    # each blob gets one label
    assert len(set(labels)) == 3


def test_detector_catches_flash_crowds():
    cfg = make_config()
    det = GaussianDetector().fit(generate(cfg, 28, seed=11), cfg)
    det.select_threshold(generate(cfg, 14, seed=12))
    test = generate(cfg, 14, seed=13)
    flags, _ = det.detect(test)
    assert flags[test["label"] == FLASH].mean() > 0.9
    assert flags[test["label"] == 0].mean() < 0.01           # few false alarms


def test_clairvoyant_bound_is_a_bound():
    cfg = make_config()
    data = generate(cfg, 14, seed=21)
    ref = run_policy(AlwaysOn(cfg), data, cfg)
    bound = run_policy(Clairvoyant(cfg), data, cfg)
    assert bound["unserved_pct"] <= ref["unserved_pct"] + 1e-12   # it never hurts service
    assert bound["energy_kwh_per_day"] < ref["energy_kwh_per_day"]
    for theta in (0.1, 0.2):                                 # cautious rules cannot beat it
        s = run_policy(Threshold(cfg, theta), data, cfg)
        if s["unserved_pct"] <= ref["unserved_pct"] + 1e-12:
            assert s["energy_kwh_per_day"] >= bound["energy_kwh_per_day"]
