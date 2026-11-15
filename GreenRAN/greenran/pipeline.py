"""Glue code: builds datasets, runs the three stages and scores controllers."""
import dataclasses

import numpy as np

from .anomaly import GaussianDetector, SafetyGuard
from .baselines import (AlwaysOn, Clairvoyant, LoadAwareThreshold, NightSchedule,
                        Threshold, pick_best)
from .config import MICRO, Config
from .dqn import MLP, DQNPolicy, N_BASE_FEATURES, train_dqn
from .env import run_policy
from .kmeans import discover_types, kmeans, daily_profiles, purity
from .traffic import generate

EARTH_SLEEP = MICRO.p_sleep / MICRO.p0      # 39 / 56 = 0.70: the light sleep mode of the EARTH model
DEEP_SLEEP = 0.10                           # assumed deep sleep: 10 % of the zero-load power


def make_config(sleep_frac=DEEP_SLEEP, min_dwell=4, lam=40.0):
    micro = dataclasses.replace(MICRO, p_sleep=MICRO.p0 * sleep_frac)
    return Config(micro=micro, min_dwell=min_dwell, lam=lam)


def datasets(cfg: Config):
    return {"fit": generate(cfg, cfg.fit_days, cfg.fit_seed),
            "val": generate(cfg, cfg.val_days, cfg.val_seed),
            "test": generate(cfg, cfg.test_days, cfg.test_seed)}


def score(policy, data, cfg, reference, guard=None):
    """Scores of a controller plus its energy saving relative to always-on."""
    s = run_policy(policy, data, cfg, guard=guard)
    s["saving_pct"] = 100.0 * (1.0 - s["energy_kwh_per_day"] / reference["energy_kwh_per_day"])
    return s


# ---------------------------------------------------------------- stage 1 + 2
def unsupervised_stage(cfg: Config, data):
    """K-means traffic typing and the Gaussian anomaly detector."""
    types = discover_types(data["fit"], cfg)
    truth = data["fit"]["true_type"]
    X, cell = daily_profiles(data["fit"], cfg)
    _, lab4, _ = kmeans(X, 4, seed=0)
    type4 = np.array([np.bincount(lab4[cell == i], minlength=4).argmax() for i in range(cfg.n_cells)])
    detector = GaussianDetector().fit(data["fit"], cfg)
    val_f1 = detector.select_threshold(data["val"])
    summary = {
        "k": int(types["k"]),
        "curve": types["curve"],
        "cell_type": types["cell_type"].tolist(),
        "true_type": truth.tolist(),
        "purity": float(purity(types["cell_type"], truth)),
        "purity_k4": float(purity(type4, truth)),
        "day_consistency": types["day_consistency"],
        "centroids": types["centroids"].tolist(),
        "anomaly": {"val_f1": float(val_f1), "test": detector.score(data["test"])},
    }
    return types, detector, summary


# ---------------------------------------------------------------- baselines
def baseline_stage(cfg: Config, data, fine=True):
    """Tune every rule on validation days, then score it once on test days."""
    val, test = data["val"], data["test"]
    ref_val = run_policy(AlwaysOn(cfg), val, cfg)
    ref_test = run_policy(AlwaysOn(cfg), test, cfg)
    limit = ref_val["unserved_pct"] + cfg.qos_budget_pct

    thetas = np.round(np.linspace(0.05, 1.0, 20 if fine else 8), 3)
    betas = np.round(np.arange(0.5, 1.0001, 0.025 if fine else 0.05), 3)
    grids = {
        "night": [NightSchedule(cfg, s, e) for s in (22.0, 23.0, 0.0, 1.0, 2.0) for e in (5.0, 6.0, 7.0, 8.0)],
        "threshold": [Threshold(cfg, th) for th in thetas],
        "load_aware": [LoadAwareThreshold(cfg, th, b) for th in (0.3, 0.5, 0.7, 0.9, 1.1) for b in betas],
    }
    out = {"always_on": {"test": score(AlwaysOn(cfg), test, cfg, ref_test)},
           "clairvoyant": {"test": score(Clairvoyant(cfg), test, cfg, ref_test)}}
    tuned = {}
    for name, cands in grids.items():
        pol, _ = pick_best(cands, val, cfg, limit)
        if pol is None:                       # nothing meets the limit: fall back to always-on
            pol = AlwaysOn(cfg)
        tuned[name] = pol
        params = {k: v for k, v in vars(pol).items() if k in ("theta", "beta", "start_h", "end_h")}
        out[name] = {"params": params, "val": score(pol, val, cfg, ref_val),
                     "test": score(pol, test, cfg, ref_test)}
    # trade-off curves on the test days (for the plot only, nothing is selected from them)
    th = tuned["load_aware"].theta if isinstance(tuned["load_aware"], LoadAwareThreshold) else 0.9
    out["frontier_load_aware"] = [
        {"beta": float(b), **score(LoadAwareThreshold(cfg, th, b), test, cfg, ref_test)} for b in betas]
    out["frontier_threshold"] = [
        {"theta": float(t), **score(Threshold(cfg, t), test, cfg, ref_test)} for t in thetas]
    return out, tuned, ref_val, ref_test


# ---------------------------------------------------------------- DQN jobs
def save_net(net, path):
    np.savez(path, **{f"W{i}": w for i, w in enumerate(net.W)}, **{f"b{i}": b for i, b in enumerate(net.b)})


def load_net(path):
    z = np.load(path)
    n = len([k for k in z.files if k.startswith("W")])
    net = MLP((1, 1), np.random.default_rng(0))
    net.W = [z[f"W{i}"] for i in range(n)]
    net.b = [z[f"b{i}"] for i in range(n)]
    return net


def dqn_job(job):
    """Train and score one agent. Runs in a worker process.

    job = dict(sleep_frac, min_dwell, lam, use_types, seed, epochs, model_path)
    """
    cfg = make_config(job["sleep_frac"], job["min_dwell"], job["lam"])
    data = datasets(cfg)
    onehot = discover_types(data["fit"], cfg)["onehot"] if job["use_types"] else None
    policy, history = train_dqn(cfg, seed=job["seed"], onehot=onehot, epochs=job.get("epochs", 60),
                                val=data["val"], eval_every=5)
    ref_val = run_policy(AlwaysOn(cfg), data["val"], cfg)
    ref_test = run_policy(AlwaysOn(cfg), data["test"], cfg)
    detector = GaussianDetector().fit(data["fit"], cfg)
    detector.select_threshold(data["val"])
    for h in history:
        h["saving_pct"] = 100.0 * (1.0 - h["energy_kwh_per_day"] / ref_val["energy_kwh_per_day"])
    if job.get("model_path"):
        save_net(policy.net, job["model_path"])
    return {"job": {k: v for k, v in job.items() if k != "model_path"},
            "history": history,
            "val": score(policy, data["val"], cfg, ref_val),
            "test": score(policy, data["test"], cfg, ref_test),
            "test_guard": score(policy, data["test"], cfg, ref_test,
                                guard=SafetyGuard(detector, data["test"], cfg))}


def load_policy(path, cfg: Config, onehot=None):
    net = load_net(path)
    assert net.W[0].shape[0] == N_BASE_FEATURES + (0 if onehot is None else onehot.shape[1])
    return DQNPolicy(net, cfg, onehot)
