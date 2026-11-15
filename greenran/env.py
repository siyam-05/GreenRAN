"""The sleep-control environment.

One macro cell gives coverage and is always on. Twelve small cells add
capacity. Every 15 minutes a controller decides, for each small cell, whether
it is ON or ASLEEP for the next 15 minutes. It only sees the past.

  * An ON small cell carries its own traffic (up to its capacity).
  * An ASLEEP small cell saves power, but its traffic falls on the macro cell.
  * If the macro cell gets more traffic than it can carry, the rest is
    UNSERVED. That is the quality-of-service cost of sleeping too much.
"""
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Obs:
    """What a controller may look at when it decides the state for interval t."""

    t: int
    lags: np.ndarray         # (N, n_lags) demand at t-1, t-2, ... (most recent first)
    slot: int                # 15-minute slot of interval t
    weekend: bool
    macro_prev: float        # macro's own traffic at t-1
    util_prev: float         # total macro load / capacity at t-1
    status: np.ndarray       # (N,) 1 = on, 0 = asleep (current state)
    locked: np.ndarray       # (N,) True = state cannot change yet (dwell time)
    next_demand: np.ndarray  # (N,) the FUTURE demand at t. Only the clairvoyant bound may read it.
    next_macro: float        # the FUTURE macro traffic at t. Same rule.


def physics(d, m, status, cfg: Config):
    """Power and unserved traffic for one interval.

    d: (N,) small-cell demand, m: macro's own demand, status: (N,) 1 = on.
    Returns (small-cell power (N,) [W], macro power [W], unserved traffic, macro load).
    """
    cap = cfg.small_capacity
    on = status.astype(bool)
    carried = np.where(on, np.minimum(d, cap), 0.0)
    load = m + float((d - carried).sum())                 # everything else lands on the macro
    unserved = max(load - cfg.macro_capacity, 0.0)
    rho = min(load, cfg.macro_capacity) / cfg.macro_capacity
    p_small = np.where(on, cfg.micro.active(carried / cap), cfg.micro.sleep())
    return p_small, cfg.macro.active(rho), unserved, load


class HetNetEnv:
    def __init__(self, data, cfg: Config):
        self.cfg = cfg
        self.d = data["demand"]
        self.m = data["macro"]
        self.slot = data["slot"]
        self.weekend = data["weekend"]
        self.T, self.N = self.d.shape
        self.reset()

    def reset(self):
        self.t = self.cfg.n_lags
        self.status = np.ones(self.N, dtype=np.int8)
        self.since = np.full(self.N, self.cfg.min_dwell)   # steps spent in the current state
        self.util_prev = float(self.m[self.t - 1]) / self.cfg.macro_capacity
        return self.observe()

    @property
    def done(self):
        return self.t >= self.T

    def observe(self):
        t = self.t
        lags = self.d[t - self.cfg.n_lags:t][::-1].T        # (N, n_lags), newest first
        return Obs(t=t, lags=lags, slot=int(self.slot[t]), weekend=bool(self.weekend[t]),
                   macro_prev=float(self.m[t - 1]), util_prev=self.util_prev,
                   status=self.status.copy(), locked=self.since < self.cfg.min_dwell,
                   next_demand=self.d[t], next_macro=float(self.m[t]))

    def step(self, request, force_on=None, ignore_dwell=False):
        """Apply the requested states for interval t and simulate it.

        request:  (N,) 1 = on, 0 = asleep.
        force_on: optional (N,) bool. These cells are switched on no matter what
                  (used by the safety guard).
        """
        cfg = self.cfg
        request = np.asarray(request, dtype=np.int8)
        if ignore_dwell:
            new = request.copy()
        else:
            new = np.where(self.since < cfg.min_dwell, self.status, request)
        if force_on is not None:
            new = np.where(force_on, 1, new).astype(np.int8)
        switched = new != self.status
        self.since = np.where(switched, 1, self.since + 1)
        self.status = new

        d, m = self.d[self.t], float(self.m[self.t])
        p_small, p_macro, unserved, load = physics(d, m, new, cfg)

        # ---- per-cell reward: "what did MY choice cost, compared with the other choice?" ----
        cap, cm = cfg.small_capacity, cfg.macro_capacity
        on = new.astype(bool)
        servable = np.minimum(d, cap)                       # traffic this cell carries when on
        load_if_on = load - servable                        # macro load if a sleeping cell were on
        macro_extra = cfg.macro.active(min(load, cm) / cm) - cfg.macro.active(np.minimum(load_if_on, cm) / cm)
        unserved_extra = unserved - np.maximum(load_if_on - cm, 0.0)
        stay_on_cost = cfg.micro.active(servable / cap) - cfg.micro.sleep()
        energy_term = np.where(on, stay_on_cost, macro_extra) / 100.0
        reward = -energy_term - cfg.lam * np.where(on, 0.0, unserved_extra) - cfg.switch_cost * switched

        self.util_prev = load / cm
        self.t += 1
        info = {"power": float(p_small.sum()) + p_macro, "unserved": unserved, "load": load,
                "demand": float(d.sum()) + m, "switches": int(switched.sum()),
                "asleep": int(self.N - new.sum()), "status": new}
        return reward, info


def run_policy(policy, data, cfg: Config, guard=None, trace=False):
    """Run one controller over a whole dataset and return its scores."""
    env = HetNetEnv(data, cfg)
    obs = env.reset()
    if guard is not None:
        guard.reset()
    ignore_dwell = getattr(policy, "ignore_dwell", False)
    energy_wh = unserved = demand = 0.0
    switches = asleep = steps = 0
    tr = {"asleep": [], "util": [], "power": [], "unserved": []} if trace else None
    while not env.done:
        request = policy.act(obs)
        force = guard.force_on(obs) if guard is not None else None
        _, info = env.step(request, force_on=force, ignore_dwell=ignore_dwell)
        energy_wh += info["power"] * cfg.step_hours
        unserved += info["unserved"]
        demand += info["demand"]
        switches += info["switches"]
        asleep += info["asleep"]
        steps += 1
        if trace:
            tr["asleep"].append(info["asleep"])
            tr["util"].append(info["load"] / cfg.macro_capacity)
            tr["power"].append(info["power"])
            tr["unserved"].append(info["unserved"])
        if not env.done:
            obs = env.observe()
    days = steps / cfg.steps_per_day
    out = {"energy_kwh_per_day": energy_wh / 1000.0 / days,
           "unserved_pct": 100.0 * unserved / demand,
           "switches_per_cell_day": switches / cfg.n_cells / days,
           "sleep_frac": asleep / (steps * cfg.n_cells)}
    if trace:
        out["trace"] = {k: np.array(v) for k, v in tr.items()}
    return out
