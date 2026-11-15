"""Rule-based controllers to compare the learned agent against.

A learned controller only means something next to good simple rules, so the
rules here are tuned on validation days in exactly the same way as the agent.
"""
import numpy as np

from .config import Config
from .env import run_policy


class AlwaysOn:
    """No energy saving at all. This is the reference."""

    name = "Always on"

    def __init__(self, cfg: Config):
        self.n = cfg.n_cells

    def act(self, obs):
        return np.ones(self.n, dtype=np.int8)


class NightSchedule:
    """Fixed timetable: every small cell sleeps between two clock times."""

    name = "Night schedule"

    def __init__(self, cfg: Config, start_h=0.0, end_h=6.0):
        self.cfg, self.start_h, self.end_h = cfg, start_h, end_h

    def act(self, obs):
        h = obs.slot * 24.0 / self.cfg.steps_per_day
        if self.start_h <= self.end_h:
            night = self.start_h <= h < self.end_h
        else:
            night = h >= self.start_h or h < self.end_h
        return np.full(self.cfg.n_cells, 0 if night else 1, dtype=np.int8)


class Threshold:
    """Sleep when the cell's latest measured traffic is below a threshold."""

    name = "Threshold"

    def __init__(self, cfg: Config, theta=0.3):
        self.theta = theta

    def act(self, obs):
        return (obs.lags[:, 0] >= self.theta).astype(np.int8)


class LoadAwareThreshold:
    """Threshold rule that also checks the macro cell has room.

    Quiet cells are put to sleep one by one, quietest first, and the rule stops
    as soon as the macro cell would be filled above `beta` of its capacity.
    """

    name = "Load-aware threshold"

    def __init__(self, cfg: Config, theta=0.3, beta=0.8):
        self.cfg, self.theta, self.beta = cfg, theta, beta

    def act(self, obs):
        d = obs.lags[:, 0]
        cap = self.cfg.small_capacity
        action = np.ones(len(d), dtype=np.int8)
        load = obs.macro_prev + float(np.maximum(d - cap, 0.0).sum())
        budget = self.beta * self.cfg.macro_capacity
        for i in np.argsort(d):
            if d[i] >= self.theta or load + d[i] > budget:
                break
            action[i] = 0
            load += d[i]
        return action


class Clairvoyant:
    """A bound, not a real controller: it knows the next 15 minutes exactly.

    With the future known, the best choice is: sleep the quietest cells first,
    while (a) sleeping still saves power and (b) the macro cell is not
    overloaded. It also ignores the 30-minute dwell rule. No controller that
    avoids overload can use less energy than this.
    """

    name = "Clairvoyant bound"
    ignore_dwell = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.macro_slope = cfg.macro.n_trx * cfg.macro.delta_p * cfg.macro.p_max / cfg.macro_capacity

    def act(self, obs):
        cfg = self.cfg
        d = obs.next_demand
        cap = cfg.small_capacity
        action = np.ones(len(d), dtype=np.int8)
        load = obs.next_macro + float(np.maximum(d - cap, 0.0).sum())
        for i in np.argsort(d):
            if d[i] > cap:
                break
            saving = cfg.micro.active(d[i] / cap) - cfg.micro.sleep() - self.macro_slope * d[i]
            if saving <= 0 or load + d[i] > cfg.macro_capacity:
                break
            action[i] = 0
            load += d[i]
        return action


def pick_best(candidates, data, cfg: Config, max_unserved_pct):
    """Among candidate controllers, return the one with the lowest energy on
    `data` that keeps unserved traffic within the limit."""
    best, best_score = None, None
    for pol in candidates:
        s = run_policy(pol, data, cfg)
        ok = s["unserved_pct"] <= max_unserved_pct
        if ok and (best_score is None or s["energy_kwh_per_day"] < best_score["energy_kwh_per_day"]):
            best, best_score = pol, s
    return best, best_score
