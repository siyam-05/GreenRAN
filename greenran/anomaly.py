"""Gaussian anomaly detection for cell traffic, written in NumPy.

Idea (same as the course): learn what "normal" looks like, model each feature
with a Gaussian, and flag a sample when its probability p(x) is below a
threshold epsilon. Epsilon is chosen on a small labelled validation set by
the best F1 score.
"""
import numpy as np

from .config import Config
from .traffic import DROP, FLASH


def fit_profile(data, cfg: Config):
    """The usual traffic of every cell at every 15-minute slot.

    Median over the fitting days, separately for weekdays and weekends.
    Returns an array (2, slots, cells): index 0 = weekday, 1 = weekend.
    """
    S = cfg.steps_per_day
    d = data["demand"].reshape(-1, S, cfg.n_cells)
    is_weekend = data["weekend"].reshape(-1, S)[:, 0]
    return np.stack([np.median(d[~is_weekend], axis=0), np.median(d[is_weekend], axis=0)])


def features(data, profile, c=0.05):
    """Two features per (time, cell):
       f1 = log of (measured traffic / usual traffic)   -> "how far from normal"
       f2 = change of f1 since the previous 15 minutes  -> "how suddenly"
    Both use only measurements available at that time (no look-ahead)."""
    expected = profile[data["weekend"].astype(int), data["slot"]]          # (T, N)
    f1 = np.log((data["demand"] + c) / (expected + c))
    f2 = np.vstack([np.zeros((1, f1.shape[1])), np.diff(f1, axis=0)])
    return np.stack([f1, f2], axis=2)                                      # (T, N, 2)


class GaussianDetector:
    def fit(self, data, cfg: Config):
        """Learn the usual profile and the Gaussian of each feature (unlabelled data)."""
        self.cfg = cfg
        self.profile = fit_profile(data, cfg)
        f = features(data, self.profile).reshape(-1, 2)
        self.mu = f.mean(axis=0)
        self.var = f.var(axis=0)
        self.log_eps = None
        return self

    def log_p(self, data):
        f = features(data, self.profile)
        return (-0.5 * np.log(2 * np.pi * self.var) - (f - self.mu) ** 2 / (2 * self.var)).sum(axis=2)

    def select_threshold(self, val):
        """Pick epsilon with the best F1 on labelled validation data."""
        lp = self.log_p(val).ravel()
        y = (val["label"].ravel() > 0)
        best_f1, best_eps = -1.0, None
        for eps in np.quantile(lp, np.linspace(0.0005, 0.05, 200)):
            f1 = prf(lp < eps, y)["f1"]
            if f1 > best_f1:
                best_f1, best_eps = f1, eps
        self.log_eps = float(best_eps)
        return best_f1

    def detect(self, data):
        """Returns (flags (T, N) bool, spike (T, N) bool: flagged AND above normal)."""
        flags = self.log_p(data) < self.log_eps
        f1 = features(data, self.profile)[:, :, 0]
        return flags, flags & (f1 > 0)

    def score(self, data):
        flags, _ = self.detect(data)
        y = data["label"]
        out = prf(flags.ravel(), y.ravel() > 0)
        out["recall_flash"] = float(flags[y == FLASH].mean())
        out["recall_drop"] = float(flags[y == DROP].mean())
        out["flagged_share"] = float(flags.mean())
        out["anomaly_share"] = float((y > 0).mean())
        return out


def prf(pred, truth):
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class SafetyGuard:
    """Uses the detector online: when a cell showed a traffic spike in the last
    interval, keep that cell ON for the next hour, whatever the controller says."""

    def __init__(self, detector, data, cfg: Config):
        _, self.spike = detector.detect(data)
        self.steps = cfg.guard_steps
        self.reset()

    def reset(self):
        self.count = np.zeros(self.spike.shape[1], dtype=int)

    def force_on(self, obs):
        self.count = np.where(self.spike[obs.t - 1], self.steps, np.maximum(self.count - 1, 0))
        return self.count > 0
