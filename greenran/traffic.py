"""Synthetic traffic for a small heterogeneous network.

Twelve small cells of four area types (residential, business, transit,
nightlife) sit under one macro cell. Each type has its own daily shape,
weekends are different from weekdays, the load is noisy, and now and then a
cell has an abnormal event (a flash crowd or a silent drop).

Everything here is simulated. No operator data is used.
"""
import numpy as np

from .config import Config

TYPES = ("residential", "business", "transit", "nightlife")

NORMAL, FLASH, DROP = 0, 1, 2


def _bump(hours, centre, width):
    """Gaussian bump on a 24-hour circle."""
    d = np.abs(hours - centre)
    d = np.minimum(d, 24.0 - d)
    return np.exp(-0.5 * (d / width) ** 2)


def _raw_shape(kind, h):
    """Un-normalised (weekday, weekend) load shapes of an area type."""
    if kind == "residential":
        wd = 0.12 + 1.00 * _bump(h, 21.0, 2.5) + 0.30 * _bump(h, 13.0, 2.5) + 0.35 * _bump(h, 7.5, 1.2)
        we = 0.15 + 1.00 * _bump(h, 21.0, 2.8) + 0.60 * _bump(h, 13.0, 3.0) + 0.20 * _bump(h, 9.5, 1.5)
    elif kind == "business":
        wd = 0.04 + 1.00 * _bump(h, 11.0, 1.8) + 0.95 * _bump(h, 15.0, 1.8)
        we = 0.04 + 0.15 * (_bump(h, 11.0, 1.8) + 0.95 * _bump(h, 15.0, 1.8))
    elif kind == "transit":
        wd = 0.05 + 1.00 * _bump(h, 8.5, 1.0) + 1.00 * _bump(h, 18.0, 1.3) + 0.30 * _bump(h, 13.0, 3.0)
        we = 0.05 + 0.45 * _bump(h, 11.0, 3.0) + 0.50 * _bump(h, 18.0, 2.5)
    elif kind == "nightlife":
        wd = 0.06 + 0.80 * _bump(h, 22.5, 2.0) + 0.35 * _bump(h, 19.0, 1.5)
        we = 0.06 + 1.00 * _bump(h, 23.5, 2.5) + 0.50 * _bump(h, 19.5, 1.5)
    else:
        raise ValueError(kind)
    return wd, we


_GRID = np.linspace(0.0, 24.0, 481)
_TOP = {k: max(float(_raw_shape(k, _GRID)[0].max()), float(_raw_shape(k, _GRID)[1].max()))
        for k in ("residential", "business", "transit", "nightlife")}


def base_shape(kind, hours, weekend):
    """Typical load of an area type at the given hours (busiest moment of the week = 1)."""
    wd, we = _raw_shape(kind, hours)
    return (we if weekend else wd) / _TOP[kind]


_MACRO_TOP = float((0.18 + 0.80 * _bump(_GRID, 14.0, 4.5) + 0.75 * _bump(_GRID, 20.5, 2.5)).max())


def macro_shape(hours, weekend):
    """Typical load of the macro cell's own users (peak = 1)."""
    s = 0.18 + 0.80 * _bump(hours, 14.0, 4.5) + 0.75 * _bump(hours, 20.5, 2.5)
    s = s / _MACRO_TOP
    return 0.9 * s if weekend else s


def make_network(cfg: Config):
    """The fixed set of cells: type, peak size and a small time shift for each."""
    rng = np.random.default_rng(cfg.network_seed)
    types = np.array([i % len(TYPES) for i in range(cfg.n_cells)])
    peak = rng.uniform(0.60, 0.95, cfg.n_cells)
    shift = rng.uniform(-0.75, 0.75, cfg.n_cells)   # hours
    return {"type": types, "peak": peak, "shift": shift}


def _ar1(rng, n_steps, n_series, phi, sigma):
    """Zero-mean AR(1) noise with stationary standard deviation `sigma`."""
    x = np.zeros((n_steps, n_series))
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - phi ** 2), (n_steps, n_series))
    x[0] = rng.normal(0.0, sigma, n_series)
    for t in range(1, n_steps):
        x[t] = phi * x[t - 1] + innov[t]
    return x


def generate(cfg: Config, n_days, seed, start_weekday=0,
             p_flash=0.05, p_drop=0.03, noise=0.22):
    """Simulate `n_days` of traffic.

    Returns a dict with
      demand  (T, N)  traffic offered in each small cell's area (small-cell units)
      macro   (T,)    traffic of the macro cell's own users
      label   (T, N)  0 normal, 1 flash crowd, 2 drop
      weekend (T,)    True on Friday/Saturday
      slot    (T,)    15-minute slot of the day, 0..95
    """
    net = make_network(cfg)
    rng = np.random.default_rng(seed)
    S, N = cfg.steps_per_day, cfg.n_cells
    T = n_days * S
    slot = np.tile(np.arange(S), n_days)
    hours = slot * 24.0 / S
    day = np.repeat(np.arange(n_days), S)
    weekday = (day + start_weekday) % 7
    weekend = np.isin(weekday, cfg.weekend_days)

    demand = np.zeros((T, N))
    for i in range(N):
        kind = TYPES[net["type"][i]]
        h = (hours - net["shift"][i]) % 24.0
        shape = np.where(weekend, base_shape(kind, h, True), base_shape(kind, h, False))
        demand[:, i] = net["peak"][i] * shape

    day_scale = np.exp(rng.normal(0.0, 0.10, (n_days, N)))            # some days are busier
    x = _ar1(rng, T, N, phi=0.85, sigma=noise)                         # short bursts
    demand = demand * day_scale[day] * np.exp(x - 0.5 * noise ** 2)

    label = np.zeros((T, N), dtype=np.int8)
    for d in range(n_days):
        for i in range(N):
            if rng.random() < p_flash:                                 # flash crowd, 06:00-22:00
                s0 = d * S + rng.integers(S // 4, S - 8)
                s1 = min(s0 + rng.integers(4, 13), T)
                demand[s0:s1, i] = demand[s0:s1, i] * rng.uniform(1.5, 2.5) + rng.uniform(0.5, 1.0)
                label[s0:s1, i] = FLASH
            if rng.random() < p_drop:                                  # traffic silently disappears
                s0 = d * S + rng.integers(0, S - 6)
                s1 = min(s0 + rng.integers(6, 17), T)
                keep = label[s0:s1, i] == NORMAL
                demand[s0:s1, i] = np.where(keep, demand[s0:s1, i] * 0.05, demand[s0:s1, i])
                label[s0:s1, i] = np.where(keep, DROP, label[s0:s1, i])

    xm = _ar1(rng, T, 1, phi=0.85, sigma=0.08)[:, 0]
    macro = 0.55 * cfg.macro_capacity * np.where(weekend, macro_shape(hours, True), macro_shape(hours, False))
    macro = macro * np.exp(xm - 0.5 * 0.08 ** 2)

    return {"demand": demand, "macro": macro, "label": label,
            "weekend": weekend, "slot": slot, "true_type": net["type"]}
