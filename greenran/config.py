"""All physical and experimental constants in one place.

Power numbers are the EARTH linear base-station power model
(Auer et al., "How much energy is needed to run a wireless network?",
IEEE Wireless Communications, 2011), LTE 10 MHz, 2x2 MIMO.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BSPower:
    """Supply power of one base station: P = N_trx * (P0 + delta_p * P_max * load)."""

    n_trx: int       # number of transceiver chains
    p_max: float     # max RF output power per chain [W]
    p0: float        # power at zero load per chain [W]
    delta_p: float   # slope of the load-dependent part
    p_sleep: float   # sleep-mode power per chain [W]

    def active(self, load):
        """Supply power [W] when switched on. `load` is in [0, 1]."""
        return self.n_trx * (self.p0 + self.delta_p * self.p_max * load)

    def sleep(self):
        """Supply power [W] in sleep mode."""
        return self.n_trx * self.p_sleep


MACRO = BSPower(n_trx=6, p_max=20.0, p0=130.0, delta_p=4.7, p_sleep=75.0)
MICRO = BSPower(n_trx=2, p_max=6.3, p0=56.0, delta_p=2.6, p_sleep=39.0)


@dataclass(frozen=True)
class Config:
    # ---- time ----
    steps_per_day: int = 96          # one step = 15 minutes
    step_hours: float = 0.25
    weekend_days: tuple = (4, 5)     # Friday, Saturday (day 0 = Monday): Bangladesh weekend

    # ---- network ----
    n_cells: int = 12                # small cells under one macro cell
    small_capacity: float = 1.0      # traffic a small cell can carry (this is the unit)
    macro_capacity: float = 4.0      # macro capacity, in small-cell units
    macro: BSPower = MACRO
    micro: BSPower = MICRO
    min_dwell: int = 4               # a cell must hold its on/off state >= 1 hour (no ping-pong)
    network_seed: int = 7            # fixes which cells exist (types, sizes)

    # ---- controller ----
    n_lags: int = 3                  # how many past demand samples the controller sees
    guard_steps: int = 4             # safety guard keeps a flagged cell on for 1 hour

    # ---- reward ----
    lam: float = 40.0                # weight of unserved traffic in the reward
    switch_cost: float = 0.02        # small penalty per on/off switch

    # ---- experiment protocol ----
    val_days: int = 56
    test_days: int = 112
    fit_days: int = 56               # days used to fit clustering and the anomaly detector
    val_seed: int = 1001
    test_seed: int = 424242
    fit_seed: int = 3003
    qos_budget_pct: float = 0.05     # max extra unserved traffic allowed (% of all demand)
