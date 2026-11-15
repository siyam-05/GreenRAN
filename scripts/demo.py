"""Short demo: train one small agent and compare it with the rules.

    python scripts/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from greenran.baselines import AlwaysOn, Clairvoyant, LoadAwareThreshold  # noqa: E402
from greenran.dqn import train_dqn  # noqa: E402
from greenran.env import run_policy  # noqa: E402
from greenran.kmeans import discover_types  # noqa: E402
from greenran.pipeline import make_config  # noqa: E402
from greenran.traffic import generate  # noqa: E402


def main():
    cfg = make_config()
    fit = generate(cfg, 28, seed=1)
    test = generate(cfg, 28, seed=2)
    types = discover_types(fit, cfg)
    print(f"K-means found {types['k']} kinds of daily traffic shape. Cell types: {types['cell_type']}")
    print("Training a DQN agent for 20 epochs...")
    agent, _ = train_dqn(cfg, seed=0, onehot=types["onehot"], epochs=20)
    agent2, _ = train_dqn(cfg, seed=0, onehot=types["onehot"], epochs=20, double=True)

    ref = run_policy(AlwaysOn(cfg), test, cfg)
    print(f"\n{'controller':<28}{'energy saved':>14}{'unserved traffic':>20}")
    for name, pol in (("Always on", AlwaysOn(cfg)),
                      ("Load-aware threshold rule", LoadAwareThreshold(cfg, 0.9, 0.8)),
                      ("DQN (short training)", agent),
                      ("Double DQN (short training)", agent2),
                      ("Clairvoyant bound", Clairvoyant(cfg))):
        s = run_policy(pol, test, cfg)
        saved = 100 * (1 - s["energy_kwh_per_day"] / ref["energy_kwh_per_day"])
        print(f"{name:<28}{saved:>13.2f}%{s['unserved_pct']:>19.4f}%")


if __name__ == "__main__":
    main()
