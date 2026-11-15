"""Runs every experiment and writes results/results.json.

    python scripts/run_experiments.py            # full run (about 25 minutes on 2 CPU cores)
    python scripts/run_experiments.py --quick    # 1 seed, short training, to check it works

Protocol: every method is tuned on validation days only. Test days are
simulated with a different random seed and are scored once at the end.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from greenran.pipeline import (DEEP_SLEEP, EARTH_SLEEP, baseline_stage, datasets,  # noqa: E402
                               dqn_job, make_config, unsupervised_stage)

ROOT = os.path.join(os.path.dirname(__file__), "..")
SETTINGS = {
    "main": {"sleep_frac": DEEP_SLEEP, "min_dwell": 4,
             "desc": "deep sleep (10% of zero-load power), on/off state held >= 1 hour"},
    "short_hold": {"sleep_frac": DEEP_SLEEP, "min_dwell": 2,
                   "desc": "deep sleep, on/off state held >= 30 minutes"},
    "earth_sleep": {"sleep_frac": EARTH_SLEEP, "min_dwell": 4,
                    "desc": "light sleep of the EARTH model (70% of zero-load power), held >= 1 hour"},
}


def clean(o):
    """Make numpy things JSON friendly."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "trace"}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results_quick" if args.quick else "results")
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    t0 = time.time()
    results = {"settings": {}}

    # ---- stage 1 + 2: clustering and anomaly detection (same traffic in every setting) ----
    cfg = make_config()
    data = datasets(cfg)
    _, _, results["unsupervised"] = unsupervised_stage(cfg, data)
    u = results["unsupervised"]
    print(f"[unsupervised] K = {u['k']}, purity {u['purity']:.2f}, anomaly test F1 {u['anomaly']['test']['f1']:.3f}",
          flush=True)

    # ---- rule-based controllers and the clairvoyant bound ----
    for name, s in SETTINGS.items():
        cfg = make_config(s["sleep_frac"], s["min_dwell"])
        base, _, ref_val, _ = baseline_stage(cfg, data, fine=not args.quick)
        results["settings"][name] = {"desc": s["desc"], "sleep_frac": s["sleep_frac"],
                                     "min_dwell": s["min_dwell"], "baselines": base,
                                     "val_unserved_limit": ref_val["unserved_pct"] + cfg.qos_budget_pct}
        b = base
        print(f"[{name}] load-aware {b['load_aware']['test']['saving_pct']:.2f}% saved, "
              f"bound {b['clairvoyant']['test']['saving_pct']:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

    # ---- how much can be saved as sleep modes get deeper? (rules + bound only) ----
    sweep = []
    for frac in ([EARTH_SLEEP, 0.1] if args.quick else [EARTH_SLEEP, 0.5, 0.3, 0.2, 0.1, 0.05]):
        cfg = make_config(frac, 4)
        base, _, _, _ = baseline_stage(cfg, data, fine=False)
        sweep.append({"sleep_frac": frac,
                      "load_aware": base["load_aware"]["test"],
                      "clairvoyant": base["clairvoyant"]["test"]})
    results["sleep_depth_sweep"] = sweep
    print(f"[sweep] done ({time.time() - t0:.0f}s)", flush=True)

    # ---- DQN agents ----
    seeds = [0] if args.quick else [0, 1, 2, 3, 4]
    epochs = 12 if args.quick else 60
    jobs = []

    def add(setting, lam, use_types, seed_list):
        s = SETTINGS[setting]
        for seed in seed_list:
            tag = f"{setting}_lam{int(lam)}_{'types' if use_types else 'notypes'}_seed{seed}"
            jobs.append({"setting": setting, "sleep_frac": s["sleep_frac"], "min_dwell": s["min_dwell"],
                         "lam": lam, "use_types": use_types, "seed": seed, "epochs": epochs,
                         "model_path": os.path.join(out_dir, "models", tag + ".npz")})

    for lam in ([40.0] if args.quick else [20.0, 40.0, 80.0]):
        add("main", lam, True, seeds)
        add("main", lam, False, seeds)
    add("short_hold", 40.0, True, seeds)
    add("short_hold", 40.0, False, seeds)
    add("earth_sleep", 40.0, True, seeds[:3])

    runs = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(dqn_job, jobs)):
            runs.append(r)
            j = r["job"]
            print(f"[dqn {i + 1}/{len(jobs)}] {j['setting']} lam={j['lam']:.0f} types={j['use_types']} "
                  f"seed={j['seed']}: val saved {r['val']['saving_pct']:.2f}%, "
                  f"unserved {r['val']['unserved_pct']:.4f}%  ({time.time() - t0:.0f}s)", flush=True)
    results["dqn_runs"] = sorted(runs, key=lambda r: (r["job"]["setting"], r["job"]["lam"],
                                                      not r["job"]["use_types"], r["job"]["seed"]))

    # ---- choose the DQN penalty weight on validation days (same rule as the baselines) ----
    limit = results["settings"]["main"]["val_unserved_limit"]
    best, fallback = None, None
    for lam in sorted({r["job"]["lam"] for r in runs if r["job"]["setting"] == "main"}):
        group = [r for r in runs if r["job"]["setting"] == "main" and r["job"]["use_types"] and r["job"]["lam"] == lam]
        energy = np.mean([r["val"]["energy_kwh_per_day"] for r in group])
        unserved = np.mean([r["val"]["unserved_pct"] for r in group])
        print(f"[select] lam={lam:.0f}: val energy {energy:.2f} kWh/day, unserved {unserved:.4f}% (limit {limit:.4f}%)")
        if unserved <= limit and (best is None or energy < best[1]):
            best = (lam, energy)
        if fallback is None or unserved < fallback[1]:
            fallback = (lam, unserved)
    # if no weight meets the limit, take the one with the least unserved traffic and say so
    results["selected_lam"] = best[0] if best else fallback[0]
    results["selected_lam_meets_limit"] = best is not None
    results["runtime_minutes"] = (time.time() - t0) / 60.0

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(clean(results), f, indent=1)
    print(f"saved {out_dir}/results.json   ({results['runtime_minutes']:.1f} min)")


if __name__ == "__main__":
    main()
