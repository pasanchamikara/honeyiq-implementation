"""
Thesis-01 evaluation: AdaptiveThresholds convergence.

Drives a long sequence of decisions (AGGRESSIVE intent, window-mode
escalation — the regime eval_escalation_mode.py showed has the highest
static R3 trigger rate, ~45%) through MatrixPolicy twice, using the
*same* underlying attacker trajectory (same seed) so the two runs are
directly comparable:

  - "static"   — RATE_THRESHOLD fixed at its default (0.80)
  - "adaptive" — AdaptiveThresholds attached, target_rate=0.10

Reports the R3 trigger rate in each successive observation_window-sized
block of decisions, and the threshold's value over time, showing whether
the controller actually pulls a ~45% natural trigger rate down toward its
10% target within its configured bound.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import numpy as np

from attacker.attack_types import AttackerIntent
from defender.matrix_policy import MatrixPolicy, RATE_THRESHOLD
from defender.adaptive_thresholds import AdaptiveThresholds
from environment.cyber_env import CyberSecurityEnv

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "thesis01_eval")
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 6000     # 30 windows of 200
WINDOW = 200
SEED = 42


def run(adaptive: bool, benign_ratio: float = 0.0):
    env = CyberSecurityEnv(
        attacker_intent=AttackerIntent.AGGRESSIVE, max_steps=N_STEPS,
        escalation_mode="window", seed=SEED, benign_ratio=benign_ratio,
    )
    controller = AdaptiveThresholds(initial_threshold=RATE_THRESHOLD,
                                     observation_window=WINDOW) if adaptive else None
    mp = MatrixPolicy(default_intent=AttackerIntent.AGGRESSIVE,
                       adaptive_thresholds=controller)

    state, info = env.reset(seed=SEED)
    r3_flags = []
    threshold_trace = []

    for step in range(N_STEPS):
        action, dinfo = mp.decide_from_state(state)
        r3_flags.append(dinfo["override_applied"] == "R3_HIGH_RATE")
        threshold_trace.append(controller.threshold if controller else RATE_THRESHOLD)
        next_state, reward, terminated, truncated, info = env.step(int(action))
        state = next_state
        if terminated or truncated:
            state, info = env.reset(seed=SEED + step + 1)

    # Per-window R3 rate and threshold-at-window-end
    windows = []
    for w_start in range(0, N_STEPS, WINDOW):
        w_flags = r3_flags[w_start:w_start + WINDOW]
        w_thresh = threshold_trace[w_start:w_start + WINDOW][-1]
        windows.append({
            "window_index": w_start // WINDOW,
            "r3_rate": float(np.mean(w_flags)),
            "threshold": float(w_thresh),
        })
    return windows


def summarize(label, benign_ratio):
    static_windows = run(adaptive=False, benign_ratio=benign_ratio)
    adaptive_windows = run(adaptive=True, benign_ratio=benign_ratio)

    print(f"\n=== {label} (benign_ratio={benign_ratio}) ===")
    print(f"{'window':>6} {'static_r3_rate':>15} {'adaptive_r3_rate':>17} {'adaptive_threshold':>19}")
    for s, a in zip(static_windows, adaptive_windows):
        print(f"{s['window_index']:>6} {s['r3_rate']:>15.4f} "
              f"{a['r3_rate']:>17.4f} {a['threshold']:>19.4f}")

    final_static_rate = float(np.mean([w["r3_rate"] for w in static_windows[-5:]]))
    final_adaptive_rate = float(np.mean([w["r3_rate"] for w in adaptive_windows[-5:]]))
    final_threshold = float(adaptive_windows[-1]["threshold"])

    print(f"\nFinal 5-window mean R3 rate — static: {final_static_rate:.4f}, "
          f"adaptive: {final_adaptive_rate:.4f}")
    print(f"Final adaptive threshold: {final_threshold:.4f} "
          f"(initial {RATE_THRESHOLD}, bound +/-0.10)")

    return {
        "label": label, "benign_ratio": benign_ratio,
        "static_windows": static_windows, "adaptive_windows": adaptive_windows,
        "final_static_rate": final_static_rate,
        "final_adaptive_rate": final_adaptive_rate,
        "final_threshold": final_threshold,
    }


if __name__ == "__main__":
    results = [
        summarize("continuous_attack", 0.0),
        summarize("mixed_traffic", 0.3),
    ]
    with open(os.path.join(OUT_DIR, "adaptive_thresholds_trace.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_DIR}/adaptive_thresholds_trace.json")
