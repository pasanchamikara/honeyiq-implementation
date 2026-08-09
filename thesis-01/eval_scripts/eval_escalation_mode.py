"""
Thesis-01 evaluation: window vs. EMA escalation tracking.

Runs the SEDM (oracle mode — decides directly from ground-truth state,
isolating the effect of the escalation signal itself from classifier
noise) across all 4 intents under both escalation_mode settings, and
reports detection rate, false-positive rate, R3 trigger rate, and the
distribution of the escalation signal itself. This directly tests whether
RATE_THRESHOLD (calibrated against window semantics) needs recalibration
under EMA mode, as flagged in docs01/environment.md.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import numpy as np

from attacker.attack_types import AttackerIntent, AttackType
from defender.matrix_policy import MatrixPolicy
from environment.cyber_env import CyberSecurityEnv

N_EPISODES = 30
N_STEPS = 200
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "thesis01_eval")
os.makedirs(OUT_DIR, exist_ok=True)


def run(intent, mode):
    env = CyberSecurityEnv(
        attacker_intent=intent, max_steps=N_STEPS,
        escalation_mode=mode, seed=SEED,
    )
    mp = MatrixPolicy(default_intent=intent)

    tp = fp = tn = fn = 0
    override_counts = {"none": 0, "R1_NORMAL_ALLOW": 0, "R2_HIGH_IMPACT": 0,
                        "R3_HIGH_RATE": 0, "R4_REPEAT_OFFENDER": 0}
    esc_values = []
    rewards = []

    for ep in range(N_EPISODES):
        state, info = env.reset(seed=SEED + ep * 13)
        ep_reward = 0.0
        for step in range(N_STEPS):
            action, dinfo = mp.decide_from_state(state)
            override_counts[dinfo["override_applied"]] += 1
            esc_values.append(float(state[19]))

            # Ground truth: attack_type one-hot at state[0:10]; index 0 = NORMAL
            is_attack = int(np.argmax(state[0:10])) != int(AttackType.NORMAL)
            predicted_attack = int(action) >= 1  # LOG or above == "treated as attack"

            if is_attack and predicted_attack:
                tp += 1
            elif is_attack and not predicted_attack:
                fn += 1
            elif not is_attack and predicted_attack:
                fp += 1
            else:
                tn += 1

            next_state, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            state = next_state
            if terminated or truncated:
                break
        rewards.append(ep_reward)

    n = tp + fp + tn + fn
    detection_rate = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    r3_rate = override_counts["R3_HIGH_RATE"] / n

    return {
        "intent": intent.name, "mode": mode,
        "detection_rate": detection_rate, "false_positive_rate": fpr,
        "r3_trigger_rate": r3_rate,
        "override_counts": override_counts,
        "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        "escalation_mean": float(np.mean(esc_values)),
        "escalation_p95": float(np.percentile(esc_values, 95)),
        "escalation_max": float(np.max(esc_values)),
        "n_steps": n,
    }


if __name__ == "__main__":
    results = []
    for intent in AttackerIntent:
        for mode in ("window", "ema"):
            r = run(intent, mode)
            results.append(r)
            print(f"{intent.name:<14} {mode:<7} "
                  f"det={r['detection_rate']:.4f} fpr={r['false_positive_rate']:.4f} "
                  f"R3_rate={r['r3_trigger_rate']:.4f} "
                  f"esc_mean={r['escalation_mean']:.4f} esc_p95={r['escalation_p95']:.4f} "
                  f"esc_max={r['escalation_max']:.4f} reward={r['mean_reward']:.2f}")

    out_path = os.path.join(OUT_DIR, "escalation_mode_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
