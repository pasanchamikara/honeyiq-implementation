"""
Thesis-01 evaluation: a statistically rigorous primary protocol.

Motivation: the original primary evaluation (Chapter 4, Table 4.1) uses
benign_ratio=0.0 -- no benign traffic is explicitly injected, so the only
NORMAL-labelled samples the false-positive-rate estimate is based on are
whatever the attacker's own Markov chain happens to produce as "NORMAL"
transitions. That is a small, uncontrolled sample of the one class the
false-positive claim is actually about. This script:

  1. Replicates evaluate.py's exact classifier-driven decision logic
     (clf_state substitution, ground-truth scored against state[0:10],
     no label lag) so the numbers are directly comparable to Table 4.1.
  2. Runs BOTH the original protocol (benign_ratio=0.0, 30 episodes x 200
     steps) and a revised primary protocol (benign_ratio=0.3, 50 episodes
     x 500 steps -- reusing the episode/step budget already validated in
     the original thesis's Table 4.8 SEM analysis for mixed-traffic
     conditions) so the two can be compared directly, side by side.
  3. Tracks raw TP/FP/TN/FN counts (not just rates) and reports Wilson
     score 95% confidence intervals for detection rate and false-positive
     rate, so a rate estimated from a handful of samples is visibly less
     certain than one estimated from thousands.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import numpy as np

from attacker.attack_types import AttackerIntent, AttackType, KillChainStage
from defender.classifier import AttackClassifier
from defender.matrix_policy import MatrixPolicy
from environment.cyber_env import CyberSecurityEnv

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "thesis01_eval")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
Z_95 = 1.959963985


def wilson_ci(successes: int, n: int, z: float = Z_95):
    """Wilson score interval for a binomial proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0.0, center - margin), min(1.0, center + margin)


def run(intent: AttackerIntent, n_episodes: int, n_steps: int, benign_ratio: float,
        classifier: AttackClassifier):
    env = CyberSecurityEnv(attacker_intent=intent, max_steps=n_steps,
                            seed=SEED, benign_ratio=benign_ratio)
    mp = MatrixPolicy(default_intent=intent)

    tp = fp = tn = fn = 0

    for ep_idx in range(n_episodes):
        ep_seed = SEED + ep_idx * 13
        state, info = env.reset(seed=ep_seed)

        for step in range(n_steps):
            features = info.get("features", {})

            true_attack_type = AttackType(int(np.argmax(state[0:10])))
            state_is_attack = true_attack_type != AttackType.NORMAL

            if features and classifier.is_fitted:
                pred_attack = classifier.predict(features)
            else:
                pred_attack = true_attack_type

            clf_state = state.copy()
            clf_state[0:10] = 0.0
            clf_state[int(pred_attack)] = 1.0

            action_obj, _ = mp.decide_from_state(clf_state)
            action_int = int(action_obj)
            predicted_positive = action_int >= 1  # LOG or above == "treated as attack"

            if state_is_attack and predicted_positive:
                tp += 1
            elif state_is_attack and not predicted_positive:
                fn += 1
            elif not state_is_attack and predicted_positive:
                fp += 1
            else:
                tn += 1

            next_state, reward, terminated, truncated, info = env.step(action_int)
            state = next_state
            if terminated or truncated:
                break

    n_attack = tp + fn
    n_benign = fp + tn
    det_rate, det_lo, det_hi = wilson_ci(tp, n_attack) if n_attack else (float("nan"),) * 3
    fpr, fpr_lo, fpr_hi = wilson_ci(fp, n_benign) if n_benign else (float("nan"),) * 3

    return {
        "intent": intent.name, "benign_ratio": benign_ratio,
        "n_episodes": n_episodes, "n_steps": n_steps,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_attack_samples": n_attack, "n_benign_samples": n_benign,
        "detection_rate": det_rate, "detection_rate_ci95": [det_lo, det_hi],
        "false_positive_rate": fpr, "false_positive_rate_ci95": [fpr_lo, fpr_hi],
    }


if __name__ == "__main__":
    clf = AttackClassifier()
    clf.load("models/classifier.joblib")

    protocols = [
        {"label": "original_primary", "benign_ratio": 0.0, "n_episodes": 30, "n_steps": 200},
        {"label": "revised_primary",  "benign_ratio": 0.3, "n_episodes": 50, "n_steps": 500},
    ]

    all_results = []
    for proto in protocols:
        print(f"\n=== {proto['label']} (benign_ratio={proto['benign_ratio']}, "
              f"{proto['n_episodes']} episodes x {proto['n_steps']} steps) ===")
        for intent in AttackerIntent:
            r = run(intent, proto["n_episodes"], proto["n_steps"], proto["benign_ratio"], clf)
            r["protocol"] = proto["label"]
            all_results.append(r)
            det, (dlo, dhi) = r["detection_rate"], r["detection_rate_ci95"]
            fpr, (flo, fhi) = r["false_positive_rate"], r["false_positive_rate_ci95"]
            print(f"{intent.name:<14} "
                  f"det={det:.4f} [{dlo:.4f},{dhi:.4f}] (n_attack={r['n_attack_samples']})  "
                  f"fpr={fpr:.4f} [{flo:.4f},{fhi:.4f}] (n_benign={r['n_benign_samples']})")

    out_path = os.path.join(OUT_DIR, "primary_benign_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")
