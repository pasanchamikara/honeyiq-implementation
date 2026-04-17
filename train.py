"""
Training script for the HoneyIQ attacker-defender simulation.

Usage:
    python train.py [--episodes 500] [--intent OPPORTUNISTIC] [--seed 42]
                    [--steps 500] [--save-dir models/] [--log-dir logs/]
                    [--eval-interval 50] [--save-interval 100]
                    [--no-classifier]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from attacker.attack_types import AttackerIntent
from defender.defender import Defender
from environment.cyber_env import CyberSecurityEnv
from evaluation.metrics import MetricsCollector, StepRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_eval_episode(
    env:     CyberSecurityEnv,
    defender: Defender,
    metrics:  MetricsCollector,
    episode:  int,
) -> dict:
    """Run one greedy (no exploration) evaluation episode."""
    state, info = env.reset()
    step_records: list[StepRecord] = []
    total_reward = 0.0

    for step in range(env.max_steps):
        action, pred_attack = defender.observe(
            state, info.get("features", {}), training=False
        )
        next_state, reward, terminated, truncated, info = env.step(action)

        rec = StepRecord(
            episode=episode, step=step, action=action, reward=reward,
            attack_type=int(info["attack_type"]),
            kill_chain_stage=int(info["kill_chain_stage"]),
            threat_level=info["threat_level"],
            is_attack=info["is_attack"],
            predicted_attack=int(pred_attack),
            loss=None,
            escalation_rate=info["escalation_rate"],
        )
        step_records.append(rec)
        metrics.record_step(episode, step, action, reward, info, pred_attack, None)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break

    record = metrics.end_episode(episode)
    return {
        "total_reward":        record.total_reward,
        "detection_rate":      record.detection_rate,
        "false_positive_rate": record.false_positive_rate,
        "avg_threat_level":    record.avg_threat_level,
        "step_records":        step_records,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    n_episodes:        int = 300,
    max_steps:         int = 500,
    intent:            AttackerIntent = AttackerIntent.OPPORTUNISTIC,
    save_dir:          str = "models/",
    log_dir:           str = "logs/",
    seed:              int = 42,
    eval_interval:     int = 50,
    save_interval:     int = 100,
    train_classifier:  bool = True,
    n_clf_samples:     int = 600,
    verbose:           bool = True,
) -> MetricsCollector:
    """
    Full training loop.

    Parameters
    ----------
    n_episodes       : number of training episodes
    max_steps        : steps per episode
    intent           : attacker intent profile
    save_dir         : directory for model checkpoints
    log_dir          : directory for logs and plots
    seed             : global random seed
    eval_interval    : frequency (in episodes) of evaluation runs
    save_interval    : frequency (in episodes) of checkpoint saves
    train_classifier : whether to train the attack classifier
    n_clf_samples    : samples per class for classifier training
    verbose          : print progress

    Returns
    -------
    MetricsCollector populated with all training episode records.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------------------
    env = CyberSecurityEnv(
        attacker_intent=intent,
        max_steps=max_steps,
        seed=seed,
    )

    defender = Defender(
        dqn_config={
            "state_dim":          24,
            "action_dim":         5,
            "lr":                 1e-3,
            "gamma":              0.99,
            "epsilon_start":      1.0,
            "epsilon_end":        0.05,
            "epsilon_decay":      0.997,
            "batch_size":         64,
            "target_update_freq": 150,
            "buffer_capacity":    15_000,
        },
        classifier_config={
            "n_estimators": 150,
            "max_depth":    20,
            "n_jobs":       1,
        },
        train_classifier=train_classifier,
        seed=seed,
    )

    if train_classifier:
        defender.initialize_classifier(n_samples_per_class=n_clf_samples)
        # Print classifier accuracy
        eval_result = defender.classifier.evaluate(n_test_per_class=200, seed=999)
        print(f"[Classifier] Test accuracy: {eval_result['accuracy']:.3f}")

    metrics      = MetricsCollector(log_dir=log_dir)
    eval_metrics = MetricsCollector(log_dir=log_dir)   # separate eval tracker

    if verbose:
        print(f"\n{'='*60}")
        print(f"HoneyIQ Training — Intent: {intent.name}")
        print(f"Episodes: {n_episodes}  |  Steps/ep: {max_steps}")
        print(f"Device: {defender.dqn_agent.device}")
        print(f"{'='*60}\n")

    start_time = time.time()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    pbar = tqdm(range(n_episodes), desc="Training", unit="ep", disable=not verbose)

    for episode in pbar:
        state, info = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            features = info.get("features", {})
            action, pred_attack = defender.observe(state, features, training=True)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            loss = defender.learn(state, action, reward, next_state, done)

            metrics.record_step(episode, step, action, reward, info, pred_attack, loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        record = metrics.end_episode(episode)

        # Progress bar postfix
        pbar.set_postfix({
            "reward": f"{record.total_reward:.1f}",
            "det":    f"{record.detection_rate:.2f}",
            "fp":     f"{record.false_positive_rate:.2f}",
            "ε":      f"{defender.epsilon:.3f}",
        })

        # Evaluation run
        if eval_interval > 0 and (episode + 1) % eval_interval == 0:
            eval_ep = 10_000 + episode  # Use offset to distinguish eval episodes
            eval_result = run_eval_episode(env, defender, eval_metrics, eval_ep)
            if verbose:
                print(
                    f"\n[Eval  ep {episode+1:>4}]  "
                    f"reward={eval_result['total_reward']:>8.1f}  "
                    f"det={eval_result['detection_rate']:.3f}  "
                    f"fp={eval_result['false_positive_rate']:.3f}"
                )

        # Checkpoint save
        if save_interval > 0 and (episode + 1) % save_interval == 0:
            defender.save(save_dir)

    # ------------------------------------------------------------------
    # Final save and reporting
    # ------------------------------------------------------------------
    defender.save(save_dir)

    elapsed = time.time() - start_time
    summary = metrics.summary_report()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training complete in {elapsed:.1f}s")
        print(f"  Mean reward     : {summary.get('mean_reward', 0):.2f}")
        print(f"  Best reward     : {summary.get('best_episode_reward', 0):.2f}")
        print(f"  Mean det. rate  : {summary.get('mean_detection_rate', 0):.3f}")
        print(f"  Mean FP rate    : {summary.get('mean_false_positive_rate', 0):.3f}")
        print(f"  Mean threat lvl : {summary.get('mean_threat_level', 0):.3f}")
        print(f"{'='*60}\n")

    metrics.save_csv()
    metrics.plot_training_curves()
    metrics.plot_kill_chain_heatmap()

    return metrics


# ---------------------------------------------------------------------------
# Multi-intent comparison
# ---------------------------------------------------------------------------

def train_all_intents(
    n_episodes: int = 200,
    max_steps:  int = 300,
    seed:       int = 42,
    base_log_dir: str = "logs/",
    base_save_dir: str = "models/",
) -> dict[str, MetricsCollector]:
    """Train a separate defender for each attacker intent and compare."""
    results = {}
    for intent in AttackerIntent:
        print(f"\n{'#'*60}")
        print(f"# Training against {intent.name} attacker")
        print(f"{'#'*60}")
        log_dir  = os.path.join(base_log_dir,  intent.name.lower())
        save_dir = os.path.join(base_save_dir, intent.name.lower())
        mc = train(
            n_episodes=n_episodes,
            max_steps=max_steps,
            intent=intent,
            save_dir=save_dir,
            log_dir=log_dir,
            seed=seed,
        )
        results[intent.name] = mc

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Intent':<16} {'MeanReward':>12} {'DetRate':>10} {'FPRate':>10} {'AvgThreat':>12}")
    print(f"{'-'*80}")
    for name, mc in results.items():
        s = mc.summary_report()
        print(f"{name:<16} {s.get('mean_reward',0):>12.2f} "
              f"{s.get('mean_detection_rate',0):>10.3f} "
              f"{s.get('mean_false_positive_rate',0):>10.3f} "
              f"{s.get('mean_threat_level',0):>12.3f}")
    print(f"{'='*80}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HoneyIQ — Attacker-Defender Training")
    p.add_argument("--episodes",      type=int,   default=300,
                   help="Number of training episodes (default: 300)")
    p.add_argument("--steps",         type=int,   default=500,
                   help="Max steps per episode (default: 500)")
    p.add_argument("--intent",        type=str,   default="OPPORTUNISTIC",
                   choices=[i.name for i in AttackerIntent],
                   help="Attacker intent profile")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--save-dir",      type=str,   default="models/")
    p.add_argument("--log-dir",       type=str,   default="logs/")
    p.add_argument("--eval-interval", type=int,   default=50)
    p.add_argument("--save-interval", type=int,   default=100)
    p.add_argument("--no-classifier", action="store_true",
                   help="Skip classifier training (faster)")
    p.add_argument("--all-intents",   action="store_true",
                   help="Train against all 4 intent profiles")
    p.add_argument("--clf-samples",   type=int,   default=600,
                   help="Samples per class for classifier training")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.all_intents:
        train_all_intents(
            n_episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed,
        )
    else:
        intent = AttackerIntent[args.intent]
        train(
            n_episodes       = args.episodes,
            max_steps        = args.steps,
            intent           = intent,
            save_dir         = args.save_dir,
            log_dir          = args.log_dir,
            seed             = args.seed,
            eval_interval    = args.eval_interval,
            save_interval    = args.save_interval,
            train_classifier = not args.no_classifier,
            n_clf_samples    = args.clf_samples,
        )
