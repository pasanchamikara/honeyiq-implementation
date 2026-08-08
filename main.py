"""
HoneyIQ — Interactive demonstration and analysis tool.

Modes:
    demo      Run a single episode with a loaded (or freshly trained) defender.
    compare   Test the trained policy against all 4 attacker intents.
    train     Quick-start training wrapper (calls train.py logic).
    analyze   Visualize attack transition matrices and feature distributions.

Usage examples:
    python main.py demo --steps 100
    python main.py compare --steps 200 --episodes 5
    python main.py train --episodes 300
    python main.py analyze
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from attacker.attack_types import (
    AttackType,
    KillChainStage,
    AttackerIntent,
    INTENSITY_LOGNORMAL_SIGMA,
    NORMAL_PERSONA_WEIGHTS,
)
from attacker.transition_model import TransitionModel
from attacker.attacker import Attacker
from defender.classifier import AttackClassifier
from defender.defender import Defender
from defender.honeypot import HoneypotAction, threat_band
from environment.cyber_env import CyberSecurityEnv
from evaluation.metrics import MetricsCollector, StepRecord


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def run_demo(
    model_dir:    str = "models/",
    intent_name:  str = "OPPORTUNISTIC",
    n_steps:      int = 100,
    render:       bool = True,
    save_plot:    bool = True,
    log_dir:      str = "logs/",
) -> None:
    """
    Run one episode with the SEDM policy, printing step-by-step output.
    """
    intent   = AttackerIntent[intent_name]
    env      = CyberSecurityEnv(attacker_intent=intent, max_steps=n_steps)
    defender = Defender(default_intent=intent)

    # Load classifier if available; SEDM needs no DQN checkpoint
    clf_path = os.path.join(model_dir, "classifier.joblib")
    if os.path.exists(clf_path):
        defender.load(model_dir)
    else:
        print("[Demo] No classifier found — initializing fresh classifier.")
        defender.initialize_classifier()

    step_records: list[StepRecord] = []
    metrics = MetricsCollector(log_dir=log_dir)

    state, info = env.reset()
    print(f"\n{'='*70}")
    print(f"  HoneyIQ DEMO  —  Attacker Intent: {intent.name}")
    print(f"  Steps: {n_steps}")
    print(f"{'='*70}")
    _print_header()

    total_reward = 0.0

    for step in range(n_steps):
        features = info.get("features", {})
        action, pred_attack = defender.observe(state, features, training=False)

        # Ground truth for THIS decision — decoded from `state`/`info`, the
        # observation the action was actually chosen from — not from the
        # `info` returned by env.step() below, which already describes the
        # *next* step. Scoring against the wrong step's label silently shifts
        # every detection/FP metric by one time step (see docs/BUGS_AND_FIXES.md, Bug 8).
        attack_type  = info.get("attack_type", AttackType.NORMAL)
        stage        = info.get("kill_chain_stage", KillChainStage.RECONNAISSANCE)
        threat_level = info.get("threat_level", 0.0)
        is_attack    = info.get("is_attack", False)
        aligned_info = info

        next_state, reward, terminated, truncated, info = env.step(action)

        rec = StepRecord(
            episode=0, step=step, action=action, reward=reward,
            attack_type=int(attack_type),
            kill_chain_stage=int(stage),
            threat_level=threat_level,
            is_attack=is_attack,
            predicted_attack=int(pred_attack),
            loss=None,
            escalation_rate=aligned_info.get("escalation_rate", 0.0),
        )
        step_records.append(rec)
        metrics.record_step(0, step, action, reward, aligned_info, pred_attack, None)

        total_reward += reward

        if render:
            band = threat_band(threat_level)
            _print_step(
                step, attack_type, stage, threat_level, band,
                action, reward, pred_attack, is_attack,
            )

        state = next_state
        if terminated or truncated:
            break

    metrics.end_episode(0)

    print(f"\n{'='*70}")
    print(f"  Episode complete.  Total reward: {total_reward:.2f}")
    print(f"{'='*70}\n")

    if save_plot and step_records:
        os.makedirs(log_dir, exist_ok=True)
        metrics.plot_attack_progression(
            step_records,
            save_path=os.path.join(log_dir, "demo_progression.png"),
        )


def _print_header() -> None:
    print(f"{'Step':>5} {'Attack Type':<18} {'Stage':<22} "
          f"{'Threat':>7} {'Band':<9} {'Action':<8} {'Reward':>7} {'PredAtk':<16} {'IsAtk'}")
    print("-" * 110)


def _print_step(
    step:        int,
    attack_type: AttackType,
    stage:       KillChainStage,
    threat:      float,
    band:        str,
    action:      int,
    reward:      float,
    pred:        AttackType,
    is_attack:   bool,
) -> None:
    action_name = HoneypotAction(action).name
    # Colour coding (ANSI)
    colors = {"benign": "\033[32m", "low": "\033[34m",
              "medium": "\033[33m", "high": "\033[91m", "critical": "\033[31m"}
    reset = "\033[0m"
    c = colors.get(band, "")
    print(
        f"{step:>5} {attack_type.name:<18} {stage.name:<22} "
        f"{c}{threat:>7.3f} {band:<9}{reset} {action_name:<8} {reward:>7.2f} "
        f"{pred.name:<16} {'YES' if is_attack else 'no'}"
    )


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def run_compare(
    model_dir:  str = "models/",
    n_episodes: int = 5,
    n_steps:    int = 200,
    log_dir:    str = "logs/",
) -> None:
    """
    Evaluate the SEDM policy against all 4 attacker intents.
    No DQN checkpoint required — the matrix policy is fully deterministic.
    """
    print(f"\n{'='*70}")
    print(f"  HoneyIQ SEDM — Multi-Intent Policy Comparison")
    print(f"  Episodes per intent: {n_episodes}  |  Steps per episode: {n_steps}")
    print(f"{'='*70}\n")

    # Load the classifier once — it's intent-independent, so reuse it across
    # every intent below instead of re-reading classifier.joblib each time.
    shared_classifier = AttackClassifier()
    clf_path = os.path.join(model_dir, "classifier.joblib")
    if os.path.exists(clf_path):
        shared_classifier.load(clf_path)
        print(f"[Defender] Classifier loaded from {clf_path}")
    else:
        print(f"[Defender] Warning: Classifier not found at {clf_path}")

    results = {}
    for intent in AttackerIntent:
        env      = CyberSecurityEnv(attacker_intent=intent, max_steps=n_steps)
        defender = Defender(default_intent=intent)
        defender.classifier = shared_classifier

        ep_rewards = []
        ep_det     = []
        ep_fp      = []

        for ep_idx in range(n_episodes):
            metrics = MetricsCollector(log_dir=log_dir)
            state, info = env.reset()

            for step in range(n_steps):
                features = info.get("features", {})
                action, pred_attack = defender.observe(state, features, training=False)
                # Score against the ground truth of the observation the action
                # was chosen from (`info`), not the *next* step's info returned
                # by env.step() below — see docs/BUGS_AND_FIXES.md, Bug 8.
                aligned_info = info
                next_state, reward, terminated, truncated, info = env.step(action)
                metrics.record_step(ep_idx, step, action, reward, aligned_info, pred_attack, None)
                state = next_state
                if terminated or truncated:
                    break

            record = metrics.end_episode(ep_idx)
            ep_rewards.append(record.total_reward)
            ep_det.append(record.detection_rate)
            ep_fp.append(record.false_positive_rate)

        results[intent.name] = {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_det":    float(np.mean(ep_det)),
            "mean_fp":     float(np.mean(ep_fp)),
        }

    # Print table
    print(f"{'Intent':<16} {'MeanReward':>12} {'DetRate':>10} {'FPRate':>10}")
    print(f"{'-'*52}")
    for name, r in results.items():
        print(f"{name:<16} {r['mean_reward']:>12.2f} "
              f"{r['mean_det']:>10.3f} {r['mean_fp']:>10.3f}")
    print()


# ---------------------------------------------------------------------------
# Analyze mode — visualize transition matrices
# ---------------------------------------------------------------------------

def analyze(
    log_dir: str = "logs/",
) -> None:
    """
    Visualize the attack type transition matrices for each attacker intent,
    plus kill chain stage progression matrices.
    """
    os.makedirs(log_dir, exist_ok=True)
    attack_names = AttackType.names()
    stage_names  = KillChainStage.names()

    n_intents = AttackerIntent.count()
    fig, axes = plt.subplots(n_intents, 2, figsize=(18, 5 * n_intents))
    fig.suptitle("Attack Transition Matrices by Attacker Intent", fontsize=14, fontweight="bold")

    for row, intent in enumerate(AttackerIntent):
        tm = TransitionModel(intent=intent)

        atk_mat   = tm.get_attack_matrix()
        stage_mat = tm.get_stage_matrix()

        # Attack type matrix
        ax = axes[row, 0]
        sns.heatmap(
            atk_mat, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=attack_names, yticklabels=attack_names,
            ax=ax, vmin=0, vmax=1, cbar=False,
            annot_kws={"size": 6},
        )
        ax.set_title(f"[{intent.name}] Attack Type Transitions", fontsize=10)
        ax.set_xlabel("Next Attack")
        ax.set_ylabel("Current Attack")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

        # Kill chain stage matrix
        ax = axes[row, 1]
        sns.heatmap(
            stage_mat, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=stage_names, yticklabels=stage_names,
            ax=ax, vmin=0, vmax=1, cbar=False,
            annot_kws={"size": 8},
        )
        ax.set_title(f"[{intent.name}] Kill Chain Stage Transitions", fontsize=10)
        ax.set_xlabel("Next Stage")
        ax.set_ylabel("Current Stage")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    save_path = os.path.join(log_dir, "transition_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analyze] Transition matrices saved to {save_path}")

    # Feature distribution comparison
    _plot_feature_distributions(log_dir)


def _plot_feature_distributions(log_dir: str) -> None:
    """Plot box plots of key features across all attack types."""
    import pandas as pd

    n_samples = 300
    records   = []
    persona_names = list(NORMAL_PERSONA_WEIGHTS.keys())
    persona_probs = list(NORMAL_PERSONA_WEIGHTS.values())
    for attack_type in AttackType:
        attacker = Attacker(seed=42)
        # Draw a fresh intensity/persona per sample instead of relying on the
        # attacker's own session profile, so these 300 "independent" samples
        # don't all collapse onto one intensity/persona (which would show up
        # as artificially tight box plots).
        sample_rng = np.random.default_rng(42 + int(attack_type) + 1000)
        for _ in range(n_samples):
            intensity = float(sample_rng.lognormal(0.0, INTENSITY_LOGNORMAL_SIGMA))
            persona = (
                str(sample_rng.choice(persona_names, p=persona_probs))
                if attack_type == AttackType.NORMAL else None
            )
            feat = attacker._simulate_features(
                attack_type, intensity=intensity, persona=persona
            )
            feat["attack_type"] = attack_type.name
            records.append(feat)

    df = __import__("pandas").DataFrame(records)

    key_features = ["dur", "sload", "spkts", "sbytes", "ct_dst_ltm"]
    fig, axes = plt.subplots(1, len(key_features), figsize=(18, 6))
    fig.suptitle("Feature Distributions by Attack Type", fontsize=13, fontweight="bold")

    for i, feat in enumerate(key_features):
        ax = axes[i]
        df.boxplot(column=feat, by="attack_type", ax=ax, vert=True)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        if i == 0:
            ax.set_ylabel("Value (log scale)")
        ax.set_yscale("symlog")

    plt.suptitle("Feature Distributions by Attack Type", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(log_dir, "feature_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analyze] Feature distributions saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HoneyIQ — Cybersecurity Attacker-Defender Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py analyze
              python main.py train --episodes 300
              python main.py demo --intent STEALTHY --steps 150
              python main.py compare --episodes 10
        """),
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # demo
    p_demo = sub.add_parser("demo", help="Run a single episode demo")
    p_demo.add_argument("--model-dir",   default="models/")
    p_demo.add_argument("--log-dir",     default="logs/")
    p_demo.add_argument("--intent",      default="OPPORTUNISTIC",
                        choices=[i.name for i in AttackerIntent])
    p_demo.add_argument("--steps",       type=int, default=100)
    p_demo.add_argument("--no-render",   action="store_true")
    p_demo.add_argument("--no-plot",     action="store_true")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare policy across all intents")
    p_cmp.add_argument("--model-dir",   default="models/")
    p_cmp.add_argument("--log-dir",     default="logs/")
    p_cmp.add_argument("--episodes",    type=int, default=5)
    p_cmp.add_argument("--steps",       type=int, default=200)

    # train
    p_train = sub.add_parser("train", help="Train the defender")
    p_train.add_argument("--episodes",      type=int, default=300)
    p_train.add_argument("--steps",         type=int, default=500)
    p_train.add_argument("--intent",        default="OPPORTUNISTIC",
                         choices=[i.name for i in AttackerIntent])
    p_train.add_argument("--seed",          type=int, default=42)
    p_train.add_argument("--save-dir",      default="models/")
    p_train.add_argument("--log-dir",       default="logs/")
    p_train.add_argument("--eval-interval", type=int, default=50)
    p_train.add_argument("--save-interval", type=int, default=100)
    p_train.add_argument("--no-classifier", action="store_true")
    p_train.add_argument("--clf-samples",   type=int, default=600)
    p_train.add_argument("--all-intents",   action="store_true")

    # analyze
    p_an = sub.add_parser("analyze", help="Visualize transition matrices")
    p_an.add_argument("--log-dir", default="logs/")

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.mode == "demo":
        run_demo(
            model_dir   = args.model_dir,
            intent_name = args.intent,
            n_steps     = args.steps,
            render      = not args.no_render,
            save_plot   = not args.no_plot,
            log_dir     = args.log_dir,
        )

    elif args.mode == "compare":
        run_compare(
            model_dir  = args.model_dir,
            n_episodes = args.episodes,
            n_steps    = args.steps,
            log_dir    = args.log_dir,
        )

    elif args.mode == "train":
        from train import train, train_all_intents, AttackerIntent as AI
        intent = AI[args.intent]
        if args.all_intents:
            train_all_intents(
                n_episodes=args.episodes,
                max_steps=args.steps,
                seed=args.seed,
            )
        else:
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

    elif args.mode == "analyze":
        analyze(log_dir=args.log_dir)
