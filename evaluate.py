"""
HoneyIQ — Comprehensive evaluation for the Stage-Escalation Decision Matrix
(SEDM) policy.

Evaluates the SEDM across all 4 attacker intents, integrates the OpenCanary
emulator (DummyHoneypot), and saves thesis-quality results.

Usage:
    python evaluate.py [--episodes 30] [--steps 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from attacker.attack_types import AttackerIntent, AttackType, KillChainStage
from defender.classifier import AttackClassifier
from defender.defender import Defender
from defender.honeypot import HoneypotAction
from defender.matrix_policy import (
    ESC_HIGH_THRESHOLD,
    ESC_LOW_THRESHOLD,
    MatrixPolicy,
)
from environment.cyber_env import CyberSecurityEnv
from evaluation.metrics import MetricsCollector

from opencanary_integration.emulator.honeypot_emulator import DummyHoneypot
from opencanary_integration.emulator.scenario import EmulatorScenario

RESULTS_DIR = os.path.join("results", "evaluation")

INTENT_COLORS = {
    "STEALTHY":      "#4CAF50",
    "AGGRESSIVE":    "#F44336",
    "TARGETED":      "#FF9800",
    "OPPORTUNISTIC": "#2196F3",
}
ACTION_COLORS = {
    "ALLOW": "#66BB6A",
    "LOG":   "#42A5F5",
    "TROLL": "#FFA726",
    "BLOCK": "#EF5350",
    "ALERT": "#AB47BC",
}
ESC_BANDS = [
    f"Low\n(<{ESC_LOW_THRESHOLD})",
    f"Medium\n({ESC_LOW_THRESHOLD}–{ESC_HIGH_THRESHOLD})",
    f"High\n(≥{ESC_HIGH_THRESHOLD})",
]
STAGE_LABELS = KillChainStage.names()
ACTION_NAMES = HoneypotAction.names()


def _save_fig(fig, out_dir: str, filename: str, label: str | None = None) -> None:
    """Finalize, save, and close a figure, with a consistent [Eval] log line."""
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] {label or filename} → {path}")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent:         str
    rewards:        List[float] = field(default_factory=list)
    det_rates:      List[float] = field(default_factory=list)
    fp_rates:       List[float] = field(default_factory=list)
    threat_levels:  List[float] = field(default_factory=list)
    action_counts:  Counter     = field(default_factory=Counter)
    kc_counts:      Counter     = field(default_factory=Counter)
    risk_scores:    List[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards))

    @property
    def mean_det(self) -> float:
        return float(np.mean(self.det_rates))

    @property
    def std_det(self) -> float:
        return float(np.std(self.det_rates))

    @property
    def mean_fp(self) -> float:
        return float(np.mean(self.fp_rates))

    @property
    def std_fp(self) -> float:
        return float(np.std(self.fp_rates))

    @property
    def mean_threat(self) -> float:
        return float(np.mean(self.threat_levels))

    @property
    def mean_risk(self) -> float:
        return float(np.mean(self.risk_scores)) if self.risk_scores else 0.0


# ---------------------------------------------------------------------------
# Matrix visualisation
# ---------------------------------------------------------------------------

def plot_decision_matrix(out_dir: str) -> None:
    """
    Visualise the 7×3 SEDM as a heatmap and save it.
    Action integers: ALLOW=0 LOG=1 TROLL=2 BLOCK=3 ALERT=4
    """
    matrix = np.array([[int(a) for a in row] for row in MatrixPolicy.get_matrix_actions()])

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = matplotlib.colors.ListedColormap(
        [ACTION_COLORS[a] for a in ACTION_NAMES]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells with action names
    for i in range(7):
        for j in range(3):
            ax.text(j, i, ACTION_NAMES[matrix[i, j]],
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white")

    ax.set_xticks(range(3))
    ax.set_xticklabels(ESC_BANDS, fontsize=9)
    ax.set_yticks(range(7))
    ax.set_yticklabels(STAGE_LABELS, fontsize=9)
    ax.set_xlabel("Escalation Risk Band", fontsize=11, labelpad=8)
    ax.set_ylabel("Kill Chain Stage", fontsize=11, labelpad=8)
    ax.set_title(
        "Stage-Escalation Decision Matrix (SEDM)\n"
        "Recommended Honeypot Action per (Stage × Escalation Band)",
        fontsize=12, fontweight="bold",
    )

    # Legend
    patches = [
        matplotlib.patches.Patch(color=ACTION_COLORS[a], label=a)
        for a in ACTION_NAMES
    ]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.28, 1.0), fontsize=9, framealpha=0.9)

    _save_fig(fig, out_dir, "sedm_decision_matrix.png", "SEDM heatmap")


def plot_escalation_risk_per_intent(out_dir: str) -> None:
    """
    Show how escalation risk varies per stage for each attacker intent.
    """
    from attacker.transition_model import TransitionModel

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle(
        "Escalation Risk per Kill Chain Stage by Attacker Intent",
        fontsize=13, fontweight="bold",
    )

    stages = list(KillChainStage)
    for ax, intent in zip(axes, AttackerIntent):
        tm    = TransitionModel(intent=intent)
        risks = []
        for s in stages:
            probs = tm.get_stage_probabilities(s)
            risks.append(float(probs[int(s) + 1:].sum()))

        colors = ["#F44336" if r >= ESC_HIGH_THRESHOLD
                  else "#FF9800" if r >= ESC_LOW_THRESHOLD
                  else "#4CAF50" for r in risks]
        bars = ax.bar(range(7), risks, color=colors, edgecolor="white", linewidth=0.6)
        ax.axhline(ESC_HIGH_THRESHOLD, color="#F44336", linestyle="--", alpha=0.7, linewidth=1.2, label="High band")
        ax.axhline(ESC_LOW_THRESHOLD, color="#FF9800", linestyle="--", alpha=0.7, linewidth=1.2, label="Med band")
        ax.set_xticks(range(7))
        ax.set_xticklabels(
            [s.name.replace("_", "\n") for s in stages],
            fontsize=7, rotation=0,
        )
        ax.set_title(intent.name, fontsize=11, fontweight="bold",
                     color=INTENT_COLORS[intent.name])
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, risks):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("P(Escalation to Next Stage)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")

    _save_fig(fig, out_dir, "escalation_risk_per_intent.png", "Escalation risk chart")


def plot_effective_policy_per_intent(out_dir: str) -> None:
    """
    Show which action the SEDM selects for each stage under each intent
    (using a representative EXPLOITS attack and escalation_rate=0.5).
    """
    mp = MatrixPolicy()
    intents = list(AttackerIntent)
    stages  = list(KillChainStage)

    # Build (4 intents × 7 stages) action matrix
    action_matrix = np.zeros((len(intents), len(stages)), dtype=int)
    for i, intent in enumerate(intents):
        for j, stage in enumerate(stages):
            action, _ = mp.decide(
                current_stage   = stage,
                current_attack  = AttackType.EXPLOITS,
                escalation_rate = 0.5,
                intent          = intent,
            )
            action_matrix[i, j] = int(action)

    fig, ax = plt.subplots(figsize=(13, 4))
    cmap = matplotlib.colors.ListedColormap(
        [ACTION_COLORS[a] for a in ACTION_NAMES]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(action_matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(len(intents)):
        for j in range(len(stages)):
            ax.text(j, i, ACTION_NAMES[action_matrix[i, j]],
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white")

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(intents)))
    ax.set_yticklabels([i.name for i in intents], fontsize=9)
    ax.set_xlabel("Kill Chain Stage", fontsize=11, labelpad=8)
    ax.set_ylabel("Attacker Intent", fontsize=11, labelpad=8)
    ax.set_title(
        "Effective SEDM Policy per Intent × Stage\n"
        "(attack=EXPLOITS, escalation_rate=0.50)",
        fontsize=12, fontweight="bold",
    )
    patches = [
        matplotlib.patches.Patch(color=ACTION_COLORS[a], label=a)
        for a in ACTION_NAMES
    ]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.18, 1.0), fontsize=9, framealpha=0.9)

    _save_fig(fig, out_dir, "effective_policy_per_intent.png", "Effective policy heatmap")


# ---------------------------------------------------------------------------
# RL-environment evaluation
# ---------------------------------------------------------------------------

def evaluate_intent(
    intent:       AttackerIntent,
    n_episodes:   int,
    n_steps:      int,
    seed:         int,
    out_dir:      str,
    honeypot:     DummyHoneypot,
    benign_ratio: float = 0.0,
    classifier:   Optional[AttackClassifier] = None,
) -> IntentResult:
    """
    Run `n_episodes` greedy evaluation episodes for one attacker intent.
    The SEDM is used — no model checkpoint is required.
    Actions are forwarded to the DummyHoneypot emulator.
    """
    result   = IntentResult(intent=intent.name)
    env      = CyberSecurityEnv(attacker_intent=intent, max_steps=n_steps,
                                seed=seed, benign_ratio=benign_ratio)
    defender = Defender(
        train_classifier=False,
        default_intent=intent,
    )
    if classifier is not None:
        defender.classifier = classifier
    else:
        # Load classifier if available
        defender.load("models/")

    fake_ip = f"10.{list(AttackerIntent).index(intent)}.0.1"
    mp      = MatrixPolicy(default_intent=intent)
    metrics = MetricsCollector(log_dir=out_dir)

    for ep_idx in range(n_episodes):
        ep_seed  = seed + ep_idx * 13
        state, info = env.reset(seed=ep_seed)

        for step in range(n_steps):
            features = info.get("features", {})

            # ── Ground truth for THIS step (no lag) ─────────────────────────
            # state[0:10] encodes the attack type that drove this observation.
            # This is the correct label to score the action against.
            true_attack_type = AttackType(int(np.argmax(state[0:10])))
            state_is_attack  = true_attack_type != AttackType.NORMAL

            # ── Classifier-in-loop: SEDM acts on noisy predicted attack type ─
            # The RF classifier predicts attack type from raw network features.
            # We replace the ground-truth one-hot in the state with the
            # classifier's prediction before passing to the SEDM.  This breaks
            # the tautology where SEDM's R1 rule and the is_attack metric share
            # the same ground-truth input, producing meaningful FP/FN values.
            if features and defender.classifier.is_fitted:
                pred_attack = defender.classifier.predict(features)
            else:
                pred_attack = true_attack_type   # fallback: no features yet

            clf_state = state.copy()
            clf_state[0:10] = 0.0
            clf_state[int(pred_attack)] = 1.0

            # SEDM decision based on classifier-predicted attack type
            action_obj, sedm_info = mp.decide_from_state(clf_state)
            action_int = int(action_obj)
            result.risk_scores.append(sedm_info["composite_risk"])

            next_state, reward, terminated, truncated, info = env.step(action_int)

            # Score against true ground truth (state_is_attack), not classifier
            info["is_attack"] = state_is_attack
            metrics.record_step(ep_idx, step, action_int, reward, info, pred_attack, None)

            attack_name = AttackType(int(info["attack_type"])).name
            stage_name  = KillChainStage(int(info["kill_chain_stage"])).name

            honeypot.apply_action_sync(
                src_ip       = fake_ip,
                action       = HoneypotAction(action_int).name,
                attack_type  = attack_name,
                stage        = stage_name,
                threat_level = float(info["threat_level"]),
                event_id     = f"{intent.name}-ep{ep_idx}-s{step}",
            )

            result.action_counts[HoneypotAction(action_int).name] += 1
            result.kc_counts[stage_name] += 1

            state = next_state
            if terminated or truncated:
                break

        rec = metrics.end_episode(ep_idx)
        result.rewards.append(rec.total_reward)
        result.det_rates.append(rec.detection_rate)
        result.fp_rates.append(rec.false_positive_rate)
        result.threat_levels.append(rec.avg_threat_level)

    return result


# ---------------------------------------------------------------------------
# OpenCanary kill-chain demo
# ---------------------------------------------------------------------------

def run_opencanary_demo(intent: str, out_dir: str) -> None:
    audit_file = os.path.join(out_dir, f"opencanary_{intent.lower()}_audit.jsonl")
    emu = EmulatorScenario(
        model_dir  = "models/",
        intent     = intent,
        audit_file = audit_file,
        verbose    = False,
    )
    emu.run_kill_chain()
    log = emu.honeypot.get_action_log()
    print(f"  [OpenCanary-Emulator] {intent}: {len(log)} kill-chain events → {audit_file}")


# ---------------------------------------------------------------------------
# Plots and tables
# ---------------------------------------------------------------------------

def plot_metric_comparison(results: Dict[str, IntentResult], out_dir: str) -> None:
    intents   = list(results.keys())
    det_means = [results[i].mean_det    for i in intents]
    det_stds  = [results[i].std_det     for i in intents]
    fp_means  = [results[i].mean_fp     for i in intents]
    fp_stds   = [results[i].std_fp      for i in intents]
    rw_means  = [results[i].mean_reward for i in intents]
    rw_stds   = [results[i].std_reward  for i in intents]

    x      = np.arange(len(intents))
    colors = [INTENT_COLORS[i] for i in intents]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HoneyIQ SEDM — Policy Evaluation Across Attacker Intents",
                 fontsize=14, fontweight="bold")

    for ax, means, stds, title, ylabel in [
        (axes[0], det_means, det_stds, "Detection Rate (TPR)", "Rate"),
        (axes[1], fp_means,  fp_stds,  "False Positive Rate (FPR)", "Rate"),
        (axes[2], rw_means,  rw_stds,  "Mean Episode Reward", "Reward"),
    ]:
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5,
                      edgecolor="white", linewidth=0.8,
                      error_kw={"elinewidth": 1.5})
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(intents, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        if title != "Mean Episode Reward":
            ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.02 if val >= 0 else -0.08),
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save_fig(fig, out_dir, "metric_comparison.png")


def plot_reward_boxplot(results: Dict[str, IntentResult], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data    = [results[i].rewards for i in results]
    labels  = list(results.keys())
    colors  = [INTENT_COLORS[i] for i in labels]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"linewidth": 2, "color": "white"},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Attacker Intent", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Episode Reward Distribution (SEDM Policy)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_dir, "reward_boxplot.png")


def plot_action_distribution(results: Dict[str, IntentResult], out_dir: str) -> None:
    intents = list(results.keys())
    fracs   = np.array([
        [results[i].action_counts[a] / max(sum(results[i].action_counts.values()), 1)
         for a in ACTION_NAMES]
        for i in intents
    ])
    fig, ax = plt.subplots(figsize=(11, 5))
    x       = np.arange(len(intents))
    bottom  = np.zeros(len(intents))
    for j, action in enumerate(ACTION_NAMES):
        bars = ax.bar(x, fracs[:, j], bottom=bottom, label=action,
                      color=ACTION_COLORS[action], edgecolor="white", linewidth=0.5)
        for bi, bar in enumerate(bars):
            frac = fracs[bi, j]
            if frac > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bottom[bi] + frac/2,
                        f"{frac:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottom += fracs[:, j]
    ax.set_xticks(x); ax.set_xticklabels(intents, fontsize=10)
    ax.set_ylabel("Action Fraction", fontsize=11)
    ax.set_title("SEDM Action Distribution by Attacker Intent",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _save_fig(fig, out_dir, "action_distribution.png")


def plot_composite_risk_distribution(results: Dict[str, IntentResult], out_dir: str) -> None:
    """Violin plot of composite risk scores across intents."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data    = [results[i].risk_scores for i in results]
    labels  = list(results.keys())
    colors  = [INTENT_COLORS[i] for i in labels]

    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmedians=True, showextrema=True)
    for i, (body, color) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(color); body.set_alpha(0.7)
    parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Composite Risk Score", fontsize=11)
    ax.set_title("SEDM Composite Risk Score Distribution",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.20, color="#66BB6A", linestyle="--", alpha=0.6, linewidth=1, label="→ALLOW")
    ax.axhline(0.40, color="#42A5F5", linestyle="--", alpha=0.6, linewidth=1, label="→LOG")
    ax.axhline(0.60, color="#FFA726", linestyle="--", alpha=0.6, linewidth=1, label="→TROLL")
    ax.axhline(0.80, color="#EF5350", linestyle="--", alpha=0.6, linewidth=1, label="→BLOCK/ALERT")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _save_fig(fig, out_dir, "composite_risk_distribution.png")


def save_summary_table(results: Dict[str, IntentResult], out_dir: str) -> None:
    rows = []
    for intent, res in results.items():
        rows.append({
            "Intent":            intent,
            "Episodes":          len(res.rewards),
            "MeanReward":        round(res.mean_reward, 2),
            "StdReward":         round(res.std_reward,  2),
            "MinReward":         round(float(np.min(res.rewards)), 2),
            "MaxReward":         round(float(np.max(res.rewards)), 2),
            "DetectionRate":     round(res.mean_det,    4),
            "StdDetRate":        round(res.std_det,     4),
            "FalsePositiveRate": round(res.mean_fp,     4),
            "StdFPRate":         round(res.std_fp,      4),
            "AvgThreatLevel":    round(res.mean_threat, 4),
            "AvgCompositeRisk":  round(res.mean_risk,   4),
        })
    df   = pd.DataFrame(rows)
    path = os.path.join(out_dir, "evaluation_summary.csv")
    df.to_csv(path, index=False)
    print(f"\n[Eval] Summary table → {path}")
    print(df.to_string(index=False))


def save_action_distribution_table(results: Dict[str, IntentResult], out_dir: str) -> None:
    """Save the per-intent action distribution (counts + %) as a CSV, alongside the plot."""
    rows = []
    for intent, res in results.items():
        total = max(sum(res.action_counts.values()), 1)
        row = {"Intent": intent}
        for a in ACTION_NAMES:
            count = res.action_counts.get(a, 0)
            row[a] = count
            row[f"{a}_%"] = round(100.0 * count / total, 1)
        rows.append(row)
    df   = pd.DataFrame(rows)
    path = os.path.join(out_dir, "action_distribution.csv")
    df.to_csv(path, index=False)
    print(f"[Eval] Action distribution table → {path}")


def save_sedm_table(out_dir: str) -> None:
    """Save the SEDM as a human-readable CSV for the thesis appendix."""
    sedm = MatrixPolicy.get_matrix()
    band_names = [
        f"Low_(<{ESC_LOW_THRESHOLD})",
        f"Medium_({ESC_LOW_THRESHOLD}-{ESC_HIGH_THRESHOLD})",
        f"High_(>={ESC_HIGH_THRESHOLD})",
    ]
    rows = []
    for stage in KillChainStage:
        row = {"Stage": stage.name}
        for band_idx, band_name in enumerate(band_names):
            row[band_name] = sedm[int(stage)][band_idx]
        rows.append(row)
    df   = pd.DataFrame(rows)
    path = os.path.join(out_dir, "sedm_table.csv")
    df.to_csv(path, index=False)
    print(f"[Eval] SEDM table → {path}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    n_episodes:    int   = 30,
    n_steps:       int   = 200,
    seed:          int   = 42,
    out_dir:       str   = RESULTS_DIR,
    benign_ratio:  float = 0.0,
) -> Dict[str, IntentResult]:

    os.makedirs(out_dir, exist_ok=True)

    shared_audit = os.path.join(out_dir, "opencanary_all_events.jsonl")
    honeypot     = DummyHoneypot(audit_file=shared_audit, verbose=False)
    results: Dict[str, IntentResult] = {}

    # Load the classifier once — it's intent-independent, so reuse it across
    # every intent below instead of re-reading classifier.joblib each time.
    shared_classifier = AttackClassifier()
    clf_path = os.path.join("models", "classifier.joblib")
    if os.path.exists(clf_path):
        shared_classifier.load(clf_path)
        print(f"[Defender] Classifier loaded from {clf_path}")
    else:
        print(f"[Defender] Warning: Classifier not found at {clf_path}")

    print(f"\n{'='*65}")
    print(f"  HoneyIQ SEDM — Comprehensive Policy Evaluation")
    print(f"  Episodes per intent : {n_episodes}")
    print(f"  Steps per episode   : {n_steps}")
    print(f"  Seed                : {seed}")
    print(f"  Benign ratio        : {benign_ratio:.0%}")
    print(f"  Output directory    : {out_dir}")
    print(f"{'='*65}\n")

    for intent in AttackerIntent:
        print(f"[Eval] Evaluating {intent.name} ({n_episodes} episodes)...")
        res = evaluate_intent(
            intent        = intent,
            n_episodes    = n_episodes,
            n_steps       = n_steps,
            seed          = seed,
            out_dir       = out_dir,
            honeypot      = honeypot,
            benign_ratio  = benign_ratio,
            classifier    = shared_classifier,
        )
        results[intent.name] = res
        print(f"  Reward={res.mean_reward:+.2f}±{res.std_reward:.2f}  "
              f"Det={res.mean_det:.3f}  FP={res.mean_fp:.3f}  "
              f"Risk={res.mean_risk:.3f}")

    print(f"\n{'='*65}")
    print("  OpenCanary Emulator — Kill-Chain Demos")
    print(f"{'='*65}")
    for intent in AttackerIntent:
        run_opencanary_demo(intent.name, out_dir)

    print(f"\n{'='*65}")
    print("  Generating plots and tables ...")
    print(f"{'='*65}")

    save_sedm_table(out_dir)
    save_summary_table(results, out_dir)
    save_action_distribution_table(results, out_dir)

    plot_decision_matrix(out_dir)
    plot_escalation_risk_per_intent(out_dir)
    plot_effective_policy_per_intent(out_dir)
    plot_metric_comparison(results, out_dir)
    plot_reward_boxplot(results, out_dir)
    plot_action_distribution(results, out_dir)
    plot_composite_risk_distribution(results, out_dir)

    print(f"\n{'='*65}")
    print(f"  All results saved to: {out_dir}")
    print(f"{'='*65}\n")
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HoneyIQ SEDM — comprehensive evaluation"
    )
    p.add_argument("--episodes",      type=int,   default=30)
    p.add_argument("--steps",         type=int,   default=200)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out-dir",       type=str,   default=RESULTS_DIR)
    p.add_argument("--benign-ratio",  type=float, default=0.0,
                   help="Fraction of steps injected as benign NORMAL traffic (0–1).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        n_episodes   = args.episodes,
        n_steps      = args.steps,
        seed         = args.seed,
        out_dir      = args.out_dir,
        benign_ratio = args.benign_ratio,
    )
