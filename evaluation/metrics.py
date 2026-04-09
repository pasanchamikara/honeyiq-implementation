"""
Metrics collection, per-episode records, and visualization for the
attacker-defender simulation.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; switch to TkAgg/Qt5Agg for live display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from attacker.attack_types import AttackType, KillChainStage
from defender.honeypot import HoneypotAction


# ---------------------------------------------------------------------------
# Per-step data
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    episode:          int
    step:             int
    action:           int
    reward:           float
    attack_type:      int
    kill_chain_stage: int
    threat_level:     float
    is_attack:        bool
    predicted_attack: int
    loss:             Optional[float]
    escalation_rate:  float


# ---------------------------------------------------------------------------
# Per-episode summary
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    episode:             int
    total_reward:        float
    steps:               int
    detection_rate:      float   # TP / (TP + FN)
    false_positive_rate: float   # FP / (FP + TN)
    avg_threat_level:    float
    avg_loss:            float
    kill_chain_dist:     Dict[str, int]  = field(default_factory=dict)
    action_dist:         Dict[str, int]  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects per-step data during training and aggregates into per-episode
    summaries.  Also provides visualization helpers.

    Parameters
    ----------
    log_dir : str
        Directory where CSV and plots are saved.
    """

    def __init__(self, log_dir: str = "logs/") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.episodes: List[EpisodeRecord] = []
        self._step_buffer: List[StepRecord] = []
        self._all_losses: List[float] = []

    # ------------------------------------------------------------------
    def record_step(
        self,
        episode:          int,
        step:             int,
        action:           int,
        reward:           float,
        info:             dict,
        predicted_attack: AttackType,
        loss:             Optional[float],
    ) -> None:
        """Append a single step record."""
        self._step_buffer.append(StepRecord(
            episode          = episode,
            step             = step,
            action           = action,
            reward           = reward,
            attack_type      = int(info.get("attack_type", AttackType.NORMAL)),
            kill_chain_stage = int(info.get("kill_chain_stage", KillChainStage.RECONNAISSANCE)),
            threat_level     = float(info.get("threat_level", 0.0)),
            is_attack        = bool(info.get("is_attack", False)),
            predicted_attack = int(predicted_attack),
            loss             = loss,
            escalation_rate  = float(info.get("escalation_rate", 0.0)),
        ))
        if loss is not None:
            self._all_losses.append(loss)

    def end_episode(self, episode: int) -> EpisodeRecord:
        """
        Aggregate the step buffer into an EpisodeRecord, then clear the
        buffer.
        """
        buf = self._step_buffer

        total_reward  = sum(r.reward for r in buf)
        avg_threat    = float(np.mean([r.threat_level for r in buf])) if buf else 0.0
        losses        = [r.loss for r in buf if r.loss is not None]
        avg_loss      = float(np.mean(losses)) if losses else 0.0

        # Confusion matrix components
        tp = sum(1 for r in buf if r.is_attack and r.action != HoneypotAction.ALLOW)
        fn = sum(1 for r in buf if r.is_attack and r.action == HoneypotAction.ALLOW)
        fp = sum(1 for r in buf if not r.is_attack and r.action != HoneypotAction.ALLOW)
        tn = sum(1 for r in buf if not r.is_attack and r.action == HoneypotAction.ALLOW)

        detection_rate      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Kill chain stage distribution
        kc_dist: Dict[str, int] = defaultdict(int)
        for r in buf:
            kc_dist[KillChainStage(r.kill_chain_stage).name] += 1

        # Action distribution
        act_dist: Dict[str, int] = defaultdict(int)
        for r in buf:
            act_dist[HoneypotAction(r.action).name] += 1

        record = EpisodeRecord(
            episode             = episode,
            total_reward        = total_reward,
            steps               = len(buf),
            detection_rate      = detection_rate,
            false_positive_rate = false_positive_rate,
            avg_threat_level    = avg_threat,
            avg_loss            = avg_loss,
            kill_chain_dist     = dict(kc_dist),
            action_dist         = dict(act_dist),
        )
        self.episodes.append(record)
        self._step_buffer.clear()
        return record

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_report(self) -> dict:
        if not self.episodes:
            return {}
        rewards      = [e.total_reward        for e in self.episodes]
        det_rates    = [e.detection_rate       for e in self.episodes]
        fp_rates     = [e.false_positive_rate  for e in self.episodes]
        threats      = [e.avg_threat_level     for e in self.episodes]
        return {
            "total_episodes":          len(self.episodes),
            "mean_reward":             float(np.mean(rewards)),
            "std_reward":              float(np.std(rewards)),
            "best_episode_reward":     float(np.max(rewards)),
            "mean_detection_rate":     float(np.mean(det_rates)),
            "mean_false_positive_rate":float(np.mean(fp_rates)),
            "mean_threat_level":       float(np.mean(threats)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_csv(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.log_dir, "metrics.csv")
        rows = []
        for e in self.episodes:
            rows.append({
                "episode":             e.episode,
                "total_reward":        e.total_reward,
                "steps":               e.steps,
                "detection_rate":      e.detection_rate,
                "false_positive_rate": e.false_positive_rate,
                "avg_threat_level":    e.avg_threat_level,
                "avg_loss":            e.avg_loss,
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[Metrics] CSV saved to {path}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        save_path: Optional[str] = None,
        rolling_window: int = 10,
    ) -> None:
        save_path = save_path or os.path.join(self.log_dir, "training_curves.png")
        if not self.episodes:
            print("[Metrics] No episodes to plot.")
            return

        episodes     = [e.episode          for e in self.episodes]
        rewards      = [e.total_reward     for e in self.episodes]
        det_rates    = [e.detection_rate   for e in self.episodes]
        fp_rates     = [e.false_positive_rate for e in self.episodes]
        threats      = [e.avg_threat_level for e in self.episodes]
        losses       = [e.avg_loss         for e in self.episodes]

        def rolling_mean(arr: list, w: int) -> np.ndarray:
            out = np.full(len(arr), np.nan)
            for i in range(len(arr)):
                start = max(0, i - w + 1)
                out[i] = np.mean(arr[start:i + 1])
            return out

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("HoneyIQ — Training Curves", fontsize=14, fontweight="bold")

        # Episode reward
        ax = axes[0, 0]
        ax.plot(episodes, rewards, alpha=0.4, color="steelblue", label="Raw")
        ax.plot(episodes, rolling_mean(rewards, rolling_window),
                color="steelblue", linewidth=2, label=f"Rolling({rolling_window})")
        ax.set_title("Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Detection rate
        ax = axes[0, 1]
        ax.plot(episodes, det_rates, color="green", alpha=0.5)
        ax.plot(episodes, rolling_mean(det_rates, rolling_window),
                color="green", linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_title("Detection Rate (TP rate)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)

        # False positive rate
        ax = axes[0, 2]
        ax.plot(episodes, fp_rates, color="red", alpha=0.5)
        ax.plot(episodes, rolling_mean(fp_rates, rolling_window),
                color="red", linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_title("False Positive Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)

        # Average threat level
        ax = axes[1, 0]
        ax.plot(episodes, threats, color="orange", alpha=0.5)
        ax.plot(episodes, rolling_mean(threats, rolling_window),
                color="orange", linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_title("Avg Threat Level per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Threat Level")
        ax.grid(True, alpha=0.3)

        # Loss curve
        ax = axes[1, 1]
        if self._all_losses:
            xs = np.linspace(0, len(episodes), len(self._all_losses))
            ax.plot(xs, self._all_losses, alpha=0.3, color="purple", linewidth=0.5)
            # Smooth
            if len(self._all_losses) > 50:
                window = max(50, len(self._all_losses) // 20)
                smooth = np.convolve(self._all_losses,
                                     np.ones(window) / window, mode="valid")
                xs_s = np.linspace(0, len(episodes), len(smooth))
                ax.plot(xs_s, smooth, color="purple", linewidth=1.5)
        ax.set_title("DQN Loss")
        ax.set_xlabel("Episode (approx.)")
        ax.set_ylabel("Huber Loss")
        ax.grid(True, alpha=0.3)

        # Per-episode average loss
        ax = axes[1, 2]
        ax.plot(episodes, losses, alpha=0.5, color="brown")
        ax.plot(episodes, rolling_mean(losses, rolling_window),
                color="brown", linewidth=2)
        ax.set_title("Avg Loss per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Loss")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Metrics] Training curves saved to {save_path}")

    def plot_kill_chain_heatmap(
        self,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Heatmap of: how often each action was taken at each kill chain stage.
        """
        save_path = save_path or os.path.join(self.log_dir, "action_stage_heatmap.png")
        if not self._step_buffer and not self.episodes:
            return

        # Rebuild from episodes' action/kc distributions
        # (full step buffer is cleared per episode, so use aggregated info)
        # Instead, plot using the last N episodes' distributions
        kc_names  = KillChainStage.names()
        act_names = HoneypotAction.names()
        matrix    = np.zeros((len(act_names), len(kc_names)), dtype=int)

        # This is an approximation since we don't store per-step kc+action pairs
        # after end_episode(); it uses the kill chain distribution per episode
        # weighted by action distribution.  For a proper heatmap, run with
        # render_mode="ansi" and collect step records separately.
        for ep in self.episodes:
            total_steps = ep.steps if ep.steps > 0 else 1
            for ai, aname in enumerate(act_names):
                act_count = ep.action_dist.get(aname, 0)
                for ki, kname in enumerate(kc_names):
                    kc_count = ep.kill_chain_dist.get(kname, 0)
                    matrix[ai, ki] += int(act_count * kc_count / total_steps)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            matrix,
            annot=True, fmt="d",
            xticklabels=kc_names,
            yticklabels=act_names,
            cmap="YlOrRd",
            ax=ax,
        )
        ax.set_title("Action vs Kill Chain Stage (aggregated)", fontsize=12)
        ax.set_xlabel("Kill Chain Stage")
        ax.set_ylabel("Honeypot Action")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Metrics] Heatmap saved to {save_path}")

    def plot_attack_progression(
        self,
        step_records: List[StepRecord],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize a single episode: attack type, kill chain stage, threat
        level, and defender actions over time.
        """
        save_path = save_path or os.path.join(self.log_dir, "episode_progression.png")
        if not step_records:
            return

        steps          = [r.step        for r in step_records]
        threats        = [r.threat_level for r in step_records]
        stages         = [r.kill_chain_stage for r in step_records]
        attack_types   = [r.attack_type for r in step_records]
        actions        = [r.action      for r in step_records]
        rewards        = [r.reward      for r in step_records]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("Episode Progression — Attacker vs Defender", fontsize=13, fontweight="bold")

        cmap_atk = plt.cm.get_cmap("tab10", AttackType.count())
        cmap_act = plt.cm.get_cmap("Set1", HoneypotAction.count())

        # Threat level
        ax = axes[0]
        ax.fill_between(steps, threats, alpha=0.3, color="red")
        ax.plot(steps, threats, color="red", linewidth=1.5)
        ax.axhline(0.75, color="darkred", linestyle="--", alpha=0.6, label="Critical")
        ax.axhline(0.55, color="orange",  linestyle="--", alpha=0.6, label="High")
        ax.axhline(0.35, color="gold",    linestyle="--", alpha=0.6, label="Medium")
        ax.set_ylabel("Threat Level")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Kill chain stage
        ax = axes[1]
        ax.step(steps, stages, where="post", color="navy", linewidth=1.5)
        ax.set_yticks(range(KillChainStage.count()))
        ax.set_yticklabels(KillChainStage.names(), fontsize=7)
        ax.set_ylabel("Kill Chain Stage")
        ax.grid(True, alpha=0.3)

        # Attack type
        ax = axes[2]
        for i, at in enumerate(AttackType):
            mask = [1 if a == int(at) else 0 for a in attack_types]
            ax.fill_between(steps, [m * (i + 0.5) for m in mask],
                            [m * (i - 0.5) for m in mask],
                            alpha=0.7, color=cmap_atk(i), label=at.name)
        ax.set_yticks(range(AttackType.count()))
        ax.set_yticklabels(AttackType.names(), fontsize=7)
        ax.set_ylabel("Attack Type")
        ax.grid(True, alpha=0.3)

        # Defender actions + reward
        ax = axes[3]
        ax2 = ax.twinx()
        for i, ha in enumerate(HoneypotAction):
            mask_x = [s for s, a in zip(steps, actions) if a == int(ha)]
            mask_y = [i] * len(mask_x)
            ax.scatter(mask_x, mask_y, color=cmap_act(i), s=10, label=ha.name, alpha=0.8)
        ax.set_yticks(range(HoneypotAction.count()))
        ax.set_yticklabels(HoneypotAction.names(), fontsize=7)
        ax.set_ylabel("Honeypot Action")
        ax2.plot(steps, rewards, color="gray", alpha=0.5, linewidth=0.8)
        ax2.set_ylabel("Reward", color="gray")
        ax.set_xlabel("Step")
        ax.legend(loc="upper left", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Metrics] Progression plot saved to {save_path}")
