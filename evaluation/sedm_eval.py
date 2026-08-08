"""
SEDM Comprehensive Evaluation
==============================
Computes a full suite of metrics that go beyond the basic detection/FP rates:

  1. Binary classification metrics: Precision, Recall, F1, F2, Specificity
  2. Response proportionality: Spearman correlation between kill-chain stage
     and mean action severity; proportionality score.
  3. Late-stage containment: fraction of INSTALLATION+ steps that get a
     weak response (ALLOW or LOG); mean steps-to-first-BLOCK/ALERT.
  4. Action distribution per kill-chain stage (aggregated over all intents).
  5. Robustness: coefficient of variation of F1 across intents.

Results are printed in a LaTeX-ready format and saved to
  logs/sedm_eval_results.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacker.attack_types import AttackType, KillChainStage, AttackerIntent
from defender.defender import Defender
from defender.honeypot import HoneypotAction
from defender.matrix_policy import MatrixPolicy
from environment.cyber_env import CyberSecurityEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_SEVERITY: Dict[int, int] = {
    HoneypotAction.ALLOW: 0,
    HoneypotAction.LOG:   1,
    HoneypotAction.TROLL: 2,
    HoneypotAction.BLOCK: 3,
    HoneypotAction.ALERT: 4,
}

# Minimum expected action severity at each kill chain stage
MIN_SEVERITY_AT_STAGE: Dict[int, int] = {
    KillChainStage.RECONNAISSANCE:   0,   # ALLOW acceptable
    KillChainStage.WEAPONIZATION:    1,   # LOG at minimum
    KillChainStage.DELIVERY:         1,   # LOG at minimum
    KillChainStage.EXPLOITATION:     2,   # TROLL at minimum
    KillChainStage.INSTALLATION:     3,   # BLOCK at minimum
    KillChainStage.COMMAND_AND_CTRL: 3,   # BLOCK at minimum
    KillChainStage.ACTIONS_ON_OBJ:   4,   # ALERT required
}

LATE_STAGES = {
    KillChainStage.INSTALLATION,
    KillChainStage.COMMAND_AND_CTRL,
    KillChainStage.ACTIONS_ON_OBJ,
}


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------

@dataclass
class FullStepRecord:
    episode:          int
    step:             int
    intent:           int
    action:           int           # HoneypotAction int
    reward:           float
    is_attack:        bool
    kill_chain_stage: int           # KillChainStage int
    attack_type:      int           # AttackType int
    threat_level:     float
    escalation_rate:  float
    composite_risk:   float


# ---------------------------------------------------------------------------
# Episode-level results
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    intent:          str
    episode:         int
    total_reward:    float
    steps:           int
    precision:       float
    recall:          float
    f1:              float
    f2:              float
    specificity:     float
    fpr:             float
    prop_score:      float          # response proportionality score
    late_miss_rate:  float          # fraction of INSTALLATION+ steps with weak response
    steps_to_contain: Optional[float]  # steps until first BLOCK/ALERT; None if never
    max_stage_reached: int
    mean_threat:     float
    mean_risk:       float
    action_counts:   Dict[str, int] = field(default_factory=dict)
    stage_counts:    Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _compute_composite_risk(state: np.ndarray) -> float:
    """Re-derive composite risk from raw state for logging."""
    from attacker.attack_types import ATTACK_SEVERITY, KILL_CHAIN_WEIGHT
    attack_idx = int(np.argmax(state[0:10]))
    stage_idx  = int(np.argmax(state[10:17]))
    esc_rate   = float(state[19])
    severity   = ATTACK_SEVERITY.get(attack_idx, 0.0)
    stage_w    = KILL_CHAIN_WEIGHT.get(stage_idx, 0.0)
    # simplified (no esc_risk without TransitionModel call)
    return float(min(0.35 * stage_w + 0.45 * severity + 0.15 * esc_rate, 1.0))


def run_intent(
    intent:            AttackerIntent,
    n_episodes:        int,
    n_steps:           int,
    benign_ratio:      float,
    seed:              int,
    model_dir:         str,
    classifier_driven: bool = False,
) -> Tuple[List[EpisodeResult], List[FullStepRecord]]:
    """
    Run n_episodes for a single intent, returning per-episode results and step records.

    Parameters
    ----------
    classifier_driven : bool
        False (default) — "oracle" mode: the SEDM decides directly from the
        environment's ground-truth state (matches ``Defender.observe``).
        True  — "realistic" mode: the SEDM decides from the RandomForest
        classifier's *predicted* attack type instead of ground truth (mirrors
        ``evaluate.py::evaluate_intent``), so the classifier is no longer
        decorative and detection metrics are not tautological.

    Alignment
    ---------
    The action for step *t* is chosen from ``state``/``info`` — the
    observation the defender actually saw.  It is scored against the
    ground-truth label of that *same* observation (decoded from ``state``
    itself, not from the ``info`` returned by the subsequent ``env.step()``
    call, which already describes step *t+1*). Pairing the action with next
    step's label would silently shift every classification metric by one
    time step.
    """

    env      = CyberSecurityEnv(
        attacker_intent=intent,
        max_steps=n_steps,
        benign_ratio=benign_ratio,
        seed=seed,
    )
    defender = Defender(default_intent=intent)
    clf_path = os.path.join(model_dir, "classifier.joblib")
    if os.path.exists(clf_path):
        defender.load(model_dir)
    else:
        defender.initialize_classifier()

    mp = MatrixPolicy(default_intent=intent)

    episode_results: List[EpisodeResult] = []
    all_step_records: List[FullStepRecord] = []

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx * 17
        state, info = env.reset(seed=ep_seed)

        step_records: List[FullStepRecord] = []

        for step_idx in range(n_steps):
            features = info.get("features", {})

            # -- Ground truth for THIS observation (no lag) --------------------
            true_attack_type = AttackType(int(np.argmax(state[0:10])))
            true_stage        = int(np.argmax(state[10:17]))
            true_is_attack    = true_attack_type != AttackType.NORMAL
            true_threat       = float(state[17])
            true_esc_rate     = float(state[19])

            if classifier_driven:
                # Decide from the classifier's predicted attack type, not
                # ground truth — breaks the tautology between the SEDM's R1
                # rule and the is_attack label.
                if features and defender.classifier.is_fitted:
                    pred_attack = defender.classifier.predict(features)
                else:
                    pred_attack = true_attack_type
                decision_state = state.copy()
                decision_state[0:10] = 0.0
                decision_state[int(pred_attack)] = 1.0
                action_obj, _ = mp.decide_from_state(decision_state)
                action = int(action_obj)
            else:
                action, _ = defender.observe(state, features, training=False)

            next_state, reward, terminated, truncated, next_info = env.step(action)

            composite_risk = _compute_composite_risk(state)

            rec = FullStepRecord(
                episode          = ep_idx,
                step             = step_idx,
                intent           = int(intent),
                action           = action,
                reward           = reward,
                is_attack        = true_is_attack,
                kill_chain_stage = true_stage,
                attack_type      = int(true_attack_type),
                threat_level     = true_threat,
                escalation_rate  = true_esc_rate,
                composite_risk   = composite_risk,
            )
            step_records.append(rec)
            all_step_records.append(rec)

            state, info = next_state, next_info
            if terminated or truncated:
                break

        # --- Episode aggregation ---
        ep_result = _aggregate_episode(intent, ep_idx, step_records)
        episode_results.append(ep_result)

    return episode_results, all_step_records


def _aggregate_episode(
    intent:       AttackerIntent,
    ep_idx:       int,
    records:      List[FullStepRecord],
) -> EpisodeResult:
    if not records:
        return EpisodeResult(
            intent=intent.name, episode=ep_idx,
            total_reward=0, steps=0, precision=0, recall=0,
            f1=0, f2=0, specificity=0, fpr=0, prop_score=0,
            late_miss_rate=0, steps_to_contain=None,
            max_stage_reached=0, mean_threat=0, mean_risk=0,
        )

    # Binary classification: treat non-ALLOW as "detected"
    tp = sum(1 for r in records if r.is_attack     and r.action != HoneypotAction.ALLOW)
    fn = sum(1 for r in records if r.is_attack     and r.action == HoneypotAction.ALLOW)
    fp = sum(1 for r in records if not r.is_attack and r.action != HoneypotAction.ALLOW)
    tn = sum(1 for r in records if not r.is_attack and r.action == HoneypotAction.ALLOW)

    precision    = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall       = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    fpr          = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) > 0 else 0.0)

    # Response proportionality score:
    # fraction of steps where action severity >= minimum expected for that stage
    prop_hits = sum(
        1 for r in records
        if ACTION_SEVERITY[r.action] >= MIN_SEVERITY_AT_STAGE[r.kill_chain_stage]
    )
    prop_score = prop_hits / len(records)

    # Late-stage miss rate: INSTALLATION+ ATTACK steps answered with ALLOW or LOG
    # (excludes benign steps, which are correctly ALLOW-ed via R1)
    late_attack_recs = [r for r in records
                        if r.kill_chain_stage >= KillChainStage.INSTALLATION and r.is_attack]
    weak_actions = {HoneypotAction.ALLOW, HoneypotAction.LOG}
    late_miss_rate = (
        sum(1 for r in late_attack_recs if r.action in weak_actions) / len(late_attack_recs)
        if late_attack_recs else 0.0
    )

    # Steps to first BLOCK/ALERT after first attack step
    strong_actions = {HoneypotAction.BLOCK, HoneypotAction.ALERT}
    attack_indices = [i for i, r in enumerate(records) if r.is_attack]
    if attack_indices:
        first_attack_step = attack_indices[0]
        contain_indices   = [i for i in range(first_attack_step, len(records))
                             if records[i].action in strong_actions]
        steps_to_contain: Optional[float] = (
            float(contain_indices[0] - first_attack_step)
            if contain_indices else None
        )
    else:
        steps_to_contain = None

    max_stage_reached = max((r.kill_chain_stage for r in records), default=0)

    # Totals
    total_reward = sum(r.reward for r in records)
    mean_threat  = float(np.mean([r.threat_level for r in records]))
    mean_risk    = float(np.mean([r.composite_risk for r in records]))

    action_counts = defaultdict(int)
    for r in records:
        action_counts[HoneypotAction(r.action).name] += 1

    stage_counts = defaultdict(int)
    for r in records:
        stage_counts[KillChainStage(r.kill_chain_stage).name] += 1

    return EpisodeResult(
        intent           = intent.name,
        episode          = ep_idx,
        total_reward     = total_reward,
        steps            = len(records),
        precision        = precision,
        recall           = recall,
        f1               = f1,
        f2               = f2,
        specificity      = specificity,
        fpr              = fpr,
        prop_score       = prop_score,
        late_miss_rate   = late_miss_rate,
        steps_to_contain = steps_to_contain,
        max_stage_reached= max_stage_reached,
        mean_threat      = mean_threat,
        mean_risk        = mean_risk,
        action_counts    = dict(action_counts),
        stage_counts     = dict(stage_counts),
    )


# ---------------------------------------------------------------------------
# Aggregate over episodes
# ---------------------------------------------------------------------------

def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    a = np.array(values)
    return float(np.mean(a)), float(np.std(a))


def aggregate_intent(results: List[EpisodeResult]) -> dict:
    def ms(key):
        vals = [getattr(r, key) for r in results]
        return _mean_std(vals)

    contain_vals = [r.steps_to_contain for r in results if r.steps_to_contain is not None]
    contain_mean = float(np.mean(contain_vals)) if contain_vals else float("nan")

    mean_r, std_r = ms("total_reward")
    mean_p, _     = ms("precision")
    mean_rec, _   = ms("recall")
    mean_f1, std_f1 = ms("f1")
    mean_f2, _    = ms("f2")
    mean_sp, _    = ms("specificity")
    mean_fpr, _   = ms("fpr")
    mean_prop, _  = ms("prop_score")
    mean_lmr, _   = ms("late_miss_rate")
    mean_thr, _   = ms("mean_threat")
    mean_risk, _  = ms("mean_risk")

    cv_f1 = (std_f1 / mean_f1) if mean_f1 > 0 else float("nan")

    return {
        "mean_reward":        mean_r,
        "std_reward":         std_r,
        "mean_precision":     mean_p,
        "mean_recall":        mean_rec,
        "mean_f1":            mean_f1,
        "std_f1":             std_f1,
        "cv_f1":              cv_f1,
        "mean_f2":            mean_f2,
        "mean_specificity":   mean_sp,
        "mean_fpr":           mean_fpr,
        "mean_prop_score":    mean_prop,
        "mean_late_miss":     mean_lmr,
        "mean_steps_to_contain": contain_mean,
        "mean_threat":        mean_thr,
        "mean_risk":          mean_risk,
    }


# ---------------------------------------------------------------------------
# Stage × Action distribution (across all intents and episodes)
# ---------------------------------------------------------------------------

def stage_action_table(all_records: List[FullStepRecord]) -> np.ndarray:
    """Return a (7, 5) matrix on attack-only steps: rows=kill chain stages, cols=actions."""
    table = np.zeros((KillChainStage.count(), HoneypotAction.count()), dtype=int)
    for r in all_records:
        if r.is_attack:
            table[r.kill_chain_stage, r.action] += 1
    return table


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def print_latex_tables(
    intent_agg:    Dict[str, dict],
    sa_table:      np.ndarray,
    spearman_rho:  float,
) -> None:

    intents = list(AttackerIntent)
    act_names   = HoneypotAction.names()
    stage_names = [
        "Reconnaissance", "Weaponization", "Delivery",
        "Exploitation", "Installation", "Command \\& Control", "Actions on Obj.",
    ]

    print("\n" + "=" * 72)
    print("  LaTeX tables — paste into thesis")
    print("=" * 72)

    # ── Table 1: Extended classification metrics ──────────────────────────
    print(r"""
%% ── Extended Classification Metrics ──────────────────────────────────
\begin{table}[ht]
  \centering
  \caption{Extended classification metrics for the SEDM policy across all
           four attacker intents (50~episodes $\times$ 500~steps each,
           benign ratio 30\%).
           Mean values over all evaluation episodes are reported.}
  \label{tab:sedm_extended_metrics}
  \begin{tabular}{lcccccc}
    \toprule
    Intent & Precision & Recall & F\textsubscript{1} & F\textsubscript{2} & Specificity & Prop.\ Score \\
    \midrule""")

    for intent in intents:
        ag = intent_agg[intent.name]
        print(
            f"    {intent.name:<14} & "
            f"{ag['mean_precision']:.3f}     & "
            f"{ag['mean_recall']:.3f}  & "
            f"{ag['mean_f1']:.3f}              & "
            f"{ag['mean_f2']:.3f}              & "
            f"{ag['mean_specificity']:.3f}       & "
            f"{ag['mean_prop_score']:.3f} \\\\"
        )

    print(r"""    \bottomrule
  \end{tabular}
\end{table}""")

    # ── Table 2: Containment and late-stage metrics ───────────────────────
    print(r"""
%% ── Containment Metrics ───────────────────────────────────────────────
\begin{table}[ht]
  \centering
  \caption{Threat containment metrics per attacker intent.
           \emph{Late-stage miss rate} is the fraction of
           Installation$+$ steps receiving a weak response (ALLOW or LOG).
           \emph{Steps-to-contain} is the mean number of steps between
           the first attack step and the first BLOCK or ALERT response
           (averaged over episodes where containment occurred).}
  \label{tab:sedm_containment}
  \begin{tabular}{lcccc}
    \toprule
    Intent & Late-Stage Miss Rate & Steps-to-Contain & Avg Threat & Avg Risk \\
    \midrule""")

    for intent in intents:
        ag = intent_agg[intent.name]
        stc = ag["mean_steps_to_contain"]
        stc_str = f"{stc:.1f}" if not (stc != stc) else "---"  # nan check
        print(
            f"    {intent.name:<14} & "
            f"{ag['mean_late_miss']:.3f}                & "
            f"{stc_str:>17} & "
            f"{ag['mean_threat']:.3f}       & "
            f"{ag['mean_risk']:.3f} \\\\"
        )

    print(r"""    \bottomrule
  \end{tabular}
\end{table}""")

    # ── Table 3: Stage × Action distribution (pooled) ────────────────────
    # Normalise to percentages per row
    row_totals = sa_table.sum(axis=1, keepdims=True).clip(min=1)
    sa_pct     = sa_table / row_totals * 100.0

    print(r"""
%% ── Stage × Action Distribution ──────────────────────────────────────
\begin{table}[ht]
  \centering
  \caption{Per-stage action distribution for \emph{attack} steps, pooled
           across all four attacker intents and all evaluation episodes.
           Benign steps (which always receive ALLOW via the R1 override)
           are excluded so the table reflects only the SEDM's response
           to genuine threats.
           Values are row-normalised percentages; each row sums to 100\%.}
  \label{tab:stage_action_dist}
  \begin{tabular}{lrrrrr}
    \toprule
    Kill Chain Stage & ALLOW & LOG & TROLL & BLOCK & ALERT \\
    \midrule""")

    for si, sname in enumerate(stage_names):
        row = sa_pct[si]
        cols = " & ".join(f"{v:5.1f}\\%" for v in row)
        print(f"    {sname:<30} & {cols} \\\\")

    print(r"""    \bottomrule
  \end{tabular}
\end{table}""")

    # ── Inline text helpers ───────────────────────────────────────────────
    print("\n%% ── Inline statistics for results text ──────────────────────────────")
    print(f"%% Spearman rho (stage vs action severity, attack-only steps): {spearman_rho:.4f}")
    for intent in intents:
        ag = intent_agg[intent.name]
        print(
            f"%%   {intent.name}: P={ag['mean_precision']:.3f}  "
            f"R={ag['mean_recall']:.3f}  F1={ag['mean_f1']:.3f}  "
            f"F2={ag['mean_f2']:.3f}  "
            f"Spec={ag['mean_specificity']:.3f}  "
            f"LMR={ag['mean_late_miss']:.3f}  "
            f"STC={ag['mean_steps_to_contain']:.1f}"
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    intent_agg:   Dict[str, dict],
    sa_table:     np.ndarray,
    all_records:  List[FullStepRecord],
    log_dir:      str,
    file_tag:     str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[Eval] matplotlib/seaborn not available — skipping plots.")
        return

    os.makedirs(log_dir, exist_ok=True)
    intents    = [i.name for i in AttackerIntent]
    act_names  = HoneypotAction.names()
    stage_names_short = ["RECON", "WEAPON", "DELIVERY", "EXPLOIT", "INSTALL", "C2", "AOO"]

    # ── F1 / Precision / Recall bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x  = np.arange(len(intents))
    w  = 0.22
    metrics = ["mean_precision", "mean_recall", "mean_f1", "mean_f2"]
    labels  = ["Precision", "Recall", "F₁", "F₂"]
    colors  = ["#2980B9", "#27AE60", "#E67E22", "#8E44AD"]
    for i, (m, lbl, c) in enumerate(zip(metrics, labels, colors)):
        vals = [intent_agg[name][m] for name in intents]
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=lbl, color=c, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(intents, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SEDM — Classification Metrics by Attacker Intent", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    p = os.path.join(log_dir, f"sedm_classification_metrics{file_tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {p}")

    # ── Stage × Action heatmap (normalised) ─────────────────────────────
    row_totals = sa_table.sum(axis=1, keepdims=True).clip(min=1)
    sa_pct     = (sa_table / row_totals * 100.0).astype(float)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        sa_pct, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=act_names, yticklabels=stage_names_short,
        ax=ax, vmin=0, vmax=100,
        annot_kws={"size": 9},
    )
    ax.set_title("Stage × Action Distribution (row-normalised %, pooled)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Honeypot Action")
    ax.set_ylabel("Kill Chain Stage")
    plt.tight_layout()
    p = os.path.join(log_dir, f"sedm_stage_action_heatmap{file_tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {p}")

    # ── Response proportionality: mean action severity per stage ─────────
    stage_sev: Dict[int, List[float]] = defaultdict(list)
    for r in all_records:
        stage_sev[r.kill_chain_stage].append(ACTION_SEVERITY[r.action])

    mean_sevs = [float(np.mean(stage_sev[s])) for s in range(KillChainStage.count())]
    min_sevs  = [MIN_SEVERITY_AT_STAGE[s] for s in range(KillChainStage.count())]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(stage_names_short, mean_sevs, "o-", color="#E67E22", linewidth=2,
            markersize=7, label="Mean action severity (SEDM)")
    ax.plot(stage_names_short, min_sevs, "s--", color="#2C3E50", linewidth=1.5,
            markersize=6, label="Min expected severity")
    ax.set_ylim(-0.2, 4.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels(act_names)
    ax.set_xlabel("Kill Chain Stage")
    ax.set_ylabel("Action Severity")
    ax.set_title("Response Proportionality — Mean Action Severity per Stage", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    p = os.path.join(log_dir, f"sedm_proportionality{file_tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {p}")

    # ── Late-stage miss rate and steps-to-contain ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    lmr_vals = [intent_agg[n]["mean_late_miss"] * 100 for n in intents]
    axes[0].bar(intents, lmr_vals, color=["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71"], alpha=0.85)
    axes[0].set_ylim(0, max(lmr_vals) * 1.3 + 1)
    axes[0].set_ylabel("Late-Stage Miss Rate (%)")
    axes[0].set_title("Late-Stage Miss Rate by Intent", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.35)

    stc_vals = [intent_agg[n]["mean_steps_to_contain"] for n in intents]
    stc_valid = [(n, v) for n, v in zip(intents, stc_vals) if v == v]  # exclude nan
    if stc_valid:
        ns, vs = zip(*stc_valid)
        axes[1].bar(ns, vs, color=["#3498DB", "#9B59B6", "#1ABC9C", "#E74C3C"][:len(vs)], alpha=0.85)
        axes[1].set_ylabel("Mean Steps to First BLOCK/ALERT")
        axes[1].set_title("Steps-to-Containment by Intent", fontsize=11, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.35)

    plt.tight_layout()
    p = os.path.join(log_dir, f"sedm_containment{file_tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    n_episodes:        int   = 50,
    n_steps:           int   = 500,
    benign_ratio:      float = 0.30,
    seed:              int   = 42,
    model_dir:         str   = "models/",
    log_dir:           str   = "logs/",
    classifier_driven: bool  = False,
) -> Dict[str, dict]:
    """
    Run the full SEDM evaluation suite.

    Two modes, selected by ``classifier_driven``:
      False (default) — "oracle" mode: the SEDM decides from ground-truth
        kill-chain state (upper bound on achievable performance).
      True  — "realistic" mode: the SEDM decides from the RandomForest
        classifier's predicted attack type, so classification noise
        propagates into the policy's decisions (matches deployment
        conditions where ground truth is never directly observable).
    """

    variant   = "classifier_driven" if classifier_driven else "oracle"
    file_tag  = "_clf" if classifier_driven else ""

    print(f"\n{'='*68}")
    print(f"  SEDM Comprehensive Evaluation — {variant.upper()} variant")
    print(f"  Episodes per intent : {n_episodes}")
    print(f"  Steps per episode   : {n_steps}")
    print(f"  Benign ratio        : {benign_ratio:.0%}")
    print(f"  Seed                : {seed}")
    print(f"{'='*68}\n")

    all_records:  List[FullStepRecord]  = []
    intent_results: Dict[str, List[EpisodeResult]] = {}

    for intent in AttackerIntent:
        print(f"[Eval] Running {n_episodes} episodes — intent: {intent.name} ...")
        ep_results, step_recs = run_intent(
            intent=intent,
            n_episodes=n_episodes,
            n_steps=n_steps,
            benign_ratio=benign_ratio,
            seed=seed,
            model_dir=model_dir,
            classifier_driven=classifier_driven,
        )
        intent_results[intent.name] = ep_results
        all_records.extend(step_recs)
        ag = aggregate_intent(ep_results)
        print(
            f"       Precision={ag['mean_precision']:.3f}  "
            f"Recall={ag['mean_recall']:.3f}  "
            f"F1={ag['mean_f1']:.3f}  "
            f"F2={ag['mean_f2']:.3f}  "
            f"Spec={ag['mean_specificity']:.3f}  "
            f"PropScore={ag['mean_prop_score']:.3f}  "
            f"LMR={ag['mean_late_miss']:.3f}"
        )

    # Aggregate
    intent_agg = {name: aggregate_intent(results) for name, results in intent_results.items()}

    # Stage × action table
    sa_table = stage_action_table(all_records)

    # Spearman rho — computed on attack-only steps to exclude R1-ALLOW noise
    try:
        from scipy.stats import spearmanr
        attack_recs = [r for r in all_records if r.is_attack]
        stages      = [r.kill_chain_stage for r in attack_recs]
        severities  = [ACTION_SEVERITY[r.action] for r in attack_recs]
        spearman_rho, spearman_p = spearmanr(stages, severities)
        spearman_rho = float(spearman_rho)
        print(f"\n[Eval] Spearman ρ (stage vs action severity, attack-only): {spearman_rho:.4f}  p={spearman_p:.2e}")
    except ImportError:
        spearman_rho = float("nan")
        print("[Eval] scipy not available — skipping Spearman rho.")

    # Save JSON
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"sedm_eval_results{file_tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_episodes":        n_episodes,
                "n_steps":           n_steps,
                "benign_ratio":      benign_ratio,
                "seed":              seed,
                "classifier_driven": classifier_driven,
            },
            "intent_aggregates": intent_agg,
            "spearman_rho":      spearman_rho,
            "stage_action_table": sa_table.tolist(),
        }, f, indent=2)
    print(f"[Eval] Results saved to {out_path}")

    # Plots
    make_plots(intent_agg, sa_table, all_records, log_dir, file_tag=file_tag)

    # LaTeX
    print_latex_tables(intent_agg, sa_table, spearman_rho)

    return intent_agg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SEDM comprehensive evaluation")
    parser.add_argument("--episodes",     type=int,   default=50)
    parser.add_argument("--steps",        type=int,   default=500)
    parser.add_argument("--benign-ratio", type=float, default=0.30)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--model-dir",    default="models/")
    parser.add_argument("--log-dir",      default="logs/")
    parser.add_argument(
        "--variant", choices=["oracle", "classifier", "both"], default="both",
        help="oracle: ground-truth state decisions. classifier: RF-predicted "
             "attack type decisions. both: run and save both (default).",
    )
    args = parser.parse_args()

    run_oracle     = args.variant in ("oracle", "both")
    run_classifier = args.variant in ("classifier", "both")

    if run_oracle:
        evaluate(
            n_episodes   = args.episodes,
            n_steps      = args.steps,
            benign_ratio = args.benign_ratio,
            seed         = args.seed,
            model_dir    = args.model_dir,
            log_dir      = args.log_dir,
            classifier_driven = False,
        )
    if run_classifier:
        evaluate(
            n_episodes   = args.episodes,
            n_steps      = args.steps,
            benign_ratio = args.benign_ratio,
            seed         = args.seed,
            model_dir    = args.model_dir,
            log_dir      = args.log_dir,
            classifier_driven = True,
        )
