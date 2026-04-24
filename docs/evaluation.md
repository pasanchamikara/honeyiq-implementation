# Evaluation & Metrics

## Overview

The `evaluation/metrics.py` module handles:

- Per-step data accumulation (`StepRecord`)
- Per-episode aggregation (`EpisodeRecord`)
- Summary statistics across all episodes
- CSV export
- Five plot types

---

## Experimental Results

### Cross-Intent SEDM Evaluation

The Stage-Escalation Decision Matrix (SEDM) was evaluated over 30 episodes per intent (200 steps each). Results from `results/evaluation/evaluation_summary.csv`:

| Intent | Episodes | Mean Reward | Std Reward | Min | Max | Detection Rate | FP Rate | Avg Threat | Avg Risk |
|---|---|---|---|---|---|---|---|---|---|
| STEALTHY | 30 | 1012.22 | 51.37 | 920.0 | 1124.5 | **99.09%** | 35.56% | 0.806 | 0.632 |
| AGGRESSIVE | 30 | 1090.84 | 19.80 | 1051.0 | 1128.5 | **99.47%** | 6.67% | 0.854 | 0.709 |
| TARGETED | 30 | 1127.10 | 20.98 | 1074.0 | 1168.0 | **99.48%** | 3.33% | 0.853 | 0.673 |
| OPPORTUNISTIC | 30 | 896.05 | 32.98 | 843.0 | 973.0 | **99.41%** | 15.00% | 0.790 | 0.663 |

**Key observations:**
- Detection rates exceed 99% across all four intent profiles, demonstrating strong and consistent threat coverage.
- False positive rates vary significantly by intent: TARGETED campaigns (3.33%) and AGGRESSIVE campaigns (6.67%) generate few false alarms because attack traffic dominates. STEALTHY campaigns produce more false positives (35.56%) because low-severity reconnaissance-stage traffic triggers LOG/TROLL responses even on borderline-benign steps.
- TARGETED yields the highest mean reward (1127.10) because the focused kill chain path is well-aligned with the SEDM's escalation-based logic.
- OPPORTUNISTIC has the lowest mean reward (896.05) despite a high detection rate; the scattered attack distribution results in more BLOCK actions on medium-severity traffic where LOG/TROLL would have been better calibrated.

### Action Distribution

From `results/evaluation/action_distribution.csv`:

| Intent | ALLOW | LOG | TROLL | BLOCK | ALERT |
|---|---|---|---|---|---|
| STEALTHY | 0.0% | 1.1% | 0.7% | 17.9% | **80.3%** |
| AGGRESSIVE | 0.0% | 0.0% | 0.0% | 4.6% | **95.4%** |
| TARGETED | 0.0% | 0.5% | 0.1% | 5.4% | **94.0%** |
| OPPORTUNISTIC | 0.1% | 1.0% | 0.4% | 24.6% | **73.8%** |

**Key observations:**
- AGGRESSIVE and TARGETED campaigns are dominated by ALERT responses (>94%), consistent with the SEDM correctly escalating against high-severity, high-frequency attacks.
- STEALTHY campaigns still produce predominantly ALERT responses (80.3%), but also include 17.9% BLOCK — reflecting intermediate-severity backdoor traffic that falls in the INSTALLATION/C2 stages with medium escalation risk.
- OPPORTUNISTIC traffic generates 24.6% BLOCK and 73.8% ALERT, with a small number of ALLOW/LOG/TROLL on the scattered lower-severity attack types (Fuzzers, Generic).
- The near-zero ALLOW rate across all intents is expected: all four intents involve continuous attack traffic with no sustained benign periods.

### SEDM Decision Matrix

From `results/evaluation/sedm_table.csv`:

| Stage | Low (<0.35) | Medium (0.35–0.65) | High (≥0.65) |
|---|---|---|---|
| RECONNAISSANCE | ALLOW | LOG | LOG |
| WEAPONIZATION | LOG | LOG | TROLL |
| DELIVERY | LOG | TROLL | TROLL |
| EXPLOITATION | TROLL | BLOCK | BLOCK |
| INSTALLATION | BLOCK | BLOCK | ALERT |
| COMMAND_AND_CTRL | BLOCK | ALERT | ALERT |
| ACTIONS_ON_OBJ | ALERT | ALERT | ALERT |

---

## Data Structures

### `StepRecord`

Dataclass capturing a single environment step:

| Field | Type | Description |
|---|---|---|
| `episode` | int | Episode index |
| `step` | int | Step within episode |
| `action` | int | HoneypotAction taken |
| `reward` | float | Reward received |
| `attack_type` | int | Ground-truth attack type |
| `kill_chain_stage` | int | Ground-truth kill chain stage |
| `threat_level` | float | Computed threat level |
| `is_attack` | bool | Whether this step was an attack |
| `predicted_attack` | int | Classifier's predicted attack type |
| `loss` | float \| None | DQN training loss (None during eval) |
| `escalation_rate` | float | Recent attack frequency |

### `EpisodeRecord`

Aggregated per-episode summary:

| Field | Type | Description |
|---|---|---|
| `episode` | int | Episode index |
| `total_reward` | float | Sum of step rewards |
| `steps` | int | Number of steps completed |
| `detection_rate` | float | TP / (TP + FN) |
| `false_positive_rate` | float | FP / (FP + TN) |
| `avg_threat_level` | float | Mean threat level |
| `avg_loss` | float | Mean DQN loss |
| `kill_chain_dist` | dict | Step counts per kill chain stage |
| `action_dist` | dict | Step counts per action |

---

## Detection Metrics

The confusion matrix is defined relative to the defender's action:

| | is_attack = True | is_attack = False |
|---|---|---|
| action ≠ ALLOW | TP (detected) | FP (false alarm) |
| action = ALLOW | FN (missed) | TN (correctly allowed) |

```
detection_rate      = TP / (TP + FN)    # True positive rate / recall
false_positive_rate = FP / (FP + TN)    # False alarm rate
```

ALLOW is used as the "no-detection" threshold because it is the only action that implies no response to a potential threat.

---

## `MetricsCollector` API

### Recording

```python
metrics = MetricsCollector(log_dir="logs/")
metrics.record_step(episode, step, action, reward, info, pred_attack, loss)
record = metrics.end_episode(episode)
```

### Summary

```python
summary = metrics.summary_report()
# Keys: total_episodes, mean_reward, std_reward, best_episode_reward,
#       mean_detection_rate, mean_false_positive_rate, mean_threat_level
```

### Persistence

```python
metrics.save_csv()   # → logs/metrics.csv
```

---

## Visualisations

### `plot_training_curves()` → `logs/training_curves.png`

6-panel figure showing episode reward, detection rate, false positive rate, average threat level, per-update DQN loss, and per-episode average DQN loss.

### `plot_kill_chain_heatmap()` → `logs/action_stage_heatmap.png`

Heatmap of (HoneypotAction × KillChainStage) showing action frequency per kill chain stage.

### `plot_attack_progression(step_records)` → `logs/demo_progression.png`

4-panel time series for a single episode: threat level, kill chain stage, attack type, and defender actions.

### Evaluation plots → `results/evaluation/`

| Plot | Content |
|---|---|
| `evaluation_summary.csv` | Per-intent summary statistics |
| `action_distribution.csv` | Per-intent action frequency breakdown |
| `sedm_table.csv` | The 7×3 SEDM decision matrix |
| `metric_comparison.png` | Side-by-side bar charts of key metrics |
| `radar_comparison.png` | Radar chart comparing intents across metrics |
| `reward_boxplot.png` | Episode reward distributions per intent |
| `effective_policy_per_intent.png` | Action distribution stacked bars |
| `composite_risk_distribution.png` | Histogram of composite risk scores |
| `escalation_risk_per_intent.png` | Escalation risk by stage and intent |
| `sedm_decision_matrix.png` | Colour-coded SEDM heatmap |
| `kill_chain_distribution.png` | Kill chain stage frequency per intent |

---

## Threat Level Score

Composite score in [0.0, 1.0]:

| Component | Weight | Source |
|---|---|---|
| Attack severity | 45% | `ATTACK_SEVERITY` map per `AttackType` |
| Kill chain stage weight | 35% | `KILL_CHAIN_WEIGHT` map per stage |
| Escalation rate | 15% | Attack frequency in recent sliding window |
| Cumulative attack count | 5% | Normalised to 100 steps |

**Threat bands:**

| Band | Range |
|---|---|
| benign | < 0.15 |
| low | 0.15 – 0.35 |
| medium | 0.35 – 0.55 |
| high | 0.55 – 0.75 |
| critical | ≥ 0.75 |

---

## Interpreting Results

### Good SEDM performance
- Detection rate > 99% across all intents
- False positive rate varies by intent (lower for high-severity intents, higher for stealthy/low-severity)
- Higher reward for intents with well-aligned escalation paths (TARGETED > AGGRESSIVE > STEALTHY > OPPORTUNISTIC)

### What the action distribution tells you
- Dominated by ALERT/BLOCK: policy is correctly responding to high kill-chain stages
- Significant LOG/TROLL: policy is engaging with early-stage reconnaissance appropriately
- Non-zero ALLOW: policy distinguishes some benign traffic correctly
- Near-zero ALLOW with high FP rate: policy is over-responding to low-threat traffic
