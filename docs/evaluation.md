# Evaluation & Metrics

## Overview

The `evaluation/metrics.py` module handles:

- Per-step data accumulation (`StepRecord`)
- Per-episode aggregation (`EpisodeRecord`)
- Summary statistics across all episodes
- CSV export
- Five plot types

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

ALLOW is used as the "no-detection" threshold because it is the only action that implies no response to a potential threat. Any other action (LOG, TROLL, BLOCK, ALERT) counts as detecting/responding to the traffic.

---

## `MetricsCollector` API

### Recording

```python
metrics = MetricsCollector(log_dir="logs/")

# Called each step
metrics.record_step(episode, step, action, reward, info, pred_attack, loss)

# Called at end of each episode — returns EpisodeRecord
record = metrics.end_episode(episode)
```

`end_episode()` flushes the internal step buffer after aggregation.

### Summary

```python
summary = metrics.summary_report()
# Keys:
#   total_episodes, mean_reward, std_reward, best_episode_reward,
#   mean_detection_rate, mean_false_positive_rate, mean_threat_level
```

### Persistence

```python
metrics.save_csv()                 # → logs/metrics.csv
```

CSV columns: `episode`, `total_reward`, `steps`, `detection_rate`, `false_positive_rate`, `avg_threat_level`, `avg_loss`.

---

## Visualizations

### `plot_training_curves()` → `logs/training_curves.png`

6-panel figure (2 rows × 3 columns):

| Panel | Content |
|---|---|
| (0,0) | Episode reward (raw + rolling mean) |
| (0,1) | Detection rate over episodes |
| (0,2) | False positive rate over episodes |
| (1,0) | Average threat level per episode |
| (1,1) | DQN loss (all individual updates, smoothed) |
| (1,2) | Average DQN loss per episode |

Rolling window: 10 episodes.

### `plot_kill_chain_heatmap()` → `logs/action_stage_heatmap.png`

Heatmap of (HoneypotAction × KillChainStage) showing how often each action was taken at each kill chain stage, aggregated across all episodes.

Note: this is an approximation — since step buffers are cleared per episode, it uses the product of action and kill chain distributions within each episode rather than exact per-step pairs.

### `plot_attack_progression(step_records)` → `logs/demo_progression.png`

4-panel time series for a single episode (sharex):

| Panel | Content |
|---|---|
| 1 | Threat level with band threshold lines (critical/high/medium) |
| 2 | Kill chain stage (step plot) |
| 3 | Attack type (colour-coded fill bands) |
| 4 | Defender actions (scatter) overlaid with reward (line) |

### `analyze()` outputs → `logs/`

Called from `main.py analyze` mode:

- `transition_matrices.png` — 4-row × 2-column grid; for each intent: attack type transition heatmap + kill chain stage transition heatmap
- `feature_distributions.png` — box plots of 5 key features (`dur`, `sload`, `spkts`, `sbytes`, `ct_dst_ltm`) across all attack types (symlog y-axis)

---

## Interpreting Results

### Good training signal

- **Detection rate** should climb from ~0.3–0.5 early to > 0.8 by end of training
- **False positive rate** should stabilise below 0.3 (the agent learns not to BLOCK/ALERT benign traffic)
- **Episode reward** should trend upward
- **DQN loss** should decrease and stabilise (not necessarily reach zero)

### What to watch for

| Symptom | Likely cause |
|---|---|
| Detection rate near 1.0, FP rate also near 1.0 | Agent always responds — never ALLOWs |
| Detection rate near 0.0 | Agent always ALLOWs — exploration not working |
| Loss oscillating without decreasing | Learning rate too high or target update too frequent |
| Flat reward after initial rise | Agent converged to a suboptimal policy (local optimum) |

### Multi-intent comparison

Use `python main.py compare` to evaluate a single trained policy against all four attacker intents. A robust policy trained on OPPORTUNISTIC (default) should generalise reasonably to STEALTHY and TARGETED, but may underperform against AGGRESSIVE (faster escalation requires quicker BLOCK/ALERT responses).
