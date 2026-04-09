# Quick Start

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU is fine for default settings)

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---|---|
| `torch` | DQN neural network |
| `gymnasium` | RL environment interface |
| `scikit-learn` | Random Forest classifier |
| `numpy` | Numerical ops, Markov sampling |
| `pandas` | Classifier training data, CSV export |
| `matplotlib` / `seaborn` | Visualizations |
| `tqdm` | Training progress bar |
| `joblib` | Classifier serialisation |

---

## CLI Modes

All modes are accessed via `main.py`:

```
python main.py <mode> [options]
```

### `analyze` — Visualize attacker behaviour

```bash
python main.py analyze
```

Generates in `logs/`:
- `transition_matrices.png` — Markov transition heatmaps for all 4 intent profiles
- `feature_distributions.png` — box plots of key network features per attack type

No training required.

### `train` — Train the defender

```bash
# Basic training (300 episodes, OPPORTUNISTIC attacker)
python main.py train

# More episodes, different intent
python main.py train --episodes 500 --intent STEALTHY

# Train separate models for all 4 intents
python main.py train --all-intents --episodes 200

# Skip classifier training (faster iteration)
python main.py train --no-classifier
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--episodes` | 300 | Number of training episodes |
| `--steps` | 500 | Max steps per episode |
| `--intent` | OPPORTUNISTIC | Attacker intent profile |
| `--seed` | 42 | Random seed |
| `--save-dir` | models/ | Model checkpoint directory |
| `--log-dir` | logs/ | Logs and plots directory |
| `--eval-interval` | 50 | Episodes between eval runs |
| `--save-interval` | 100 | Episodes between checkpoints |
| `--no-classifier` | — | Skip RF classifier training |
| `--clf-samples` | 600 | Samples per class for classifier |
| `--all-intents` | — | Train against all 4 intents |

Training output (example):
```
============================================================
HoneyIQ Training — Intent: OPPORTUNISTIC
Episodes: 300  |  Steps/ep: 500
Device: cpu
============================================================

[Classifier] Test accuracy: 0.942

Training: 100%|████| 300/300 [ep] [reward=312.4, det=0.87, fp=0.18, ε=0.050]

[Eval  ep  50]  reward=     89.2  det=0.623  fp=0.312
[Eval  ep 100]  reward=    201.5  det=0.771  fp=0.241
[Eval  ep 150]  reward=    278.3  det=0.841  fp=0.198
...

============================================================
Training complete in 142.3s
  Mean reward     : 187.34
  Best reward     : 398.12
  Mean det. rate  : 0.812
  Mean FP rate    : 0.183
  Mean threat lvl : 0.374
============================================================
```

### `demo` — Run a single episode

```bash
# Default: OPPORTUNISTIC attacker, 100 steps
python main.py demo

# Specify intent and steps
python main.py demo --intent AGGRESSIVE --steps 150

# Suppress console output
python main.py demo --no-render --no-plot
```

Output (per-step table):
```
Step  Attack Type        Stage                   Threat Band      Action   Reward PredAtk          IsAtk
----------------------------------------------------------------------------------------------------------------------
    1 NORMAL             RECONNAISSANCE            0.000 benign    ALLOW     1.50 NORMAL           no
    2 RECONNAISSANCE     RECONNAISSANCE            0.164 low       LOG       2.00 RECONNAISSANCE   YES
    3 ANALYSIS           WEAPONIZATION             0.234 low       LOG       1.50 ANALYSIS         YES
    4 EXPLOITS           EXPLOITATION              0.612 high      BLOCK     3.50 EXPLOITS         YES
```

Saves `logs/demo_progression.png`.

### `compare` — Multi-intent evaluation

```bash
python main.py compare --episodes 10 --steps 200
```

Requires a trained model in `models/`. Prints a summary table:

```
Intent           MeanReward    DetRate     FPRate
----------------------------------------------------
STEALTHY             198.40      0.812      0.203
AGGRESSIVE           134.70      0.763      0.241
TARGETED             221.30      0.849      0.189
OPPORTUNISTIC        245.90      0.871      0.172
```

---

## Programmatic Usage

```python
from attacker.attack_types import AttackerIntent
from defender.defender import Defender
from environment.cyber_env import CyberSecurityEnv
from evaluation.metrics import MetricsCollector

env      = CyberSecurityEnv(attacker_intent=AttackerIntent.STEALTHY, max_steps=200)
defender = Defender()
defender.load("models/")

metrics = MetricsCollector(log_dir="logs/")
state, info = env.reset()

for step in range(200):
    features = info.get("features", {})
    action, pred = defender.observe(state, features, training=False)
    state, reward, terminated, truncated, info = env.step(action)
    metrics.record_step(0, step, action, reward, info, pred, None)
    if terminated or truncated:
        break

record = metrics.end_episode(0)
print(f"Detection rate: {record.detection_rate:.3f}")
print(f"False positive: {record.false_positive_rate:.3f}")
```

---

## Saved Files

After training:

```
models/
├── dqn_agent.pt          # DQN weights + optimizer state + metadata
└── classifier.joblib     # RandomForest + scaler + feature names

logs/
├── metrics.csv           # Per-episode metrics table
├── training_curves.png   # 6-panel training progress
├── action_stage_heatmap.png
├── demo_progression.png  # (after demo mode)
└── transition_matrices.png  # (after analyze mode)
```

---

## Recommended Workflow

1. **Explore** the attacker behaviour first: `python main.py analyze`
2. **Train** a baseline model: `python main.py train --episodes 300`
3. **Evaluate** performance: check `logs/training_curves.png`
4. **Demo** the policy: `python main.py demo --intent STEALTHY`
5. **Compare** across all intents: `python main.py compare`
6. **Iterate**: try `--episodes 500`, `--intent AGGRESSIVE`, or `--all-intents`
