# Quick Start

## Requirements

- Python 3.10+
- No `torch` requirement for anything documented here — it's only needed
  by the orphaned `defender/dqn.py` (see [`dqn_practicality.md`](dqn_practicality.md))

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies actually exercised by the current implementation:

| Package | Purpose |
|---|---|
| `gymnasium` | `CyberSecurityEnv`'s base class |
| `scikit-learn` | RandomForest attack classifier |
| `numpy` | Markov sampling, feature simulation, state vectors |
| `pandas` | Classifier training data, CSV export |
| `matplotlib` / `seaborn` | Visualizations |
| `joblib` | Classifier serialization |
| `pydantic` | `OpenCanaryEvent` model |

## The SEDM policy needs no training

Unlike the original DQN-based defender, `MatrixPolicy` requires **no
model checkpoint** — it's a deterministic lookup table. Only the
classifier (`models/classifier.joblib`) needs to exist on disk, and even
that's optional (without it, the pipeline falls back to logtype-based
attack-type mapping / `AttackType.NORMAL`).

## `main.py` — demo, compare, analyze

```bash
python main.py demo --intent AGGRESSIVE --steps 100
python main.py compare --episodes 10 --steps 200
python main.py analyze
```

- **`demo`** — one episode, step-by-step console table, saves
  `logs/demo_progression.png`.
- **`compare`** — evaluates the shared classifier against all 4 intents,
  prints a summary table. Loads the classifier once, shares it across
  intents.
- **`analyze`** — Markov transition-matrix heatmaps
  (`logs/transition_matrices.png`) and feature-distribution box plots
  (`logs/feature_distributions.png`, now showing real per-sample
  intensity/persona variance — see [`attacker.md`](attacker.md)).

`train` mode still exists and still accepts a `dqn_config` dict
(silently ignored) for backward compatibility with older scripts — it's
not needed to use the SEDM.

## `evaluate.py` — full SEDM evaluation

```bash
python evaluate.py --episodes 30 --steps 200 --seed 42
```

See [`evaluation.md`](evaluation.md) for full output details.

## `evaluation/sedm_eval.py` — extended classification metrics

```bash
python -m evaluation.sedm_eval --episodes 50 --steps 500 --variant both
```

## Live OpenCanary pipeline (emulated, no live honeypot needed)

```bash
python -m opencanary_integration.emulator.scenario --scenario kill_chain --src-ip 10.0.0.1
python -m opencanary_integration.emulator.scenario --scenario random --events 20
```

See [`opencanary_integration.md`](opencanary_integration.md).

## Programmatic usage

**Synthetic training environment:**

```python
from attacker.attack_types import AttackerIntent
from defender.defender import Defender
from environment.cyber_env import CyberSecurityEnv

env      = CyberSecurityEnv(attacker_intent=AttackerIntent.STEALTHY, max_steps=200)
defender = Defender()
defender.load("models/")

state, info = env.reset()
for step in range(200):
    features = info.get("features", {})
    action, pred = defender.observe(state, features, training=False)
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

**Live pipeline, one event at a time:**

```python
from opencanary_integration.emulator.scenario import EmulatorScenario
from opencanary_integration.emulator.event_generator import OpenCanaryEventGenerator

scenario = EmulatorScenario(model_dir="models/", intent="OPPORTUNISTIC")
generator = OpenCanaryEventGenerator(seed=42)

for event in generator.generate_kill_chain(src_ip="10.0.0.1"):
    decision = scenario.run_event(event)
    print(decision["action"], decision["stage"], decision["reputation"])
```

**Opting into the new mechanisms directly:**

```python
from defender.matrix_policy import MatrixPolicy, RATE_THRESHOLD
from defender.adaptive_thresholds import AdaptiveThresholds
from environment.cyber_env import CyberSecurityEnv

# EMA escalation instead of the window
env = CyberSecurityEnv(escalation_mode="ema", escalation_ema_alpha=0.15)

# Bounded auto-tuning of the R3 alert-fatigue threshold
policy = MatrixPolicy(
    adaptive_thresholds=AdaptiveThresholds(initial_threshold=RATE_THRESHOLD),
)

# Manual reputation lookup (normally comes from SessionTracker in the live pipeline)
action, info = policy.decide_from_state(state, reputation=0.75)
```

## Saved files

```
models/
├── classifier.joblib     # RandomForest + scaler + feature names
└── dqn_agent.pt          # Legacy — not read by anything active

logs/
├── metrics.csv, training_curves.png, action_stage_heatmap.png
├── demo_progression.png       (after `demo`)
├── transition_matrices.png    (after `analyze`)
├── feature_distributions.png  (after `analyze`)
└── sedm_eval_results{,_clf}.json  (after evaluation/sedm_eval.py)

results/evaluation/
├── sedm_table.csv, evaluation_summary.csv, action_distribution.csv
├── *.png (7 plots — see evaluation.md)
└── opencanary_*_audit.jsonl   (kill-chain demo audit logs)
```

## Recommended workflow

1. `python main.py analyze` — see what the attacker's behavior and
   synthetic traffic actually look like.
2. `python evaluate.py --episodes 30 --steps 200` — full SEDM evaluation
   across all 4 intents.
3. `python -m evaluation.sedm_eval --variant both` — deeper classification
   metrics, oracle vs. classifier-driven.
4. `python -m opencanary_integration.emulator.scenario --scenario kill_chain`
   — see the live pipeline (session tracking, reputation, dispatch) in
   action end-to-end.
5. Read [`dynamic_response.md`](dynamic_response.md) and
   [`parameter_selection.md`](parameter_selection.md) before tuning
   `REPUTATION_THRESHOLD`, `AdaptiveThresholds`, or `escalation_ema_alpha`
   for a specific deployment.
