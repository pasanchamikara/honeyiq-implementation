# HoneyIQ

A cybersecurity attacker-defender simulation that trains a honeypot-based defender using **Deep Q-Network (DQN)** reinforcement learning against a **Markov-chain-driven attacker** progressing through the cyber kill chain.

---

## Overview

HoneyIQ models the cybersecurity problem as a two-agent game:

- **Attacker** — follows a stochastic Markov chain through 10 attack types and 7 kill chain stages, generating synthetic UNSW-NB15-style network flow features at each step.
- **Defender** — observes the environment state, classifies incoming traffic with a RandomForest, and selects one of 5 honeypot actions via a DQN policy.

The defender learns to maximize cumulative reward by correctly matching its response (ALLOW / LOG / TROLL / BLOCK / ALERT) to the actual threat level.

---

## Project Structure

```
honeyiq-implementation/
├── main.py                    # CLI: demo, compare, train, analyze
├── train.py                   # Training loop and multi-intent trainer
│
├── attacker/
│   ├── attack_types.py        # AttackType, KillChainStage, AttackerIntent enums;
│   │                          #   severity weights; UNSW-NB15 feature distributions
│   ├── transition_model.py    # Intent-shaped Markov chains (attack + stage)
│   └── attacker.py            # AttackerAgent — steps through the kill chain,
│                              #   samples attack transitions, simulates features
│
├── defender/
│   ├── honeypot.py            # HoneypotAction enum; threat-level formula; reward function
│   ├── classifier.py          # AttackClassifier (RandomForest on synthetic data)
│   ├── dqn.py                 # DQNNetwork, ReplayBuffer, DQNAgent
│   └── defender.py            # Defender orchestrator (classifier + DQN)
│
├── environment/
│   └── cyber_env.py           # CyberSecurityEnv (Gymnasium) — bridges attacker & defender
│
├── evaluation/
│   └── metrics.py             # MetricsCollector, StepRecord, EpisodeRecord, plots
│
├── notebooks/                 # Jupyter notebooks (one per layer)
│   ├── 01_attacker_model.ipynb
│   ├── 02_defender_model.ipynb
│   ├── 03_environment_and_metrics.ipynb
│   └── 04_training_and_evaluation.ipynb
│
├── assets/                    # Architecture diagrams
├── docs/                      # Extended documentation
├── models/                    # Saved checkpoints (dqn_agent.pt, classifier.joblib)
└── logs/                      # CSV metrics and PNG plots
```

---

## Components

### Attacker

#### Attack types (`attacker/attack_types.py`)
Ten categories drawn from the UNSW-NB15 dataset:

| # | Type | Severity | Primary Kill Chain Stage |
|---|---|---|---|
| 0 | NORMAL | 0.00 | Reconnaissance |
| 1 | RECONNAISSANCE | 0.20 | Reconnaissance |
| 2 | ANALYSIS | 0.25 | Weaponization |
| 3 | FUZZERS | 0.35 | Delivery |
| 4 | EXPLOITS | 0.70 | Exploitation |
| 5 | BACKDOORS | 0.80 | Installation |
| 6 | SHELLCODE | 0.75 | Exploitation |
| 7 | GENERIC | 0.40 | Delivery |
| 8 | DOS | 0.85 | Actions on Objectives |
| 9 | WORMS | 0.90 | Command & Control |

Each attack type has parametric feature distributions for 15 UNSW-NB15 network flow fields (`dur`, `sbytes`, `dbytes`, `sttl`, `dttl`, `sloss`, `dloss`, `sload`, `dload`, `spkts`, `dpkts`, `swin`, `dwin`, `ct_srv_src`, `ct_dst_ltm`).

#### Attacker intents (`attacker/attack_types.py`)
Four intent profiles that bias the transition probabilities:

| Intent | Behaviour |
|---|---|
| STEALTHY | Low-and-slow; favours recon and backdoors; avoids noisy attacks |
| AGGRESSIVE | Fast escalation; high-impact attacks (DoS, Worms, Exploits) |
| TARGETED | Focused exploit chain → shellcode → backdoor → lateral movement |
| OPPORTUNISTIC | Scattered; elevated fuzzer and generic attack rates |

#### Transition model (`attacker/transition_model.py`)
Two separate row-stochastic matrices are maintained:
- **Attack matrix** (10 × 10): next attack type given current
- **Stage matrix** (7 × 7): next kill chain stage given current

Base matrices are multiplied element-wise by intent-specific modifiers, then row-normalized. The attacker agent always takes the max of the sampled stage and the attack's primary stage to prevent unrealistic regression.

---

### Defender

#### Honeypot actions & reward (`defender/honeypot.py`)

| Action | Optimal for |
|---|---|
| ALLOW | Benign traffic only |
| LOG | Low threats (intelligence gathering) |
| TROLL | Medium threats (tarpit / fake data) |
| BLOCK | High threats (firewall) |
| ALERT | Critical threats (immediate escalation) |

Threat level is computed as a weighted composite:
```
threat = 0.45 × attack_severity
       + 0.35 × kill_chain_weight
       + 0.15 × escalation_rate   (sliding window fraction of recent attacks)
       + 0.05 × min(1, attack_count / 100)
```

Rewards are looked up from a 5 × 5 action-by-threat-band matrix. Late kill chain stages (Installation, C2, Actions on Objectives) amplify negative rewards by 1.5×. Specific bonuses apply for high-value honeypot interactions (trolling backdoors/worms, blocking worms, logging reconnaissance).

#### Attack classifier (`defender/classifier.py`)
A `RandomForestClassifier` (scikit-learn) trained on synthetic data generated directly by the attacker's feature simulator. Key properties:
- `class_weight='balanced'` handles class imbalance
- `StandardScaler` normalizes features before training and inference
- `fit_from_simulation()` generates data and fits in one call
- `evaluate()` reports per-class precision/recall/F1 on a held-out test set

#### DQN agent (`defender/dqn.py`)

| Component | Detail |
|---|---|
| Architecture | Fully connected: 24 → 256 → 128 → 64 → 5, with `LayerNorm` + ReLU per layer |
| Exploration | Epsilon-greedy with exponential decay (`ε_start=1.0`, `ε_end=0.05`, `decay=0.997`) |
| Replay buffer | Fixed-capacity circular deque (default 15,000 transitions) |
| Loss | Huber loss (SmoothL1) for robustness to outlier rewards |
| Optimization | Adam optimizer with gradient clipping (`max_norm=10`) |
| Target network | Hard copy from policy net every `target_update_freq` steps |

#### Defender orchestrator (`defender/defender.py`)
Wraps the classifier and DQN agent with a clean API:
- `observe(state, features, training)` → classifies features, selects DQN action
- `learn(state, action, reward, next_state, done)` → stores transition, triggers update
- `save(model_dir)` / `load(model_dir)` → persists both components independently

---

### Environment (`environment/cyber_env.py`)

A standard **Gymnasium** environment (`gym.Env`).

**State vector** (24 floats):
```
[0:10]   attack_type one-hot       (10 classes)
[10:17]  kill_chain_stage one-hot  (7 stages)
[17]     threat_level              float [0, 1]
[18]     attack_count_normalized   float [0, 1]  — min(1, count/100)
[19]     escalation_rate           float [0, 1]  — sliding window
[20:24]  attacker_intent one-hot   (4 intents)
```

**Action space**: `Discrete(5)` — one per `HoneypotAction`.

Each call to `step(action)` advances the attacker by one step, computes threat level and reward, and returns the next state.

---

### Metrics (`evaluation/metrics.py`)

`MetricsCollector` accumulates `StepRecord` objects during an episode and aggregates them into an `EpisodeRecord` at episode end. Tracked quantities:

| Metric | Definition |
|---|---|
| Detection rate | TP / (TP + FN) — attacks where action ≠ ALLOW |
| False positive rate | FP / (FP + TN) — benign traffic blocked/alerted |
| Avg threat level | Mean threat across all steps |
| Avg DQN loss | Mean Huber loss per episode |

Built-in visualizations: training curves, kill-chain action heatmap, single-episode attack progression.

---

## Quickstart

### Install dependencies
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Train
```bash
python main.py train --episodes 300 --intent OPPORTUNISTIC
```

### Run a demo episode
```bash
python main.py demo --intent STEALTHY --steps 150
```

### Compare the policy across all attacker intents
```bash
python main.py compare --episodes 5 --steps 200
```

### Visualize transition matrices and feature distributions
```bash
python main.py analyze
```

---

## Notebooks

Interactive walkthroughs in `notebooks/`:

| Notebook | Contents |
|---|---|
| `01_attacker_model.ipynb` | Enumerations, transition matrix heatmaps, feature distributions, trajectory visualization |
| `02_defender_model.ipynb` | Reward matrix, classifier training & evaluation, DQN architecture, epsilon schedule |
| `03_environment_and_metrics.ipynb` | Gym API walkthrough, random-policy episode, episode metric plots |
| `04_training_and_evaluation.ipynb` | Full training run, training curves, demo/compare/analyze modes |

Launch with:
```bash
jupyter notebook notebooks/
```

---

## Extended Documentation

See [`docs/`](docs/) for in-depth coverage of each component, theoretical background (RL, DQN, Markov chains, kill chain model), API reference, and architecture diagrams.
