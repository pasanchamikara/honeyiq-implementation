# HoneyIQ

A cybersecurity attacker-defender simulation that evaluates a honeypot-based defender using a **Stage-Escalation Decision Matrix (SEDM)** — an interpretable, deterministic policy — against a **Markov-chain-driven attacker** progressing through the Lockheed Martin Cyber Kill Chain.

---

## Overview

HoneyIQ models the cybersecurity problem as a two-agent game:

- **Attacker** — follows a stochastic Markov chain through 10 attack types and 7 kill chain stages, generating synthetic UNSW-NB15-style network flow features at each step. Four intent profiles (Stealthy, Aggressive, Targeted, Opportunistic) bias the transition probabilities to produce qualitatively distinct campaigns.
- **Defender** — observes the environment state, classifies incoming traffic with a RandomForest, and selects one of 5 honeypot actions via the **SEDM policy** (ALLOW / LOG / TROLL / BLOCK / ALERT).

The SEDM maps the current kill chain stage and a Markov-chain-derived escalation risk score to an optimal action, with override rules for high-impact attack types and elevated attack frequency.

---

## Key Results

Evaluated over 30 episodes per intent (200 steps each):

| Intent | Mean Reward | Detection Rate | False Positive Rate | Avg Threat Level |
|---|---|---|---|---|
| STEALTHY | 1012.22 ± 51.37 | **99.09%** | 35.56% | 0.806 |
| AGGRESSIVE | 1090.84 ± 19.80 | **99.47%** | 6.67% | 0.854 |
| TARGETED | 1127.10 ± 20.98 | **99.48%** | 3.33% | 0.853 |
| OPPORTUNISTIC | 896.05 ± 32.98 | **99.41%** | 15.00% | 0.790 |

The SEDM achieves near-perfect detection rates across all four attacker intent profiles, demonstrating strong cross-intent generalisation.

### Action Distribution

| Intent | ALLOW | LOG | TROLL | BLOCK | ALERT |
|---|---|---|---|---|---|
| STEALTHY | 0.0% | 1.1% | 0.7% | 17.9% | 80.3% |
| AGGRESSIVE | 0.0% | 0.0% | 0.0% | 4.6% | 95.4% |
| TARGETED | 0.0% | 0.5% | 0.1% | 5.4% | 94.0% |
| OPPORTUNISTIC | 0.1% | 1.0% | 0.4% | 24.6% | 73.8% |

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
│   ├── matrix_policy.py       # MatrixPolicy (SEDM) — primary decision policy
│   ├── dqn.py                 # DQNNetwork, ReplayBuffer, DQNAgent (baseline)
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
├── logs/                      # Training CSV metrics and PNG plots
└── results/                   # Evaluation outputs (per-intent CSVs and plots)
    └── evaluation/
        ├── evaluation_summary.csv
        ├── action_distribution.csv
        ├── sedm_table.csv
        └── *.png               # Visualisation plots
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
| 4 | GENERIC | 0.40 | Delivery |
| 5 | EXPLOITS | 0.70 | Exploitation |
| 6 | SHELLCODE | 0.75 | Exploitation |
| 7 | BACKDOORS | 0.80 | Installation |
| 8 | DOS | 0.85 | Actions on Objectives |
| 9 | WORMS | 0.90 | Command & Control |

Each attack type has parametric feature distributions for 15 UNSW-NB15 network flow fields.

#### Attacker intents (`attacker/attack_types.py`)
Four intent profiles that bias the Markov transition probabilities:

| Intent | Behaviour |
|---|---|
| STEALTHY | Low-and-slow; favours recon and backdoors; avoids noisy attacks |
| AGGRESSIVE | Fast escalation; high-impact attacks (DoS, Worms, Exploits) |
| TARGETED | Focused exploit chain → shellcode → backdoor → lateral movement |
| OPPORTUNISTIC | Scattered; elevated fuzzer and generic attack rates |

---

### Defender

#### Stage-Escalation Decision Matrix (`defender/matrix_policy.py`)

The primary decision policy. Maps (kill chain stage, escalation risk band) → honeypot action:

| Stage / Band | Low (<0.35) | Medium (0.35–0.65) | High (≥0.65) |
|---|---|---|---|
| RECONNAISSANCE | ALLOW | LOG | LOG |
| WEAPONIZATION | LOG | LOG | TROLL |
| DELIVERY | LOG | TROLL | TROLL |
| EXPLOITATION | TROLL | BLOCK | BLOCK |
| INSTALLATION | BLOCK | BLOCK | ALERT |
| COMMAND_AND_CTRL | BLOCK | ALERT | ALERT |
| ACTIONS_ON_OBJ | ALERT | ALERT | ALERT |

**Escalation risk** is computed from the intent-specific Markov chain as P(next stage > current stage).

**Override rules** (applied after matrix lookup):
- R1: Normal traffic → always ALLOW
- R2: DOS or WORMS → upgrade action one level
- R3: Escalation rate > 0.80 → upgrade action one level

#### Honeypot actions & reward (`defender/honeypot.py`)

| Action | Optimal for |
|---|---|
| ALLOW | Benign traffic only |
| LOG | Low threats (intelligence gathering) |
| TROLL | Medium threats (tarpit / fake data) |
| BLOCK | High threats (firewall) |
| ALERT | Critical threats (immediate escalation) |

#### Attack classifier (`defender/classifier.py`)
A `RandomForestClassifier` (scikit-learn) trained on synthetic data generated by the attacker's feature simulator. Uses `class_weight='balanced'` and `StandardScaler` normalisation.

#### DQN agent (`defender/dqn.py`)
Baseline deep learning policy (24 → 256 → 128 → 64 → 5), retained for comparison. Uses experience replay, target network, Huber loss, and epsilon-greedy exploration.

---

### Environment (`environment/cyber_env.py`)

A standard **Gymnasium** environment. State vector (24 floats):
```
[0:10]   attack_type one-hot       (10 classes)
[10:17]  kill_chain_stage one-hot  (7 stages)
[17]     threat_level              float [0, 1]
[18]     attack_count_normalized   float [0, 1]
[19]     escalation_rate           float [0, 1]
[20:24]  attacker_intent one-hot   (4 intents)
```

Composite threat level:
```
T = 0.45 × attack_severity + 0.35 × kill_chain_weight
  + 0.15 × escalation_rate + 0.05 × min(1, attack_count/100)
```

---

## Quickstart

### Install dependencies
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Train (DQN baseline)
```bash
python main.py train --episodes 300 --intent OPPORTUNISTIC
```

### Run a demo episode
```bash
python main.py demo --intent STEALTHY --steps 150
```

### Compare SEDM policy across all attacker intents
```bash
python main.py compare --episodes 30 --steps 200
```

### Visualize transition matrices and feature distributions
```bash
python main.py analyze
```

---

## Notebooks

| Notebook | Contents |
|---|---|
| `01_attacker_model.ipynb` | Enumerations, transition matrix heatmaps, feature distributions, trajectory visualisation |
| `02_defender_model.ipynb` | Reward matrix, classifier training & evaluation, DQN architecture, SEDM decision matrix |
| `03_environment_and_metrics.ipynb` | Gym API walkthrough, random-policy episode, episode metric plots |
| `04_training_and_evaluation.ipynb` | SEDM evaluation, cross-intent comparison, result visualisation |

---

## Extended Documentation

See [`docs/`](docs/) for in-depth coverage of each component, theoretical background, API reference, and architecture diagrams.

See [`results/evaluation/`](results/evaluation/) for evaluation CSVs and visualisation plots.
