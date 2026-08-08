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

Evaluated over 30 episodes per intent (200 steps each), classifier-driven SEDM
(policy decisions are made from the RandomForest's *predicted* attack type,
not ground truth — see [Methodology note](#evaluation-methodology-note) below):

| Intent | Mean Reward | Detection Rate | False Positive Rate | Avg Threat Level |
|---|---|---|---|---|
| STEALTHY | 1010.22 ± 51.52 | **99.93%** | 0.00% | 0.806 |
| AGGRESSIVE | 1090.84 ± 19.80 | **100.0%** | 0.00% | 0.854 |
| TARGETED | 1126.10 ± 21.54 | **99.97%** | 0.00% | 0.853 |
| OPPORTUNISTIC | 895.05 ± 33.30 | **99.97%** | 0.00% | 0.790 |

The SEDM achieves near-perfect detection with zero measured false positives
across all four attacker intent profiles. This is **not** a claim of
real-world perfection — it is a direct consequence of two facts, both
made explicit rather than left implicit:
(1) the R1 override unconditionally maps a predicted-NORMAL sample to
ALLOW, and (2) the RandomForest classifier separates NORMAL from
attack traffic with 100% precision/recall on the synthetic UNSW-NB15-style
feature distributions (99.85% overall 10-class accuracy — see
[`logs/classifier_eval_report.json`](logs/classifier_eval_report.json)).
Because the feature simulator generates attack classes from disjoint
parametric distributions by construction, this ceiling is expected to be
*optimistic* relative to real, noisy network telemetry. A feature-noise
robustness sweep confirms the mechanism directly: injecting multiplicative
Gaussian noise onto the simulated features degrades classifier accuracy from
99.85% (0% noise) to 87.65% (50% noise) and raises the NORMAL→attack false
positive rate from 0.00% to 10.00% (see
[`logs/classifier_noise_robustness.json`](logs/classifier_noise_robustness.json),
Figure `classifier_noise_robustness.png`) — i.e. false positives are
recovered as soon as the input signal is made realistically imperfect,
confirming the zero-FP result is a property of the (clean) synthetic data,
not an artefact that would necessarily hold against real traffic captures.

### Action Distribution

| Intent | ALLOW | LOG | TROLL | BLOCK | ALERT |
|---|---|---|---|---|---|
| STEALTHY | 2.0% | 0.3% | 0.9% | 1.7% | 95.1% |
| AGGRESSIVE | 0.7% | 0.0% | 0.1% | 1.7% | 97.6% |
| TARGETED | 0.6% | 0.0% | 0.1% | 1.5% | 97.8% |
| OPPORTUNISTIC | 0.8% | 0.0% | 0.2% | 2.8% | 96.2% |

### Extended Metrics (Mixed Traffic, 50 episodes × 500 steps, 30% benign ratio)

Precision/Recall/F1/Specificity/late-stage-miss/steps-to-contain, computed
with the action correctly paired to the *same* observation's ground-truth
label (see the methodology note below), for both an **oracle** SEDM
(decides from ground-truth state) and a **classifier-driven** SEDM (decides
from the RandomForest's prediction):

| Intent | Variant | Precision | Recall | F1 | Specificity | Prop. Score | Late-Miss | Steps-to-Contain |
|---|---|---|---|---|---|---|---|---|
| STEALTHY | Oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.693 | 0.000 | 2.22 |
| STEALTHY | Classifier | 1.000 | 1.000 | 1.000 | 0.999 | 0.693 | 0.000 | 2.3 |
| AGGRESSIVE | Oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.698 | 0.000 | 0.24 |
| AGGRESSIVE | Classifier | 1.000 | 1.000 | 1.000 | 0.999 | 0.698 | 0.000 | 0.2 |
| TARGETED | Oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.699 | 0.000 | 0.60 |
| TARGETED | Classifier | 1.000 | 0.999 | 0.999 | 0.999 | 0.699 | 0.001 | 0.6 |
| OPPORTUNISTIC | Oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.698 | 0.000 | 1.08 |
| OPPORTUNISTIC | Classifier | 0.999 | 1.000 | 1.000 | 0.999 | 0.698 | 0.000 | 1.1 |

Spearman ρ (kill-chain stage vs. action severity, attack-only steps) = **0.492** (p < 1e-200).

### Evaluation Methodology Note

Both `evaluate.py` and `evaluation/sedm_eval.py` score each action against
the ground-truth label of the **same observation** the action was chosen
from — not the label of the *next* environment step. Pairing an action with
the wrong step's label silently shifts every detection/FP metric by one time
step; this was caught and fixed as **Bug 8** in
[`docs/BUGS_AND_FIXES.md`](docs/BUGS_AND_FIXES.md), and the same fix was
subsequently applied to `main.py`'s `demo`/`compare` paths and to
`evaluation/sedm_eval.py`, which had regressed on this point when the
extended-metrics evaluation was added.

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
├── docs/                      # Extended documentation (architecture, API, bug log, parameter selection)
├── models/                    # Saved checkpoints (dqn_agent.pt, classifier.joblib)
├── logs/                      # Training CSV metrics and PNG plots
├── results/                   # Evaluation outputs (per-intent CSVs and plots)
│   └── evaluation/
│       ├── evaluation_summary.csv
│       ├── action_distribution.csv
│       ├── sedm_table.csv
│       └── *.png               # Visualisation plots
│
└── thesis/                    # Thesis-writing artifacts — isolated from the implementation above
    ├── latex/                 # LaTeX thesis chapters, figures, bibliography
    ├── doc0/                  # Chapter-by-chapter Markdown mirror of latex/chapters/, kept in sync
    ├── slides/                # Defence slides
    └── Thesis___HoneyIQ/
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
