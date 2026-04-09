# Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CyberSecurityEnv                         │
│  (gymnasium.Env wrapper — bridges attacker and defender)        │
│                                                                 │
│  ┌──────────────────────┐       ┌───────────────────────────┐  │
│  │      Attacker        │──────▶│       Defender            │  │
│  │                      │ state │                           │  │
│  │  TransitionModel     │       │  AttackClassifier (RF)    │  │
│  │  (Markov chain)      │ feat  │  DQNAgent                 │  │
│  │                      │──────▶│  (policy net + target net)│  │
│  │  feature simulation  │       │                           │  │
│  └──────────────────────┘       └───────────────────────────┘  │
│           │                               │                     │
│           │ attack_type, stage, features  │ action              │
│           ▼                               ▼                     │
│  compute_threat_level() ──────▶ compute_reward()               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    MetricsCollector
                    (StepRecord, EpisodeRecord, plots)
```

---

## Data Flow per Step

```
1. env.step(action) is called

2. Attacker.step()
   ├── TransitionModel.next_attack(current_attack)   → new attack type
   ├── TransitionModel.next_stage(current_stage)     → new kill chain stage
   │   (floored to avoid stage regression)
   └── _simulate_features(attack_type)               → 15 network features

3. compute_threat_level(attack_type, stage,
                        escalation_rate, attack_count)  → threat ∈ [0,1]

4. compute_reward(action, threat_level,
                  is_attack, stage, attack_type)        → reward

5. _build_state(...)                                    → 24-dim state vector

6. Return (next_state, reward, terminated, truncated, info)
```

```
Training loop:
┌──────────────────────────────────────────────────────────────────┐
│  for each episode:                                               │
│    state, info = env.reset()                                     │
│    for each step:                                                │
│      features = info["features"]                                 │
│      action, pred = defender.observe(state, features, train=True)│
│        ├── classifier.predict(features)  → pred_attack           │
│        └── dqn.select_action(state)      → action (ε-greedy)    │
│      next_state, reward, done, _, info = env.step(action)        │
│      loss = defender.learn(s, a, r, s', done)                    │
│        ├── replay_buffer.push(...)                               │
│        └── dqn.update()                                          │
│             ├── sample mini-batch                                │
│             ├── compute TD targets (target net)                  │
│             ├── SmoothL1 loss                                    │
│             ├── Adam optimizer step                              │
│             ├── decay epsilon                                    │
│             └── hard-copy to target net every 150 steps          │
│      metrics.record_step(...)                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### `attacker/`

| File | Responsibility |
|---|---|
| `attack_types.py` | Defines `AttackType`, `KillChainStage`, `AttackerIntent` enums; severity/weight dicts; per-attack feature distributions |
| `transition_model.py` | Constructs intent-modified Markov transition matrices; samples next attack/stage |
| `attacker.py` | Stateful agent — advances step-by-step through the kill chain, generating features |

### `defender/`

| File | Responsibility |
|---|---|
| `dqn.py` | `DQNNetwork` (feed-forward NN), `ReplayBuffer` (circular deque), `DQNAgent` (select/store/update) |
| `classifier.py` | `AttackClassifier` wrapping sklearn `RandomForestClassifier`; synthetic data generation, predict/predict_proba |
| `honeypot.py` | `HoneypotAction` enum; `compute_threat_level()`; `compute_reward()` with full reward matrix |
| `defender.py` | `Defender` orchestrator — wraps classifier + DQN; `observe()`, `learn()`, `save()`, `load()` |

### `environment/`

| File | Responsibility |
|---|---|
| `cyber_env.py` | `CyberSecurityEnv(gym.Env)` — `reset()`, `step()`, `_build_state()`, `render()` |

### `evaluation/`

| File | Responsibility |
|---|---|
| `metrics.py` | `StepRecord` / `EpisodeRecord` dataclasses; `MetricsCollector` — accumulation, CSV export, 6 plot types |

---

## State Vector Layout (24 dimensions)

```
Indices  Width  Content
0–9      10     attack_type one-hot       (NORMAL=0 … WORMS=9)
10–16     7     kill_chain_stage one-hot  (RECON=0 … ACTIONS_ON_OBJ=6)
17        1     threat_level              float [0, 1]
18        1     attack_count / 100        float [0, 1]
19        1     escalation_rate           float [0, 1]
20–23     4     attacker_intent one-hot   (STEALTHY=0 … OPPORTUNISTIC=3)
```

---

## Hyperparameter Summary

| Parameter | Value | Location |
|---|---|---|
| State dimension | 24 | `environment/cyber_env.py` |
| Action dimension | 5 | `environment/cyber_env.py` |
| Hidden layers | [256, 128, 64] | `defender/dqn.py` |
| Learning rate | 1e-3 | `train.py` |
| Discount factor γ | 0.99 | `train.py` |
| ε start | 1.0 | `train.py` |
| ε end | 0.05 | `train.py` |
| ε decay | 0.997 | `train.py` |
| Batch size | 64 | `train.py` |
| Target update freq | 150 steps | `train.py` |
| Replay buffer capacity | 15,000 | `train.py` |
| RF n_estimators | 150 | `train.py` |
| RF max_depth | 20 | `train.py` |
| Classifier samples/class | 600 | `train.py` |
| Escalation window | 20 steps | `environment/cyber_env.py` |
