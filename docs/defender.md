# Defender Module

## Overview

The defender subsystem has four layers:

1. **Attack Classifier** (`classifier.py`) — Random Forest identifying attack types from raw network features
2. **Stage-Escalation Decision Matrix** (`matrix_policy.py`) — primary deterministic policy mapping kill chain stage and escalation risk to honeypot actions
3. **DQN Agent** (`dqn.py`) — Deep Q-Network baseline retained for comparison
4. **Defender Orchestrator** (`defender.py`) — wraps classifier + DQN; handles inference, learning, and persistence

The reward function and honeypot action definitions live in `honeypot.py`.

---

## `honeypot.py`

### `HoneypotAction` (IntEnum)

| Value | Name | Semantics |
|---|---|---|
| 0 | ALLOW | Let traffic through untouched |
| 1 | LOG | Record and monitor the session |
| 2 | TROLL | Respond with fake data / tarpit the attacker |
| 3 | BLOCK | Drop / firewall the connection |
| 4 | ALERT | Trigger a high-priority security alert |

### `compute_threat_level(attack_type, kill_chain_stage, escalation_rate, attack_count) → float`

Returns a composite threat level in [0, 1]:

```
T = 0.45 × ATTACK_SEVERITY[attack_type]
  + 0.35 × KILL_CHAIN_WEIGHT[kill_chain_stage]
  + 0.15 × escalation_rate
  + 0.05 × min(1, attack_count / 100)
```

### `threat_band(threat_level) → str`

| Band | Range |
|---|---|
| benign | < 0.15 |
| low | 0.15 – 0.35 |
| medium | 0.35 – 0.55 |
| high | 0.55 – 0.75 |
| critical | ≥ 0.75 |

### `compute_reward(action, threat_level, is_attack, kill_chain_stage, attack_type) → float`

Base reward from `_REWARD_MATRIX[action][band]`:

| Action | benign | low | medium | high | critical |
|---|---|---|---|---|---|
| ALLOW | +1.0 | +0.5 | -1.0 | -3.0 | -6.0 |
| LOG | +0.2 | +1.5 | +2.0 | +1.0 | -1.0 |
| TROLL | -1.0 | +1.0 | +3.0 | +2.5 | +0.5 |
| BLOCK | -2.0 | -0.5 | +1.5 | +3.5 | +5.0 |
| ALERT | -3.0 | -1.0 | +0.5 | +2.0 | +6.0 |

**Modifiers applied after base reward:**

1. Late-stage amplifier — if stage ∈ {INSTALLATION, C2, ACTIONS_ON_OBJ} and reward < 0: `reward *= 1.5`
2. TROLL + (BACKDOORS | SHELLCODE | WORMS): `reward += 0.8`
3. BLOCK + WORMS: `reward += 1.0`
4. LOG + RECONNAISSANCE: `reward += 0.5`
5. ALLOW + not is_attack: `reward += 0.5`

---

## `matrix_policy.py` — Stage-Escalation Decision Matrix (SEDM)

The SEDM is the **primary decision policy** in HoneyIQ. It replaces stochastic RL with a deterministic, interpretable decision procedure grounded in the kill chain model.

### Algorithm

**Step 1 — Escalation risk**: Query the intent-specific Markov transition model for the probability of advancing to a strictly higher kill chain stage:

```
esc_risk = Σ P(next_stage = s') for all s' > current_stage
```

**Step 2 — Band classification**:
- Low:    esc_risk < 0.35
- Medium: 0.35 ≤ esc_risk < 0.65
- High:   esc_risk ≥ 0.65

**Step 3 — Matrix lookup** (7 stages × 3 bands → action):

| Stage / Band | Low | Medium | High |
|---|---|---|---|
| RECONNAISSANCE | ALLOW | LOG | LOG |
| WEAPONIZATION | LOG | LOG | TROLL |
| DELIVERY | LOG | TROLL | TROLL |
| EXPLOITATION | TROLL | BLOCK | BLOCK |
| INSTALLATION | BLOCK | BLOCK | ALERT |
| COMMAND_AND_CTRL | BLOCK | ALERT | ALERT |
| ACTIONS_ON_OBJ | ALERT | ALERT | ALERT |

**Step 4 — Override rules** (applied after matrix lookup):
- **R1**: AttackType.NORMAL → always ALLOW
- **R2**: AttackType ∈ {DOS, WORMS} → upgrade action one severity level
- **R3**: escalation_rate > 0.80 → upgrade action one severity level

Upgrade order: ALLOW → LOG → TROLL → BLOCK → ALERT

**Step 5 — Composite risk score** (logged, does not affect action):

```
risk = 0.35 × stage_weight + 0.35 × escalation_risk
     + 0.15 × attack_severity + 0.15 × escalation_rate
```

### Intent-awareness

The SEDM uses the intent-specific TransitionModel to compute escalation risk. The same matrix applies across all intents, but the escalation risk values differ, so the SEDM naturally adapts to each attacker profile.

### API

```python
policy = MatrixPolicy(default_intent=AttackerIntent.OPPORTUNISTIC)

# From environment state vector:
action, info = policy.decide_from_state(state)   # state: np.ndarray (24,)

# From first principles:
action, info = policy.decide(
    current_stage=KillChainStage.EXPLOITATION,
    current_attack=AttackType.EXPLOITS,
    escalation_rate=0.7,
    intent=AttackerIntent.AGGRESSIVE
)

# info dict contains:
# stage, attack_type, intent, escalation_risk, escalation_band,
# base_action, override_applied, final_action, composite_risk
```

### Evaluation Results

Across 30 × 200-step evaluation episodes per intent:

| Intent | Detection Rate | False Positive Rate | Mean Reward |
|---|---|---|---|
| STEALTHY | 99.09% | 35.56% | 1012.22 |
| AGGRESSIVE | 99.47% | 6.67% | 1090.84 |
| TARGETED | 99.48% | 3.33% | 1127.10 |
| OPPORTUNISTIC | 99.41% | 15.00% | 896.05 |

---

## `dqn.py`

### `DQNNetwork`

Fully-connected feed-forward network:

```
Input(state_dim=24)
  → Linear(24, 256) → LayerNorm(256) → ReLU
  → Linear(256, 128) → LayerNorm(128) → ReLU
  → Linear(128, 64) → LayerNorm(64) → ReLU
  → Linear(64, action_dim=5)
```

### `DQNAgent`

Epsilon-greedy DQN with experience replay (15,000 capacity), target network (hard copy every 150 steps), Huber loss, Adam optimizer (lr=1e-3), gradient clipping (max_norm=10).

Training dynamics (300 episodes, OPPORTUNISTIC intent, logs/metrics.csv):
- Episode 0: detection rate 87.6%, reward 1006
- Episode 1: detection rate 97.4%, reward 2051
- Episodes 2–9: detection rate 98.0–99.2%, reward 2178–2317
- Episodes 100+: detection rate consistently >98.5%, reward 2200–2360

---

## `classifier.py`

### `AttackClassifier`

Wraps `sklearn.ensemble.RandomForestClassifier`:
- 150 estimators, max_depth=20
- `class_weight="balanced"` for uniform class treatment
- `StandardScaler` for feature normalisation
- Trained on 600 synthetic samples per class (6,000 total)

```python
clf = AttackClassifier(n_estimators=150, max_depth=20)
clf.fit_from_simulation(n_samples_per_class=600, seed=42)
attack_type = clf.predict(features_dict)
result = clf.evaluate(n_test_per_class=200, seed=999)
```

---

## `defender.py`

### `Defender`

Top-level orchestrator integrating classifier + DQN agent.

```python
defender = Defender(dqn_config={...}, classifier_config={...})
action, pred = defender.observe(state, features, training=True)
loss = defender.learn(state, action, reward, next_state, done)
defender.save("models/")
defender.load("models/")
```
