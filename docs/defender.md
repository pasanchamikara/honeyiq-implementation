# Defender Module

## Overview

The defender subsystem has three layers:

1. **Attack Classifier** (`classifier.py`) — Random Forest identifying attack types from raw network features
2. **DQN Agent** (`dqn.py`) — Deep Q-Network selecting honeypot actions from the state vector
3. **Defender Orchestrator** (`defender.py`) — combines both; handles inference, learning, and persistence

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

- **LayerNorm**: normalises activations per-sample (not per-batch), stable with small batch sizes
- **Kaiming uniform** weight init preserves variance under ReLU
- **No output activation**: Q-values are unbounded

### `ReplayBuffer`

Circular deque of `Transition` namedtuples `(state, action, reward, next_state, done)`.

- Capacity: 15,000 transitions
- Sampling: uniform random without replacement
- Returns batches as stacked `torch.Tensor` objects on the correct device

### `DQNAgent`

Main agent class. Owns:
- `policy_net` — trained online
- `target_net` — frozen copy updated every 150 steps
- `replay_buffer`
- `optimizer` (Adam)
- `loss_fn` (SmoothL1 / Huber)

#### `select_action(state, training=True) → int`

```python
if training and random() < epsilon:
    return random_action()            # explore
return argmax(policy_net(state))     # exploit
```

#### `update() → float | None`

1. Return `None` if buffer has fewer than 64 transitions
2. Sample mini-batch of 64
3. Compute current Q-values: `Q(s, a; θ) = policy_net(s).gather(action)`
4. Compute TD targets: `y = r + γ · max_{a'} Q(s', a'; θ⁻) · (1 - done)`
5. Loss: `SmoothL1(Q(s,a), y)`
6. Backprop with gradient clip (max norm 10.0)
7. Decay epsilon: `ε ← max(ε_min, ε · 0.997)`
8. Every 150 steps: `target_net ← policy_net` (hard copy)

#### Persistence

```python
agent.save("models/dqn_agent.pt")   # saves state dicts + metadata
agent.load("models/dqn_agent.pt")
```

Checkpoint keys: `policy_net`, `optimizer`, `epsilon`, `steps_done`, `state_dim`, `action_dim`, `gamma`, `epsilon_end`, `epsilon_decay`.

---

## `classifier.py`

### `AttackClassifier`

Wraps `sklearn.ensemble.RandomForestClassifier` with:
- `StandardScaler` for feature normalisation
- Synthetic data generation via `Attacker._simulate_features()`
- `class_weight="balanced"` for uniform class treatment

#### Training

```python
clf = AttackClassifier(n_estimators=150, max_depth=20, n_jobs=1)
clf.fit_from_simulation(n_samples_per_class=600, seed=42)
# or manually:
X, y = clf.generate_training_data(600, seed=42)
clf.fit(X, y)
```

`generate_training_data` creates a balanced dataset: 600 samples × 10 classes = 6,000 rows.

#### Inference

```python
attack_type  = clf.predict(features_dict)           # → AttackType
proba        = clf.predict_proba(features_dict)     # → np.ndarray (10,)
batch_labels = clf.predict_batch(X_df)              # → np.ndarray (N,)
```

`predict_proba` ensures all 10 classes are present in the output even if some were absent from the training split.

#### Evaluation

```python
result = clf.evaluate(n_test_per_class=200, seed=999)
# result["accuracy"]  → float
# result["report"]    → sklearn classification report dict
```

#### Persistence

```python
clf.save("models/classifier.joblib")    # joblib dump of model + scaler + feature_names
clf.load("models/classifier.joblib")
```

---

## `defender.py`

### `Defender`

Top-level orchestrator holding one `DQNAgent` and one `AttackClassifier`.

#### Initialisation

```python
defender = Defender(
    dqn_config={
        "state_dim": 24, "action_dim": 5,
        "lr": 1e-3, "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.997,
        "batch_size": 64, "target_update_freq": 150, "buffer_capacity": 15_000,
    },
    classifier_config={"n_estimators": 150, "max_depth": 20, "n_jobs": 1},
    train_classifier=True,
    seed=42,
)
defender.initialize_classifier(n_samples_per_class=600)
```

#### `observe(state, features, training=True) → (action, predicted_attack)`

1. `predicted_attack = classifier.predict(features)` (or NORMAL if unfitted)
2. `action = dqn.select_action(state, training)`
3. Returns both — the action is executed in the env; the prediction is for logging

#### `learn(state, action, reward, next_state, done) → float | None`

1. `replay_buffer.push(...)` via `dqn.store_transition`
2. `dqn.update()` — returns loss or None

#### `save(model_dir) / load(model_dir)`

Saves/loads both components independently:
- `{model_dir}/dqn_agent.pt`
- `{model_dir}/classifier.joblib`

#### Introspection

```python
defender.epsilon          # current exploration rate
defender.steps_done       # total gradient steps
defender.q_values(state)  # raw Q-values as np.ndarray (5,)
```
