# Defender & SEDM Policy

The defender subsystem has three active layers (a fourth, DQN, is
orphaned — see below):

1. **Attack Classifier** (`classifier.py`) — RandomForest identifying
   attack types from raw network features.
2. **Stage-Escalation Decision Matrix** (`matrix_policy.py`) — the actual
   decision policy: kill-chain stage + escalation risk (+ optionally,
   reputation) → honeypot action.
3. **Defender Orchestrator** (`defender.py`) — thin wrapper combining the
   classifier and `MatrixPolicy`, plus save/load.

Threat level, honeypot action definitions, and the reward function live
in `honeypot.py`.

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

Base reward from a 5×5 action-by-band matrix:

| Action | benign | low | medium | high | critical |
|---|---|---|---|---|---|
| ALLOW | +1.0 | +0.5 | -1.0 | -3.0 | -6.0 |
| LOG | +0.2 | +1.5 | +2.0 | +1.0 | -1.0 |
| TROLL | -1.0 | +1.0 | +3.0 | +2.5 | +0.5 |
| BLOCK | -2.0 | -0.5 | +1.5 | +3.5 | +5.0 |
| ALERT | -3.0 | -1.0 | +0.5 | +2.0 | +6.0 |

Modifiers applied after the base lookup: late-stage amplifier (`×1.5` for
negative rewards at INSTALLATION/C2/ACTIONS_ON_OBJ), TROLL +
BACKDOORS/SHELLCODE/WORMS (`+0.8`), BLOCK + WORMS (`+1.0`), LOG +
RECONNAISSANCE (`+0.5`), ALLOW + not `is_attack` (`+0.5`).

This reward exists for episode-metric bookkeeping (`state[reward]`
tracking, plots) and DQN-era compatibility — `MatrixPolicy` never learns
from it, so changing this formula has zero effect on what action gets
chosen.

## `classifier.py` — `AttackClassifier`

RandomForest (150 estimators, max_depth 20, `class_weight="balanced"`,
`StandardScaler` normalization), trained on synthetic data from
`Attacker._simulate_features()`. Training-data generation draws a fresh
intensity/persona per sample (see [`attacker.md`](attacker.md)) so
"independent" samples of one class actually vary the way real traffic
would.

```python
clf = AttackClassifier(n_estimators=150, max_depth=20)
clf.fit_from_simulation(n_samples_per_class=600, seed=42)
attack_type = clf.predict(features_dict)
```

Currently shipped `models/classifier.joblib` was retrained on the
session-coherent, persona-diverse generator; holdout accuracy on a fresh
seed was 99.4%.

## `matrix_policy.py` — `MatrixPolicy` (the SEDM)

### Algorithm

**1. Escalation risk** — sum of the intent-specific Markov
`TransitionModel`'s probability of landing on any stage strictly beyond
the current one:

```
esc_risk = Σ P(next_stage = s') for all s' > current_stage
```

**2. Band classification**: Low (`esc_risk < 0.35`), Medium (`0.35 ≤
esc_risk < 0.65`), High (`esc_risk ≥ 0.65`).

**3. Matrix lookup** (7 stages × 3 bands → base action):

| Stage / Band | Low | Medium | High |
|---|---|---|---|
| RECONNAISSANCE | ALLOW | LOG | LOG |
| WEAPONIZATION | LOG | LOG | TROLL |
| DELIVERY | LOG | TROLL | TROLL |
| EXPLOITATION | TROLL | BLOCK | BLOCK |
| INSTALLATION | BLOCK | BLOCK | ALERT |
| COMMAND_AND_CTRL | BLOCK | ALERT | ALERT |
| ACTIONS_ON_OBJ | ALERT | ALERT | ALERT |

**4. Override rules**, applied **in this order** (first match wins):

| Rule | Condition | Effect | Notes |
|---|---|---|---|
| **R4** | `reputation ≥ REPUTATION_THRESHOLD` (0.60) | Upgrade one level | Checked *first* — a flagged source IP stays escalated even on a benign-looking event. See [`dynamic_response.md`](dynamic_response.md). |
| **R1** | `attack_type == NORMAL` | Force ALLOW | Always applied unless R4 already fired. |
| **R2** | `attack_type ∈ {DOS, WORMS}` | Upgrade one level | Rapid, high-impact spreading attacks. |
| **R3** | `escalation_rate > RATE_THRESHOLD` (0.80, or the current `AdaptiveThresholds.threshold` if one is attached) | Upgrade one level | Sustained high-frequency campaign. |

"Upgrade one level" = `HoneypotAction(min(int(action) + 1, ALERT))` —
ALLOW→LOG→TROLL→BLOCK→ALERT, clamped at ALERT.

**5. Composite risk score** (logged only, never affects the action):

```
risk = 0.35 × stage_weight + 0.35 × escalation_risk
     + 0.15 × attack_severity + 0.15 × escalation_rate
```

### R4's reasoning, spelled out

R4 checks *before* R1 on purpose: without it, a source IP that had already
crossed the reputation threshold would still get auto-ALLOWed the moment
it sent one benign-looking packet, resetting to full trust every time —
undermining the entire point of tracking reputation. This is a real,
deliberate behavior change from the R1-always-wins-first design that
predates R4; see [`dynamic_response.md`](dynamic_response.md) for the
`ReputationTracker` that feeds this rule and the security trade-off it
implies (shared/dynamic IPs stay penalized for the reputation half-life
even after they stop being malicious).

### Intent-awareness

The same 7×3 matrix and the same four override rules apply across all
intents; what differs per intent is `esc_risk`, because it's computed from
the intent-specific `TransitionModel`. This is why the SEDM adapts to
attacker behavior without needing four separate matrices.

### API

```python
policy = MatrixPolicy(
    default_intent=AttackerIntent.OPPORTUNISTIC,
    adaptive_thresholds=None,   # or an AdaptiveThresholds instance
)

# From environment state vector:
action, info = policy.decide_from_state(state, reputation=0.0)

# From first principles:
action, info = policy.decide(
    current_stage=KillChainStage.EXPLOITATION,
    current_attack=AttackType.EXPLOITS,
    escalation_rate=0.7,
    intent=AttackerIntent.AGGRESSIVE,
    reputation=0.0,
)

# info dict:
# stage, attack_type, intent, escalation_risk, escalation_band,
# base_action, reputation, override_applied, final_action, composite_risk
```

`reputation` and `adaptive_thresholds` both default to values that
reproduce the pre-R4/pre-adaptive behavior exactly — nothing changes
unless a caller opts in.

## `defender.py` — `Defender`

```python
defender = Defender(
    classifier_config={...},
    train_classifier=True,
    default_intent=AttackerIntent.OPPORTUNISTIC,
    dqn_config=None,   # legacy — accepted, silently ignored (see below)
)
action, predicted_attack = defender.observe(state, features, reputation=0.0)
defender.save("models/")
defender.load("models/")
```

`defender.epsilon` and `defender.steps_done` are stub properties returning
`0.0`/`0` — kept because `train.py` (a barely-touched legacy script) still
reads them for its progress display and still passes a `dqn_config` dict
into the constructor.

## `dqn.py` — orphaned

`defender/__init__.py` no longer imports it, and nothing else in the
active code path references `DQNAgent`/`DQNNetwork`/`ReplayBuffer`.
`train.py` still builds a full `dqn_config` dict and passes it to
`Defender()`, where it's accepted and silently discarded. `torch` remains
in `requirements.txt` purely because `dqn.py` still imports it. See
[`dqn_practicality.md`](dqn_practicality.md) for why this hasn't been
revived, and what its historical training results looked like.
