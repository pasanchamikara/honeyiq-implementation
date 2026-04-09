# Environment

## Overview

`CyberSecurityEnv` is a [Gymnasium](https://gymnasium.farama.org/) environment that wraps the `Attacker` and exposes a standard RL interface to the `Defender`. It handles:

- State vector construction
- Threat level and reward computation
- Episode lifecycle (reset, step, truncation)
- Escalation rate tracking via a sliding window

---

## Spaces

### Observation Space

```python
spaces.Box(low=0.0, high=0.0, shape=(24,), dtype=np.float32)
```

24-dimensional vector — see [Architecture: State Vector Layout](architecture.md#state-vector-layout-24-dimensions).

### Action Space

```python
spaces.Discrete(5)
```

Values 0–4 correspond to `HoneypotAction` (ALLOW, LOG, TROLL, BLOCK, ALERT).

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `attacker_intent` | `AttackerIntent` | `OPPORTUNISTIC` | Intent profile for the attacker in this environment instance |
| `max_steps` | `int` | 500 | Maximum steps before truncation |
| `escalation_window` | `int` | 20 | Sliding window size for `escalation_rate` |
| `seed` | `int \| None` | None | Random seed passed to the attacker |
| `render_mode` | `str \| None` | None | `"human"` prints step summaries; `"ansi"` returns string |

---

## `reset(seed=None, options=None) → (state, info)`

Resets the attacker to the beginning of the kill chain (NORMAL attack, RECONNAISSANCE stage). Returns the initial state and an info dict with zero-valued fields.

Initial info dict:
```python
{
    "step": 0,
    "attack_type": AttackType.NORMAL,
    "kill_chain_stage": KillChainStage.RECONNAISSANCE,
    "threat_level": 0.0,
    "is_attack": False,
    "features": {},
    "escalation_rate": 0.0,
    "attack_count": 0,
}
```

---

## `step(action) → (next_state, reward, terminated, truncated, info)`

Per-step logic:

1. **Attacker advances**: `attacker.step()` samples the next attack type, kill chain stage, and network features
2. **Escalation rate**: updated in a sliding window of size 20: `sum(recent_is_attack) / window_size`
3. **Threat level**: computed from attack type, stage, escalation rate, and cumulative attack count
4. **Reward**: computed from the defender's action against the threat
5. **State**: rebuilt as a 24-dim one-hot + continuous vector
6. **Termination**: `terminated` is always `False` (no early termination); `truncated` is `True` when `step_count >= max_steps`

### Info Dict Fields

| Key | Type | Description |
|---|---|---|
| `step` | int | Current step number |
| `attack_type` | AttackType | Current attacker's attack type |
| `kill_chain_stage` | KillChainStage | Current kill chain stage |
| `threat_level` | float | Composite threat level [0, 1] |
| `is_attack` | bool | True iff attack_type ≠ NORMAL |
| `features` | dict | 15 network flow features |
| `escalation_rate` | float | Recent attack frequency [0, 1] |
| `attack_count` | int | Cumulative non-NORMAL attacks this episode |
| `action_name` | str | Name of the action taken |
| `next_probs` | np.ndarray | Attacker's next-step attack type probabilities (10,) |
| `stage_probs` | np.ndarray | Attacker's next-step kill chain stage probabilities (7,) |

---

## `_build_state(...)` — State Construction

```python
state = np.zeros(24, dtype=np.float32)
state[int(attack_type)]           = 1.0   # indices 0–9
state[10 + int(kill_chain_stage)] = 1.0   # indices 10–16
state[17] = threat_level
state[18] = min(1.0, attack_count / 100.0)
state[19] = escalation_rate
state[20 + int(intent)]           = 1.0   # indices 20–23
```

---

## Rendering

`render_mode="human"` prints to stdout:

```
============================================================
Step    5 | Attack: RECONNAISSANCE    Stage: WEAPONIZATION
         Threat: 0.183  Escalation: 0.200  Action: LOG
============================================================
```

`render_mode="ansi"` returns the formatted string instead of printing.

---

## Episode Lifecycle Example

```python
env = CyberSecurityEnv(
    attacker_intent=AttackerIntent.STEALTHY,
    max_steps=200,
    seed=42,
)

state, info = env.reset()

for step in range(200):
    action = defender.observe(state, info["features"], training=False)[0]
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Design Notes

### Why is `terminated` always False?

The environment does not have a natural terminal condition — there is no "attacker defeated" state. Episodes always end by truncation at `max_steps`. This is consistent with the Gymnasium convention where `truncated=True` signals a time limit, not task completion.

### Why does the defender's action not affect the attacker's trajectory?

The attacker is an autonomous Markov chain independent of the defender's responses. This models a realistic scenario where the defender's honeypot actions slow/deter but do not completely stop an attacker. The reward function captures whether the response was appropriate, but the attacker always gets to take their next step regardless.

### Escalation Rate

The 20-step sliding window gives a short-term frequency signal distinct from the cumulative `attack_count`. An attacker who has been quiet and then suddenly floods with attacks will have a high escalation rate (up to 1.0) even if their total attack count is still moderate.
