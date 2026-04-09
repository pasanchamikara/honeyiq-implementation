# Attacker Module

## Overview

The attacker subsystem simulates a threat actor progressing through the Lockheed Martin Cyber Kill Chain. It provides:

1. **Attack type and kill chain stage enumerations** with associated severity weights
2. **Per-attack network feature distributions** based on UNSW-NB15
3. **Intent-specific Markov transition matrices** for realistic sequencing
4. **A stateful agent** (`Attacker`) that advances step-by-step

---

## `attack_types.py`

### Enumerations

#### `AttackType` (IntEnum)

Ten attack categories derived from UNSW-NB15:

| Value | Name | Severity | Description |
|---|---|---|---|
| 0 | NORMAL | 0.00 | Legitimate traffic — no attack |
| 1 | RECONNAISSANCE | 0.20 | Port scanning, host discovery, banner grabbing |
| 2 | ANALYSIS | 0.25 | Protocol analysis, vulnerability scanning |
| 3 | FUZZERS | 0.35 | Random/malformed input to find crashes |
| 4 | EXPLOITS | 0.70 | Exploiting known vulnerabilities |
| 5 | BACKDOORS | 0.80 | Persistent covert access mechanisms |
| 6 | SHELLCODE | 0.75 | Injected executable payload |
| 7 | GENERIC | 0.40 | Generic attacks not fitting other categories |
| 8 | DOS | 0.85 | Denial-of-service flooding |
| 9 | WORMS | 0.90 | Self-propagating malware |

#### `KillChainStage` (IntEnum)

Seven Cyber Kill Chain phases:

| Value | Name | Weight | Primary attacks |
|---|---|---|---|
| 0 | RECONNAISSANCE | 0.10 | NORMAL, RECONNAISSANCE |
| 1 | WEAPONIZATION | 0.20 | ANALYSIS |
| 2 | DELIVERY | 0.35 | FUZZERS, GENERIC |
| 3 | EXPLOITATION | 0.55 | EXPLOITS, SHELLCODE |
| 4 | INSTALLATION | 0.70 | BACKDOORS |
| 5 | COMMAND_AND_CTRL | 0.85 | WORMS |
| 6 | ACTIONS_ON_OBJ | 1.00 | DOS |

#### `AttackerIntent` (IntEnum)

Four intent profiles that modulate the Markov chain:

| Value | Name | Behaviour |
|---|---|---|
| 0 | STEALTHY | Low-and-slow; prefers recon and backdoors; avoids noisy attacks |
| 1 | AGGRESSIVE | Fast escalation; prefers DoS, worms, exploits; moves quickly through stages |
| 2 | TARGETED | Focused exploit chain; exploits → shellcode → backdoors → lateral movement |
| 3 | OPPORTUNISTIC | Scattered; prefers generic/fuzzers; moderate noise across all attack types |

### Feature Distributions

`FEATURE_DISTRIBUTIONS` maps each `AttackType` to a dict of 15 feature specs. Each spec is a tuple `(distribution_name, *params)`:

| Distribution | Parameters | Usage |
|---|---|---|
| `"uniform"` | (low, high) | `dur`, `sload`, `dload` |
| `"lognormal"` | (mean_log, sigma_log) | `sbytes`, `dbytes` |
| `"poisson"` | (lambda) | `spkts`, `dpkts`, `sloss`, `dloss`, `ct_*` |
| `"choice"` | ([values]) | `sttl`, `dttl`, `swin`, `dwin` |
| `"constant"` | (value) | Rarely used |

Notable per-attack characteristics:

- **Reconnaissance**: Short `dur` (0.0001–0.5s), high `ct_dst_ltm` (λ=30) — scanning many targets
- **DoS**: Extremely high `sload` (50k–500k bps), near-zero `dur`, `spkts` λ=500
- **Backdoors**: Long `dur` (10–3600s), low `sload` (10–500 bps) — persistent and stealthy
- **Worms**: High `ct_dst_ltm` (λ=40) — rapidly spreading to new hosts

---

## `transition_model.py`

### Base Matrices

#### Attack Transition Matrix (10×10)

The base matrix encodes realistic attack progressions. Key patterns:

- From **NORMAL**: high probability of staying normal (0.60) or starting reconnaissance (0.28)
- From **RECONNAISSANCE**: likely to proceed to ANALYSIS (0.30) or FUZZERS (0.18)
- From **EXPLOITS**: likely to escalate to BACKDOORS (0.30) or SHELLCODE (0.25)
- From **DOS**: high self-loop (0.62) — floods tend to continue
- From **WORMS**: high probability of DOS (0.35) or persistence via BACKDOORS (0.18)

#### Stage Transition Matrix (7×7)

Mostly forward-progressing with small regression probability (models defender push-back):

- From **RECONNAISSANCE**: 60% → WEAPONIZATION, 30% self-loop
- From **EXPLOITATION**: 60% → INSTALLATION
- From **ACTIONS_ON_OBJ**: 70% self-loop — late-stage attackers entrench

### Intent Modifiers

Modifiers are applied element-wise to the base matrices, then rows are re-normalised:

```python
result = clip(base * modifier, 0, ∞)
result /= result.sum(axis=1, keepdims=True)
```

**STEALTHY**:
- Attack: ×2.0 on RECON/ANALYSIS/BACKDOORS columns; ×0.15 on DOS/WORMS/FUZZERS; ×1.5 on diagonal
- Stage: ×2.0 on diagonal; ×0.2 on skip-one-forward transitions

**AGGRESSIVE**:
- Attack: ×2.5 on DOS/WORMS/EXPLOITS/SHELLCODE; ×0.2 on RECON/ANALYSIS/NORMAL; ×0.4 on diagonal
- Stage: ×0.3 on diagonal; ×2.0 on one-step forward; ×1.5 on two-step forward

**TARGETED**:
- Attack: ×2.0 on EXPLOITS/SHELLCODE/BACKDOORS; ×3.0 on BACKDOORS→WORMS; ×0.15 on GENERIC/FUZZERS/DOS/NORMAL
- Stage: ×2.0 on EXPLOITATION/INSTALLATION/C2 self-loops; ×3.0 on RECON→WEAPONIZATION

**OPPORTUNISTIC**:
- Attack: ×2.0 on GENERIC/FUZZERS; ×1.3 on all non-NORMAL; ×0.3 on NORMAL
- Stage: ×1.3 on one-step forward; ×1.5 on one-step back

### `TransitionModel` API

```python
tm = TransitionModel(intent=AttackerIntent.STEALTHY, seed=42)

# Sample next state
next_attack = tm.next_attack(current_attack)      # → AttackType
next_stage  = tm.next_stage(current_stage)        # → KillChainStage

# Inspect matrices
atk_matrix   = tm.get_attack_matrix()             # np.ndarray (10, 10)
stage_matrix = tm.get_stage_matrix()              # np.ndarray (7, 7)
probs = tm.get_attack_probabilities(current_attack)  # np.ndarray (10,)
```

---

## `attacker.py`

### `Attacker`

Stateful agent wrapping the `TransitionModel` and feature simulation.

```python
attacker = Attacker(intent=AttackerIntent.OPPORTUNISTIC, seed=42)
attacker.reset()

info = attacker.step()
```

**`step()` return value:**

```python
{
    "attack_type":        AttackType,
    "kill_chain_stage":   KillChainStage,
    "intent":             AttackerIntent,
    "attack_count":       int,        # cumulative non-NORMAL attacks
    "step_count":         int,
    "features":           dict,       # 15 network features
    "is_attack":          bool,       # False iff attack_type == NORMAL
    "next_probabilities": np.ndarray, # shape (10,) — next attack probs
    "stage_probabilities":np.ndarray, # shape (7,) — next stage probs
}
```

**Stage constraint logic:**

```python
primary = ATTACK_PRIMARY_STAGE[self.current_attack]
sampled  = transition_model.next_stage(self.current_stage)
self.current_stage = max(sampled, primary - 1)   # prevent unrealistic regression
```

**`_simulate_features(attack_type)`** samples each of the 15 features independently from its distribution, clips negatives to 0.0, and returns a `dict[str, float]`.
