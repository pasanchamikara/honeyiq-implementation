# Environment

`environment/cyber_env.py` wraps `Attacker` in a `gymnasium.Env`, giving
one episode = one attacker session. It's the environment used by
`main.py`, `evaluate.py`, and `evaluation/sedm_eval.py`; the live pipeline
does not use it directly, but reuses its `encode_state()` function (see
below) so both sides of the system build identical state vectors.

## State vector (24 dims) — `encode_state()`

```python
def encode_state(
    attack_type, kill_chain_stage, threat_level,
    attack_count, escalation_rate, intent,
) -> np.ndarray:   # shape (24,)
```

| Indices | Content |
|---|---|
| `[0:10]` | `attack_type` one-hot (10 classes) |
| `[10:17]` | `kill_chain_stage` one-hot (7 stages) |
| `[17]` | `threat_level` — float `[0, 1]` |
| `[18]` | `attack_count` normalized by `/100`, clamped to 1.0 |
| `[19]` | `escalation_rate` — float `[0, 1]` (window- or EMA-based, see below) |
| `[20:24]` | `attacker_intent` one-hot (4 intents) |

This function is shared between `CyberSecurityEnv` and
`opencanary_integration/engine/state_builder.py` — there is exactly one
place the 24-dim layout is defined.

## Episode lifecycle

```python
env = CyberSecurityEnv(
    attacker_intent=AttackerIntent.OPPORTUNISTIC,
    max_steps=500,
    escalation_window=20,
    escalation_mode="window",       # or "ema"
    escalation_ema_alpha=0.15,
    benign_ratio=0.0,
    seed=None,
)
state, info = env.reset(seed=...)

for _ in range(max_steps):
    next_state, reward, terminated, truncated, info = env.step(action)
```

`step()`:
1. Advances the underlying `Attacker` by one step.
2. With probability `benign_ratio`, overrides the step with `NORMAL`-class
   traffic while preserving the attacker's actual kill-chain position in
   the state vector (simulates legitimate users occasionally contacting
   the honeypot alongside a real campaign).
3. Updates both escalation signals (see below).
4. Computes `threat_level` (`defender.honeypot.compute_threat_level`) and
   `reward` (`defender.honeypot.compute_reward`) — `reward` is returned
   for episode-metric bookkeeping; `MatrixPolicy` never trains on it, so
   it has no effect on the chosen action.
5. Builds `next_state` via `encode_state()`.

## Escalation tracking: window vs. EMA

Two signals are always computed on every step, regardless of which one
feeds `state[19]`:

```python
info["escalation_window_rate"]   # hard sliding window (always present)
info["escalation_ema"]           # severity-weighted EMA (always present)
info["escalation_rate"]          # whichever one escalation_mode selected
```

**`"window"`** (default — identical to the original implementation):
fraction of the last `escalation_window` steps (default 20) that were
*any* attack:

```python
window_rate = sum(recent_attacks) / len(recent_attacks)   # recent_attacks: deque of 0/1
```

Binary per step (an OPPORTUNISTIC-vs-WORMS step counts the same as a
RECONNAISSANCE step) and has a hard cutoff — a step that falls out of the
window stops influencing the rate immediately, rather than fading out.

**`"ema"`** (opt-in): a severity-weighted exponential moving average,
using the existing `ATTACK_SEVERITY` weights (`attacker/attack_types.py`)
instead of a plain 0/1:

```python
severity = ATTACK_SEVERITY[int(attack_type)]        # 0.0 (NORMAL) .. 0.90 (WORMS)
ema = alpha * severity + (1 - alpha) * ema           # alpha = escalation_ema_alpha, default 0.15
```

O(1) per step (no deque bookkeeping), decays smoothly instead of a hard
window cutoff, and reflects *how bad* recent attacks were, not just how
many occurred. Because it's severity-weighted, its ceiling is the max
severity actually occurring (0.90 for sustained WORMS activity) rather
than 1.0 — see the calibration note below.

Both modes are implemented identically in
`opencanary_integration/engine/session_tracker.py` for the live pipeline
(as `SessionState.escalation_rate` / `SessionState.escalation_ema`); see
[`opencanary_integration.md`](opencanary_integration.md).

### Calibration note

`RATE_THRESHOLD = 0.80` in `defender/matrix_policy.py` (the R3 override
trigger) is calibrated against **window-fraction semantics** — "80% of
the last 20 steps were attacks." Under `"ema"` mode the signal behaves
differently near that threshold (it's severity-bounded, not
frequency-bounded), so `"ema"` mode should be spot-checked via
`evaluation/sedm_eval.py` before being relied on anywhere R3's trigger
behavior matters operationally. `"window"` remains the default everywhere
for exactly this reason.

## Reward and threat level

Both live in `defender/honeypot.py` — see [`defender.md`](defender.md)
for the full formulas. Briefly: `threat_level` is a weighted blend of
attack severity, kill-chain stage, escalation rate, and cumulative attack
count; `reward` looks up a 5×5 action-by-threat-band matrix with a few
situational bonuses. Neither one drives `MatrixPolicy`'s decisions —
`threat_level` is informational (surfaced in `state[17]` and used for
band/plot displays), and `reward` exists purely for episode-metric
bookkeeping and compatibility with the (now orphaned) DQN-era training
loop.
