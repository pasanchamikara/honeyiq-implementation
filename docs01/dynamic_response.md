# Dynamic Response

Before this round of work, every threshold and the SEDM table itself were
static constants — nothing in HoneyIQ remembered a source IP across
sessions, and nothing adapted to observed traffic. This document covers
the two mechanisms added to change that, and — just as importantly — what
they deliberately do **not** claim to do.

Both mechanisms are non-learning: a human can hand-verify every line of
arithmetic involved. See [`dqn_practicality.md`](dqn_practicality.md) for
why a learning-based approach was rejected for this instead.

## Cross-session reputation (`opencanary_integration/engine/reputation.py`)

### The problem it solves

`SessionState`'s window/EMA escalation signals are scoped to *one*
session and expire with it (`SessionTracker`'s TTL sweep, default 300s).
A source IP that attacks, goes quiet past the TTL, and comes back is
treated as brand new — there's no memory of it having been malicious
before.

### `ReputationTracker`

```python
class ReputationTracker:
    def __init__(
        self,
        decay_half_life_seconds: float = 6 * 3600,
        offense_increment:       float = 0.25,
        max_score:                float = 1.0,
        stale_after_seconds:      float = 30 * 24 * 3600,
        sweep_interval_seconds:   int   = 300,
    ) -> None: ...

    def record_offense(self, src_ip: str, severity: float) -> float: ...
    def get(self, src_ip: str) -> float: ...
    def reset(self, src_ip: str) -> None: ...
```

- **In-memory, per-process, no external dependency** — a plain dict keyed
  by source IP, same architectural style as `SessionTracker`.
- **Half-life decay, applied lazily on read**:
  `score * 0.5 ** (elapsed_seconds / half_life)`. At the default 6-hour
  half-life, a score of 0.5 decays to ~0.25 after 6 hours of silence from
  that IP, ~0.125 after 12 hours, and so on — a formula anyone can check
  with a calculator, not a learned function.
- **`record_offense(src_ip, severity)`** decays the existing entry, adds
  `offense_increment * severity` (severity is `ATTACK_SEVERITY[attack_type]`,
  so `NORMAL` traffic contributes 0 to the increment but still triggers
  the lazy decay), clamps to `max_score`, and returns the resulting score.
- **A throttled staleness sweep** (same pattern as
  `SessionTracker._expire_old_sessions`) drops entries untouched for 30
  days, bounding memory growth without needing a background thread.

`SessionTracker` owns one `ReputationTracker` (public attribute
`session_tracker.reputation`, so an operator/inspection tool can query or
manually pardon an IP) and calls `record_offense()` on every
`SessionTracker.update()`, storing the result on `SessionState.reputation`.

### R4 — the override rule it feeds

See [`defender.md`](defender.md) for the full override-rule table. The
short version: `MatrixPolicy._apply_overrides` checks
`reputation >= REPUTATION_THRESHOLD` (default 0.60) **before** R1, and if
it fires, upgrades the action one level via the same `_upgrade()` helper
R2/R3 use — it does not jump to a fixed severity, it escalates *from*
whatever the matrix/other rules would have chosen.

### Threading `reputation` through the call stack

`reputation` is an explicit, out-of-band parameter — **not** part of the
24-dim state vector. `STATE_DIM = 24` is load-bearing for `encode_state`,
classifier training, and the RL environment's observation space, and
reputation has no meaning inside `CyberSecurityEnv` anyway (no persistent
IP identity exists across synthetic episodes there). So it flows:

```
SessionTracker.update() → session.reputation
    → EmulatorScenario.run_event(): self.policy.decide(state, reputation=session.reputation)
    → PolicyEngine.decide(state, features, reputation)
    → MatrixPolicy.decide_from_state(state, reputation) → decide(..., reputation)
```

`Defender.observe()` also gained a `reputation: float = 0.0` parameter for
symmetry, but nothing in `evaluate.py`/`sedm_eval.py`/`main.py` has
cross-episode IP identity, so it's always `0.0` there — R4 simply never
fires in the synthetic evaluation harness unless a caller explicitly wires
something in.

### Trade-off, stated plainly

R4 firing before R1 means a shared or dynamic IP that was briefly
malicious stays penalized for the full decay half-life even after it
genuinely stops being malicious (e.g. it's reassigned to a different, innocent
user via DHCP or CGNAT). This is the standard fail2ban-style trade-off —
security posture over avoiding all possible false positives on IP churn —
and was made explicitly rather than left as an accident of implementation
order.

## Adaptive `RATE_THRESHOLD` (`defender/adaptive_thresholds.py`)

### What it is not

`ESC_LOW_THRESHOLD`/`ESC_HIGH_THRESHOLD` (the SEDM band cuts) are
**deliberately excluded** from any adaptive mechanism. They bucket
`esc_risk`, which comes purely from the static, hand-authored Markov
`TransitionModel` — there is no live observational signal in this
codebase whose drift would indicate those two cuts are miscalibrated.
Building something that looks like learning against a fixed, enumerable,
non-observational input would be theater, not a real improvement, so it
wasn't built.

`RATE_THRESHOLD` (the R3 trigger) is different — it gates on
`escalation_rate`, a genuine function of observed traffic. But the *only*
feedback actually available in this codebase is **R3's own firing
frequency**, not whether any individual trigger was the right call (there
is no ground-truth label pipeline — see
[`dqn_practicality.md`](dqn_practicality.md)). So `AdaptiveThresholds` is
framed honestly: it is **not** a correctness improvement. It's an
alert-fatigue safety valve — keeping R3's trigger rate near an
operator-chosen target so a BLOCK/ALERT queue doesn't lose signal value by
firing on nearly every step, or go quiet and stop being useful.

### `AdaptiveThresholds`

```python
@dataclass
class AdaptiveThresholds:
    initial_threshold: float = 0.80
    target_rate:        float = 0.10
    tolerance:           float = 0.03
    step:                 float = 0.01
    bound:                float = 0.10    # max drift from initial_threshold
    observation_window:  int   = 200

    def record(self, r3_condition: bool) -> None: ...
    @property
    def threshold(self) -> float: ...
```

A plain **deadband controller**: once `observation_window` decisions have
been recorded, if R3's observed trigger rate exceeds `target_rate +
tolerance`, nudge the threshold up by `step` (fire less often); if it's
below `target_rate - tolerance`, nudge it down by `step` (fire more
often); otherwise leave it alone. Always clamped to
`[initial_threshold - bound, initial_threshold + bound]`.

### Wiring

```python
policy = MatrixPolicy(
    default_intent=AttackerIntent.AGGRESSIVE,
    adaptive_thresholds=AdaptiveThresholds(initial_threshold=RATE_THRESHOLD),
)
```

`default None` = static `RATE_THRESHOLD`, byte-for-byte today's prior
behavior. Every existing construction site
(`evaluate.py`/`sedm_eval.py`/`main.py`/`defender.py`/`policy_engine.py`/
`scenario.py`) keeps constructing `MatrixPolicy(default_intent=...)` with
no `adaptive_thresholds` argument.

`_apply_overrides` evaluates `r3_condition = escalation_rate >
rate_threshold` **before** the R1/R2/R4 short-circuits, and calls
`adaptive.record(r3_condition)` unconditionally when an `AdaptiveThresholds`
is attached — this is so the controller observes R3's *counterfactual*
trigger rate on every decision, not a censored subset where R1/R2/R4
already preempted it. Observing only the un-preempted cases would bias the
rate estimate low.

### Verified behavior

Constructing `AdaptiveThresholds(initial_threshold=RATE_THRESHOLD,
observation_window=50)` and driving 60 high-escalation-rate decisions
through a `MatrixPolicy` with it attached moved the threshold up from
0.80 toward its bound (0.90) — confirming the controller responds to a
sustained high trigger rate and stays within its configured bound.
