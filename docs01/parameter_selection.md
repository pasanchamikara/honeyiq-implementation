# Parameter Selection

Every tunable constant added or changed in this round, with its default
and rationale. For parameters unchanged from the original SEDM
implementation (thresholds, reward matrix, classifier hyperparameters),
see `docs/parameter_selection.md` and `thesis/doc0/appendix_b_hyperparams.md`
— they're still accurate.

## Synthetic traffic realism

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| `INTENSITY_LOGNORMAL_SIGMA` | 0.35 | `attack_types.py` | Session intensity multiplier spread — wide enough for real variation, narrow enough that most sessions stay within ~2x of the median |
| `INTENSITY_SCALED_FEATURES` | 8 features | `attack_types.py` | Only volume-shaped features (`sbytes`, `dbytes`, `sload`, `dload`, `spkts`, `dpkts`, `ct_srv_src`, `ct_dst_ltm`) scale with intensity — TTL/window/duration/loss aren't physically meaningful to scale this way |
| `NORMAL_PERSONA_WEIGHTS` | 70/20/10 | `attack_types.py` | `casual_user` dominant (matches real traffic mix), `crawler`/`monitoring_probe` present but minority — not equal weighting, since equal weighting would overrepresent automated traffic relative to human users |

## Escalation tracking

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| `escalation_mode` | `"window"` | `cyber_env.py`, `session_tracker.py` | Preserves exact prior behavior; `"ema"` needs its own calibration pass against `RATE_THRESHOLD` before being relied on operationally (see below) |
| `DEFAULT_EMA_ALPHA` | 0.15 | `cyber_env.py`, `session_tracker.py` | ~13-step effective memory (`1/alpha`) — comparable to the 20-step window it complements, but with smooth decay instead of a hard cutoff |

## Dynamic response — reputation (R4)

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| `REPUTATION_THRESHOLD` | 0.60 | `matrix_policy.py` | Requires substantial sustained/severe offense history before overriding R1 — not a hair-trigger; roughly 2-3 high-severity offenses (0.25 increment × severity) before firing |
| `offense_increment` | 0.25 | `reputation.py` | A single WORMS-severity (0.90) offense contributes `0.25 × 0.90 ≈ 0.23` — several repeated high-severity offenses needed to cross 0.60, not one |
| `decay_half_life_seconds` | 21,600 (6h) | `reputation.py` | Long enough that a source IP doesn't "reset" between short breaks in activity; short enough that reputation doesn't accumulate forever from stale history |
| `max_score` | 1.0 | `reputation.py` | Matches the `[0, 1]` convention used everywhere else in the system (severity, escalation_rate, esc_risk) |
| `stale_after_seconds` | 2,592,000 (30d) | `reputation.py` | Bounds memory growth from one-off/scanner IPs without needing a database |
| `sweep_interval_seconds` | 300 | `reputation.py` | Same throttled-sweep pattern as `SessionTracker` — avoids scanning the whole reputation dict on every single event |

**All five reputation constants are placeholder values with no empirical
backing from real attack data** — they encode a reasonable *shape* of
behavior (gradual escalation, meaningful but not permanent memory) but
should be calibrated against real or more realistic synthetic traffic
before being trusted operationally. This is called out explicitly rather
than presented as tuned.

## Dynamic response — `AdaptiveThresholds`

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| `target_rate` | 0.10 | `adaptive_thresholds.py` | R3 firing on ~10% of decisions is a reasonable "not every step, not silent" starting point — an operational choice, not derived from data |
| `tolerance` | 0.03 | `adaptive_thresholds.py` | Deadband width — prevents the controller from chasing statistical noise in the observed rate |
| `step` | 0.01 | `adaptive_thresholds.py` | Small relative to `RATE_THRESHOLD`'s 0-1 range — gradual correction, not overshoot-prone |
| `bound` | 0.10 | `adaptive_thresholds.py` | Caps total drift at ±0.10 from the initial threshold — the controller can meaningfully respond but can't wander arbitrarily far from the hand-chosen default |
| `observation_window` | 200 | `adaptive_thresholds.py` | Large enough to smooth over per-episode variance before nudging; small enough to respond within a reasonable number of decisions |

These five are explicitly about **alert-fatigue control, not
correctness** — see [`dynamic_response.md`](dynamic_response.md) for why
there's no honest way to tune them against a "correctness" signal in this
codebase today.

## What was deliberately left non-adaptive

| Parameter | Why it stays static |
|---|---|
| `ESC_LOW_THRESHOLD` (0.35) | Buckets `esc_risk`, which comes purely from the static Markov `TransitionModel` — zero live observational signal to adapt against |
| `ESC_HIGH_THRESHOLD` (0.65) | Same reasoning |
| The 7×3 `_SEDM` table itself | No mechanism proposed or built to adapt table cells — would need a much stronger correctness signal than anything available (see [`dqn_practicality.md`](dqn_practicality.md)) |
