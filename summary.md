# HoneyIQ: Synthetic Traffic, Escalation Tracking, and Dynamic Response — Summary & Findings

This document summarizes a round of practical improvements made to the SEDM
implementation, covering three implementation areas plus an architectural
question about whether a learning-based approach (DQN or otherwise) would be
practical for dynamic behavior adjustment.

## Motivation

The existing SEDM implementation had three concrete practical gaps:

1. **Synthetic traffic was too uniform.** Every feature was sampled
   independently every step from static per-attack-type distributions — no
   session ever "felt like" one coherent attacker, and all benign traffic
   looked identical regardless of context.
2. **Escalation tracking was a blunt, binary sliding window.** `escalation_rate`
   was "fraction of the last 20 steps that were *any* attack" — a hard
   window cutoff that discarded how severe each event actually was.
3. **There was no dynamic behavior adjustment at all.** All thresholds and
   the SEDM table itself were static constants; nothing remembered a source
   IP across sessions, and nothing adapted to observed traffic.

A fourth question was whether reintroducing a DQN (or other learning-based
approach) would be a practical way to add that dynamic adjustment.

## What was implemented

### A. Synthetic traffic generation

- **Per-session intensity.** `Attacker` now draws a persistent lognormal
  intensity scalar (and, for benign sessions, a persona) once per episode,
  applied consistently to volume-shaped features (`sbytes`, `dbytes`,
  `sload`, `dload`, `spkts`, `dpkts`, `ct_srv_src`, `ct_dst_ltm`) for the
  whole episode — so a session reads as one coherent attacker machine
  instead of independent per-step noise. TTL/window-size/duration/loss
  features are left untouched (scaling those by "intensity" isn't
  physically meaningful).
- **Benign traffic personas.** Three personas (`casual_user`, `crawler`,
  `monitoring_probe`) replace the single fixed `NORMAL` distribution, so
  benign traffic has real variety — important for realistic false-positive
  evaluation.
- **Randomized event payloads.** `OpenCanaryEventGenerator`'s handful of
  literal templates (e.g. `ssh_brute`'s 3 hardcoded username/password
  pairs) now resolve usernames, passwords, paths, user-agents, and numeric
  counts from wordlists/jitter ranges instead of an enumerable fixed list.
- **Training-data consistency fix.** `AttackClassifier.generate_training_data()`
  and `main.py`'s feature-distribution plotter were updated to draw a
  fresh intensity/persona per training sample — otherwise every "independent"
  sample of a class would have silently shared one session profile and
  understated real feature variance.
- **Classifier retrained** on the new generator (`models/classifier.joblib`,
  both copies). Holdout accuracy on a fresh seed: **99.4%**.

### B. Escalation tracking

- Added a **severity-weighted exponential moving average (EMA)** alongside
  the existing sliding window, in both `environment/cyber_env.py` (training
  env) and `opencanary_integration/engine/session_tracker.py` (live
  pipeline). The EMA uses the existing `ATTACK_SEVERITY` weights instead of
  a plain 0/1 boolean, so it reflects *how bad* recent attacks were, not
  just how many occurred, and decays smoothly instead of a hard window
  cutoff.
- **Additive, not a replacement.** The existing window-based `escalation_rate`
  is untouched and stays the default (`escalation_mode="window"`); `"ema"`
  is opt-in. Both signals are always computed and exposed in `info`
  (`escalation_window_rate`, `escalation_ema`), regardless of which one
  feeds the state vector.
- **Calibration caveat, documented in-code:** `RATE_THRESHOLD = 0.80` is
  calibrated against window-fraction semantics. Under `"ema"` mode the
  signal is bounded by the max severity actually occurring (WORMS = 0.90),
  so it behaves differently near the threshold — `"ema"` should be
  spot-checked via `evaluation/sedm_eval.py` before being relied on
  anywhere R3's trigger behavior matters.

### C. Dynamic behavior adjustment

- **Repeat-offender reputation (new R4 override).** A new
  `ReputationTracker` (`opencanary_integration/engine/reputation.py`)
  tracks a time-decayed offense score per source IP that persists *across*
  sessions (unlike `SessionState`, which expires with its TTL) — a
  half-life decay formula, in-memory, no external dependency, analogous to
  fail2ban's escalating memory of a host. This feeds a new **R4** override
  rule in `MatrixPolicy`, checked **before R1**: a source IP that crosses
  the reputation threshold stays escalated even on a benign-looking event.
- **Adaptive `RATE_THRESHOLD` — narrowly scoped, honestly framed.** A new
  `AdaptiveThresholds` controller (`defender/adaptive_thresholds.py`) nudges
  `RATE_THRESHOLD` within a bounded range to keep R3's trigger rate near an
  operator-chosen target. This is explicitly **not** framed as a
  correctness improvement — the only signal available is R3's own firing
  frequency, not whether any given trigger was right — so it's scoped as an
  alert-fatigue safety valve instead. `ESC_LOW_THRESHOLD`/`ESC_HIGH_THRESHOLD`
  were deliberately **excluded** from any adaptive mechanism: they bucket
  `esc_risk`, which comes purely from the static, hand-authored Markov
  `TransitionModel` with zero live observational input in this codebase —
  there is no honest signal to adapt them against.
- **Everything defaults off.** `adaptive_thresholds=None` and
  `reputation=0.0` are the defaults everywhere; no existing construction
  site or call site changes behavior unless explicitly opted in.

### D. Is a DQN (or other learning-based) approach practical?

**No — not for this ask, grounded in this project's own history, not a
generic RL-vs-rules argument:**

- DQN was already tried in this exact codebase and removed. It converged to
  a degenerate "always escalate, never ALLOW" policy: detection rate jumped
  87.6% → 97.4% in a single episode while false-positive rate stayed
  pinned near 1.0 — a trivial shortcut, not real discrimination, because
  training episodes were almost entirely attack traffic (severe class
  imbalance) and training only covered one attacker intent.
- The evaluation harness itself had **two independent, hard-to-catch
  label/timestep-alignment bugs** (`docs/BUGS_AND_FIXES.md`, Bug 8 and Bug
  9) that silently inflated measured FPR by up to 10x before being caught —
  in code that only had to compute descriptive statistics *after the fact*.
  A reward signal for an RL agent is computed by structurally similar code,
  on every step, and directly shapes what gets learned rather than just
  mis-reporting a number afterward.
- This project's own thesis discussion chapter already reached the same
  conclusion (citing Rudin 2019): prefer an interpretable model until a
  black-box model demonstrates clear, validated superiority *and* can be
  explained post-hoc.
- **The reputation override and the narrowly-scoped adaptive threshold are
  the practical near-term path instead** — both are a few lines of
  auditable arithmetic, not a training loop or a reward function that can
  silently teach the wrong lesson.
- If a learning component is ever wanted, the only thing worth considering
  is a **small contextual bandit — not a Q-network** — over a tiny action
  space (e.g. 3–5 preset `RATE_THRESHOLD` values), driven by a genuine
  reward signal: an analyst's own post-hoc confirmed-true/false-positive
  label on an audit-log entry. That labeling ingestion path doesn't exist
  today; it would need to be built first rather than manufacturing a proxy
  signal in the meantime.
- Low-risk hygiene noted but out of scope: `defender/dqn.py` is fully
  orphaned dead code, `train.py` still passes a `dqn_config` dict into
  `Defender()` that's silently ignored, and `torch` in `requirements.txt`
  is unused weight.

## Verification

All changes were verified with the dependencies actually installed (numpy,
pandas, scikit-learn, matplotlib, seaborn, joblib, gymnasium — installed
locally for this work since they weren't previously available in this
environment):

- **Syntax check** on all 14 touched/new files — all pass.
- **Functional smoke tests**: session profile stability within an episode
  and variation across resets; event-generator payload diversity (7 distinct
  usernames observed across 20 `ssh_brute` generations); EMA vs. window mode
  parity checks; `SessionTracker` EMA/reputation accumulation; reputation
  half-life decay math; R4 firing before R1 with high reputation and
  correctly upgrading the *matrix's* base action (not jumping to a fixed
  severity); `AdaptiveThresholds` moving within its bound and staying put
  when unused (default construction unaffected).
- **`evaluate.py` full run** (defaults, all new params off): ran cleanly
  end-to-end across all 4 intents, produced all expected tables/plots.
- **`evaluation/sedm_eval.py`** (oracle variant): ran cleanly, produced
  consistent LaTeX tables.
- **`main.py analyze`** and **`main.py demo`**: both ran cleanly, exercising
  the fixed feature-distribution plotting and the live `Defender.observe()`
  path with the new `reputation` parameter defaulting correctly.
- **Live pipeline CLI** (`opencanary_integration.emulator.scenario`): ran a
  full kill-chain sequence from one source IP; actions visibly escalated
  ALLOW→TROLL→BLOCK→ALERT as the same IP repeated attacks, confirming the
  reputation-aware pipeline works end-to-end.
- **Classifier retrain**: regenerated training data with the new generator,
  retrained, verified round-trip save/load and inference.
- Smoke-test output written to `results/`/`logs/` during verification was
  discarded (`git checkout -- results/ logs/`) so the repo's real evaluation
  artifacts aren't polluted with 3-episode smoke-test data.

## Files changed

**Modified:**
`attacker/attack_types.py`, `attacker/attacker.py`, `defender/classifier.py`,
`defender/defender.py`, `defender/matrix_policy.py`, `environment/cyber_env.py`,
`main.py`, `opencanary_integration/emulator/event_generator.py`,
`opencanary_integration/emulator/scenario.py`,
`opencanary_integration/engine/policy_engine.py`,
`opencanary_integration/engine/session_tracker.py`,
`opencanary_integration/engine/state_builder.py`,
`models/classifier.joblib`, `models/opportunistic/classifier.joblib`

**New:**
`defender/adaptive_thresholds.py`, `opencanary_integration/engine/reputation.py`

Every new parameter defaults to today's exact prior behavior — nothing
requires opting in to keep working as before. Nothing has been committed.
The full implementation plan (with detailed rationale, rejected alternatives,
and open design decisions) is preserved at
`C:\Users\pasan\.claude\plans\foamy-nibbling-penguin.md`.
