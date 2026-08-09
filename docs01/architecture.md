# Architecture

HoneyIQ has **two parallel pipelines** that share the same core decision
policy but differ in how they get an attacker and a state vector:

1. **The training/evaluation environment** (`environment/cyber_env.py`) —
   a self-contained Gymnasium environment where a simulated `Attacker`
   drives one episode at a time. Used by `main.py`, `evaluate.py`, and
   `evaluation/sedm_eval.py`.
2. **The live OpenCanary integration pipeline**
   (`opencanary_integration/`) — ingests OpenCanary-shaped JSON events
   (real or emulated), tracks per-source-IP session state across many
   concurrent sessions, and calls the same policy. Used by
   `opencanary_integration.emulator.scenario` and, in a real deployment,
   would sit behind a real OpenCanary honeypot process.

Both pipelines converge on the same 24-dimensional state vector
(`environment.cyber_env.encode_state`) and the same decision policy
(`defender.matrix_policy.MatrixPolicy`) — there is exactly one place the
"what should we do about this attacker" logic lives.

## Component diagram

```
┌─────────────────────────────── Training / Eval pipeline ───────────────────────────────┐
│                                                                                          │
│  Attacker (Markov chain)  ──step()──▶  CyberSecurityEnv  ──encode_state()──▶  state[24] │
│  (attacker/attacker.py)      │            (environment/cyber_env.py)              │     │
│                               │                                                     │     │
│                       synthetic features                                            ▼     │
│                               │                                              MatrixPolicy  │
│                               ▼                                          (defender/matrix_ │
│                     AttackClassifier (RandomForest)                          policy.py)    │
│                     (defender/classifier.py)  ──predicted_attack──▶  Defender.observe()    │
│                                                                                    │        │
│                                                                              HoneypotAction  │
└────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── Live OpenCanary pipeline ────────────────────────────────┐
│                                                                                           │
│  OpenCanaryEvent (real or            map_logtype()          SessionTracker              │
│  OpenCanaryEventGenerator)  ────▶  (ingest/logtype_map.py)  ────▶  (engine/session_       │
│  (emulator/event_generator.py)                                     tracker.py)           │
│                                                                       │        │          │
│                                                            SessionState   ReputationTracker│
│                                                            (window/EMA    (engine/         │
│                                                             escalation)    reputation.py)  │
│                                                                       │        │          │
│                                                                       ▼        │          │
│                                                          state_builder.build_state()      │
│                                                            (engine/state_builder.py)       │
│                                                                       │                    │
│                                                                       ▼                    │
│                                              PolicyEngine.decide(state, reputation)        │
│                                                 (engine/policy_engine.py)                  │
│                                                                       │                    │
│                                                             MatrixPolicy (same class)       │
│                                                                       │                    │
│                                                                       ▼                    │
│                                                    DummyHoneypot.apply_action_sync()        │
│                                                    (emulator/honeypot_emulator.py)          │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Why two pipelines share one policy

`MatrixPolicy` never depends on where its inputs came from — it only reads
a kill-chain stage, an attack type, an escalation rate, an intent, and
(optionally) a reputation score. `encode_state()` was factored out of
`CyberSecurityEnv` specifically so both pipelines build the state vector
identically (see [`defender.md`](defender.md) and
[`opencanary_integration.md`](opencanary_integration.md) for each side's
call path). This means anything validated in the synthetic training
environment — the matrix itself, the override rules, the threat/reward
formulas — carries over unchanged to the live pipeline, and anything that
only makes sense in a live deployment (reputation, which needs a real
source IP identity that persists across many separate sessions) stays an
optional, out-of-band parameter rather than being forced into the shared
state vector.

## Data flow: one decision, start to finish (live pipeline)

1. An `OpenCanaryEvent` arrives (from a real OpenCanary log line, or from
   `OpenCanaryEventGenerator` for demos/testing).
2. `map_logtype()` converts its numeric `logtype` code into an `AttackType`.
3. `SessionTracker.update(src_ip, attack_type)` finds or creates a
   `SessionState` for that IP, appends to its sliding window, updates its
   severity-weighted EMA, and records the event in the source IP's
   persistent `ReputationTracker` entry.
4. `state_builder.build_state(session, escalation_mode=...)` produces the
   24-dim vector, choosing window- or EMA-based escalation depending on
   mode.
5. `PolicyEngine.decide(state, features, reputation=session.reputation)`
   calls `MatrixPolicy.decide_from_state()`, which computes escalation
   risk from the Markov `TransitionModel`, looks up the SEDM table, and
   applies R1–R4 in order.
6. The resulting `HoneypotAction` is dispatched to `DummyHoneypot`
   (currently the only working dispatch target — see
   [`opencanary_integration.md`](opencanary_integration.md) for the status
   of the `dispatcher/` package).

## Module responsibilities at a glance

| Module | Responsibility |
|---|---|
| `attacker/` | Simulate an attacker's kill-chain progression and synthetic network features |
| `environment/` | Gymnasium wrapper: episode lifecycle, reward, state encoding, escalation tracking |
| `defender/classifier.py` | Predict attack type from raw network features |
| `defender/matrix_policy.py` | The actual decision policy (SEDM + overrides) |
| `defender/adaptive_thresholds.py` | Optional, bounded threshold auto-tuning for one rule |
| `defender/defender.py` | Thin orchestrator: classifier + MatrixPolicy, save/load |
| `opencanary_integration/ingest/` | Parse and classify incoming events |
| `opencanary_integration/engine/` | Session/reputation state, state building, policy invocation |
| `opencanary_integration/emulator/` | Synthetic event generation + in-memory honeypot for demos/tests |
| `opencanary_integration/dispatcher/` | Intended real-dispatch layer — currently orphaned, see notes |
| `evaluation/` | Metrics collection and the extended SEDM evaluation suite |
