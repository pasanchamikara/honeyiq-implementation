# OpenCanary Integration

`opencanary_integration/` is the live/near-real-time pipeline: it turns a
stream of OpenCanary-shaped JSON events (real or emulated) into
`MatrixPolicy` decisions, tracking per-source-IP state across many
concurrent sessions. It has zero dependency on the training environment
or the DQN — everything here was built for (and works with) the SEDM
policy directly.

```
opencanary_integration/
├── ingest/       OpenCanaryEvent model, logtype → AttackType mapping
├── engine/       SessionTracker, ReputationTracker, state_builder,
│                 EscalationPredictor, PolicyEngine
├── emulator/     OpenCanaryEventGenerator, DummyHoneypot, EmulatorScenario
└── dispatcher/   DummyDispatcher — currently orphaned/broken (see below)
```

## `ingest/`

**`models.py`** — `OpenCanaryEvent`, a Pydantic model matching OpenCanary's
JSON log-line schema (`dst_host`, `dst_port`, `logdata`, `logtype`,
`node_id`, `src_host`, `src_port`, `utc_time`, `local_time`). Has a
`service_name` property mapping numeric `logtype` codes to a
human-readable service name (`"SSH"`, `"HTTP"`, etc.).

**`logtype_map.py`** — `map_logtype(event) -> AttackType` (a dict keyed by
`event.logtype`, defaulting to `GENERIC`) and `initial_stage_for(attack_type)
-> KillChainStage`, which reuses `attacker.attack_types.ATTACK_PRIMARY_STAGE`
directly rather than duplicating it.

## `engine/`

### `SessionTracker` / `SessionState` (`session_tracker.py`)

Per-source-IP session state, keyed by `src_ip`, with a throttled TTL sweep
(default 300s TTL, checked at most once per 60s — not on every call).

```python
@dataclass
class SessionState:
    src_ip: str
    current_attack:  AttackType
    current_stage:   KillChainStage
    attack_count:    int
    event_count:     int
    recent_attacks:  deque            # window escalation source
    escalation_ema:  float            # EMA escalation source
    last_seen:       datetime
    inferred_intent: AttackerIntent
    reputation:      float           # cross-session, see reputation.py

    @property
    def escalation_rate(self) -> float:  # window-based, backward compatible
        ...

tracker = SessionTracker(
    ttl_seconds=3600, escalation_window=20,
    escalation_ema_alpha=0.15, sweep_interval_seconds=60,
)
session = tracker.update(src_ip, attack_type)   # find-or-create, updates all signals
tracker.reputation.get(src_ip)                  # query without recording an offense
```

`update()` does four things per event: (1) find-or-create the
`SessionState` and advance `current_stage` monotonically forward if the
new attack type implies a later stage than currently recorded, (2) append
to the window deque, (3) update the severity-weighted EMA, (4) call
`self.reputation.record_offense(src_ip, severity)` and store the result on
`session.reputation`. See [`environment.md`](environment.md) for the
window/EMA formulas (identical here) and
[`dynamic_response.md`](dynamic_response.md) for `ReputationTracker`.

### `state_builder.py`

```python
def build_state(session: SessionState, escalation_mode: str = "window") -> np.ndarray:
```

Computes `threat_level` (`defender.honeypot.compute_threat_level`) and
calls `environment.cyber_env.encode_state()` — the same function the
training environment uses — so both sides of the system produce
byte-identical state vectors for the same inputs. `escalation_mode`
selects `session.escalation_rate` (window) or `session.escalation_ema`
(EMA); the same rate feeds both the threat-level computation and
`state[19]`, so the two stay internally consistent.

### `EscalationPredictor` (`escalation_predictor.py`)

A thin wrapper around `TransitionModel` exposing the probability vectors
the pipeline needs for display/audit purposes (`next_attack_probs`,
`next_stage_probs`, `most_likely_next_stage`, `most_likely_next_attack`),
plus `escalation_risk(stage, probs=None)` — the same forward-looking
"P(advance beyond current stage)" formula `MatrixPolicy` uses internally.
Accepts an already-computed `probs` array to avoid recomputing the same
transition row twice when the caller (`EmulatorScenario`) already has it.

### `PolicyEngine` (`policy_engine.py`)

```python
engine = PolicyEngine(model_dir="models/", default_intent="OPPORTUNISTIC")
action, predicted_attack = engine.decide(state, features=None, reputation=0.0)
info = engine.decision_info(state)   # full SEDM breakdown for explainability
```

Wraps a `MatrixPolicy` + an `AttackClassifier` (loaded from
`model_dir/classifier.joblib`, with `reload()` for hot-swapping a
retrained model without restarting the process). `decide()` calls
`MatrixPolicy.decide_from_state(state, reputation=reputation)` for the
action, and separately runs the classifier on raw `features` if provided
— the classifier's prediction here is informational (audit/logging), not
an input to the SEDM decision itself.

## `emulator/`

This package lets the whole pipeline run and be demoed with **no live
OpenCanary process, no network sockets, no FastAPI/gRPC** — everything is
in-memory.

### `OpenCanaryEventGenerator` (`event_generator.py`)

Produces `OpenCanaryEvent` objects matching real OpenCanary JSON output,
for ~19 named scenarios (`ssh_brute`, `port_scan`, `http_dir_scan`, ...).
See [`attacker.md`](attacker.md) for the wordlist/jitter-range payload
randomization added in this round.

```python
gen = OpenCanaryEventGenerator(node_id="honeypot-01", seed=42)
event = gen.generate("ssh_brute", src_ip=None)     # random realistic src_ip if None
events = gen.generate_kill_chain(src_ip="10.0.0.1")  # 7-event recon→shellcode sequence
```

### `DummyHoneypot` (`honeypot_emulator.py`)

Mirrors the interface a real OpenCanary config-mutation/reload layer
would have, but only updates in-memory per-IP state and prints/logs.

```python
honeypot = DummyHoneypot(audit_file="audit.jsonl", verbose=True)
honeypot.apply_action_sync(src_ip, action="BLOCK", attack_type=..., stage=..., threat_level=...)
honeypot.schedule_reload_sync(urgent=True)
honeypot.get_ip_status(src_ip)     # blocked/trolled/logged/alerted flags, notes
honeypot.close()                   # flush + close the audit file handle
```

The audit file handle is opened once in `__init__` and kept open (flushed
after every write) rather than reopened per event; `close()` and
`__del__` handle cleanup.

### `EmulatorScenario` (`scenario.py`)

The full pipeline glued together, runnable as a CLI:

```bash
python -m opencanary_integration.emulator.scenario --scenario kill_chain --src-ip 10.0.0.1
python -m opencanary_integration.emulator.scenario --scenario random --events 20
```

```python
scenario = EmulatorScenario(
    model_dir="models/", intent="OPPORTUNISTIC",
    audit_file=None, verbose=True, escalation_mode="window",
)
decision = scenario.run_event(event)
# {event_id, src_ip, logtype, service, attack_type, stage, threat_level,
#  escalation_risk, reputation, action, stage_probs, attack_probs}
```

`run_event()` is the reference implementation of the full data-flow
diagram in [`architecture.md`](architecture.md): `map_logtype` →
`SessionTracker.update` → `build_state` → `PolicyEngine.decide` →
`DummyHoneypot.apply_action_sync` + `schedule_reload_sync`.

Verified live: running a full 7-event kill-chain sequence from one source
IP shows actions escalating ALLOW→TROLL→BLOCK→ALERT as that IP repeats
attacks — both the matrix/override logic and (in the background) the
reputation accumulation are exercised correctly.

## `dispatcher/` — currently orphaned

`dummy_dispatcher.py` defines `DummyDispatcher`, intended as a real
network-dispatch layer sitting behind a `Pipeline` abstraction that
doesn't exist in this codebase. It imports `DecisionPayload` from
`opencanary_integration.dispatcher.models`, **which does not exist** —
importing `opencanary_integration.dispatcher` (or `dummy_dispatcher`
directly) currently raises `ModuleNotFoundError`. Nothing outside the
`dispatcher` package imports it, so this has no effect on
`EmulatorScenario` or any tested code path (which use `DummyHoneypot`
directly, not `DummyDispatcher`). Flagged here for accuracy, not fixed —
out of scope for this round of work.
