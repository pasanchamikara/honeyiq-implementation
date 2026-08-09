# Attacker & Synthetic Traffic

`attacker/` simulates one attacker session at a time: what kill-chain
stage and attack type it's currently at, how it's likely to move next, and
what its network traffic looks like on the wire.

## Enumerations (`attack_types.py`)

```python
class AttackType(IntEnum):
    NORMAL=0, RECONNAISSANCE=1, ANALYSIS=2, FUZZERS=3, EXPLOITS=4,
    BACKDOORS=5, SHELLCODE=6, GENERIC=7, DOS=8, WORMS=9

class KillChainStage(IntEnum):
    RECONNAISSANCE=0, WEAPONIZATION=1, DELIVERY=2, EXPLOITATION=3,
    INSTALLATION=4, COMMAND_AND_CTRL=5, ACTIONS_ON_OBJ=6

class AttackerIntent(IntEnum):
    STEALTHY=0, AGGRESSIVE=1, TARGETED=2, OPPORTUNISTIC=3
```

`ATTACK_SEVERITY` (0.00–0.90) and `KILL_CHAIN_WEIGHT` (0.10–1.00) drive the
threat/reward formulas in `defender/honeypot.py` and the composite risk
score in `defender/matrix_policy.py`; `ATTACK_PRIMARY_STAGE` anchors each
attack type to the kill-chain stage it's naturally associated with, so the
attacker's stage never drifts implausibly out of sync with its attack type.

## The Markov transition model (`transition_model.py`)

Two base row-stochastic matrices — one 10×10 over attack types, one 7×7
over kill-chain stages — are each modified per intent (multiplicative
element-wise scaling, then renormalized) so the four intents produce
qualitatively different campaigns from the same underlying structure:

| Intent | Attack-type bias | Stage-progression bias |
|---|---|---|
| STEALTHY | Favors RECON/ANALYSIS/BACKDOORS, high self-loop weight | Slow — long dwell time per stage, few forward jumps |
| AGGRESSIVE | Favors DOS/WORMS/EXPLOITS/SHELLCODE | Fast — low self-loop weight, strong forward push |
| TARGETED | Favors the EXPLOITS→SHELLCODE→BACKDOORS chain | Direct path, skips early stages |
| OPPORTUNISTIC | Favors GENERIC/FUZZERS, moderate noise everywhere | Random walk, mild forward bias |

`TransitionModel.get_stage_probabilities(stage)` is what
`MatrixPolicy._escalation_risk` sums over (all probability mass on stages
strictly after the current one) to get the forward-looking escalation
risk used by the SEDM band lookup — this is a separate signal from the
backward-looking `escalation_rate` described in
[`environment.md`](environment.md).

## `Attacker` — session-coherent traffic generation

```python
attacker = Attacker(intent=AttackerIntent.STEALTHY, seed=42)
info = attacker.step()   # advances one step, returns dict (see api_reference.md)
attacker.reset()         # new episode: re-seeds RNG, draws a new session profile
```

### Per-session intensity and persona

Previously, every call to `_simulate_features()` drew all 15 UNSW-NB15-style
features independently from a static per-attack-type distribution — a
session never had a consistent "character," and every `NORMAL` packet used
the same fixed distribution regardless of context.

`Attacker` now draws a **session profile** once per episode
(`_draw_session_profile()`, called from `__init__` and `reset()`):

- **`_intensity`** — a `lognormal(0, 0.35)` scalar applied multiplicatively
  to the volume-shaped features (`sbytes`, `dbytes`, `sload`, `dload`,
  `spkts`, `dpkts`, `ct_srv_src`, `ct_dst_ltm`). TTL, window size,
  duration, and loss counts are left alone — scaling those by "session
  intensity" wouldn't be physically meaningful. The result: a whole
  episode reads as one coherent attacker machine (consistently "big" or
  "small" traffic) instead of independent per-step noise.
- **`_benign_persona`** — which of three `NORMAL`-traffic distributions
  (see below) this session's benign packets are drawn from, when
  `attack_type == AttackType.NORMAL`.

Both are reproducible under a fixed seed (the same seed reproduces the
same profile on every `reset()`), and both can be overridden explicitly:

```python
features = attacker._simulate_features(
    AttackType.EXPLOITS, intensity=2.0, persona=None,
)
```

Callers that need many *independent* samples of one attack type — not one
session's worth — must pass fresh `intensity`/`persona` per sample this
way. Two call sites do this: `AttackClassifier.generate_training_data()`
and `main.py`'s feature-distribution plotter. Without the explicit
override, every "independent" training sample of a class would silently
share one session's profile, understating the real feature variance the
classifier needs to learn from.

### Benign traffic personas

```python
NORMAL_PERSONA_WEIGHTS = {
    "casual_user": 0.70, "crawler": 0.20, "monitoring_probe": 0.10,
}
```

| Persona | Character | Distinguishing features |
|---|---|---|
| `casual_user` | The original baseline `NORMAL` distribution, unchanged | Moderate everything |
| `crawler` | Short, fast, high-volume requests to one service | High `ct_srv_src`, low `ct_dst_ltm`, near-zero loss |
| `monitoring_probe` | Tiny payloads, checks many hosts | Very short `dur`, high `ct_dst_ltm`, zero loss |

Persona never touches `attack_type` or the kill-chain state — `NORMAL` is
still `NORMAL` regardless of which persona generated it; only the 15 raw
feature values differ. This matters for realistic false-positive
evaluation: a classifier or policy that's only ever seen one flavor of
benign traffic will behave differently against real, varied benign traffic
than one trained/tested against a mix.

## Randomized OpenCanary event payloads

The live-pipeline side of traffic generation —
`opencanary_integration/emulator/event_generator.py` — used to have a
small, fixed list of literal payload templates per scenario (e.g.
`ssh_brute` had exactly 3 hardcoded username/password pairs). Templates
now use sentinel strings and jitter ranges instead:

```python
"ssh_brute": [
    {"logtype": 22000, "dst_port": 22,
     "logdata": {"USERNAME": "$USERNAME", "PASSWORD": "$PASSWORD"}},
],
```

`_resolve_field()` resolves `"$USERNAME"`/`"$PASSWORD"`/`"$PATH"`/
`"$USERAGENT"` sentinels against small wordlists, and `(low, high)` tuples
(e.g. `port_scan`'s `HOST_COUNT`) into a jittered random int in that
range — anything else passes through unchanged. `map_logtype()` only ever
reads `event.logtype`, never `logdata`, so this is purely cosmetic realism
and doesn't change any downstream classification/policy behavior. See
[`opencanary_integration.md`](opencanary_integration.md) for how this
feeds into the rest of the live pipeline.
