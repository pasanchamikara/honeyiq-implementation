# HoneyIQ Documentation (v2 — SEDM + Live Pipeline + Dynamic Response)

This is a from-scratch, up-to-date documentation set for HoneyIQ as it
exists today. The original [`docs/`](../docs/) set predates the SEDM
rewrite and the `opencanary_integration/` live pipeline — several of its
files (`defender.md`, `README.md`) still describe the DQN as the active
policy, which it no longer is. `docs01/` supersedes it for anyone working
on the current codebase; `docs/` is kept as historical record.

HoneyIQ is a cybersecurity attacker-defender simulation for adaptive
honeypot management. A synthetic **Markov-chain attacker** progresses
through the Lockheed Martin Cyber Kill Chain while emitting realistic
network-flow features; a deterministic **Stage-Escalation Decision Matrix
(SEDM)** decides how the honeypot should respond at every step — with no
training loop, no neural network, and every decision traceable to a table
cell and a named rule.

## Documentation Index

| Document | Contents |
|---|---|
| [Architecture](architecture.md) | Two parallel pipelines (training env vs. live OpenCanary pipeline), component diagram, data flow |
| [Attacker & Synthetic Traffic](attacker.md) | Attack types, kill chain stages, intent profiles, Markov transition model, per-session intensity/persona traffic realism |
| [Environment](environment.md) | `CyberSecurityEnv` (Gymnasium wrapper), state vector, window vs. EMA escalation tracking |
| [Defender & SEDM Policy](defender.md) | Attack classifier, `MatrixPolicy`, honeypot actions, reward function, R1–R4 override rules |
| [Dynamic Response](dynamic_response.md) | Cross-session reputation tracking (R4), the `AdaptiveThresholds` controller, and what "dynamic" honestly means here |
| [OpenCanary Integration](opencanary_integration.md) | The live/near-real-time pipeline: ingest → session tracking → policy → dispatch |
| [Evaluation & Metrics](evaluation.md) | `evaluate.py`, `evaluation/sedm_eval.py`, `MetricsCollector`, how to read the numbers |
| [Quick Start](quickstart.md) | Installation, running every entry point, programmatic usage |
| [API Reference](api_reference.md) | Class and function signatures, current as of this implementation round |
| [Parameter Selection](parameter_selection.md) | Every tunable constant in the system, its default, and its rationale |
| [Is DQN Practical?](dqn_practicality.md) | Why a learning-based approach was rejected for dynamic adjustment, and what would change that |

## What changed since `docs/`

Three areas were added on top of the existing SEDM implementation:

1. **Synthetic traffic realism** — `Attacker` sessions now have a
   persistent intensity/persona profile instead of resampling every
   feature independently each step; benign traffic has three distinct
   personas instead of one blob; OpenCanary event payloads are drawn from
   wordlists instead of a handful of literal templates.
2. **Escalation tracking** — a severity-weighted EMA is available
   alongside the original hard sliding window, in both the training
   environment and the live pipeline.
3. **Dynamic behavior adjustment** — a cross-session `ReputationTracker`
   feeds a new R4 override rule, and a narrowly-scoped `AdaptiveThresholds`
   controller keeps one specific threshold from causing alert fatigue.
   Both are deliberately non-learning — see
   [Is DQN Practical?](dqn_practicality.md) for why.

Every addition defaults to the prior exact behavior; nothing needs to be
opted into for existing code to keep working.

## Project Layout (current)

```
honeyiq/
├── main.py                          # CLI: demo, compare, train, analyze
├── train.py                         # Training loop (legacy DQN-shaped config accepted, ignored)
├── evaluate.py                      # SEDM evaluation across 4 intents + OpenCanary demo
├── requirements.txt
│
├── attacker/
│   ├── attack_types.py              # Enums, severity weights, feature distributions,
│   │                                 #   intensity constants, benign personas
│   ├── transition_model.py          # Markov chain (attack type + kill chain stage)
│   └── attacker.py                  # Attacker — session-coherent traffic generation
│
├── defender/
│   ├── classifier.py                # RandomForest attack classifier
│   ├── honeypot.py                  # Actions, threat level, reward function
│   ├── matrix_policy.py             # MatrixPolicy (SEDM) — R1–R4 overrides
│   ├── adaptive_thresholds.py       # AdaptiveThresholds — bounded RATE_THRESHOLD nudge
│   ├── defender.py                  # Defender — orchestrates classifier + MatrixPolicy
│   └── dqn.py                       # Orphaned — nothing imports this anymore
│
├── environment/
│   └── cyber_env.py                 # CyberSecurityEnv (Gymnasium), encode_state()
│
├── evaluation/
│   ├── metrics.py                   # MetricsCollector, StepRecord, EpisodeRecord
│   └── sedm_eval.py                 # Extended classification-metrics evaluation suite
│
├── opencanary_integration/          # Live/near-real-time pipeline (no DQN dependency)
│   ├── ingest/                      # OpenCanaryEvent model, logtype → AttackType mapping
│   ├── engine/                      # SessionTracker, ReputationTracker, state_builder,
│   │                                 #   EscalationPredictor, PolicyEngine
│   ├── emulator/                    # OpenCanaryEventGenerator, DummyHoneypot, EmulatorScenario
│   └── dispatcher/                  # DummyDispatcher — currently orphaned/broken, see docs
│
├── models/                          # classifier.joblib (+ legacy dqn_agent.pt)
├── logs/, results/                  # CSV metrics, JSON audit logs, PNG plots
├── docs/                            # Original docs — partly stale, kept for history
├── docs01/                          # This documentation set
└── thesis/                          # Thesis chapters, slides, LaTeX source
```
