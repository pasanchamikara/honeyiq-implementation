# HoneyIQ Documentation

HoneyIQ is a cybersecurity attacker-defender simulation that uses Deep Reinforcement Learning to train a honeypot-based defender against a Markov-chain-driven attacker.

## Documentation Index

| Document | Contents |
|---|---|
| [Theoretical Background](theoretical_background.md) | Reinforcement learning, DQN, Markov chains, cyber kill chain, honeypots, UNSW-NB15 |
| [Architecture](architecture.md) | Component diagram, data flow, module responsibilities |
| [Attacker Module](attacker.md) | Attack types, kill chain stages, intent profiles, transition model, feature simulation |
| [Defender Module](defender.md) | DQN agent, attack classifier, honeypot actions, reward function |
| [Environment](environment.md) | Gymnasium wrapper, state space, action space, episode lifecycle |
| [Evaluation & Metrics](evaluation.md) | Metrics, detection rate, false positive rate, visualizations |
| [Quick Start](quickstart.md) | Installation, training, demo, analysis |
| [API Reference](api_reference.md) | Class and function signatures |

## Project Layout

```
honeyiq/
├── main.py                  # CLI entry point (demo, compare, train, analyze)
├── train.py                 # Training loop
├── requirements.txt
├── attacker/
│   ├── attack_types.py      # Enums, severity weights, feature distributions
│   ├── transition_model.py  # Markov chain (attack + kill chain stage)
│   └── attacker.py          # AttackerAgent
├── defender/
│   ├── dqn.py               # DQNNetwork, ReplayBuffer, DQNAgent
│   ├── classifier.py        # RandomForest attack classifier
│   ├── honeypot.py          # Actions, threat level, reward function
│   └── defender.py          # DefenderAgent (orchestrator)
├── environment/
│   └── cyber_env.py         # CyberSecurityEnv (Gymnasium)
├── evaluation/
│   └── metrics.py           # MetricsCollector, StepRecord, EpisodeRecord
├── models/                  # Saved checkpoints (dqn_agent.pt, classifier.joblib)
└── logs/                    # CSV metrics and PNG plots
```
