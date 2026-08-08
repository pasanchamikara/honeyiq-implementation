# Appendix B — Full Hyperparameter and Configuration Listing

Table B.1 lists every configurable parameter in HoneyIQ with its default value, the file in which it is defined, and a brief rationale.

**Table B.1 — Complete HoneyIQ parameter listing with rationale.**

**DQN Network (`defender/dqn.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| state_dim | 24 | `dqn.py` | Observation vector length |
| action_dim | 5 | `dqn.py` | HoneypotAction count |
| hidden_layers | [256,128,64] | `dqn.py` | Wide-to-narrow funnel |
| normalisation | LayerNorm | `dqn.py` | Batch-size-agnostic |
| activation | ReLU | `dqn.py` | Standard deep RL choice |
| weight_init | Kaiming uniform | `dqn.py` | Preserves gradient variance |

**DQN Training (`train.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| learning_rate | $10^{-3}$ | `train.py` | Adam optimiser |
| gamma | 0.99 | `train.py` | Emphasises future rewards |
| epsilon_initial | 1.0 | `train.py` | Full exploration at start |
| epsilon_min | 0.05 | `train.py` | Retains 5% exploration |
| epsilon_decay | 0.997 | `train.py` | Exponential annealing |
| buffer_capacity | 15,000 | `train.py` | 30 episodes of 500 steps |
| batch_size | 64 | `train.py` | Standard mini-batch |
| target_update_freq | 150 steps | `train.py` | ≈3 updates/episode |
| grad_clip_norm | 10.0 | `train.py` | Prevents gradient explosion |
| loss_function | SmoothL1 | `train.py` | Robust to outlier Q-errors |
| training_episodes | 300 | `train.py` | Plateau reached by ep. 50 |
| steps_per_episode | 500 | `train.py` | Training episode length |
| training_intent | OPPORTUNISTIC | `train.py` | Broadest state coverage |

**SEDM Policy (`defender/matrix_policy.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| esc_low_threshold | 0.35 | `matrix_policy.py` | Low/Medium band boundary |
| esc_high_threshold | 0.65 | `matrix_policy.py` | Medium/High band boundary |
| rate_threshold | 0.80 | `matrix_policy.py` | R3 override trigger |
| high_impact_attacks | DOS, WORMS | `matrix_policy.py` | R2 override trigger |
| default_intent | OPPORTUNISTIC | `matrix_policy.py` | Fallback intent |

**Random Forest (`defender/classifier.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| n_estimators | 150 | `train.py` | Ensemble size |
| max_depth | 20 | `train.py` | Prevents overfitting |
| n_jobs | 1 | `train.py` | Single-threaded |
| class_weight | balanced | `classifier.py` | Handles class imbalance |
| scaler | StandardScaler | `classifier.py` | Zero-mean, unit-variance |
| train_samples/class | 600 | `train.py` | 6,000 total samples |
| eval_samples/class | 200 | `train.py` | 2,000 test samples |
| training_seed | 42 | `train.py` | Reproducibility |
| eval_seed | 999 | `classifier.py` | Independent test set |

**Environment (`environment/cyber_env.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| train_max_steps | 500 | `cyber_env.py` | Training episode length |
| eval_max_steps | 200 | `main.py` | Evaluation episode length |
| escalation_window | 20 steps | `cyber_env.py` | Sliding window size |
| threat_w_severity | 0.45 | `honeypot.py` | Dominant signal |
| threat_w_stage | 0.35 | `honeypot.py` | Kill chain progress |
| threat_w_esc_rate | 0.15 | `honeypot.py` | Campaign tempo |
| threat_w_count | 0.05 | `honeypot.py` | Cumulative pressure |
| attack_count_norm | 100 | `honeypot.py` | Saturation threshold |

**Threat Bands (`defender/honeypot.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| benign threshold | $<0.15$ | `honeypot.py` | Very low severity |
| low threshold | $0.15$–$0.35$ | `honeypot.py` | Reconnaissance-level |
| medium threshold | $0.35$–$0.55$ | `honeypot.py` | Delivery-level |
| high threshold | $0.55$–$0.75$ | `honeypot.py` | Exploitation-level |
| critical threshold | $\geq 0.75$ | `honeypot.py` | C2/AoO-level |

**Evaluation (`main.py`, `evaluate.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| eval_episodes | 30 | `main.py` | Per-intent episodes |
| eval_steps | 200 | `main.py` | Steps per eval episode |
| random_seed | 42 | `main.py` | Reproducibility |

**Attacker (`attacker/transition_model.py`)**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| default_intent | OPPORTUNISTIC | `attacker.py` | Can be overridden |
| attacker_seed | 42 | `attacker.py` | Numpy random seed |
| kill_chain_floor | primary stage − 1 | `attacker.py` | Prevents stage regression |

## B.1 SEDM Matrix Values

Table B.2 lists the complete 7×3 Stage-Escalation Decision Matrix for reference.

**Table B.2 — Complete Stage-Escalation Decision Matrix.**

| Kill Chain Stage | Low ($\rho<0.35$) | Medium ($0.35\le\rho<0.65$) | High ($\rho\ge0.65$) |
|---|---|---|---|
| RECONNAISSANCE | ALLOW | LOG | LOG |
| WEAPONIZATION | LOG | LOG | TROLL |
| DELIVERY | LOG | TROLL | TROLL |
| EXPLOITATION | TROLL | BLOCK | BLOCK |
| INSTALLATION | BLOCK | BLOCK | ALERT |
| COMMAND_&_CTRL | BLOCK | ALERT | ALERT |
| ACTIONS_ON_OBJ | ALERT | ALERT | ALERT |

## B.2 Override Rule Specifications

The three override rules are applied in order after the matrix lookup:

1. **R1 (Normal traffic)**: If `AttackType` = NORMAL, return ALLOW regardless of matrix result. Priority: highest (always applied first).
2. **R2 (High-impact attacks)**: If `AttackType` ∈ {DOS, WORMS} and R1 was not triggered, upgrade action by one level in the sequence: ALLOW → LOG → TROLL → BLOCK → ALERT. Rationale: DOS and WORMS cause rapid, widespread damage requiring an immediately escalated response.
3. **R3 (High attack rate)**: If escalation_rate > 0.80 and R1 and R2 were not triggered, upgrade action by one level. Rationale: more than 80% of the last 20 steps being attacks indicates a sustained campaign warranting escalated containment.

## B.3 Reward Matrix Values

Table B.3 lists the complete 5×5 base reward matrix.

**Table B.3 — Complete base reward matrix (action × threat band).**

| Action | Benign (<0.15) | Low (0.15–0.35) | Medium (0.35–0.55) | High (0.55–0.75) | Critical (≥0.75) |
|---|---|---|---|---|---|
| ALLOW | +1.0 | +0.5 | −1.0 | −3.0 | −6.0 |
| LOG | +0.2 | +1.5 | +2.0 | +1.0 | −1.0 |
| TROLL | −1.0 | +1.0 | +3.0 | +2.5 | +0.5 |
| BLOCK | −2.0 | −0.5 | +1.5 | +3.5 | +5.0 |
| ALERT | −3.0 | −1.0 | +0.5 | +2.0 | +6.0 |

Additional modifiers: late-stage amplifier (1.5× for negative rewards at stages ≥4); TROLL + BACKDOORS/SHELLCODE/WORMS (+0.8); BLOCK + WORMS (+1.0); LOG + RECONNAISSANCE (+0.5); ALLOW + non-attack (+0.5).
