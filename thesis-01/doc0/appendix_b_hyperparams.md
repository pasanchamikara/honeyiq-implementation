# Appendix B — Full Hyperparameter and Configuration Listing

Table B.1 lists every configurable parameter in HoneyIQ with its default value, the file in which it is defined, and a brief rationale. Parameters introduced in this version are listed in §B.4–B.7.

**Table B.1 — Complete HoneyIQ parameter listing with rationale (parameters unchanged from the original system).**

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
| rate_threshold | 0.80 | `matrix_policy.py` | R3 override trigger (calibrated for window-mode escalation tracking, §3.9, §4.10) |
| high_impact_attacks | DOS, WORMS | `matrix_policy.py` | R2 override trigger |
| default_intent | OPPORTUNISTIC | `matrix_policy.py` | Fallback intent |
| reputation_threshold | 0.60 | `matrix_policy.py` | R4 override trigger — new in this version, §B.6 |

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
| eval_max_steps | 200 (original), 500 (revised, §4.2) | `main.py`, `evaluate.py` | Evaluation episode length |
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
| eval_episodes | 30 (original), 50 (revised primary, §4.2) | `main.py`, `evaluate.py` | Per-intent episodes; SEM analysis (Table 4.8) |
| eval_steps | 200 (original), 500 (revised primary) | `main.py`, `evaluate.py` | Steps per eval episode |
| eval_benign_ratio | 0.0 (original), 0.3 (revised primary, §4.2) | `evaluate.py` | Fraction of steps overridden with NORMAL-class traffic |
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

Four override rules are applied, in the following priority order, after the matrix lookup (extended in this version with R4, listed first per its evaluation priority — §3.4):

1. **R4 (Cross-session reputation, new in this version)**: If the requesting source's decayed cross-session reputation score (§B.6) is ≥ `REPUTATION_THRESHOLD` (0.60), upgrade the action by one level and stop. Checked *before* R1. Rationale: a source with sufficient accumulated offense history should not regain unconditional trust via a single benign-looking event.
2. **R1 (Normal traffic)**: If `AttackType` = NORMAL and R4 did not trigger, return ALLOW regardless of matrix result. Rationale: confirmed benign traffic from a source with no accumulated reputation is never blocked.
3. **R2 (High-impact attacks)**: If `AttackType` ∈ {DOS, WORMS} and neither R4 nor R1 triggered, upgrade action by one level in the sequence: ALLOW → LOG → TROLL → BLOCK → ALERT. Rationale: DOS and WORMS cause rapid, widespread damage requiring an immediately escalated response.
4. **R3 (High attack rate)**: If escalation_rate exceeds `RATE_THRESHOLD` (or the current value of an attached `AdaptiveThresholds` controller, §B.7) and no earlier rule triggered, upgrade action by one level. Rationale: sustained high-frequency attack activity indicates an ongoing campaign that requires escalated containment.

A system with `reputation` always equal to 0.0 — the default everywhere except where §4.11 states otherwise — reproduces the original three-rule (R1, R2, R3) behaviour exactly, since R4's condition can never hold.

## B.3 Reward Matrix Values

Table B.3 lists the complete 5×5 base reward matrix (unchanged in this version).

**Table B.3 — Complete base reward matrix (action × threat band).**

| Action | Benign (<0.15) | Low (0.15–0.35) | Medium (0.35–0.55) | High (0.55–0.75) | Critical (≥0.75) |
|---|---|---|---|---|---|
| ALLOW | +1.0 | +0.5 | −1.0 | −3.0 | −6.0 |
| LOG | +0.2 | +1.5 | +2.0 | +1.0 | −1.0 |
| TROLL | −1.0 | +1.0 | +3.0 | +2.5 | +0.5 |
| BLOCK | −2.0 | −0.5 | +1.5 | +3.5 | +5.0 |
| ALERT | −3.0 | −1.0 | +0.5 | +2.0 | +6.0 |

Additional modifiers: late-stage amplifier (1.5× for negative rewards at stages ≥4); TROLL + BACKDOORS/SHELLCODE/WORMS (+0.8); BLOCK + WORMS (+1.0); LOG + RECONNAISSANCE (+0.5); ALLOW + non-attack (+0.5).

## B.4 Traffic-Realism Parameters (new in this version, §3.2.4)

**Table B.4 — Synthetic traffic realism parameters.**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| intensity_lognormal_sigma | 0.35 | `attack_types.py` | Spread of the per-session volume multiplier — wide enough for real variation, narrow enough that most sessions stay within ~2× the median |
| intensity_scaled_features | 8 features (`sbytes`, `dbytes`, `sload`, `dload`, `spkts`, `dpkts`, `ct_srv_src`, `ct_dst_ltm`) | `attack_types.py` | Only volume-shaped features scale with intensity; TTL/window/duration/loss are not physically scaled by a session-level factor |
| normal_persona_weights | casual_user 0.70, crawler 0.20, monitoring_probe 0.10 | `attack_types.py` | `casual_user` dominant (matches a real human-majority traffic mix); the other two present but minority, avoiding overrepresentation of automated traffic |

## B.5 Escalation-Tracking Parameters (new in this version, §3.9)

**Table B.5 — Window and EMA escalation-tracking parameters.**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| escalation_mode | "window" | `cyber_env.py`, `session_tracker.py` | Preserves exact prior behaviour; "ema" is opt-in pending the recalibration identified in §5.6 |
| escalation_ema_alpha | 0.15 | `cyber_env.py`, `session_tracker.py` | ~13-step effective memory (1/α), comparable to the 20-step window it complements |

## B.6 Cross-Session Reputation Parameters (new in this version, §3.10.1)

**Table B.6 — `ReputationTracker` parameters. All are unvalidated defaults chosen for a reasonable behavioural shape (§5.4) rather than fit to real attack-frequency data.**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| offense_increment | 0.25 | `reputation.py` | A single WORMS-severity (0.90) offense contributes ≈0.23; several repeated high-severity offenses needed to cross the R4 threshold, not one |
| decay_half_life_seconds | 21,600 (6 h) | `reputation.py` | Long enough that a source does not "reset" between short breaks in activity; short enough that history does not accumulate forever |
| max_score | 1.0 | `reputation.py` | Matches the [0,1] convention used elsewhere (severity, escalation_rate, escalation_risk) |
| stale_after_seconds | 2,592,000 (30 d) | `reputation.py` | Bounds memory growth from one-off/scanner sources without needing a database |
| sweep_interval_seconds | 300 | `reputation.py` | Throttled sweep, same pattern as `SessionTracker`'s TTL expiry |

§4.11 measures the resulting behaviour precisely: under these defaults, reputation crosses `REPUTATION_THRESHOLD` (0.60) at exactly the fourth prior EXPLOITS-severity ($s_a=0.70$) offense.

## B.7 Bounded Threshold Controller Parameters (new in this version, §3.10.2)

**Table B.7 — `AdaptiveThresholds` parameters.**

| Parameter | Default | Source | Rationale |
|---|---|---|---|
| target_rate | 0.10 | `adaptive_thresholds.py` | R3 firing on ~10% of decisions is a reasonable "not every step, not silent" operational choice, not derived from data |
| tolerance | 0.03 | `adaptive_thresholds.py` | Deadband width — prevents the controller from chasing statistical noise in the observed rate |
| step | 0.01 | `adaptive_thresholds.py` | Small relative to `RATE_THRESHOLD`'s 0–1 range — gradual correction |
| bound | 0.10 | `adaptive_thresholds.py` | Caps total drift at ±0.10 from the initial threshold; §4.12 shows this bound is the binding constraint under continuous-attack (saturated) conditions specifically |
| observation_window | 200 | `adaptive_thresholds.py` | Large enough to smooth over per-episode variance before nudging |

These five parameters govern an explicitly-scoped alert-fatigue safety valve, not a correctness mechanism (§3.10.2) — they should not be read as tuned against any detection-accuracy objective, because no such objective is computable in this system without a ground-truth feedback pathway that does not currently exist (§5.3, §5.6).
