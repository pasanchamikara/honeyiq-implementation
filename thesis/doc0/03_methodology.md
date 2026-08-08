# Chapter 3 — Methodology

This chapter describes the complete design of the HoneyIQ simulation framework. §3.1 presents the high-level architecture. §3.2 describes the attacker module. §3.3 defines the composite threat-level metric and reward function. §3.4 presents the Stage-Escalation Decision Matrix in detail. §3.5 describes the DQN baseline. §3.6 documents the Gymnasium environment. §3.7 covers the Random Forest classifier. §3.8 describes the evaluation infrastructure.

## 3.1 System Architecture

HoneyIQ is a closed-loop simulation in which an *Attacker* and a *Defender* interact through a Gymnasium-compatible environment (`CyberSecurityEnv`).

**High-level architecture of the HoneyIQ simulation framework** (`CyberSecurityEnv`, a `gymnasium.Env` wrapper):

- **Attacker:** `TransitionModel` (Markov chains) → feature simulation → $(a_t, k_t, \mathbf{x}_t)$
- **Threat:** `compute_threat_level`$(a_t, k_t, r_t, n_t) \rightarrow T \in [0,1]$
- **State:** $s_t \in \mathbb{R}^{24}$ (one-hot + scalars)
- **Defender:** `AttackClassifier` (RF) + `MatrixPolicy` (SEDM) → action ∈ {0,…,4}
- **Reward:** `compute_reward`$(a_t, T, k_t, \text{attack\_type}) \rightarrow r_t$
- **MetricsCollector**: StepRecord → EpisodeRecord → CSV + plots

At each discrete time step $t$, the following sequence of operations is executed:

1. The attacker advances its Markov chains to produce a new $(a_t, k_t)$ pair and samples a 15-dimensional feature vector $\mathbf{x}_t$.
2. The environment computes the threat level $T_t$ and constructs the 24-dimensional state vector $s_t$.
3. The defender observes $s_t$ and selects a honeypot action.
4. The reward $r_t$ is computed and returned to the defender.
5. Metrics are recorded.

This one-step interaction repeats for a configurable number of steps (200 in evaluation, 500 in DQN training) before the episode terminates.

## 3.2 Attacker Module

### Attack Type and Kill Chain Enumerations

The discrete state spaces of the attacker are defined by two `IntEnum` classes.

**AttackType.** Ten categories inspired by the UNSW-NB15 classification taxonomy: NORMAL (0), RECONNAISSANCE (1), ANALYSIS (2), FUZZERS (3), GENERIC (4), EXPLOITS (5), SHELLCODE (6), BACKDOORS (7), DOS (8), WORMS (9). Each category is assigned a severity weight $s_i \in [0, 1]$.

**Table 3.1 — Attack types, severity weights, primary kill-chain stages, and kill-chain stage weights.**

| Index | Attack Type | Severity $s_i$ | Primary Stage | Stage Weight $w_k$ |
|---|---|---|---|---|
| 0 | NORMAL | 0.00 | Reconnaissance (0) | 0.10 |
| 1 | RECONNAISSANCE | 0.20 | Reconnaissance (0) | 0.10 |
| 2 | ANALYSIS | 0.25 | Weaponization (1) | 0.20 |
| 3 | FUZZERS | 0.35 | Delivery (2) | 0.35 |
| 4 | GENERIC | 0.40 | Delivery (2) | 0.35 |
| 5 | EXPLOITS | 0.70 | Exploitation (3) | 0.55 |
| 6 | SHELLCODE | 0.75 | Exploitation (3) | 0.55 |
| 7 | BACKDOORS | 0.80 | Installation (4) | 0.70 |
| 8 | DOS | 0.85 | Actions on Obj (6) | 1.00 |
| 9 | WORMS | 0.90 | Cmd & Ctrl (5) | 0.85 |

**KillChainStage.** Seven stages mapped to the Lockheed Martin Kill Chain: RECONNAISSANCE (0), WEAPONIZATION (1), DELIVERY (2), EXPLOITATION (3), INSTALLATION (4), COMMAND_AND_CTRL (5), ACTIONS_ON_OBJ (6). Stage weights $w_k$ increase monotonically from 0.10 to 1.00, reflecting the escalating danger of later kill-chain activity.

### Markov Transition Model

Two row-stochastic base transition matrices are maintained: $\mathbf{T}_A \in \mathbb{R}^{10 \times 10}$ (attack types) and $\mathbf{T}_K \in \mathbb{R}^{7 \times 7}$ (kill chain stages). At each step the attacker samples:

$$a_{t+1} \sim \text{Categorical}\bigl(\mathbf{T}_A^{(\pi)}[a_t, \cdot]\bigr), \qquad \tilde{k}_{t+1} \sim \text{Categorical}\bigl(\mathbf{T}_K^{(\pi)}[k_t, \cdot]\bigr)$$

A kill-chain floor constraint prevents unrealistic backward jumps:

$$k_{t+1} = \max\!\bigl(\tilde{k}_{t+1},\; \text{primary\_stage}(a_{t+1}) - 1\bigr)$$

This ensures that attack types associated with late kill chain stages cannot appear at very early stages, maintaining temporal consistency.

#### Intent Profiles

Four intent profiles $\pi \in \{\text{STEALTHY}, \text{AGGRESSIVE}, \text{TARGETED}, \text{OPPORTUNISTIC}\}$ are encoded as element-wise modifier matrices applied before row-renormalisation.

**Table 3.2 — Attacker intent profiles, behavioural characteristics, and primary attack preferences.**

| Intent | Attack Preference | Kill Chain Speed |
|---|---|---|
| STEALTHY | Recon, Analysis, Backdoors | Slow; high self-loop probability |
| AGGRESSIVE | DoS, Worms, Exploits | Fast; strong forward bias |
| TARGETED | Exploits → Shellcode → Backdoors | Direct exploitation chain |
| OPPORTUNISTIC | Generic, Fuzzers (scattered) | Moderate forward bias |

**STEALTHY**: the modifier matrix amplifies the self-loop probability for RECONNAISSANCE and ANALYSIS, and increases the transition probability from ANALYSIS to BACKDOORS. Stage transitions are biased towards self-loops (the attacker dwells in each stage longer), producing long-duration, low-severity campaigns.

**AGGRESSIVE**: forward transitions are amplified for DOS, WORMS, and EXPLOITS. Stage transitions are biased strongly forward, producing rapid escalation through the kill chain. The result is short, high-intensity campaigns that quickly reach ACTIONS_ON_OBJ.

**TARGETED**: transitions from EXPLOITS to SHELLCODE, SHELLCODE to BACKDOORS, and BACKDOORS to WORMS are amplified, encoding a specific attack chain. The kill chain transitions are biased towards a fast path from EXPLOITATION through INSTALLATION to COMMAND_AND_CTRL.

**OPPORTUNISTIC**: transitions to FUZZERS and GENERIC are amplified, producing scattered attack-type sequences. Stage transitions maintain moderate forward bias, resulting in a campaign that explores multiple attack types without strong directionality.

### Network Feature Simulation

At each step the attacker samples a 15-dimensional feature vector $\mathbf{x}_t \in \mathbb{R}^{15}$ from parametric distributions conditioned on the current attack type. The fifteen features are: `dur`, `sbytes`, `dbytes`, `sttl`, `dttl`, `sloss`, `dloss`, `sload`, `dload`, `spkts`, `dpkts`, `swin`, `dwin`, `ct_srv_src`, `ct_dst_ltm`.

Selected distributions illustrate the per-attack signature design:

- **RECONNAISSANCE**: `dur` ~ Exponential(rate=5), `ct_dst_ltm` ~ Poisson(30), reflecting rapid port-scanning activity.
- **DoS**: `sload` ~ Uniform(50,000, 500,000) bps, `dur` ~ Exponential(rate=10), capturing the high-bandwidth, short-burst pattern of flooding attacks.
- **BACKDOORS**: `dur` ~ Uniform(10, 3600) s, `sload` ~ Uniform(50, 500) bps, modelling long-lived, low-bandwidth covert channels.
- **WORMS**: `ct_dst_ltm` ~ Poisson(40), `spkts` ~ Poisson(200), reflecting spreading behaviour.
- **EXPLOITS**: `sbytes` ~ Uniform(1000, 50,000), `sload` ~ Uniform(10,000, 100,000) bps, capturing the payload-heavy pattern of exploit delivery.

The 15 features are passed to the Random Forest classifier for attack-type identification. They are not directly included in the DQN state vector, which uses ground-truth labels.

> **Data format generality note:** although the 15-field schema above is named after UNSW-NB15 for familiarity, neither the classifier nor the SEDM has a code-level dependency on UNSW-NB15 or NSL-KDD specifically — both operate on a plain `dict[str, float]` matching this schema, which maps onto other flow-export formats (NetFlow v9/IPFIX, Zeek `conn.log`, CICFlowMeter). See `docs/parameter_selection.md` §6 for the full field-mapping table.

## 3.3 Composite Threat Level and Reward Function

### Composite Threat Level

The threat level $T_t \in [0, 1]$ aggregates four signals into a single normalised index:

$$T_t = 0.45 \cdot s_{a_t} + 0.35 \cdot w_{k_t} + 0.15 \cdot r_t + 0.05 \cdot \min\!\left(1,\, \frac{n_t}{100}\right)$$

where:

- $s_{a_t} \in [0, 0.90]$ is the attack-type severity weight (Table 3.1).
- $w_{k_t} \in [0.10, 1.00]$ is the kill-chain stage weight.
- $r_t \in [0, 1]$ is the *escalation rate*: the fraction of the last 20 steps that contained attack traffic. It captures the *tempo* of the campaign independently of the specific attack types.
- $n_t$ is the cumulative attack count; the capped normalisation $\min(1, n_t / 100)$ provides a measure of cumulative campaign pressure that saturates after 100 attacks.

The weight assignment reflects a domain-informed prioritisation. Attack severity (45%) is the dominant signal because the attack type directly indicates the attacker's capability and intent. Kill chain stage (35%) captures the attacker's progress through the campaign, which determines the urgency of response. Escalation rate (15%) and cumulative count (5%) provide temporal context that distinguishes an isolated event from an ongoing campaign.

**Threat bands.** The threat level is partitioned into five bands used by the reward function:

| Band | Range |
|---|---|
| Benign | $T < 0.15$ |
| Low | $0.15 \leq T < 0.35$ |
| Medium | $0.35 \leq T < 0.55$ |
| High | $0.55 \leq T < 0.75$ |
| Critical | $T \geq 0.75$ |

### Reward Function

The reward function $R(a, T, k, \text{attack\_type})$ encodes domain knowledge as a structured signal with three layers.

**Layer 1: Base matrix.** The base reward is looked up from a 5×5 action-by-threat-band matrix.

**Table 3.3 — Base reward matrix (action × threat band). Values encode the desirability of each action–threat pairing.**

| Action | Benign | Low | Medium | High | Critical |
|---|---|---|---|---|---|
| ALLOW | +1.0 | +0.5 | −1.0 | −3.0 | −6.0 |
| LOG | +0.2 | +1.5 | +2.0 | +1.0 | −1.0 |
| TROLL | −1.0 | +1.0 | +3.0 | +2.5 | +0.5 |
| BLOCK | −2.0 | −0.5 | +1.5 | +3.5 | +5.0 |
| ALERT | −3.0 | −1.0 | +0.5 | +2.0 | +6.0 |

The matrix encodes graduated proportional responses: ALLOW is rewarded for benign traffic but incurs increasing penalties as threat level rises. ALERT receives the highest reward for critical threats (+6.0) and the largest penalty for benign traffic (−3.0), reflecting its asymmetric cost structure. TROLL peaks at Medium threat (+3.0), incentivising engagement with medium-severity attacks for intelligence gathering.

**Layer 2: Late kill-chain amplifier.** At kill chain stages Installation (4), Command & Control (5), and Actions on Objectives (6), negative rewards are amplified by a factor of 1.5×:

$$r \leftarrow 1.5 \cdot r \quad \text{if } r < 0 \text{ and } k \geq 4$$

This modifier encodes the escalating cost of defensive errors at advanced campaign stages, where allowing or under-responding to an attack is substantially more damaging than at early stages.

**Layer 3: Attack-type bonuses.** Specific action–attack-type combinations receive domain-knowledge bonuses:

- TROLL on BACKDOORS, SHELLCODE, or WORMS: +0.8. These persistent attack types offer the highest intelligence value when the defender engages them for extended periods.
- BLOCK on WORMS: +1.0. Rapid containment of spreading malware is particularly valuable.
- LOG on RECONNAISSANCE: +0.5. Silent logging of reconnaissance provides intelligence without alerting the attacker.
- ALLOW on confirmed non-attack: +0.5. Correct passthrough of benign traffic is explicitly rewarded to discourage over-aggressive responses.

## 3.4 Stage-Escalation Decision Matrix (SEDM)

The SEDM is a transparent, deterministic policy that maps the observable environment state to a honeypot action through a five-step procedure.

### Design Rationale

The SEDM is motivated by three observations:

1. The Markov chain structure provides *exact* escalation probabilities that encode the attacker's likely next move without requiring trajectory data.
2. Kill chain stage is the single most informative variable for honeypot response: the appropriate action is qualitatively different at Reconnaissance (observe passively) versus Actions on Objectives (escalate immediately).
3. Operational security practitioners require policies whose reasoning can be audited, documented, and modified by hand. A 7×3 matrix with three override rules satisfies this requirement; a 256-node neural network does not.

### Algorithm

**Step 1 — Escalation risk.** Query the intent-specific Markov transition model for the probability of advancing to a strictly higher kill chain stage from the current stage $k$:

$$\rho(k, \pi) = \sum_{k' > k} T^{(\pi)}_{kk'}$$

This quantity is computed analytically from the stored transition matrix. Under the STEALTHY intent, stage transitions are biased towards self-loops, producing low escalation risk values. Under the AGGRESSIVE intent, strong forward transitions produce high escalation risk.

**Step 2 — Band classification.** Discretise $\rho$ into three bands:

$$\text{band} = \begin{cases} 0 \; (\text{Low}) & \rho < 0.35 \\ 1 \; (\text{Medium}) & 0.35 \leq \rho < 0.65 \\ 2 \; (\text{High}) & \rho \geq 0.65 \end{cases}$$

The thresholds 0.35 and 0.65 partition the escalation risk into three roughly equal tertiles. The Low band covers stages with strong self-loops (primarily Reconnaissance under Stealthy intent); High covers stages with strong forward momentum (primarily Delivery through Installation under Aggressive intent).

**Step 3 — Matrix lookup.** Read the base action from the 7×3 SEDM (Table 3.4): $a_{\text{base}} = \text{SEDM}[k][\text{band}]$.

**Table 3.4 — Stage-Escalation Decision Matrix. Rows are kill chain stages; columns are escalation risk bands.**

| Stage | Low ($\rho<0.35$) | Medium ($0.35\le\rho<0.65$) | High ($\rho\ge0.65$) |
|---|---|---|---|
| RECONNAISSANCE | ALLOW | LOG | LOG |
| WEAPONIZATION | LOG | LOG | TROLL |
| DELIVERY | LOG | TROLL | TROLL |
| EXPLOITATION | TROLL | BLOCK | BLOCK |
| INSTALLATION | BLOCK | BLOCK | ALERT |
| COMMAND_&_CTRL | BLOCK | ALERT | ALERT |
| ACTIONS_ON_OBJ | ALERT | ALERT | ALERT |

The matrix encodes three design principles:

- **Proportionality**: responses escalate with both kill chain stage and escalation risk. No ALERT is issued before Exploitation even under high escalation risk; no ALLOW is issued after Weaponization.
- **Intelligence over containment at low stages**: LOG and TROLL are preferred over BLOCK at early stages to maximise intelligence before committing to containment.
- **Decisive containment at late stages**: ALERT is the dominant action from Installation onwards under medium or high escalation risk, reflecting the low tolerance for delay at advanced kill chain positions.

**Step 4 — Override rules.** Three override rules are applied sequentially after the matrix lookup:

- **R1 (Normal traffic)**: If `AttackType` = NORMAL, return ALLOW regardless of the matrix result. This ensures that confirmed benign traffic is never blocked.
- **R2 (High-impact attack types)**: If `AttackType` ∈ {DOS, WORMS}, upgrade the action by one level in the sequence ALLOW → LOG → TROLL → BLOCK → ALERT. These attack types cause rapid, widespread damage and warrant an immediately more aggressive response.
- **R3 (High attack frequency)**: If the escalation rate $r_t > 0.80$, upgrade the action by one level. A sliding window showing that more than 80% of recent steps involved attacks indicates a sustained, ongoing campaign that requires escalated containment.

Override rules R2 and R3 are applied conditionally: R2 is checked first, and R3 is checked only if R2 was not triggered. This precedence prevents double-upgrading within a single step.

**Step 5 — Composite risk score.** A composite risk score $\rho_c \in [0, 1]$ is computed for logging and analysis. It does not affect the action selection:

$$\rho_c = 0.35 \cdot w_k + 0.35 \cdot \rho(k, \pi) + 0.15 \cdot s_a + 0.15 \cdot r_t$$

### Intent-Awareness

The SEDM is intent-aware through the escalation risk computation. The base matrix $\text{SEDM}[k][\text{band}]$ is identical across all intents, but the band assigned to a given stage differs:

- Under STEALTHY, stage transitions are slow. RECONNAISSANCE has low escalation risk ($\rho < 0.35$) → ALLOW. The defender correctly passes low-threat reconnaissance traffic.
- Under AGGRESSIVE, stage transitions are fast. Even RECONNAISSANCE may have medium escalation risk ($\rho \geq 0.35$) → LOG, reflecting the attacker's tendency to escalate quickly.
- Under TARGETED, the direct exploitation path produces consistently high escalation risk from EXPLOITATION onwards → BLOCK/ALERT dominates.

The escalation risk is the key mechanism through which the SEDM adapts to different attacker behaviours without any learning or parameter updates.

### Worked Example

Consider an observation with: stage = EXPLOITATION (3), attack type = EXPLOITS, escalation rate = 0.65, intent = AGGRESSIVE.

1. **Escalation risk**: Under AGGRESSIVE intent, $T^{(\text{AGG})}_{3,4} + T^{(\text{AGG})}_{3,5} + T^{(\text{AGG})}_{3,6} = 0.72$ (high forward bias in the transition matrix).
2. **Band**: $\rho = 0.72 \geq 0.65 \Rightarrow$ band = High.
3. **Matrix lookup**: SEDM[3][2] = BLOCK.
4. **Override check**: EXPLOITS ∉ {DOS, WORMS} (R2 not triggered); escalation rate 0.65 ≤ 0.80 (R3 not triggered).
5. **Final action**: BLOCK.
6. **Composite risk**: $\rho_c = 0.35 \times 0.55 + 0.35 \times 0.72 + 0.15 \times 0.70 + 0.15 \times 0.65 = 0.654$.

## 3.5 DQN Baseline

The DQN baseline provides a learned policy for comparison with the SEDM. Its architecture and training hyperparameters are summarised below.

**Table 3.5 — DQN hyperparameters used in all training experiments.**

| Parameter | Value |
|---|---|
| State dimension | 24 |
| Action space | 5 (`HoneypotAction`) |
| Hidden layers | [256, 128, 64] |
| Normalisation per layer | `LayerNorm` |
| Activation | `ReLU` |
| Weight initialisation | Kaiming uniform |
| Replay buffer capacity | 15,000 transitions |
| Mini-batch size | 64 |
| Discount factor $\gamma$ | 0.99 |
| Learning rate | $10^{-3}$ (Adam) |
| Target network update interval | 150 steps (hard copy) |
| $\varepsilon$ initial / final / decay | 1.0 / 0.05 / 0.997 |
| Gradient clip $\ell_2$ norm | 10.0 |
| Loss function | Huber (SmoothL1) |
| Training episodes | 300 |
| Steps per episode | 500 |
| Training intent | OPPORTUNISTIC |

### Training Protocol

Training proceeds for 300 episodes of 500 steps each, using the OPPORTUNISTIC attacker intent. OPPORTUNISTIC was chosen as the training distribution because its scattered attack-type sequence provides broad coverage of the state space, encouraging the policy network to learn a generalised Q-function rather than one optimised for a specific escalation pattern.

At each step:

1. The defender observes the 24-dimensional state vector.
2. An action is selected via $\varepsilon$-greedy exploration.
3. The transition $(s, a, r, s', \text{done})$ is pushed to the replay buffer.
4. If the buffer contains at least 64 transitions, one gradient update is performed with a randomly sampled mini-batch.
5. Epsilon is decayed: $\varepsilon \leftarrow \max(0.05, 0.997\varepsilon)$.
6. Every 150 steps, the target network is hard-copied from the policy network.

Metrics recorded per episode: total reward, detection rate, false-positive rate, average threat level, average training loss. Training metrics are saved to `logs/metrics.csv` and visualised as six-panel training curves.

## 3.6 Gymnasium Environment (`CyberSecurityEnv`)

`CyberSecurityEnv` extends `gymnasium.Env` (Towers et al., 2024) and implements the standard Gymnasium interface.

### Observation Space

The observation space is `Box(24,)` with `float32` dtype. The 24-dimensional vector is constructed as follows:

$$s_t = \underbrace{[e_{a_t}^{(10)}]}_{\text{attack type (one-hot)}} \;\oplus\; \underbrace{[e_{k_t}^{(7)}]}_{\text{stage (one-hot)}} \;\oplus\; \underbrace{\left[T_t,\;\tfrac{n_t}{100},\;r_t\right]}_{\text{scalars}} \;\oplus\; \underbrace{[e_{\pi}^{(4)}]}_{\text{intent (one-hot)}}$$

where $e_x^{(d)}$ denotes a $d$-dimensional one-hot vector for category $x$. The scalar features ($T_t$, $n_t/100$, $r_t$) lie in $[0, 1]$ by construction. Including the attacker intent in the state vector allows both the DQN and the SEDM to condition their responses on the inferred attacker type, even though the SEDM queries the Markov chain directly.

### Action Space

The action space is `Discrete(5)`, one element per `HoneypotAction`. The integer-to-action mapping is: 0 = ALLOW, 1 = LOG, 2 = TROLL, 3 = BLOCK, 4 = ALERT.

### Episode Lifecycle

`reset()` re-initialises the attacker Markov chains to their starting states, clears the escalation history window, and resets the attack counter. It returns the initial state vector and an `info` dictionary containing the initial features and attack-type ground truth.

`step(action)` executes one simulation step:

1. Calls `attacker.step()` to advance the Markov chains and sample a feature vector.
2. Computes the threat level $T_t$ (§3.3).
3. Computes the reward $r_t$ (§3.3).
4. Constructs the next state vector $s_{t+1}$.
5. Increments the step counter; terminates the episode if `max_steps` is reached.
6. Returns `(next_state, reward, terminated, truncated, info)`.

Episodes terminate by truncation (`max_steps` reached) rather than by natural termination conditions. The environment does not model attacker defeat or withdrawal, consistent with the assumption that the attacker's Markov chain is independent of the defender's responses.

## 3.7 Random Forest Attack Classifier

The `AttackClassifier` wraps a `RandomForestClassifier` from scikit-learn with a `StandardScaler` preprocessing step.

### Training Data Generation

Training data is generated entirely from the parametric feature distributions (§3.2). For each of the ten attack types, 600 samples are generated independently:

$$\mathcal{D}_{\text{train}} = \bigcup_{a=0}^{9} \bigl\{(\mathbf{x}, a) \,:\, \mathbf{x} \sim p(\cdot \mid a)\bigr\}_{600}$$

yielding a balanced dataset of 6,000 samples. `class_weight='balanced'` provides a further guard against any residual class imbalance. The evaluation set (2,000 samples, 200 per class) is generated with a different random seed (999) to ensure independence from training data.

### Role in the System

The classifier output is logged alongside the ground-truth attack type at each step for comparison and monitoring. Two decision modes are exercised in Chapter 4: an **oracle** mode where the SEDM reads the attack type directly from the environment state vector (ground truth), and a **classifier-driven** mode where the SEDM decides from the classifier's *predicted* attack type instead, while metrics are still scored against ground truth. The classifier-driven mode is the more realistic operating condition, since ground truth is never directly observable in a real deployment. Re-evaluating the classifier-driven pipeline under realistic (non-synthetic) input noise is identified as a key direction for future work (§5.6).

## 3.8 Evaluation Infrastructure

### Metrics

Two granularity levels are maintained:

**Step-level (`StepRecord`).** A dataclass capturing per-step data: `episode, step, action, reward, attack_type, kill_chain_stage, threat_level, is_attack, predicted_attack, loss, escalation_rate`.

**Episode-level (`EpisodeRecord`).** Aggregated per-episode statistics:

$$\text{detection\_rate} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \qquad \text{false\_positive\_rate} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

where a True Positive (TP) is any step where the attack was detected (action ≠ ALLOW) and the traffic was genuinely malicious. A False Positive (FP) is any step where the defender responded with a non-ALLOW action on benign traffic.

> Every metric in Chapter 4 scores each defender action against the ground-truth label of the *same* observation the action was chosen from, decoded before the environment advances — see the methodology note at the start of Chapter 4 and `docs/BUGS_AND_FIXES.md` (Bug 8/9) for the one-step label-lag bug this corrected.

### Experimental Protocol

**SEDM evaluation.** The SEDM is evaluated for 30 episodes per attacker intent (120 total), each episode lasting 200 steps. No training is required; the policy is fully specified by Table 3.4 and the Markov chain transition model. Results are aggregated per intent and reported as mean ± standard deviation over the 30 episodes.

**DQN training.** The DQN agent is trained for 300 episodes of 500 steps each using the OPPORTUNISTIC intent. All hyperparameters are listed in Table 3.5. Training metrics are logged to `logs/metrics.csv`.

**Reproducibility.** All random seeds are fixed: seed 42 for training data generation and environment resets; seed 999 for classifier evaluation. The NumPy and PyTorch random states are seeded at the start of each run.
