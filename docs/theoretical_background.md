# Theoretical Background

This document explains the key concepts behind HoneyIQ.

---

## 1. Reinforcement Learning (RL)

Reinforcement learning is a paradigm where an **agent** learns to make decisions by interacting with an **environment** and receiving scalar **rewards**. The goal is to find a **policy** π(a|s) — a mapping from states to actions — that maximises the expected cumulative discounted reward:

```
G_t = Σ_{k=0}^{∞} γ^k · r_{t+k+1}
```

where γ ∈ [0, 1) is the **discount factor** controlling how much future rewards are valued relative to immediate ones.

### Key RL terms

| Term | Meaning in HoneyIQ |
|---|---|
| Agent | The `Defender` |
| Environment | `CyberSecurityEnv` |
| State s | 24-dimensional observation vector |
| Action a | One of 5 `HoneypotAction` values |
| Reward r | Computed by `compute_reward()` |
| Episode | One attacker campaign (up to `max_steps`) |
| Policy π | Learned by the DQN |

---

## 2. Deep Q-Networks (DQN)

### Q-Learning

Q-learning learns the **action-value function** Q(s, a) — the expected return from taking action a in state s and then following the optimal policy:

```
Q*(s, a) = r + γ · max_{a'} Q*(s', a')     (Bellman optimality equation)
```

The optimal policy is simply:

```
π*(s) = argmax_a Q*(s, a)
```

### DQN Extensions

Plain Q-learning does not scale to high-dimensional states. DQN (Mnih et al., 2015) addresses this with three ideas:

#### 2.1 Neural Network Function Approximator

Instead of a lookup table, Q(s, a; θ) is parameterised by a neural network with weights θ. HoneyIQ uses a fully connected network:

```
Input(24) → [Linear → LayerNorm → ReLU] × 3 → Linear(5)
```

Hidden layer sizes: 256 → 128 → 64. **LayerNorm** (instead of BatchNorm) stabilises training with small batch sizes and avoids dependencies between samples.

**Weight initialisation** uses Kaiming uniform (He initialization), which preserves gradient variance when using ReLU activations.

#### 2.2 Experience Replay

Transitions (s, a, r, s', done) are stored in a circular **replay buffer** of capacity 15,000. Mini-batches of 64 are sampled uniformly at random for each gradient update.

Benefits:
- Breaks temporal correlations in sequential data (IID assumption for SGD)
- Each transition can be used multiple times (sample efficiency)
- Stabilises training by averaging over many states

#### 2.3 Target Network

A second network — the **target network** — provides stable TD targets:

```
y_i = r_i + γ · max_{a'} Q(s'_i, a'; θ⁻)
```

The target network parameters θ⁻ are hard-copied from the policy network every 150 steps. Without this, the moving target problem causes oscillations or divergence.

#### 2.4 Huber Loss

The TD error is minimised with **SmoothL1 (Huber) loss**:

```
L(δ) = { 0.5 δ²          if |δ| ≤ 1
        { |δ| - 0.5       otherwise
```

This is less sensitive to outlier rewards than MSE while still being differentiable everywhere.

#### 2.5 Epsilon-Greedy Exploration

During training the agent selects a random action with probability ε (exploration) and the greedy action otherwise (exploitation). ε is annealed from 1.0 to 0.05 using exponential decay:

```
ε ← max(ε_min, ε · ε_decay)       ε_decay = 0.997
```

#### 2.6 Gradient Clipping

Gradients are clipped to a max norm of 10.0 before the optimizer step, preventing exploding gradients.

---

## 3. Markov Decision Process (MDP)

The cybersecurity environment is formalised as an MDP:

- **S** — state space (24-dimensional continuous box)
- **A** — action space (5 discrete actions)
- **P(s' | s, a)** — transition probability (determined by the attacker's Markov chain; the defender's action does not change the attacker's next state)
- **R(s, a, s')** — reward function
- **γ = 0.99** — discount factor

Because the defender's actions do not directly alter the attacker's trajectory, the environment is **partially adversarial**: the attacker evolves independently, and the defender must react.

---

## 4. Markov Chains (Attacker Model)

### 4.1 Discrete-Time Markov Chain

An attacker's behaviour is modelled as a **Discrete-Time Markov Chain (DTMC)**:

```
P(X_{t+1} = j | X_t = i, X_{t-1}, ...) = P(X_{t+1} = j | X_t = i)
```

The Markov property holds: the future state depends only on the present state, not history. Two independent chains are used:

1. **Attack type chain** — 10×10 row-stochastic transition matrix
2. **Kill chain stage chain** — 7×7 row-stochastic transition matrix

### 4.2 Intent-Specific Modifiers

The base transition matrices are multiplied element-wise by intent-specific modifier matrices, then rows are renormalized. This skews the probability mass without changing the matrix dimensions.

| Intent | Attack preference | Kill chain speed |
|---|---|---|
| STEALTHY | Recon, Analysis, Backdoors | Slow — high self-loops |
| AGGRESSIVE | DoS, Worms, Exploits, Shellcode | Fast — low self-loops, boost forward |
| TARGETED | Exploits → Shellcode → Backdoors | Direct path through exploitation stages |
| OPPORTUNISTIC | Generic, Fuzzers — scattered | Moderate forward bias with regression |

### 4.3 Kill Chain Stage Constraint

After sampling the next stage from the Markov chain, the result is floored to `max(sampled_stage, primary_stage - 1)` where `primary_stage` is the stage associated with the new attack type. This prevents unrealistic regression (e.g., returning to Reconnaissance while performing a Backdoor attack).

---

## 5. Lockheed Martin Cyber Kill Chain

The **Cyber Kill Chain** (Hutchins et al., 2011) models an adversary's campaign as a sequence of seven phases. HoneyIQ maps attack types to these stages:

| Stage | Index | Description | Primary attack types |
|---|---|---|---|
| Reconnaissance | 0 | Information gathering | NORMAL, RECONNAISSANCE |
| Weaponization | 1 | Building the attack tool | ANALYSIS |
| Delivery | 2 | Transmitting the weapon | FUZZERS, GENERIC |
| Exploitation | 3 | Triggering the vulnerability | EXPLOITS, SHELLCODE |
| Installation | 4 | Establishing persistence | BACKDOORS |
| Command & Control | 5 | Establishing C2 channel | WORMS |
| Actions on Objectives | 6 | Achieving goals (exfil, DoS) | DOS |

**Kill chain weight** increases linearly with stage (0.10 to 1.00) and contributes 35% to the composite threat level, reflecting that later-stage attacks are more dangerous and costly to contain.

---

## 6. Honeypot Technology

A **honeypot** is a decoy system designed to attract, detect, and study attackers. The key insight is that any connection to a honeypot is inherently suspicious — there is no legitimate reason for a non-attacker to interact with it.

### 6.1 Honeypot Action Space

HoneyIQ abstracts honeypot responses into five actions:

| Action | Int | Purpose | Optimal threat band |
|---|---|---|---|
| ALLOW | 0 | Let traffic through untouched | Benign |
| LOG | 1 | Record and monitor the session | Low |
| TROLL | 2 | Respond with fake data / tarpit | Medium |
| BLOCK | 3 | Drop / firewall the connection | High |
| ALERT | 4 | Trigger a high-priority security alert | Critical |

**Tarpitting** (TROLL) is a technique where the defender slows down the attacker by responding with artificially delayed or fake data, wasting attacker resources and gathering intelligence simultaneously.

### 6.2 Reward Design

The reward matrix encodes domain knowledge about correct responses:

- **ALLOW** on benign traffic: +1.0 (correct, no disruption)
- **ALLOW** on critical traffic: -6.0 (catastrophic miss)
- **BLOCK** on benign traffic: -2.0 (false positive, disrupts legitimate users)
- **ALERT** on critical traffic: +6.0 (correct, high-priority response)
- **Late kill chain penalty**: negative rewards are multiplied by 1.5 for stages ≥ INSTALLATION, reflecting higher stakes

**Attack-type bonuses** reward honeypot-specific strategies:
- TROLL on BACKDOORS, SHELLCODE, WORMS: +0.8 (trolling persistent attackers gathers valuable intelligence)
- BLOCK on WORMS: +1.0 (containment bonus)
- LOG on RECONNAISSANCE: +0.5 (intelligence value of early-stage data)

---

## 7. Threat Level Computation

The composite threat level T ∈ [0, 1] is a weighted combination:

```
T = 0.45 × attack_severity
  + 0.35 × kill_chain_weight
  + 0.15 × escalation_rate
  + 0.05 × min(1, attack_count / 100)
```

| Component | Weight | Source |
|---|---|---|
| Attack severity | 45% | Per-type constant (0.0 for NORMAL → 0.90 for WORMS) |
| Kill chain stage weight | 35% | Stage constant (0.10 → 1.00) |
| Escalation rate | 15% | Fraction of attacks in last 20 steps |
| Cumulative attack count | 5% | Sustained pressure signal |

Threat bands:

| Band | Threshold |
|---|---|
| Benign | < 0.15 |
| Low | 0.15 – 0.35 |
| Medium | 0.35 – 0.55 |
| High | 0.55 – 0.75 |
| Critical | ≥ 0.75 |

---

## 8. UNSW-NB15 Dataset

The network feature distributions in HoneyIQ are inspired by the **UNSW-NB15** dataset (Moustafa & Slay, 2015), a widely used network intrusion detection benchmark containing 9 attack categories and normal traffic.

### 8.1 Feature Set

HoneyIQ uses 15 features derived from UNSW-NB15 fields:

| Feature | Description |
|---|---|
| `dur` | Flow duration (seconds) |
| `sbytes` | Source-to-destination bytes |
| `dbytes` | Destination-to-source bytes |
| `sttl` | Source IP TTL |
| `dttl` | Destination IP TTL |
| `sloss` | Source packet retransmission / drop count |
| `dloss` | Destination packet retransmission / drop count |
| `sload` | Source bits per second |
| `dload` | Destination bits per second |
| `spkts` | Source-to-destination packet count |
| `dpkts` | Destination-to-source packet count |
| `swin` | Source TCP window size |
| `dwin` | Destination TCP window size |
| `ct_srv_src` | Connections with same service and source in last 100 |
| `ct_dst_ltm` | Connections with same destination address in last 100 |

### 8.2 Per-Attack Feature Distributions

Features are sampled from parametric distributions that reflect real attack signatures:

- **Reconnaissance**: Very short duration, small packets, very high `ct_dst_ltm` (scanning many destinations)
- **DoS**: Extremely high `sload` (50k–500k bps), near-zero `dur`, massive `spkts` (Poisson λ=500)
- **Backdoors**: Long `dur` (10s–3600s), low `sload` (stealthy), balanced `swin`/`dwin`
- **Worms**: High `ct_dst_ltm` (λ=40, spreading), high `sload`, high packet counts

---

## 9. Random Forest Classifier

The attack classifier uses a **Random Forest** (Breiman, 2001) — an ensemble of decision trees trained on bootstrap samples with feature subsampling.

### 9.1 Key Properties

- **Ensemble diversity**: Each tree is trained on a bootstrap sample and selects from √n_features at each split
- **Class weighting**: `class_weight="balanced"` adjusts for potential class imbalance in generated data
- **Probability calibration**: `predict_proba` returns mean vote fractions across all trees
- **Feature scaling**: Input features are standardised with `StandardScaler` before training

### 9.2 Training Data Generation

Training data is entirely synthetic, generated from the parametric feature distributions. This eliminates the need for a real labelled dataset and allows the classifier to be re-trained on-demand with any number of samples per class (default: 600).

### 9.3 Role in the Defender

The classifier provides a **predicted attack type** at each step. This prediction is used for introspection and logging but is **not directly fed into the DQN state vector** — the DQN state is built from ground-truth environment information. In a real deployment the classifier output would replace the ground-truth labels.

---

## 10. Stage-Escalation Decision Matrix (SEDM)

The SEDM is the primary decision policy in HoneyIQ. It replaces stochastic RL with a
deterministic, interpretable decision procedure grounded in the Markov chain model.

### Algorithm

1. **Escalation risk**: Query the intent-specific transition model for P(next stage > current stage):
   ```
   esc_risk = Σ P(next_stage = s') for all s' > current_stage
   ```

2. **Band classification**:
   - Low:    esc_risk < 0.35
   - Medium: 0.35 ≤ esc_risk < 0.65
   - High:   esc_risk ≥ 0.65

3. **Matrix lookup** (7 stages × 3 bands → action):

   | Stage / Band | Low | Medium | High |
   |---|---|---|---|
   | RECONNAISSANCE | ALLOW | LOG | LOG |
   | WEAPONIZATION | LOG | LOG | TROLL |
   | DELIVERY | LOG | TROLL | TROLL |
   | EXPLOITATION | TROLL | BLOCK | BLOCK |
   | INSTALLATION | BLOCK | BLOCK | ALERT |
   | COMMAND_AND_CTRL | BLOCK | ALERT | ALERT |
   | ACTIONS_ON_OBJ | ALERT | ALERT | ALERT |

4. **Override rules**:
   - R1: NORMAL traffic → always ALLOW
   - R2: DOS or WORMS → upgrade action one level
   - R3: escalation_rate > 0.80 → upgrade action one level

5. **Composite risk score** (for logging, does not affect action):
   ```
   risk = 0.35 × stage_weight + 0.35 × esc_risk
        + 0.15 × attack_severity + 0.15 × escalation_rate
   ```

### Evaluation Results (30 episodes × 200 steps per intent)

| Intent | Detection Rate | False Positive Rate | Mean Reward |
|---|---|---|---|
| STEALTHY | 99.09% | 35.56% | 1012.22 |
| AGGRESSIVE | 99.47% | 6.67% | 1090.84 |
| TARGETED | 99.48% | 3.33% | 1127.10 |
| OPPORTUNISTIC | 99.41% | 15.00% | 896.05 |

---

## 11. State Space Encoding

The 24-dimensional state vector uses a mix of one-hot encoding and normalised continuous values:

```
[0:10]   attack_type one-hot          (10 classes)
[10:17]  kill_chain_stage one-hot     (7 stages)
[17]     threat_level                 float ∈ [0, 1]
[18]     attack_count / 100           float ∈ [0, 1]
[19]     escalation_rate              float ∈ [0, 1]
[20:24]  attacker_intent one-hot      (4 intents)
```

One-hot encoding for categorical variables avoids imposing a false ordinal relationship. Continuous features are normalised to [0, 1] to improve gradient flow.

---

## References

- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Hutchins, E. M., Cloppert, M. J., & Amin, R. M. (2011). Intelligence-driven computer network defense. *Proceedings of the 6th International Conference on Information Warfare and Security*.
- Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS*.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
