# Chapter 2 — Background and Related Work

This chapter surveys the theoretical and empirical foundations of HoneyIQ. §2.1 introduces reinforcement learning and Markov Decision Processes. §2.2 covers Deep Q-Networks. §2.3 presents Discrete-Time Markov Chains. §2.4 describes the Lockheed Martin Cyber Kill Chain. §2.5 reviews honeypot technology. §2.6 introduces the UNSW-NB15 dataset. §2.7 covers Random Forests. §2.8 discusses interpretable machine learning. §2.9 reviews related work.

## 2.1 Reinforcement Learning

Reinforcement learning (RL) is a computational framework in which an *agent* learns to make decisions through interaction with an *environment* (Sutton & Barto, 2018). Unlike supervised learning, the agent receives no labelled examples; instead, it receives a scalar *reward* signal $r_t$ after each action, and its objective is to discover a *policy* $\pi$ that maximises the expected cumulative discounted return:

$$G_t = \sum_{k=0}^{\infty} \gamma^k \cdot r_{t+k+1}$$

where $\gamma \in [0, 1)$ is the *discount factor* controlling the relative weight of immediate versus future rewards. In HoneyIQ, $\gamma = 0.99$, reflecting the importance of long-term campaign containment over myopic step-by-step responses.

### Markov Decision Process Formalisation

The interaction is formalised as a Markov Decision Process (MDP) $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:

- $\mathcal{S}$ — the *state space*. In HoneyIQ this is a continuous 24-dimensional observation space combining one-hot attack type, one-hot kill chain stage, threat level, attack count, escalation rate, and attacker intent.
- $\mathcal{A}$ — the *action space*: five discrete honeypot responses (ALLOW, LOG, TROLL, BLOCK, ALERT).
- $P(s' \mid s, a)$ — the *transition dynamics*. In HoneyIQ, the transition is determined by the attacker's Markov chain, which advances independently of the defender's action. This reflects the assumption that the attacker cannot observe the defender's specific response in real time.
- $R(s, a, s')$ — the *reward function* encoding domain knowledge about correct honeypot behaviour (§3.3).
- $\gamma = 0.99$ — discount factor.

The Markov property states that the future is conditionally independent of the past given the present state: $P(s_{t+1} \mid s_t, a_t, \ldots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)$. This holds exactly in HoneyIQ because the 24-dimensional state vector encodes all information relevant to predicting the next state.

### Value Functions and the Bellman Equation

The *state-value function* $V^\pi(s)$ gives the expected return from state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \,\Big|\, s_t = s\right]$$

The *action-value function* $Q^\pi(s, a)$ extends this to state-action pairs:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \,\Big|\, s_t = s,\, a_t = a\right]$$

The optimal action-value function $Q^*$ satisfies the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}\!\left[r + \gamma \cdot \max_{a'} Q^*(s', a') \,\Big|\, s, a\right]$$

The optimal policy is $\pi^*(s) = \arg\max_a Q^*(s, a)$.

### Q-Learning

Q-learning (Watkins & Dayan, 1992) estimates $Q^*$ through incremental updates from observed transitions $(s, a, r, s')$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\!\left[r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)\right]$$

where $\alpha \in (0, 1]$ is the learning rate. Under mild technical conditions, Q-learning converges to $Q^*$ in the tabular setting, but the table grows exponentially with the size of the state space, making it infeasible for continuous or high-dimensional observations.

## 2.2 Deep Q-Networks (DQN)

DQN (Mnih et al., 2015) resolves the scalability limitation by approximating $Q(s, a; \theta)$ with a neural network parameterised by $\theta$. The seminal work demonstrated human-level performance on 49 Atari games using raw pixel inputs, establishing DQN as a practical algorithm for large state spaces.

### Neural Network Approximator

In HoneyIQ, the policy network is a three-hidden-layer multi-layer perceptron:

$$\text{Input}(24) \xrightarrow{\text{Linear+LayerNorm+ReLU}} 256 \xrightarrow{\text{Linear+LayerNorm+ReLU}} 128 \xrightarrow{\text{Linear+LayerNorm+ReLU}} 64 \xrightarrow{\text{Linear}} \mathbb{R}^5$$

The network receives a 24-dimensional state vector and outputs five Q-values, one per honeypot action. Layer normalisation (Ba et al., 2016) is applied after each linear layer; it normalises activations across features rather than across the batch, making it appropriate for small replay-buffer mini-batches and environments where the input distribution shifts during training.

### Experience Replay

Consecutive environment transitions are highly correlated, which violates the independent and identically distributed (i.i.d.) assumption that underlies stochastic gradient descent. DQN addresses this by storing transitions $(s, a, r, s', \text{done})$ in a *replay buffer* (circular deque of capacity 15,000) and sampling uniformly at random for each gradient update. This decorrelates the training distribution and allows each transition to contribute to multiple updates.

### Target Network

If the same network provides both the Q-value predictions and the Bellman targets, the targets shift with every gradient step, leading to oscillation and divergence. DQN introduces a *target network* $Q(s, a; \theta^-)$ whose parameters are periodically synchronised from the policy network:

$$y_i = r_i + \gamma \cdot \max_{a'} Q(s'_i, a'; \theta^-)$$

In HoneyIQ, $\theta^-$ is hard-copied from the policy network every 150 steps, providing target stability over a window that spans multiple episodes given the 200-step episode length.

### Huber Loss and Gradient Clipping

The temporal-difference (TD) error is minimised with the Huber (SmoothL1) loss:

$$\mathcal{L}(\delta) = \begin{cases} \tfrac{1}{2}\delta^2 & |\delta| \le 1 \\ |\delta| - \tfrac{1}{2} & \text{otherwise} \end{cases}, \quad \delta = y_i - Q(s_i, a_i; \theta)$$

Huber loss is quadratic near zero (matching MSE for small errors) and linear for large errors (making it less sensitive to outlier Q-values than MSE). Gradients are additionally clipped to a maximum $\ell_2$ norm of 10.0 to prevent explosive updates in early training.

### Epsilon-Greedy Exploration

DQN uses an $\varepsilon$-greedy policy: with probability $\varepsilon$ the agent takes a random action (exploration); otherwise it selects the action with the highest estimated Q-value (exploitation). $\varepsilon$ is annealed exponentially:

$$\varepsilon_{t+1} \leftarrow \max\!\left(\varepsilon_{\min},\; \varepsilon_t \cdot d\right), \quad d = 0.997,\quad \varepsilon_{\min} = 0.05$$

Starting from $\varepsilon = 1.0$, the agent explores almost uniformly for the first ≈100 episodes before transitioning to predominantly exploitative behaviour. This warm-up period is essential: the replay buffer must accumulate diverse transitions before the Q-value surface is well-defined.

### Limitations of DQN in Security Contexts

Despite its empirical success, DQN presents two significant challenges for operational cybersecurity deployment.

First, **opacity**: the Q-values produced by a multi-layer perceptron are not directly interpretable by a human analyst. When a DQN agent chooses BLOCK over LOG for a given traffic pattern, it is impossible to trace the decision to a human-readable rule without post-hoc explanation techniques (Lundberg & Lee, 2017), which introduce their own approximation errors.

Second, **training instability**: the interaction of function approximation, bootstrapping (the TD target depends on the same function being updated), and off-policy data can produce oscillating or diverging Q-value estimates (van Hasselt et al., 2016). While the stabilising mechanisms (replay buffer, target network, gradient clipping) substantially mitigate this in practice, convergence is not guaranteed and must be verified empirically.

These limitations motivate the SEDM as an alternative policy design (§3.4).

## 2.3 Markov Chains

A Discrete-Time Markov Chain (DTMC) (Norris, 1998) is a stochastic process $\{X_t\}_{t \geq 0}$ over a finite state space $\mathcal{X}$ that satisfies the Markov property:

$$P(X_{t+1} = j \mid X_t = i,\, X_{t-1} = i_{t-1},\, \ldots) = P(X_{t+1} = j \mid X_t = i) \eqqcolon T_{ij}$$

The $|\mathcal{X}| \times |\mathcal{X}|$ matrix $\mathbf{T}$ of transition probabilities is *row-stochastic*: $T_{ij} \geq 0$ and $\sum_j T_{ij} = 1$ for all $i$.

In HoneyIQ, the attacker's behaviour is modelled with two parallel DTMCs:

- $\mathbf{T}_A$ (10×10): attack-type transitions, governing movement between the ten attack categories.
- $\mathbf{T}_K$ (7×7): kill-chain-stage transitions, governing progression through the seven kill chain stages.

Intent-specific modifier matrices $\mathbf{M}_A^{(\pi)}$ and $\mathbf{M}_K^{(\pi)}$ are applied element-wise to the base matrices before row-renormalisation:

$$\tilde{T}^{(\pi)}_{ij} = T_{ij} \cdot M^{(\pi)}_{ij}, \qquad T^{(\pi)}_{ij} = \frac{\tilde{T}^{(\pi)}_{ij}}{\sum_k \tilde{T}^{(\pi)}_{ik}}$$

This design separates the *shared attack grammar* (encoded in the base matrices) from the *intent-specific tendency* (encoded in the modifiers), allowing four qualitatively different attack campaigns to arise from the same underlying model.

The SEDM exploits the Markov chain structure directly: the escalation risk at stage $k$ under intent $\pi$ is:

$$\rho(k, \pi) = \sum_{k' > k} T^{(\pi)}_{kk'}$$

the probability that the attacker advances to a strictly higher kill-chain stage in the next step. This quantity is computed analytically from the transition matrix rather than estimated from trajectory data, making the SEDM's risk assessment exact under the model.

## 2.4 Lockheed Martin Cyber Kill Chain

The Cyber Kill Chain (Hutchins et al., 2011) models an adversary's campaign as a sequential progression through seven phases (Table 2.1).

**Table 2.1 — Cyber Kill Chain stages, associated attack types in HoneyIQ, and kill-chain weight values.**

| Index | Stage | Weight | Primary Attack Types |
|---|---|---|---|
| 0 | Reconnaissance | 0.10 | NORMAL, RECONNAISSANCE |
| 1 | Weaponization | 0.20 | ANALYSIS |
| 2 | Delivery | 0.35 | FUZZERS, GENERIC |
| 3 | Exploitation | 0.55 | EXPLOITS, SHELLCODE |
| 4 | Installation | 0.70 | BACKDOORS |
| 5 | Command & Control | 0.85 | WORMS |
| 6 | Actions on Obj. | 1.00 | DOS |

Originally derived from military targeting doctrine, the framework has become a standard reference model for structuring threat intelligence and mapping defensive countermeasures.

**Reconnaissance (Stage 0).** The attacker observes the target network: port scanning, service enumeration, and open-source intelligence (OSINT) gathering. In HoneyIQ, this maps to RECONNAISSANCE attack type with short-duration, high-connection-count traffic patterns. The stage weight of 0.10 reflects that reconnaissance alone does not cause direct harm.

**Weaponization (Stage 1).** The attacker crafts an exploit or malware tailored to discovered vulnerabilities. ANALYSIS traffic (e.g., fuzzing sub-threshold probes) characterises this phase.

**Delivery (Stage 2).** The weapon is transmitted to the target via phishing, drive-by download, or network exploitation. FUZZERS and GENERIC attack types with elevated packet rates represent delivery-phase traffic in HoneyIQ.

**Exploitation (Stage 3).** The delivered exploit executes code on the target system. EXPLOITS and SHELLCODE attacks, characterised by high severity and specific connection patterns, represent this phase. The stage weight (0.55) reflects the significant danger of active code execution.

**Installation (Stage 4).** A persistent mechanism (backdoor, rootkit) is installed to maintain access. BACKDOORS in HoneyIQ exhibit long-duration, low-bandwidth connections characteristic of covert persistence channels.

**Command and Control (Stage 5).** The attacker establishes a command channel to issue instructions and exfiltrate data. WORMS represent C2 activity in HoneyIQ, with high connection counts to multiple destinations.

**Actions on Objectives (Stage 6).** The attacker executes the final mission objective: data exfiltration, destruction, or disruption. DOS attacks with extreme packet rates characterise this stage. The stage weight of 1.00 reflects maximum threat.

**Kill Chain Constraint.** The Lockheed Martin Kill Chain is a forward-progressing model. HoneyIQ enforces a floor constraint to prevent unrealistic stage regression: the sampled next stage is bounded below by the attack type's primary stage minus one. This models the attacker's ability to re-use lower-stage techniques while ensuring they cannot regress arbitrarily.

**MITRE ATT&CK.** The MITRE ATT&CK framework (Strom et al., 2018) provides a complementary view at a finer granularity, cataloguing specific techniques within each kill chain phase. HoneyIQ's attack-type taxonomy maps to MITRE ATT&CK at the category level: RECONNAISSANCE corresponds to the Discovery tactic; EXPLOITATION to the Execution and Privilege Escalation tactics; WORMS to the Command and Control tactic.

## 2.5 Honeypot Technology

A honeypot is an intentionally vulnerable decoy system designed to attract attackers and study their behaviour without risk to production assets (Spitzner, 2003; Cheswick, 1992).

### Taxonomy

Honeypots are commonly classified along two dimensions:

**Interaction level.** *Low-interaction* honeypots emulate limited services (e.g., an SSH banner) with minimal actual functionality, reducing risk but limiting intelligence value. *High-interaction* honeypots run full operating systems and services, capturing rich attacker behaviour but requiring careful isolation to prevent compromise of production infrastructure.

**Deployment purpose.** *Research* honeypots aim to understand attacker techniques and gather threat intelligence. *Production* honeypots are deployed within live networks to detect intrusions and divert attackers.

### Tarpitting and Deception

A distinctive honeypot capability is the ability to actively deceive the attacker rather than passively observe (Provos & Holz, 2004). Tarpitting (the TROLL action in HoneyIQ) involves:

- Responding with synthetic, plausible-but-incorrect data (fake credentials, bogus file systems, fabricated network topology).
- Introducing artificial delays to waste attacker time and resources.
- Feeding false intelligence that misdirects subsequent attack steps.

Tarpitting is most valuable against persistent attack types (backdoors, worms, shellcode) where the attacker maintains a long-duration session and can be engaged for extended observation.

### Response Actions in HoneyIQ

HoneyIQ abstracts honeypot management into five actions that span a spectrum from passive to aggressive:

1. **ALLOW**: pass traffic without logging or intervention. Appropriate for confirmed benign traffic to avoid false alarms.
2. **LOG**: record the session for later analysis without alerting the attacker. Appropriate for low-severity, early-stage reconnaissance.
3. **TROLL**: engage with fake responses to maximise intelligence while wasting attacker resources. Appropriate for medium-severity attacks where session duration is valuable.
4. **BLOCK**: terminate the connection via firewall rules. Appropriate for high-severity attacks that should not be permitted to continue.
5. **ALERT**: issue an immediate high-priority security alert for human analyst review. Appropriate for critical-severity attacks at late kill-chain stages.

### Game-Theoretic Perspectives

The honeypot management problem has been studied from a game-theoretic perspective (Shi et al., 2021), modelling it as a Stackelberg or Nash equilibrium problem between attacker and defender. These approaches provide formal optimality guarantees but require explicit modelling of the attacker's utility function, which is typically unknown in practice. RL and decision-matrix approaches avoid this requirement by learning or designing responses from simulated interactions.

## 2.6 UNSW-NB15 Dataset

The UNSW-NB15 benchmark (Moustafa & Slay, 2015) was created at the Australian Centre for Cyber Security to overcome the limitations of the widely-criticised KDD Cup 99 dataset (Tavallaee et al., 2009). It contains network flow records captured from a testbed environment running nine attack categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) alongside normal traffic, totalling approximately 2.5 million records with 49 features.

### Feature Set

The 49 UNSW-NB15 features cover five categories:

- **Flow-level features**: duration (`dur`), bytes sent/received (`sbytes`, `dbytes`), packets sent/received (`spkts`, `dpkts`), load in bits/s (`sload`, `dload`).
- **IP-level features**: time-to-live values (`sttl`, `dttl`), packet loss counts (`sloss`, `dloss`).
- **TCP-level features**: window sizes (`swin`, `dwin`).
- **Connection-level features**: `ct_srv_src` (connections to the same service from the same source), `ct_dst_ltm` (connections to the same destination in the last time interval).
- **Content features**: protocol, service, state.

HoneyIQ uses 15 of these features, selected for their discriminatory power across attack categories. Rather than fitting distributions to the actual UNSW-NB15 records, the simulation samples from *parametric approximations* that reproduce the qualitative signatures of each attack type (e.g., high `sload` for DoS, long `dur` for Backdoors).

### Attack Signatures in HoneyIQ

**Table 2.2 — Characteristic UNSW-NB15 feature patterns per attack type in HoneyIQ.**

| Attack Type | Distinctive Feature Pattern |
|---|---|
| NORMAL | Short `dur`, moderate balanced `sbytes`/`dbytes` |
| RECONNAISSANCE | High `ct_dst_ltm` (λ=30), many short connections |
| ANALYSIS | Moderate `sload`, medium `ct_srv_src` |
| FUZZERS | High `spkts` (Poisson λ=100), varied payloads |
| GENERIC | Balanced traffic, similar to normal but higher packet counts |
| EXPLOITS | High `sbytes`, spike in `sload` (10k–100k bps) |
| SHELLCODE | Low `dur`, very high `sload`, small payloads |
| BACKDOORS | Long `dur` (10–3600 s), low stealthy `sload` |
| DOS | Extreme `sload` (50k–500k bps), near-zero `dur` |
| WORMS | High `ct_dst_ltm` (λ=40), spreading pattern |

## 2.7 Random Forests

A Random Forest (Breiman, 2001) is an ensemble of decision trees, each trained on a bootstrap sample of the training data with a randomly selected subset of features considered at each split. Aggregating predictions across many uncorrelated trees reduces variance without increasing bias, yielding classifiers that are robust to overfitting and effective on high-dimensional data with mixed feature types.

In HoneyIQ, the Random Forest classifier serves as a supporting component for attack-type identification:

- **Training data**: 6,000 synthetic samples (600 per class) generated from the parametric feature distributions.
- **Class balance**: `class_weight='balanced'` reweights the contribution of each sample inversely proportional to its class frequency, preventing the majority class from dominating gradients.
- **Feature normalisation**: `StandardScaler` zero-centres and unit-scales each feature before training and inference.
- **Configuration**: 150 trees, maximum depth 20.

Because training data is generated directly from the feature distributions, the classifier has an inherent advantage over real-world settings: there is no distribution shift between training and evaluation data. High accuracy (>95% on held-out synthetic data) is therefore expected, and the classifier primarily serves as a preprocessing step for logging and introspection rather than as the primary decision-making mechanism.

> **Note (corrected results, Chapter 4):** the current held-out accuracy of the classifier is 99.85% (10-class), well above the ">95%" figure anticipated in this background chapter, and the SEDM's evaluation now runs in a classifier-driven decision mode rather than treating the classifier purely as a logging/introspection component — see §3.7 and Chapter 4.

## 2.8 Interpretable Machine Learning

The past decade has seen rapid progress in predictive performance of neural networks at the cost of reduced interpretability. In high-stakes domains such as healthcare, criminal justice, and cybersecurity, this trade-off is often unacceptable (Rudin, 2019).

### Motivation for Interpretability in Security

Cybersecurity deployments have specific interpretability requirements:

- **Auditability**: security operations centres (SOCs) must be able to explain to auditors and regulators why a particular IP was blocked or why an alert was raised.
- **Trust and override**: analysts must be able to identify when a policy is making systematic errors and override its decisions.
- **Debugging**: when a policy allows a known attack (false negative) or blocks benign traffic (false positive), the responsible decision rule should be identifiable and correctable.
- **Adversarial robustness**: opaque policies are potentially vulnerable to adversarial inputs crafted to manipulate neural network activations; interpretable policies are more resistant because their logic is explicit.

### Decision Tables as Interpretable Policies

A decision table maps a discretisation of the input space to actions. It is interpretable by construction: any decision can be traced to a specific cell in the table and the associated input conditions. Decision tables have been used in medical diagnosis, industrial control systems, and credit scoring. In the security domain, they underpin classic IDS rule engines such as Snort.

The SEDM in HoneyIQ is a 7×3 decision table (stage × band) with three supplementary override rules. This structure is small enough to be printed on a single page, audited by a practitioner in minutes, and modified without retraining.

### Trade-offs

Decision tables achieve interpretability by discretising a continuous space. This discretisation introduces step discontinuities at band boundaries: two states with escalation risks of 0.349 and 0.351 receive different base actions (Low vs. Medium band), even though the underlying risk is nearly identical. The override rules partially mitigate this by providing additional conditions that can upgrade the action regardless of the band boundary. Chapter 5 (Discussion) examines this trade-off empirically.

## 2.9 Related Work

### RL-Based Intrusion Response

Several studies have applied reinforcement learning to network intrusion response. Malialis & Kudenko (2015) applied Q-learning agents to distributed denial-of-service (DDoS) mitigation, demonstrating that independent per-router agents could collectively throttle attack traffic without centralised coordination. Elderman et al. (2017) studied adversarial RL in a network intrusion simulation, showing that a defender agent trained against a fixed attacker degrades when the attacker adapts. Hammar & Stadler (2021) framed intrusion prevention as an optimal stopping problem, using RL to determine when to block a suspicious connection. Li et al. (2019) combined multi-agent RL with inverse reinforcement learning to infer attacker utility functions and design countervailing responses.

HoneyIQ extends this body of work by (1) modelling attacker intent as a latent variable that governs the Markov transition structure, enabling evaluation of policy robustness across qualitatively distinct campaigns, and (2) introducing the SEDM as an interpretable alternative to learned policies that achieves comparable detection performance.

### Attacker Simulation

Realistic attacker simulation is a prerequisite for evaluating any defensive policy (Applebaum et al., 2017). Agent-based simulators such as CyberBattle Gym and CAGE Challenge environments model network topologies and service vulnerabilities explicitly. HoneyIQ abstracts topology to focus on the temporal dynamics of attack progression, using the Markov chain to produce plausible kill-chain sequences without requiring a specific network graph. The UNSW-NB15-inspired feature distributions provide a bridge to real-world traffic statistics while remaining computationally tractable.

### Honeypot Optimisation

Shi et al. (2021) modelled honeypot placement as a bilevel optimisation problem and used RL to find strategies that maximise the attacker's deception cost. Franco et al. (2021) provided a comprehensive survey of honeypot architectures and their deployment in production environments. The HoneyIQ action abstraction (ALLOW / LOG / TROLL / BLOCK / ALERT) captures the essential operational decisions without committing to a specific honeypot technology, making the framework applicable to a range of deployment contexts.

### Partially Observable Defence

Miehling et al. (2018) studied optimal defence under partial observability using a POMDP framework with Bayesian attack graphs. HoneyIQ's primary and extended evaluations (Chapter 4) report both an oracle mode, treating the attacker's state (attack type, kill chain stage, intent) as fully observable via the ground-truth state vector, and a classifier-driven mode that decides from the Random Forest classifier's predicted attack type instead, introducing a bounded form of partial observability. Re-evaluating this classifier-driven mode under realistic (non-synthetic) classification noise remains a priority direction for future work (§5.6).

> **Editorial note on this chapter:** two passages in the original chapter (§2.6, §2.9's closing paragraph) described the classifier's role and accuracy using language written before the classifier-driven evaluation mode existed and before the corrected evaluation pipeline was re-run (see Chapter 4 and `docs/BUGS_AND_FIXES.md`, Bug 8/9). The notes above adjust the reading of those passages for consistency with the corrected results without rewriting the original prose; see `docs/parameter_selection.md` for full technical detail.
