# Chapter 5 — Discussion

This chapter interprets the experimental results in the context of the research objectives, analyses the interpretability–performance trade-off, discusses limitations of the simulation framework, and examines key design decisions. §5.1 analyses SEDM behaviour and the role of the Markov chain. §5.2 interprets DQN training dynamics. §5.3 addresses the interpretability–performance trade-off. §5.4 discusses limitations. §5.5 examines design choices. §5.6 outlines future work directions.

## 5.1 SEDM Performance and Markov Chain Adaptation

### Why >99% Detection is Achievable

The near-perfect detection rates across all four intent profiles are remarkable given that the SEDM has no trainable parameters and requires no labelled data. Three structural factors explain this result.

**Factor 1: Evaluation protocol.** The 200-step evaluation episodes involve continuous attack traffic; there are no extended benign periods. Because the kill chain stages visited in all four intents are predominantly Installation, C2, and Actions on Objectives (as shown by the kill chain distribution in Chapter 4, §4.4), the SEDM's matrix almost always returns BLOCK or ALERT — both of which count as detections. In other words, the high detection rate is partly a consequence of the evaluation scenario: it is easy to achieve high detection when attacks are dense and the kill chain has already progressed.

**Factor 2: Conservative matrix design.** The SEDM never returns ALLOW for stages above RECONNAISSANCE under any escalation risk band. Even in the Low band at WEAPONIZATION, the action is LOG rather than ALLOW. This conservatism ensures that any step beyond RECONNAISSANCE is responded to with at least a LOG action, which counts as a detection.

**Factor 3: Intent-adaptive escalation risk.** The Markov chain escalation risk computation (§3.4, Step 1) correctly classifies each intent's typical trajectory into the appropriate band. For STEALTHY at RECONNAISSANCE, the low escalation risk correctly returns ALLOW (or LOG in the Medium band), matching the low-threat nature of the traffic. For AGGRESSIVE at EXPLOITATION, the high escalation risk correctly returns BLOCK or ALERT, matching the high-threat nature of the traffic. The alignment between the Markov model and the evaluation scenarios means that the band classification is correct for the vast majority of steps.

### False Positive Rate: Structural Causes

Under the corrected evaluation alignment (each action scored against the ground-truth label of the observation it was actually chosen from, rather than the following step's label), the measured false positive rate is 0.00% across all four intents under classifier-driven decisions. An earlier draft of this evaluation, produced before the one-step label-lag bug (documented as Bug 8/9 in `docs/BUGS_AND_FIXES.md`) was fixed, reported false positive rates varying by a factor of ~10 across intents (3.33% for TARGETED vs. 35.56% for STEALTHY); that variation did not reflect real SEDM behaviour and has been withdrawn.

The 0.00% figure is not a claim that the SEDM is immune to false positives in general. It follows from two specific, checked properties of the current experimental setup: (1) the R1 override unconditionally maps a predicted-NORMAL sample to ALLOW, so any false positive must originate upstream in the classifier mistaking a true NORMAL sample for an attack type; and (2) the Random Forest classifier achieves 100% precision and recall on the NORMAL class specifically (99.85% accuracy overall), because the synthetic feature simulator draws each attack type from a disjoint parametric distribution by construction. Because real network telemetry does not offer this clean separability, a feature-noise robustness sweep (Chapter 4, §4.8) injects multiplicative Gaussian noise onto the classifier's input features and recovers a non-zero, monotonically increasing false-positive rate (0.00% at ≤10% noise, rising to 10.00% at 50% noise). This sweep, not the noiseless 0.00% headline figure in isolation, is the evidence that should inform expectations about false-positive behaviour against real, noisy traffic.

### Reward Ordering: TARGETED > AGGRESSIVE > STEALTHY > OPPORTUNISTIC

The reward ordering across intents (Chapter 4, Table 4.1) reflects the alignment between the SEDM's action preferences and the reward matrix.

TARGETED receives the highest reward because its concentrated exploitation path consistently generates high-threat steps that are met with ALERT (reward +6.0 for critical traffic) with very few false positives (ALLOW on benign, +1.0) or suboptimal responses (TROLL on critical, +0.5).

OPPORTUNISTIC receives the lowest reward because its scattered attack types frequently produce medium-severity steps that receive TROLL (+3.0 for medium) or BLOCK (+1.5 for medium) when ALERT would yield higher reward for critical steps. The SEDM does not see the future of the OPPORTUNISTIC campaign and calibrates to the current stage's escalation risk, which for medium-band Delivery-stage steps recommends TROLL. If the same step were at INSTALLATION with high escalation risk, ALERT would be recommended and the reward would be higher.

### Determinism vs. Stochasticity

The SEDM is fully deterministic: given the same state, it always returns the same action. This determinism is a feature for operational deployability (the policy is predictable and auditable) but could be a vulnerability against an adaptive attacker who can fingerprint the policy and craft states that trigger suboptimal responses.

The override rules (particularly R3, which responds to high attack frequency) partially mitigate this by creating a response mode that adapts to the tempo of the campaign, even if the per-step action is fixed for a given state. However, an adversary who maintains escalation rate just below 0.80 would avoid triggering R3. Future work should investigate whether small perturbations to the threshold values or the addition of a randomised component would improve robustness against adaptive adversaries.

## 5.2 DQN Training Dynamics Interpretation

### Rapid Detection Rate Improvement

The detection rate jump from 87.6% (episode 0) to 97.4% (episode 1) reflects the dense attack structure of the OPPORTUNISTIC training scenario combined with the DQN's rapid initial learning. After episode 0, the replay buffer contains 500 transitions covering a wide range of states; the first gradient update trains the network to associate non-ALLOW actions with positive rewards for attack states. This coarse association suffices to achieve >97% detection because the vast majority of states in OPPORTUNISTIC campaigns are high-threat attack states where any non-ALLOW action yields positive reward.

The implication is that the apparent high detection rate of the DQN does not reflect a nuanced understanding of the threat landscape; rather, it reflects a learned bias toward non-ALLOW responses for the dense attack distribution of OPPORTUNISTIC traffic. This interpretation is supported by the near-constant false-positive rate of 1.0 throughout training: the DQN has effectively learned to never ALLOW, which maximises reward on the training distribution but would fail on a more balanced traffic mixture.

### Loss Dynamics

The increasing loss trend through training (from ≈0.97 to ≈2.4) is not indicative of divergence. It reflects the expansion of the Q-value range as the policy learns to differentiate high-reward from low-reward state-action pairs. The Bellman target $y_i = r_i + \gamma \max_{a'} Q(s', a'; \theta^-)$ grows as the Q-function improves, causing the absolute Huber loss to increase even as the relative TD error decreases. The stable variance in the later training phase (no upward spikes after episode 100) confirms that the stabilising mechanisms (target network, gradient clipping) are effective.

### DQN Limitations in This Setting

The DQN baseline reveals two limitations specific to the HoneyIQ setting.

**Class imbalance in training data.** The OPPORTUNISTIC training episodes contain predominantly attack traffic. The DQN is never exposed to episodes with significant benign traffic, so it cannot learn to discriminate benign from malicious states. The effective strategy of "always respond with a non-ALLOW action" achieves high training reward but is not a robust policy.

**Transfer gap.** Training exclusively on OPPORTUNISTIC intent means the Q-function is calibrated to the specific attack distribution of OPPORTUNISTIC campaigns. Evaluating against STEALTHY or TARGETED would require the DQN to generalise to different threat-level time series; without intent-conditioned training or multi-intent mixed training, this generalisation is likely to be imperfect. The SEDM avoids this limitation entirely through the analytical escalation risk computation.

## 5.3 Interpretability and Performance Trade-off

### The Case for SEDM

The SEDM achieves >99% detection across all four intent profiles with zero training overhead, full transparency, and an auditable decision procedure. These properties are highly valuable for operational deployment:

- **No training data requirement.** The SEDM requires only the Markov transition model, which encodes assumed attacker behaviour. It can be deployed immediately without a historical attack corpus.
- **Analytical escalation risk.** The escalation risk is computed from the transition matrix rather than estimated from data, making it exact under the model assumptions and independent of observation noise.
- **Full traceability.** Every action can be traced to a specific matrix cell and override rule. An analyst can reproduce any decision with pencil and paper.
- **Easy modification.** Security practitioners can update the matrix entries to reflect new threat intelligence (e.g., raising the response at WEAPONIZATION for a specific campaign) without retraining any model.

### The Case for DQN

The DQN baseline offers adaptive flexibility that the SEDM cannot provide:

- **Reward-optimised responses.** Given sufficient training data and a well-designed reward function, the DQN can discover optimal action strategies that are not immediately obvious to human designers.
- **Continuous adaptation.** Online DQN training (with a sliding replay buffer) allows the policy to adapt as the attack distribution evolves, provided new transitions are added to the buffer.
- **Nuanced feature sensitivity.** In a deployment with richer state representations (e.g., including classifier probability vectors, connection metadata), the DQN can exploit fine-grained patterns that a fixed matrix cannot capture.

### Practical Recommendation

The results suggest a two-stage deployment strategy:

1. **Initial deployment**: Use the SEDM as the primary policy. Its immediate operability, interpretability, and strong baseline performance (>99% detection) make it the appropriate default.
2. **Refinement**: Train a DQN agent alongside the SEDM using live honeypot interactions. If the DQN demonstrably outperforms the SEDM on detection rate or false positive rate under rigorous evaluation, and its decisions can be explained post-hoc via SHAP or similar techniques, it can be used to augment or eventually replace the SEDM for specific intent profiles.

This strategy mirrors the human-AI collaboration model recommended by Rudin (2019) for high-stakes decision-making: start with an interpretable model, and only adopt a black-box model when there is clear evidence of superior performance and adequate explanation mechanisms.

## 5.4 Limitations

**Synthetic data only.** Feature distributions are parametric approximations designed to reproduce the qualitative signatures of UNSW-NB15 attack types. They do not capture long-range temporal correlations, protocol interactions, or the noise characteristics of real network traffic. The simulation-to-reality gap means that results cannot be directly extrapolated to live network deployments without validation on real honeypot logs.

**Defender action does not affect attacker trajectory.** In the current formulation, the attacker's Markov chain evolves independently of the defender's actions. In practice, BLOCK or ALERT responses may deter, delay, or redirect an attacker. Modelling the attacker as a best-responding agent — using multi-agent RL, game-theoretic formulations, or a secondary attacker Markov chain that conditions on the observed defender actions — would produce more realistic adversarial dynamics.

**Classifier integration is validated only on synthetic data.** The primary and extended evaluations in Chapter 4 now run the SEDM in a classifier-driven mode, deciding from the Random Forest classifier's *predicted* attack type while scoring against ground truth, alongside an oracle mode that decides from ground-truth labels directly. This confirms that the SEDM remains robust (detection ≥99.9%, classification metrics ≥0.999) under the classifier's residual misclassification rate. What remains untested is robustness under *realistic* classification noise: the classifier's near-perfect accuracy (99.85%) is itself a property of the synthetic feature simulator's clean class separability. The feature-noise robustness sweep (§4.8) shows this ceiling degrades predictably as input fidelity decreases, but the SEDM's escalation-risk computation has not been re-evaluated end-to-end under noisy classifier input — only the classifier's raw accuracy has been. Closing this gap, and eventually validating against a classifier trained on real (non-synthetic) network features, is the genuinely open item here.

**No network topology.** HoneyIQ models all interactions as single-session point contacts without spatial structure. Real network intrusions involve lateral movement, multi-hop attacks, and topological constraints that are not captured. Extending the simulation to a graph-based network topology would increase realism at the cost of significantly higher state-space complexity.

**Static Markov transition matrices.** The attacker's transition matrices are fixed for each intent and do not adapt during an episode. Real adversaries update their strategies based on the defender's responses. An adversary that observes consistent BLOCK actions on EXPLOITS might switch to a different attack type or modify its escalation speed. A dynamic Markov model whose transition matrices evolve in response to defender feedback would capture this adaptive behaviour.

**Episode termination by truncation.** Episodes terminate only when `max_steps` is reached; there is no natural termination condition corresponding to attacker success or defeat. In a real scenario, an attacker who achieves Actions on Objectives (data exfiltration, system destruction) would terminate the campaign. Modelling goal-conditioned episode termination would allow the evaluation to capture the probability of attacker success, a more operationally relevant metric than episode reward.

## 5.5 Design Choices and Trade-offs

**LayerNorm over BatchNorm.** LayerNorm normalises activations across features for each sample independently, making it batch-size-agnostic. For DQN training with mini-batches of 64 samples drawn from a diverse replay buffer, BatchNorm statistics would be computed from a heterogeneous mix of state types, potentially introducing instability. LayerNorm avoids this dependency. Empirically, no training instability attributable to normalisation was observed during the 300-episode training runs.

**Hard target network update.** A hard copy of the policy network parameters every 150 steps was chosen over a Polyak soft update ($\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$, $\tau \approx 0.005$) for implementation simplicity. Both approaches stabilise training; the hard update introduces discrete target shifts every 150 steps but avoids the hyperparameter sensitivity of choosing $\tau$. At the episode length of 500 steps, 150-step updates correspond to roughly three target updates per episode, providing stable targets over a substantial fraction of each episode.

**Composite threat-level weight assignment.** The weight assignment (attack severity 45%, kill chain stage 35%, escalation rate 15%, cumulative count 5%) was designed by domain reasoning rather than optimisation. Attack severity receives the highest weight because it is the most direct indicator of the attacker's current capability and impact. Kill chain stage receives the second-highest weight because it captures the campaign context. The 5% weight on cumulative attack count is a minor adjustment that differentiates isolated probes from sustained campaigns without dominating the threat signal.

An alternative approach would optimise these weights jointly with the reward matrix using Bayesian optimisation or multi-objective optimisation, with detection rate and false-positive rate as objectives. This is left for future work.

**SEDM band thresholds.** The Low/Medium ($\rho = 0.35$) and Medium/High ($\rho = 0.65$) thresholds were chosen to create roughly equal-width bands in the [0, 1] risk space. Intent-specific analysis confirms that the thresholds correctly separate the intent profiles: STEALTHY typically falls in the Low band, AGGRESSIVE in the High band, and TARGETED/OPPORTUNISTIC in the Medium band for most stages. Shifting the thresholds could alter the balance between false positives (lower thresholds) and false negatives (higher thresholds) for specific stages.

**Synthetic classifier training data.** Generating classifier training data from the parametric feature distributions avoids the need for a real labelled dataset and enables instant re-training with any desired class balance. The risk is classifier over-fit to synthetic data that may not generalise to real traffic. In the HoneyIQ evaluation, this risk is moot because the evaluation features are also synthetic; in a real deployment, the classifier would need to be retrained on real network data.

## 5.6 Future Work Directions

### Classifier Integration Under Realistic Noise

The SEDM's decision pipeline already supports deciding from the Random Forest classifier's predicted attack type rather than ground truth (Chapter 4), and detection remains >99.9% under this classifier-driven mode. The most immediate remaining extension is to re-run this evaluation with the classifier's input features corrupted by the same realistic measurement noise used in the feature-noise robustness sweep (§4.8), to test whether the SEDM's escalation-risk computation and downstream action distribution degrade gracefully as classifier accuracy falls from 99.85% toward the 87.65% observed at 50% feature noise. A confidence-weighted band assignment — where uncertain classifier predictions widen the escalation risk estimate — could provide a principled way to handle the resulting ambiguous observations.

### Adaptive Attacker Modelling

Modelling the attacker as a best-responding agent (e.g., using multi-agent RL or a game-theoretic Stackelberg formulation) would introduce genuine adversarial dynamics. An attacker trained against the SEDM might learn to manipulate the escalation risk computation by dwelling at low-risk stages longer than the Markov chain predicts, or by generating traffic that bypasses the override rules. Evaluating SEDM robustness against such an adaptive adversary would provide a stronger performance guarantee.

### Multi-Intent Training for DQN

Exposing the DQN to a mixture of intent profiles during training — sampling a random intent at the start of each episode — would broaden the state distribution and reduce the tendency to over-specialise on OPPORTUNISTIC traffic. An intent-conditioned DQN, where the current intent is provided as input (as it is in the 24-dimensional state vector), could be compared with the SEDM on cross-intent generalisation.

### Real Network Feature Distributions

Fitting the 15 feature distributions to the actual UNSW-NB15 records (using maximum likelihood estimation or kernel density estimation per attack category) would reduce the simulation-to-reality gap. Alternatively, a conditional generative model (e.g., a conditional VAE or normalising flow) trained on real packet data would enable richer, correlated feature generation.

### Network Topology Extension

Extending HoneyIQ to a multi-node network topology with explicit honeypot placement decisions would introduce a spatial dimension to the problem. The SEDM could be extended to consider both the kill chain stage and the network position of the attacker, enabling topology-aware response policies.

### Distributional Reinforcement Learning

Replacing the standard DQN with a distributional RL variant (e.g., C51 or QR-DQN) would allow the agent to maintain a distribution over returns rather than a point estimate. The resulting uncertainty estimates could improve decisions at threat-band boundaries, where the optimal action is least clear.

### OpenCanary Integration

HoneyIQ includes initial scaffolding for integration with OpenCanary, an open-source honeypot framework. Replaying real OpenCanary session logs through the SEDM would provide an out-of-distribution test of generalisation to genuine attack traffic, bridging the simulation-to-reality gap without requiring live deployment.
