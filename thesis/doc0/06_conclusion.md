# Chapter 6 — Conclusion

## 6.1 Summary of the Work

This thesis presented HoneyIQ, a simulation framework for adaptive honeypot management in which a structured, intent-driven attacker — modelled on the Lockheed Martin Cyber Kill Chain (Hutchins et al., 2011) — is opposed by two defender policies: the Stage-Escalation Decision Matrix (SEDM) and a Deep Q-Network (DQN) baseline.

The work was motivated by three observations. First, contemporary cyber adversaries conduct structured, multi-stage campaigns that progress predictably through reconnaissance, weaponization, delivery, exploitation, installation, command-and-control, and actions on objectives. Second, honeypots provide a uniquely valuable defensive layer that can detect, observe, and deceive attackers without disrupting production systems, but their effectiveness depends on the quality of the response policy. Third, learned policies based on deep reinforcement learning — while capable of high performance — are fundamentally opaque and therefore difficult to deploy in operational security environments that require auditability and explainability.

To address these challenges, this thesis made the following contributions.

### HoneyIQ Simulation Framework

A fully self-contained, Gymnasium-compatible simulation environment implementing:

- A 24-dimensional state space combining one-hot attack type, kill chain stage, and attacker intent with continuous threat level, attack count, and escalation rate.
- A five-action honeypot policy interface (ALLOW, LOG, TROLL, BLOCK, ALERT) mapping response severity from passive passthrough to immediate escalation.
- A domain-informed reward function with three layers: a 5×5 base matrix, a late-stage amplifier for kill chain stages 4–6, and attack-type-specific bonuses encoding honeypot best practices.
- A comprehensive metrics layer computing per-step and per-episode statistics including detection rate, false-positive rate, and composite risk, with six visualisation types.

All experiments are fully reproducible without access to live network data.

### Intent-Aware Markov Attacker

A coupled pair of Discrete-Time Markov Chains governing attack-type and kill-chain-stage transitions, with four intent profiles encoded as element-wise transition modifiers:

- **Stealthy**: slow, low-severity campaigns biased toward reconnaissance and backdoor activity.
- **Aggressive**: fast, high-severity campaigns biased toward DoS, Worms, and Exploits.
- **Targeted**: focused exploitation chains progressing from Exploits through Shellcode to Backdoors.
- **Opportunistic**: scattered campaigns with moderate escalation speed and diverse attack types.

Network features are sampled from per-attack parametric distributions calibrated to UNSW-NB15 signatures, providing a bridge between the simulation and real-world traffic characteristics. The kill-chain floor constraint ensures temporal consistency by preventing unrealistic stage regression.

### Stage-Escalation Decision Matrix (SEDM)

The principal technical contribution: an interpretable, deterministic, zero-training-overhead policy implementing a five-step decision algorithm:

1. Compute escalation risk analytically from the intent-specific Markov transition matrix.
2. Classify the risk into Low, Medium, or High band.
3. Perform a 7×3 matrix lookup.
4. Apply three prioritised override rules (R1: Normal → ALLOW; R2: DOS/WORMS → upgrade; R3: high rate → upgrade).
5. Log a composite risk score for situational awareness.

The SEDM's transparency is its defining property: every decision is traceable to a specific matrix entry and override condition, enabling full audit without post-hoc explanation approximations.

### Composite Threat-Level Metric

A weighted combination (attack severity 45%, kill chain stage 35%, escalation rate 15%, cumulative count 5%) providing a normalised threat index that drives both the reward function and the SEDM's composite risk score. The weight assignment reflects domain reasoning about the relative informational value of each signal.

### DQN Baseline and Comparison

A full DQN implementation (24 → 256 → 128 → 64 → 5 with LayerNorm, experience replay, target network, Huber loss, and epsilon-greedy exploration) trained for 300 episodes on the Opportunistic intent. The DQN achieves comparable detection rates (>98.5%) to the SEDM but requires 150,000 training steps, produces an opaque policy, and exhibits undefined false-positive rate characteristics due to the dense-attack training distribution.

## 6.2 Key Findings

**Finding 1: Analytical escalation risk enables exact, training-free adaptation.** The Markov chain structure provides exact, per-intent escalation probabilities that the SEDM uses directly. This avoids the data dependency of RL-based approaches while correctly capturing the intent-specific differences in campaign tempo. The result is a policy that adapts to four qualitatively different attack profiles without any training or parameter tuning.

**Finding 2: Detection rates above 99% are achievable with a deterministic matrix.** The SEDM achieves detection rates of 99.93–100.0% across all four attacker intents under classifier-driven decisions. This performance is competitive with the DQN baseline and substantially exceeds what a random policy would achieve. The result challenges the common assumption that high detection rate requires a learned, adaptive policy.

**Finding 3: The near-zero false positive rate is a measured property of the current synthetic setup, not a general immunity claim.** The measured false positive rate is 0.00% across all four intents under classifier-driven decisions, following from the R1 override's unconditional NORMAL→ALLOW mapping combined with the classifier's near-perfect (99.85%) separation of the synthetic attack-type distributions. This is a property of the evaluation scenario's clean feature separability rather than evidence that the SEDM eliminates false positives under any conditions: a feature-noise robustness sweep shows the false-positive rate rises predictably and monotonically as the classifier's input signal is made more realistic (up to 10.00% at 50% injected feature noise), and this degradation is essentially uniform across intents rather than intent-specific.

**Finding 4: Interpretability and detection performance are compatible.** The SEDM achieves detection rates comparable to the DQN with full interpretability, zero training overhead, and superior cross-intent consistency. This finding supports the recommendation of Rudin (2019) to prefer interpretable models in high-stakes domains when the problem structure permits a good interpretable solution — and the kill chain model clearly provides such a structure.

**Finding 5: DQN specialises to the training distribution.** The DQN's effective strategy of "almost never ALLOW" achieves high detection on the Opportunistic training distribution but would not generalise to scenarios with more balanced benign/attack traffic. This specialisation is an artefact of the dense-attack training environment, not an inherent limitation of DQN, but it illustrates the importance of training distribution design for RL-based security policies.

## 6.3 Implications for Practice

The results of this thesis suggest several practical implications for honeypot management system design:

1. **Start with an interpretable policy.** The SEDM's performance, immediate operability, and auditability make it the appropriate default for initial deployment. A security team can review, understand, and modify the 7-row matrix without requiring machine learning expertise.
2. **Use the kill chain model explicitly.** The strong correlation between kill chain stage and appropriate response severity (ALLOW at Reconnaissance, ALERT at C2 and beyond) is a robust design principle that should be reflected in any honeypot policy, whether interpretable or learned.
3. **Calibrate expectations on false positives by input signal quality, not attacker profile.** Under the current evaluation, false positive rate does not vary meaningfully by attacker intent (0.00% for all four); it is driven instead by classifier input fidelity. Deployments should budget for false positives proportional to the expected noise in their feature-extraction pipeline (e.g., flow export quality, sampling rate) rather than assuming a single intent profile is inherently noisier than another.
4. **Use RL for refinement, not as a starting point.** A DQN trained alongside a deployed SEDM — using live honeypot interactions as the data source — could discover refinements to the matrix that are not immediately obvious to human designers. This staged approach separates the need for immediate interpretability (SEDM) from the potential for long-term optimisation (DQN).

## 6.4 Closing Remarks

HoneyIQ demonstrates that the mathematical structure of the Markov attacker model — specifically, the analytically computable escalation probabilities derived from intent-specific transition matrices — provides sufficient information to design a high-quality, interpretable honeypot policy without any machine learning. The Stage-Escalation Decision Matrix translates the probabilistic dynamics of the kill chain into a human-readable response protocol that aligns with honeypot best practices: engaging low-severity traffic passively, tarpitting persistent attackers for intelligence, and escalating decisively as campaigns reach critical kill-chain stages.

The simulation infrastructure provides a principled, self-contained foundation for future research. Replacing synthetic feature distributions with real data fits, re-evaluating the already-integrated classifier-driven decision pipeline under realistic feature-measurement noise, modelling an adaptive attacker who responds to the defender's actions, and extending to a multi-node network topology are natural next steps that would progressively close the gap between simulation and operational deployment.

As networked systems face increasingly sophisticated adversaries, the combination of structured attacker models, analytically-grounded decision policies, and rigorous simulation-based evaluation offers a principled path toward honeypot management that is simultaneously effective, trustworthy, and deployable in the operational environments that need it most.
