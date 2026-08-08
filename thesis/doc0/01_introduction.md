# Chapter 1 — Introduction

## 1.1 Motivation and Context

The modern networked enterprise operates in an environment of persistent, structured adversarial pressure. The global cost of cybercrime exceeded USD 8 trillion in 2023 and is projected to reach USD 10.5 trillion annually by 2025 (CISA, 2023). These figures reflect not merely the direct cost of successful intrusions — data theft, ransomware payments, system downtime — but also the growing investment required to maintain defensive infrastructure against an ever-more-sophisticated threat landscape.

Contemporary adversaries operate with strategic intent. Advanced Persistent Threat (APT) groups sponsored by nation-states, organised criminal syndicates, and increasingly automated malware ecosystems conduct multi-stage campaigns that unfold over days, weeks, or months. These campaigns follow recognisable patterns: initial reconnaissance of the target network, followed by weapon development, delivery of malicious payloads, exploitation of discovered vulnerabilities, establishment of persistent backdoors, command-and-control communication, and finally the execution of the mission objective. The Lockheed Martin Cyber Kill Chain (Hutchins et al., 2011) formalises this progression into seven sequential phases and has become the standard reference model for structuring both threat intelligence and defensive countermeasures.

### The Limitations of Static Defences

Traditional network defences are designed around the principle of known-threat identification. Signature-based Intrusion Detection Systems (IDS) such as Snort and Suricata maintain databases of known malicious patterns and alert when network traffic matches a signature. This approach is effective against catalogued threats but fundamentally blind to novel or zero-day exploits (Tavallaee et al., 2009). When an adversary deploys a new technique not yet in the signature database, the IDS produces no alert.

Fixed-threshold anomaly detectors address this limitation by flagging deviations from a statistical baseline rather than matching known patterns. However, the baseline calibration problem is severe in practice: thresholds that are sensitive enough to detect subtle intrusions tend to generate unmanageable volumes of false alarms, overwhelming security operations centre (SOC) analysts. Thresholds set conservatively to reduce false alarms miss the stealthy, low-and-slow campaigns favoured by sophisticated adversaries.

Rule-based honeypot management systems face a related challenge. A honeypot administrator must specify in advance how the system should respond to each observed traffic pattern: what level of engagement warrants logging versus blocking versus alerting. These rules are effective when the anticipated threat matches the rule designer's expectations, but adversaries adapt. An attacker who discovers that certain traffic patterns trigger immediate blocking may modify their approach to stay below the detection threshold while still achieving their objectives.

### Honeypots as Adaptive Decoys

Honeypots offer a qualitatively different defensive capability. A decoy system placed within or at the perimeter of a network attracts attacker attention by appearing to be a valuable, vulnerable target. Any interaction with the honeypot is inherently suspicious: legitimate users have no reason to access a system that serves no production purpose. This property eliminates the false positive problem for the detection *event* itself — but the question of *how to respond* to a detected interaction remains.

The response decision is non-trivial for several reasons. First, not all honeypot interactions are equally dangerous. A port scan that visits a honeypot is suspicious but low-severity; a successful exploit that installs a backdoor is critical. The response should be proportional: aggressive blocking of low-severity interactions wastes the intelligence value of honeypot engagement and may alert the attacker that they have been detected. Second, the optimal response depends on the attacker's position in the kill chain: intelligence gathering at RECONNAISSANCE stage is best served by quiet logging, while command-and-control activity at Stage 5 warrants immediate escalation. Third, the response should account for the attacker's likely *next* move: if the attacker is likely to escalate to a more dangerous stage, the defender should pre-empt that escalation rather than waiting for it to occur.

### Reinforcement Learning for Adaptive Defence

Reinforcement learning (RL) provides a principled framework for learning adaptive responses from experience (Sutton & Barto, 2018). An RL agent interacts with a simulated environment, observes the consequences of its actions, and adjusts its policy to maximise cumulative reward. Applied to honeypot management, an RL agent could potentially discover response strategies that generalise across attacker behaviours and optimally balance intelligence gathering against containment.

Deep Q-Networks (DQN) (Mnih et al., 2015) extend Q-learning to high-dimensional state spaces by using a neural network to approximate the action-value function. DQN demonstrated human-level performance on Atari games and has subsequently been applied to a range of sequential decision problems including robotic control, traffic management, and network defence (Elderman et al., 2017). However, DQN presents two significant challenges for operational cybersecurity:

1. **Opacity**: The Q-values produced by a multi-layer perceptron cannot be directly audited by a human analyst. When a DQN agent chooses to BLOCK rather than LOG a specific traffic pattern, there is no traceable explanation. Post-hoc explanation methods such as SHAP (Lundberg & Lee, 2017) provide approximations, but these are not the policy itself.
2. **Data dependency**: DQN requires a substantial corpus of interaction data before converging to a useful policy. In a real honeypot deployment, collecting sufficient labelled attack data to train a DQN without the simulation-to-reality gap is itself a major challenge.

### The Case for Interpretable Policies

In high-stakes operational domains, interpretability is not optional. Security operations centres are subject to regulatory requirements for incident reporting; the decision to block an IP address or raise a major alert must be defensible to auditors, management, and sometimes courts. A policy that cannot be explained is a policy that cannot be trusted with critical decisions.

Rudin (2019) argues forcefully that in such domains, interpretable models should be preferred over black-box models: not only are they more auditable, they are often comparably accurate when the problem structure permits a good interpretable solution. The cybersecurity kill chain provides exactly this structure: the seven-stage model defines a clear ordinal risk progression, and the Markov chain provides exact, computable escalation probabilities. These quantities are sufficient to define a transparent policy that captures the key decision dimensions without requiring opaque function approximation.

## 1.2 Problem Statement

This thesis addresses three interrelated challenges in adaptive honeypot management:

**Challenge 1: Realistic attacker simulation.** Any evaluation of a honeypot policy requires a realistic adversary. Replaying captured attack logs does not support counterfactual evaluation: if the defender changes its responses, a static replay cannot model how the attacker would adapt. A synthetic attacker must produce plausible network traffic, follow a coherent escalation logic, and support multiple qualitatively distinct intent profiles that exercise the defender across a range of threat scenarios. The attacker must be parameterisable (to allow controlled experiments) and grounded in real-world data (to maintain ecological validity).

**Challenge 2: Intent-aware adaptive defence.** A static rule base is insufficient for adaptive adversaries. An effective honeypot policy must map observable signals — current attack type, kill chain stage, escalation rate — to appropriate responses in a way that generalises across different attacker behaviours and profiles. The policy must balance *sensitivity* (responding to genuine threats) against *specificity* (avoiding false alarms on borderline traffic) across a range of threat-level distributions.

**Challenge 3: Interpretability without sacrificing effectiveness.** RL policies trained with DQN or similar algorithms achieve strong empirical performance but offer no insight into their decision-making. Operational deployment requires policies that can be inspected, audited, and understood by human practitioners without relying on post-hoc approximations. The challenge is to design a policy that is simultaneously interpretable, computationally tractable, and effective across multiple attacker profiles.

## 1.3 Research Objectives

This thesis addresses the following primary research objectives:

1. **Attacker modelling.** Design a Markov-chain attacker grounded in the Lockheed Martin Cyber Kill Chain, with per-intent transition modifiers and UNSW-NB15-inspired parametric feature distributions, capable of producing qualitatively distinct attack campaigns from a shared base model.
2. **SEDM design.** Develop the Stage-Escalation Decision Matrix (SEDM), a transparent, deterministic honeypot policy that maps kill chain stage and Markov-chain-derived escalation risk to a ranked response action, with override rules for high-impact attack types and elevated attack frequency.
3. **Threat quantification.** Define a composite threat-level metric that integrates multiple attack signals (type severity, kill chain stage, escalation rate, cumulative pressure) into a single normalised index that drives the reward function and the SEDM composite risk score.
4. **DQN baseline.** Implement a DQN-based defender as a learned baseline, characterise its training dynamics, and compare its performance properties with the SEDM in terms of detection rate, false-positive rate, and interpretability.
5. **Empirical evaluation.** Evaluate both policies using detection rate, false-positive rate, episode reward, and action distribution across all four attacker intents, and characterise the interpretability–performance trade-off in the context of operational cybersecurity requirements.

## 1.4 Principal Contributions

The principal contributions of this work are as follows:

**HoneyIQ simulation framework.** A fully self-contained attacker–defender simulation built on the Gymnasium RL interface (Towers et al., 2024). The framework implements a 24-dimensional state space combining one-hot encoded attack type, kill chain stage, and attacker intent with continuous threat level, attack count, and escalation rate. A five-action honeypot policy interface (ALLOW, LOG, TROLL, BLOCK, ALERT) is connected to a domain-informed reward function with three layers of domain knowledge. A comprehensive metrics and visualisation layer enables reproducible evaluation across all four attacker intents. All results can be reproduced without access to live network data.

**Intent-aware Markov attacker.** A coupled pair of Discrete-Time Markov Chains governing attack-type and kill-chain-stage transitions. Four intent profiles (Stealthy, Aggressive, Targeted, Opportunistic) are encoded as element-wise transition modifiers that produce qualitatively different attack campaigns from a shared base transition structure. Network features are sampled from per-attack parametric distributions calibrated to reproduce UNSW-NB15 attack signatures, providing a bridge between the simulation and real-world traffic statistics.

**Stage-Escalation Decision Matrix (SEDM).** An interpretable, deterministic, zero-training-overhead policy that: (1) computes escalation risk analytically from the intent-specific Markov transition matrix; (2) classifies the risk into three bands; (3) performs a 7×3 matrix lookup; and (4) applies three prioritised override rules. The SEDM is fully traceable from observable inputs to final action, printable on a single page, and modifiable by hand without retraining any model.

**Composite threat-level metric.** A weighted combination of attack severity (45%), kill chain stage weight (35%), escalation rate (15%), and cumulative attack count (5%) partitioned into five threat bands, driving both the reward function and the SEDM composite risk score. The weight assignment is grounded in domain reasoning about the relative informational value of each signal.

**Empirical evaluation across four intent profiles.** Across 30 evaluation episodes per attacker intent (120 total, 24,000 steps), the SEDM achieves detection rates of 99.93% (Stealthy), 100.0% (Aggressive), 99.97% (Targeted), and 99.97% (Opportunistic), with a measured false positive rate of 0.00% for all four intents under classifier-driven decisions — a consequence of the classifier's near-perfect separation of the synthetic attack-type distributions rather than a general immunity to false positives, as a feature-noise robustness sweep in Chapter 4 subsequently confirms. These results are obtained with zero training overhead and full interpretability, demonstrating that the structure of the kill chain model permits a high-quality policy to be designed analytically.

## 1.5 Scope and Limitations

The scope of this thesis is intentionally bounded to enable rigorous evaluation:

- The environment is a simulation; results are not validated against live network traffic or real honeypot deployments.
- Feature distributions are parametric approximations inspired by UNSW-NB15 rather than fits to the actual dataset records.
- The attacker's Markov chain evolves independently of the defender's actions; the defender cannot deter or redirect the attacker.
- The primary evaluation reports both an oracle mode (the SEDM decides from ground-truth attack labels, an upper bound) and a classifier-driven mode (the SEDM decides from the Random Forest classifier's predicted attack type, scored against ground truth); the classifier itself is trained and evaluated only on synthetic feature distributions.
- Network topology is abstracted; all interactions are single-session point contacts without spatial or topological structure.

These limitations are acknowledged in Chapter 5 (Discussion) and form the basis for future work directions in §5.6.

## 1.6 Thesis Outline

The remainder of this thesis is organised as follows.

**Chapter 2 (Background)** surveys the theoretical and empirical foundations of HoneyIQ: reinforcement learning and Markov Decision Processes, Deep Q-Networks, Discrete-Time Markov Chains, the Lockheed Martin Cyber Kill Chain, honeypot technology, the UNSW-NB15 dataset, Random Forests, and interpretable machine learning. Related work in RL-based intrusion response, attacker simulation, and honeypot optimisation is reviewed.

**Chapter 3 (Methodology)** describes the complete system design: the attacker module (Markov chains, intent profiles, feature simulation), the composite threat-level metric and reward function, the SEDM policy design and five-step algorithm, the DQN baseline architecture and training protocol, the Gymnasium environment, the Random Forest classifier, and the evaluation infrastructure.

**Chapter 4 (Results)** presents quantitative results from five sets of experiments: (1) SEDM cross-intent evaluation across all four attacker profiles, (2) per-intent action distribution analysis, (3) composite risk and escalation risk characterisation, (4) DQN training dynamics, and (5) parameter-selection and robustness analysis. All numerical results are derived from the experiment logs in `results/evaluation/` and `logs/`.

**Chapter 5 (Discussion)** interprets the experimental findings, analyses the interpretability–performance trade-off, examines the structural reasons for the SEDM's cross-intent consistency, discusses limitations of the simulation, reviews key design decisions, and outlines future work directions.

**Chapter 6 (Conclusion)** summarises the thesis contributions and key findings.

**Appendix A** provides the complete public API reference for each module, including function signatures and return types. **Appendix B** lists all configurable parameters used in training and evaluation, with their default values and rationale.
