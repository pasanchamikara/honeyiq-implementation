# HoneyIQ Thesis — Markdown Edition

This folder is a Markdown conversion of the HoneyIQ MSc thesis chapters (source LaTeX in `../latex/chapters/`). All numeric results reflect the evaluation-alignment bug fix (Bug 8/9) described below — they supersede any older PDF/notes that still cite pre-fix figures.

## Chapters

1. [`00_abstract.md`](00_abstract.md) — Thesis abstract: HoneyIQ overview, the SEDM contribution, and headline evaluation numbers.
2. [`01_introduction.md`](01_introduction.md) — Motivation, problem statement, research objectives, principal contributions, and scope/limitations.
3. [`02_background.md`](02_background.md) — Theoretical foundations: RL/MDPs, DQN, Markov chains, the Cyber Kill Chain, honeypot technology, UNSW-NB15, Random Forests, interpretable ML, and related work.
4. [`03_methodology.md`](03_methodology.md) — Full system design: attacker module, threat/reward model, the SEDM algorithm, the DQN baseline, the Gymnasium environment, the classifier, and evaluation infrastructure.
5. [`04_results.md`](04_results.md) — Quantitative results: SEDM cross-intent evaluation, action distribution, risk analysis, DQN training dynamics, extended precision/proportionality/containment metrics, and the parameter-selection/robustness analysis.
6. [`05_discussion.md`](05_discussion.md) — Interpretation of results, interpretability–performance trade-off, limitations, design-choice rationale, and future work.
7. [`06_conclusion.md`](06_conclusion.md) — Summary of contributions, key findings, and practical implications.
8. [`appendix_a_api.md`](appendix_a_api.md) — Public API reference for the attacker, defender, environment, and evaluation modules.
9. [`appendix_b_hyperparams.md`](appendix_b_hyperparams.md) — Complete hyperparameter listing, the full SEDM matrix, override rules, and the reward matrix.

## About the corrected numbers

An earlier version of the evaluation pipeline paired each defender action with the *following* time step's ground-truth label instead of the label of the observation the action was actually chosen from. This one-step lag silently corrupted false-positive, detection-rate, containment, and classification-metric figures across the thesis (while leaving reward and gross action-distribution figures largely unaffected). All chapters in this folder reflect the corrected, re-run evaluation — most visibly:

- False positive rate: was reported as varying 3.33%–35.56% across attacker intents; is now 0.00% for all four intents (a consequence of classifier/R1-override separability, not general immunity — see the feature-noise robustness sweep in Chapter 4, §4.8).
- Late-stage miss rate: was reported as ≈30% with a "camouflaging attacker" narrative; is now ≈0.000%, and that narrative has been withdrawn.
- Steps-to-containment: was 1.24–3.22 steps; is now 0.2–2.3 steps.
- Spearman correlation (stage vs. action severity): was ρ = 0.115; is now ρ = 0.492.

For the full bug narrative and before/after numbers, see [`../../docs/BUGS_AND_FIXES.md`](../../docs/BUGS_AND_FIXES.md) (Bug 8 and Bug 9 entries). For the empirical/design justification behind each protocol parameter (episode counts, escalation-risk thresholds, classifier hyperparameters, and the feature-noise robustness sweep), and the note on the feature schema's generality beyond UNSW-NB15, see [`../../docs/parameter_selection.md`](../../docs/parameter_selection.md).
