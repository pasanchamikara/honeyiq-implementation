# Literature Supporting the Deployment Defense, Continued

This picks up where [`../questions/literature_references.md`](../questions/literature_references.md)
left off. That first pass covered the simulation-to-reality gap, cloud
honeypot deployment, interpretability versus RL, adaptive adversaries,
ethics, and kill-chain background; this second pass fills in the gaps it
left, mainly the classifier's own literature, the Gymnasium framework the
whole environment is built on, the TLS fingerprinting question raised
directly in [`../questions/deployment_defense_questions.md`](../questions/deployment_defense_questions.md),
and a couple of papers that don't sit comfortably with the thesis's own
conclusions and are worth engaging with rather than leaving out. As
before, none of this is in `thesis/latex/bibliography.bib` yet.

## Benchmarking the classifier against other UNSW-NB15 work

Three papers are useful for putting the thesis's own RandomForest numbers
in context rather than presenting the 99.85% accuracy figure as if it
exists in isolation. Zoghi and Serpen's 2024 paper on reducing the margin
of error from data overlap and imbalance on UNSW-NB15 tackles the same
problem `defender/classifier.py` handles with `class_weight='balanced'`,
from a different angle, and is worth citing alongside the noise-robustness
sweep in §4.8. A 2025 Springer chapter evaluates RandomForest and deep
learning intrusion detection specifically inside a cloud environment,
which is a useful anchor for the deployment discussion rather than only
the offline evaluation. A 2024 paper in Algorithms reports RandomForest
figures on UNSW-NB15 (97.80% F1, 98.63% accuracy, 1.36% false alarm rate)
that are close enough to this thesis's own results to be worth a direct
comparison, showing how much of the ceiling is a property of the dataset
rather than something specific to this classifier setup.

- https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8242
- https://link.springer.com/chapter/10.1007/978-981-97-8329-8_6
- https://www.open-access.bcu.ac.uk/15332/1/algorithms-17-00064.pdf

## Citing the framework the environment is built on

This is a straightforward gap worth closing regardless of the deployment
question: `environment/cyber_env.py` is built directly on Gymnasium, and
the paper describing it (Towers et al., 2024, arXiv:2407.17032) is likely
missing from the bibliography despite being the environment's actual
foundation. PenGym, a Gymnasium-based penetration-testing training
framework, is a reasonable related-work comparator for HoneyIQ's own
environment. A 2025 paper on adversarial RL for offensive and defensive
agents in a simulated zero-sum network (arXiv:2510.05157) is worth
reading with one specific contrast in mind: its attacker best-responds to
the defender, while HoneyIQ's Markov chain deliberately doesn't, which is
already flagged as a limitation in §5.4. Citing this paper is a clean way
to show that limitation is a known, addressable gap rather than an
oversight.

- https://arxiv.org/pdf/2407.17032
- https://github.com/cyb3rlab/PenGym
- https://arxiv.org/pdf/2510.05157

## The TLS fingerprinting question, properly sourced

This is the reading behind the certificate discussion in the deployment
questions file, and probably the two most directly useful references in
either literature pass. A 2026 MDPI review of honeypot fingerprinting,
detection, and evasion techniques surveys TLS certificate fingerprinting
specifically as a detection vector, and is the strongest single citation
for the claim that a static or self-signed certificate is a known tell.
Vetterl and Clayton's paper on multistage honeypot fingerprinting is the
canonical source behind that claim: it demonstrated, at internet scale,
that hard-coded certificates and banner metadata are actually used to
identify honeypots in the wild. Between the two, this settles the
certbot-versus-self-signed question with something more solid than
intuition.

- https://www.mdpi.com/1999-5903/18/4/190
- https://dl.acm.org/doi/10.1145/3584976

## Kill chain versus MITRE ATT&CK, one more pass

Two more references for the background chapter's framing of why this
thesis uses the Lockheed Martin kill chain rather than ATT&CK. A 2023
systematisation-of-knowledge paper on the ATT&CK framework is a solid
anchor for describing what ATT&CK actually is before contrasting it with
the kill chain's linear staging. A more recent comparison paper looking
specifically at operational technology defence weighs the kill chain,
ATT&CK, and the Diamond Model against each other, which is useful if the
background chapter wants a paragraph justifying the kill chain choice
rather than just asserting it.

- https://arxiv.org/pdf/2304.07411
- https://www.researchgate.net/publication/387233519

## Reward design, as a parallel to the threat-level weighting

Two papers speak to the same design question §5.5 already discusses:
whether the composite threat-level weights (45% severity, 35% kill-chain
stage, 15% escalation, 5% count) should have been hand-designed at all, or
optimised. A 2023 ACM AISec workshop paper on reward shaping for
autonomous cyber security agents is directly relevant to that discussion.
A 2025 ESORICS workshop paper on risk-aware SOC alert handling proposes a
reward formulation built from threat criticality, confidence, and
isolation cost, which is structurally close to HoneyIQ's own composite
score and is a good citation for the future-work suggestion about
Bayesian or multi-objective weight optimisation.

- https://dl.acm.org/doi/10.1145/3605764.3623916
- https://dl.acm.org/doi/10.1007/978-3-032-16165-9_6

## Papers that argue the other side

Two references worth including precisely because they complicate the
thesis's own position rather than support it. NEMESIS, a 2025 hybrid
intrusion detection system combining deep Q-learning with a RandomForest
classifier, is structurally close to how `defender/defender.py`
orchestrates its own classifier and DQN, and is a strong direct comparator
for related work. More pointedly, ARCS, a 2025 adaptive RL framework for
cybersecurity incident response, explicitly benchmarks itself against
"traditional rule-based approaches" and reports faster resolution times
and higher effectiveness for the learned policy. That's a real
counter-argument to §5.3's case for SEDM, and it's better to address it
directly in the discussion chapter than to let an examiner raise it first.

- https://www.sciencedirect.com/science/article/pii/S1877050925035446
- https://www.mdpi.com/2076-3417/15/2/951

## Cloud and digital-twin honeypots, for the deployment framing

One more reference for the deployment discussion specifically: TwinFedPot,
a 2025 paper on distilling honeypot intelligence into a digital twin
model, is a recent example of honeypot deployment feeding a live,
continuously updated model rather than a one-off static evaluation. It's
a useful way to frame the Oracle Cloud deployment as a plausible first
step toward retraining the classifier on real data eventually, which
connects back to the "synthetic classifier training data" limitation
already named in §5.4 and §5.5.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC12349280/

## What to prioritise

If only a few of these get added before the viva, the Gymnasium citation
is the easiest win since it plugs an actual gap rather than adding
colour, the Vetterl and Clayton fingerprinting paper is the one that
directly answers a question likely to be asked out loud, and the ARCS
paper is the one that shouldn't be quietly skipped, since it argues
against the thesis and needs a considered response rather than silence.
