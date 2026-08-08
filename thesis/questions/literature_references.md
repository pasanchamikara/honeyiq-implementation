# Literature Supporting the Deployment Defense

This is the reading behind [`deployment_defense_questions.md`](deployment_defense_questions.md),
organised to follow the same sections rather than as a flat list. A second
pass of references, found afterwards, continues directly from this file in
[`../questions-new/literature_references_new.md`](../questions-new/literature_references_new.md).
Nothing here has been added to `thesis/latex/bibliography.bib` yet; treat
this as a shortlist to check and cite properly, since some entries are
preprints without final venue details.

## The simulation-to-reality gap

The obvious starting point is Moustafa and Slay's original UNSW-NB15
paper, since the entire feature simulator this thesis builds on traces
back to it; it's worth confirming it's actually in the bibliography given
how load-bearing it is. Two more recent papers are useful for framing how
far synthetic training data can be trusted to generalise: a 2025
neurosymbolic paper on transfer learning for robust intrusion detection
(arXiv:2506.04454), and a 2024 paper on oversampling and feature
extraction for imbalanced intrusion data (arXiv:2401.12262), which is a
reasonable parallel to the class-imbalance problem the DQN runs into when
trained only on OPPORTUNISTIC traffic (§5.2 of the discussion chapter).

- https://www.researchgate.net/publication/287330529
- https://arxiv.org/pdf/2506.04454
- https://arxiv.org/pdf/2401.12262

## Honeypots on real cloud infrastructure

For the practical side of running OpenCanary somewhere real, Palumbi's
2026 write-up of deploying and testing an OpenCanary honeypot is a decent
grounding reference for the methodology section, and Trapster's
comparison of honeypot platforms is useful for justifying OpenCanary over
something heavier like T-Pot, mainly on resource cost and interaction
depth. A 2025 paper on an LLM-based LDAP honeypot (arXiv:2509.16682) is
worth a mention in future work, since it's a recent example of a
canary-style honeypot extended with generative, adaptive responses,
which is roughly the direction a DQN-driven response layer would take
this project.

- https://medium.com/@nicholaspalumbi/active-defense-deploying-and-testing-an-opencanary-honeypot-fa1393988438
- https://trapster.cloud/en/blog/best-honeypots
- https://arxiv.org/pdf/2509.16682

## Interpretable policy versus reinforcement learning

This is the literature backing §5.3's central argument. Potteiger et
al.'s 2024 work on evolving behaviour trees for cyber defence combines
exactly the two properties the thesis argues for separately, RL's
adaptability and rule-based interpretability, so it's probably the single
strongest citation to add here (found via the Kim Hammar curated reading
list below). A 2026 paper on explainable multi-agent RL for cyber defence
speaks directly to the black-box accountability problem the thesis raises
against DQN, and a 2025 survey of explainable RL is a useful source for
framing the interpretability-performance trade-off with the current
vocabulary the field uses. Hammar and Stadler's own line of work on
optimal-stopping-based intrusion response is close in spirit to SEDM and
belongs in related work rather than only being cited for the DQN
comparison.

- https://github.com/Kim-Hammar/awesome-rl-for-cybersecurity
- https://www.sciencedirect.com/science/article/abs/pii/S0957417426016544
- https://arxiv.org/pdf/2604.04442
- https://arxiv.org/pdf/2507.12599

## Adaptive adversaries against a fixed policy

This is the reading for the fingerprinting risk the discussion chapter
already flags in §5.1 and picks back up in the "adaptive attacker
modelling" future work item (§5.6). A 2025/2026 paper on adaptive
honeypot allocation using Bayesian Stackelberg games is the most directly
relevant, since it models exactly the scenario of an attacker learning
and exploiting a defender's fixed strategy. A companion paper on
moving-target defence built on stochastic games and honeypots offers a
concrete mitigation direction, randomising rather than fixing the
response, which maps onto the future-work suggestion about perturbing
SEDM's thresholds. A 2024 survey of cyber deception techniques is good
general background, and Hammar and Stadler's 2023 paper on learning
near-optimal intrusion responses against dynamic attackers is the closest
match for modelling an attacker that best-responds to the defender, which
HoneyIQ's Markov chain currently doesn't do.

- https://arxiv.org/pdf/2505.16043
- https://www.sciencedirect.com/science/article/pii/S0167404826001252
- https://dl.acm.org/doi/10.1016/j.ins.2025.122488
- https://dl.acm.org/doi/10.1016/j.cose.2024.103792
- https://arxiv.org/pdf/2301.06085

## Ethics and legality of live data collection

If the deployment is discussed as anything beyond a pure engineering
demo, a 2025 Springer chapter on the privacy and legal implications of
processing honeypot data (covering GDPR and POPI-style obligations) is
close to essential reading before the viva, not just a nice-to-have
citation. Honeyquest's 2024 methodology for measuring how enticing a
deception technique actually is (arXiv:2408.10796) is also relevant if
the discussion touches on whether a real certificate makes the honeypot
more convincing, since it gives a way to talk about that question with
some methodological weight behind it rather than just asserting it.

- https://link.springer.com/chapter/10.1007/978-3-032-09660-9_2
- https://arxiv.org/pdf/2408.10796

## Kill-chain and Markov attacker modelling as background

For the background chapter rather than the deployment argument directly:
a 2021 ACM paper on multi-stage attack detection via kill-chain state
machines is the closest prior formalisation to compare SEDM's stage-based
matrix against. A 2024 MDPI paper on an autonomous attack response
framework (AARF) trains defence agents against a modelled attack chain
and is a reasonable comparator for the dual SEDM/DQN setup here. A 2024
paper on MITRE ATT&CK-driven decoy selection (arXiv:2404.12783) offers a
contrasting framework worth a sentence in background, and a 2024
systematic review on AI's impact on the cyber kill chain rounds this out.

- https://dl.acm.org/doi/10.1145/3474374.3486918
- https://www.mdpi.com/2227-7390/12/10/1508
- https://arxiv.org/pdf/2404.12783
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11665572/

## A caveat on where these came from

A few of the sources above (worldmetrics.org, securityhive.io,
canarytrap.com type pages) are vendor or industry blog content rather
than peer-reviewed work; they're fine for grounding the practical
deployment section but shouldn't be leaned on for the literature review's
academic argument, and they've mostly been left out of the list above for
that reason. One gap worth flagging honestly: a dedicated survey on the
simulation-to-reality gap specifically for network intrusion detection
never turned up. The transfer-learning and class-imbalance papers in the
first section are the closest available anchors, and a follow-up search
along the lines of "sim-to-real reinforcement learning cybersecurity"
would be worth doing if that gap needs a stronger citation than what's
here.
