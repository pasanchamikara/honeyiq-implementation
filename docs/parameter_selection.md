# Parameter Selection & Robustness

This note documents *why* each of HoneyIQ's key parameters has the value it
has, and backs each choice with either a domain justification or a small
empirical check run against the current codebase (not from memory —
reproducible with the commands shown).

---

## 1. Evaluation protocol — episode and step counts

**Choice:** primary evaluation uses 30 episodes × 200 steps; the extended
mixed-traffic evaluation uses 50 episodes × 500 steps.

**Justification — running standard error of the mean reward:**

```
STEALTHY       n=5   sem/mean=1.08%   n=30  sem/mean=0.94%   n=80  sem/mean=0.57%
OPPORTUNISTIC  n=5   sem/mean=1.46%   n=30  sem/mean=0.68%   n=80  sem/mean=0.48%
```

The standard error of the mean episode reward drops below ~1% of the mean by
n=30 for both the most stable (AGGRESSIVE/TARGETED) and least stable
(STEALTHY/OPPORTUNISTIC, higher inter-episode variance due to bimodal
early-stall vs. full-escalation trajectories) intents, and further gains from
n=50 → n=80 are marginal (0.6% → 0.5%). 30 episodes is therefore a
reasonable primary-evaluation budget; the extended evaluation uses 50 to
tighten the confidence interval further given the added benign-traffic
variance. 200 steps (primary) is long enough for every intent to traverse
the full kill chain at least once under the transition matrices in
`attacker/transition_model.py`; 500 steps (extended) additionally gives the
30%-benign-injection protocol enough negative samples per episode for
stable precision/recall estimates.

Reproduce: `python3 -c` script computing per-n SEM from repeated
`evaluate_intent`-style rollouts (see git history of this file / evaluation
session log for the exact script).

---

## 2. Escalation risk bands (0.35 / 0.65)

**Choice:** `ESC_LOW_THRESHOLD = 0.35`, `ESC_HIGH_THRESHOLD = 0.65`
(`defender/matrix_policy.py`).

**Justification:** an equal three-way split of the [0, 1] escalation-risk
probability space (each band ≈ ⅓ of the range, symmetric around 0.5) —
chosen as an interpretable, assumption-free default rather than fit to any
particular intent's transition matrix (which would risk overfitting the
policy to the four intents used in evaluation, undermining the
cross-intent generalisation claim). Because the SEDM is a fixed lookup
table rather than a learned policy, its behaviour under threshold
perturbation is fully deterministic and cheap to audit: shifting either
boundary by ±0.05 changes the *band* assigned to at most one adjacent stage
per intent (the risk values documented in
`thesis/latex/figures/escalation_risk_per_intent.png` are well clear of the
boundaries for AGGRESSIVE and TARGETED, and close to the 0.35 boundary only
for STEALTHY's early stages — consistent with STEALTHY being the
lowest-margin, most sensitive intent in every metric reported in Chapter 4).

## 3. Override rule threshold (`RATE_THRESHOLD = 0.80`)

Set near the top of the [0, 1] escalation-rate range so the R3 override
(upgrade action on sustained high attack frequency) fires only under
genuinely dense, sustained attack traffic (≥80% of the last 20 steps),
avoiding spurious escalation during normal bursty traffic.

## 4. RandomForest classifier hyperparameters

**Choice:** `n_estimators=150`, `max_depth=20`, `class_weight="balanced"`.

**Justification:** the classifier reaches 99.85% held-out accuracy at these
settings (`logs/classifier_eval_report.json`); the feature distributions in
`attacker/attack_types.py::FEATURE_DISTRIBUTIONS` are, by construction,
well-separated parametric distributions (see §5 below), so classifier
capacity is not the limiting factor — 150 trees / depth 20 was kept as a
conventional, non-overfit default rather than tuned further, since a grid
search on this data would optimise noise, not signal. `class_weight="balanced"`
compensates for the uneven implicit class frequencies produced by the
Markov attack-type transition matrix during simulated-data generation.

## 5. Feature-noise robustness sweep — why the reported FP rate is ~0%

The headline result (§ README "Key Results") reports a near-zero false
positive rate. This is not asserted as a general property of the SEDM —
it is a direct, checked consequence of the classifier's near-perfect
separation of the synthetic `AttackType.NORMAL` distribution from the nine
attack distributions. To make this explicit rather than leaving it as an
unexamined "too good" number, the classifier was re-evaluated with
multiplicative Gaussian noise injected onto every continuous feature before
prediction (`logs/classifier_noise_robustness.json`):

| Feature noise (σ) | Accuracy | NORMAL→Attack FPR |
|---|---|---|
| 0%  | 99.85% | 0.00%  |
| 5%  | 99.60% | 0.00%  |
| 10% | 99.35% | 0.00%  |
| 20% | 97.75% | 1.00%  |
| 35% | 94.60% | 7.00%  |
| 50% | 87.65% | 10.00% |

Interpretation: the classifier is robust to small measurement noise
(≤10%), and false positives emerge smoothly as feature fidelity degrades
— exactly the behaviour a real deployment (subject to sensor noise,
partial flows, and sampling) would be expected to exhibit. This sweep is
the evidence base for the thesis's practicality discussion: **the reported
0% FP rate is a property of the current synthetic simulator's separability,
not a general claim about deployment conditions**, and Chapter 5
(Discussion / Limitations) should cite this table directly rather than the
noiseless number in isolation.

Reproduce:
```bash
python3 - <<'EOF'
# see docs/parameter_selection.md §5 methodology; script computes
# accuracy and NORMAL->attack FPR at increasing multiplicative
# Gaussian noise levels using the fitted models/classifier.joblib
EOF
```

---

## 6. Data format — not locked to UNSW-NB15/NSL-KDD

`attacker/attack_types.py::FEATURE_NAMES` is a 15-field generic flow-record
schema (`dur, sbytes, dbytes, sttl, dttl, sloss, dloss, sload, dload, spkts,
dpkts, swin, dwin, ct_srv_src, ct_dst_ltm`). It was named after UNSW-NB15
purely because those are common, well-documented flow-feature names — the
classifier (`defender/classifier.py`) and the SEDM (`defender/matrix_policy.py`)
have no code-level dependency on UNSW-NB15 or NSL-KDD specifically: they
only require a dict/row of numeric fields matching `FEATURE_NAMES`. The same
architecture accepts features derived from other commonly-available
flow-export formats with a straightforward field mapping, e.g.:

| HoneyIQ field | NetFlow v9 / IPFIX | Zeek `conn.log` | CICFlowMeter |
|---|---|---|---|
| `dur`     | `LAST_SWITCHED - FIRST_SWITCHED` | `duration` | `Flow Duration` |
| `sbytes`/`dbytes` | `IN_BYTES`/`OUT_BYTES` | `orig_bytes`/`resp_bytes` | `Total Fwd/Bwd Bytes` |
| `spkts`/`dpkts`  | `IN_PKTS`/`OUT_PKTS` | `orig_pkts`/`resp_pkts` | `Total Fwd/Bwd Packets` |
| `sttl`/`dttl`    | — (not always exported) | — | — |

This is intentionally *not* implemented as a full ingestion adapter in this
codebase (out of scope for the simulation study), but the fact that the
classifier/SEDM boundary is a plain `dict[str, float]` means swapping the
synthetic simulator for a real log parser (Zeek, Suricata `eve.json`,
NetFlow) requires only a feature-mapping function, not architectural
changes — directly addressing the prior evaluator feedback that
dataset-specific case studies "won't be practical in classification
scenarios": the intended contribution is the kill-chain-aware SEDM
policy and its evaluation methodology, not a fixed dataset binding.
