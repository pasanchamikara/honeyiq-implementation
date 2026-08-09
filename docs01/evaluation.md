# Evaluation & Metrics

Three evaluation surfaces exist:

1. **`evaluate.py`** — the primary evaluation script: SEDM across all 4
   intents, plus an OpenCanary emulator kill-chain demo, plus every plot
   and table used elsewhere in the docs/thesis.
2. **`evaluation/sedm_eval.py`** — a deeper, classification-metrics-focused
   evaluation suite (precision/recall/F1/F2, proportionality, containment
   speed) run in two variants (oracle vs. classifier-driven).
3. **`evaluation/metrics.py`** — the shared `MetricsCollector` both use for
   per-step/per-episode bookkeeping.

## `evaluate.py`

```bash
python evaluate.py --episodes 30 --steps 200 --seed 42 --benign-ratio 0.0
```

For each of the 4 `AttackerIntent`s: runs `n_episodes` greedy episodes
through `CyberSecurityEnv` + `MatrixPolicy`, forwarding every action to a
shared `DummyHoneypot` and logging all decisions to one combined audit
JSONL. Loads the classifier **once** and shares it across all 4 intents
(rather than re-reading `classifier.joblib` from disk 4 times).

Ground-truth alignment: the action for step *t* is chosen from the
observation the defender actually saw, and scored against *that same*
observation's ground-truth label (decoded from `state`, not from the
`info` returned by the subsequent `env.step()`, which already describes
step *t+1*). This one-step lag was previously a real, twice-independently-
introduced bug (see `docs/BUGS_AND_FIXES.md`, Bug 8/9) — the fix is now
baked into how `evaluate_intent()` reads ground truth.

**Outputs** (in `results/evaluation/`): `sedm_table.csv`,
`evaluation_summary.csv`, `action_distribution.csv`, plus PNGs —
`sedm_decision_matrix.png`, `escalation_risk_per_intent.png`,
`effective_policy_per_intent.png`, `metric_comparison.png`,
`reward_boxplot.png`, `action_distribution.png`,
`composite_risk_distribution.png` — and per-intent OpenCanary audit
JSONLs from the kill-chain demo section.

## `evaluation/sedm_eval.py`

```bash
python -m evaluation.sedm_eval --episodes 50 --steps 500 --benign-ratio 0.30 --variant both
```

`--variant`:
- **`oracle`** — SEDM decides from ground-truth state directly (upper
  bound; matches `Defender.observe`'s default path).
- **`classifier`** — SEDM decides from the classifier's *predicted*
  attack type instead, so classification noise propagates into the
  policy's decisions and detection metrics stop being tautological
  (oracle mode scores the SEDM's ground-truth-driven R1 rule against the
  same ground truth it used to decide — a known pitfall documented in
  `docs/BUGS_AND_FIXES.md`).
- **`both`** (default) — runs and saves both.

Loads the classifier once per `evaluate()` call and shares it across all 4
intents (same optimization as `evaluate.py`).

Computes, per intent: precision/recall/F1/F2/specificity/FPR, a response
**proportionality score** (fraction of steps where action severity ≥ the
minimum expected severity for that kill-chain stage), a **late-stage miss
rate** (INSTALLATION+ attack steps answered with ALLOW/LOG), **steps to
first containment** (first BLOCK/ALERT), and a Spearman correlation
between kill-chain stage and mean action severity across all
intents/episodes pooled. Prints LaTeX-ready tables and saves
`logs/sedm_eval_results{,_clf}.json`.

## `MetricsCollector` (`evaluation/metrics.py`)

```python
metrics = MetricsCollector(log_dir="logs/")
metrics.record_step(episode, step, action, reward, info, predicted_attack, loss=None)
record = metrics.end_episode(episode)   # → EpisodeRecord, clears the step buffer
metrics.save_csv()
metrics.plot_training_curves()
metrics.plot_kill_chain_heatmap()
metrics.plot_attack_progression(step_records)
```

`EpisodeRecord.detection_rate` = TP / (TP + FN); `false_positive_rate` =
FP / (FP + TN), both computed from `StepRecord.is_attack` vs. whether the
action taken was ≥ LOG (a proxy for "the defender treated this as an
attack"). Every `record_step()` call reads `info` by key
(`attack_type`, `kill_chain_stage`, `threat_level`, `is_attack`,
`escalation_rate`) and extracts scalars immediately — it never holds a
reference to the `info` dict itself, so callers are free to mutate/reuse
their `info` object across steps without aliasing risk.

## Reading escalation-mode/reputation/adaptive-threshold results

None of `evaluate.py`/`sedm_eval.py` opt into `escalation_mode="ema"`,
`reputation`, or `adaptive_thresholds` by default — every number in
`results/evaluation/` and `logs/sedm_eval_results*.json` reflects the
original window-based, non-adaptive, zero-reputation behavior. To evaluate
the new mechanisms quantitatively:

- **EMA mode**: pass `escalation_mode="ema"` when constructing
  `CyberSecurityEnv` (would need a small script or a new CLI flag —
  neither `evaluate.py` nor `sedm_eval.py` currently exposes one).
- **Reputation (R4)**: `CyberSecurityEnv`/`evaluate.py`/`sedm_eval.py` have
  no cross-episode IP identity concept, so R4 can't be exercised in the
  synthetic harness as-is — it's only reachable through
  `opencanary_integration.emulator.scenario`, which does track IP
  identity across events. A quantitative before/after comparison would
  need a small harness that replays repeated attacker sessions from the
  same synthetic "IP" and measures how often R4 fires.
- **Adaptive thresholds**: construct a `MatrixPolicy` with an
  `AdaptiveThresholds` attached and inspect `.threshold` over time (see
  the verified-behavior example in
  [`dynamic_response.md`](dynamic_response.md)).

These are documented as open follow-up work, not implemented evaluation
scripts — see [`dqn_practicality.md`](dqn_practicality.md) for the related
point about not having a ground-truth signal to score R3/R4 against even
if such a harness existed.
