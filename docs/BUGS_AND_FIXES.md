# HoneyIQ — Bug Report & Fix Log

Date: 2026-04-25  
Branch: `rule-based-matrix-policy`

---

## Bug 1 — Dependencies not installed (Critical)

**File:** `venv/` (environment state)  
**Severity:** Critical — blocks all execution

### Description
The virtual environment contained only `pip 24.0`. None of the packages listed in `requirements.txt` were installed:
- `gymnasium`, `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `joblib`

Every import in the codebase fails with `ModuleNotFoundError`.

### Root cause
The venv was created but `pip install -r requirements.txt` was never run.

### Fix
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Bug 2 — `defender/__init__.py` imports stale DQN symbols (High)

**File:** `defender/__init__.py`, line 2  
**Severity:** High — imports `torch` unconditionally on every use of the `defender` package

### Description
```python
from .dqn import DQNAgent, DQNNetwork, ReplayBuffer  # ← stale after DQN→SEDM migration
```

After the architecture was migrated from DQN to SEDM, the `dqn.py` imports were left in `__init__.py`. Since `dqn.py` imports `torch` at the module level, any code that does `from defender.defender import Defender` (or any other defender submodule import) triggers `defender/__init__.py`, which in turn imports `torch`. This:

1. Adds a multi-gigabyte `torch` dependency to pure-evaluation code paths that don't use the DQN at all.
2. Causes `ImportError` if `torch` is absent from the environment.
3. Contradicts the architectural decision to replace the DQN with the interpretable SEDM.

### Fix
Remove `DQNAgent`, `DQNNetwork`, `ReplayBuffer` from `defender/__init__.py` exports. They are dead exports — nothing in the active code path (`evaluate.py`, `opencanary_integration/`) uses them.

---

## Bug 3 — `train.py` references non-existent `defender.dqn_agent` attribute (High)

**File:** `train.py`, line 162  
**Severity:** High — `AttributeError` on every training run

### Description
```python
print(f"Device: {defender.dqn_agent.device}")
```

The `Defender` class was refactored to remove the DQN agent (`dqn_agent` attribute). It now only holds `self.classifier` (RandomForest) and `self.matrix_policy` (SEDM). This line raises:

```
AttributeError: 'Defender' object has no attribute 'dqn_agent'
```

### Fix
Remove the line. The SEDM policy is device-agnostic (pure Python/numpy), so there is no device to report.

---

## Bug 4 — `session_tracker.py` uses deprecated `datetime.utcnow()` (Medium)

**File:** `opencanary_integration/engine/session_tracker.py`, lines 69 and 84  
**Severity:** Medium — `DeprecationWarning` in Python 3.12+, may error in future Python versions

### Description
```python
session.last_seen = datetime.utcnow()   # line 69
cutoff = datetime.utcnow() - self._ttl  # line 84
```

`datetime.utcnow()` was deprecated in Python 3.12 (PEP 657). It returns a naive datetime without timezone info, making comparisons with timezone-aware datetimes fail. The correct replacement is `datetime.now(timezone.utc)`.

### Fix
Replace both calls with `datetime.now(timezone.utc)` and add `timezone` to the import.

---

## Bug 5 — `evaluation/metrics.py` uses deprecated `plt.cm.get_cmap()` (Low)

**File:** `evaluation/metrics.py`, line 374  
**Severity:** Low — `MatplotlibDeprecationWarning` in matplotlib 3.7+

### Description
```python
cmap_atk = plt.cm.get_cmap("tab10", AttackType.count())
cmap_act = plt.cm.get_cmap("Set1", HoneypotAction.count())
```

`plt.cm.get_cmap()` was deprecated in matplotlib 3.7. The replacement is `matplotlib.colormaps[name].resampled(n)`.

### Fix
Use the new API: `matplotlib.colormaps["tab10"].resampled(AttackType.count())`.

---

## Bug 6 — `evaluate.py` creates a new `MetricsCollector` on every episode (Medium)

**File:** `evaluate.py`, line 291 (inside `evaluate_intent` episode loop)  
**Severity:** Medium — wasteful but does not corrupt results

### Description
```python
for ep_idx in range(n_episodes):
    metrics = MetricsCollector(log_dir=out_dir)   # ← new collector every episode
    ...
    rec = metrics.end_episode(ep_idx)
```

`MetricsCollector.__init__` calls `os.makedirs` and initialises data structures on every iteration. The collector is only used for its `record_step` / `end_episode` flow and then discarded. While the primary evaluation data (rewards, detection rates) is captured in the `IntentResult` object outside the loop, the repeated instantiation is wasteful and means no cross-episode metrics accumulation occurs inside the collector.

### Fix
Move the `MetricsCollector` instantiation outside the episode loop (create once per intent, not once per episode).

---

## Bug 7 — `pydantic` dependency missing from `requirements.txt` (Critical)

**File:** `requirements.txt`, `opencanary_integration/ingest/models.py`  
**Severity:** Critical — `ModuleNotFoundError` on import, blocks OpenCanary emulator

### Description
`opencanary_integration/ingest/models.py` imports:
```python
from pydantic import BaseModel, field_validator
```

But `pydantic` was not listed in `requirements.txt`. Any code path that touches the OpenCanary emulator (including `evaluate.py` via `DummyHoneypot` and `EmulatorScenario`) fails with:

```
ModuleNotFoundError: No module named 'pydantic'
```

This is triggered because `opencanary_integration/emulator/__init__.py` eagerly imports `OpenCanaryEventGenerator`, which imports `OpenCanaryEvent` from `models.py`.

### Fix
Add `pydantic>=2.0.0` to `requirements.txt` and install it.

---

## Bug 8 — One-step lag corrupts FP/FN metrics in `evaluate_intent` (High)

**File:** `evaluate.py`, `evaluate_intent()` step loop  
**Severity:** High — produces deeply misleading FP and detection-rate figures

### Description
The action is chosen from `state` (previous env step's observation) but scored against `info["is_attack"]` (current env step's truth):

```python
action_int, _ = defender.observe(state, ...)     # action ← step t−1's state
next_state, reward, ..., info = env.step(...)
metrics.record_step(..., info, ...)              # scored against step t's is_attack
```

When the attacker transitions ATTACK→NORMAL, the SEDM (having seen ATTACK in `state`) returns LOG/BLOCK, but `is_attack=False` in `info` → counted as FP. Episodes have only 0–10 normal-traffic steps (tiny denominator), so a single FP event drives one episode to 50–100% FPR.

**Before fix:**

| Intent | FP Rate | std |
|--------|---------|-----|
| STEALTHY | 35.6% | 36.6% |
| OPPORTUNISTIC | 15.0% | 34.5% |
| AGGRESSIVE | 6.7% | 24.9% |
| TARGETED | 3.3% | 18.0% |

The STEALTHY vs. TARGETED 10× difference is a measurement artifact, not a real behaviour difference.

### Fix
Align `is_attack` with the attack type encoded in the state the action was based on:

```python
state_is_attack = AttackType(int(np.argmax(state[0:10]))) != AttackType.NORMAL
aligned_info = dict(info)
aligned_info["is_attack"] = state_is_attack
metrics.record_step(..., aligned_info, ...)
```

**After fix — all intents:**

| Metric | Value |
|--------|-------|
| FP Rate | 0.000 ± 0.000 |
| Detection Rate | 1.000 ± 0.000 |

Provably correct: SEDM R1 makes FP=0 by design (NORMAL state always → ALLOW), and no intent generates RECON esc_risk < 0.35 (the only other ALLOW branch), so FN=0 as well.

**Follow-up (2026-04-26):** exact FP=0/Det=1 is a *tautology* under
oracle decisions — the SEDM's R1 rule and the `is_attack` label are both
driven by the same ground-truth `state[0:10]`. `evaluate.py::evaluate_intent`
was subsequently changed to decide from the RandomForest classifier's
*predicted* attack type (`clf_state`) while still scoring against ground
truth, breaking the tautology. This produces the FP≈0.00–0.07% figures in
the current `results/evaluation/evaluation_summary.csv` (classifier is
99.85% accurate — see `docs/parameter_selection.md` §5), rather than an
exact, un-testable 0.000. See `docs/parameter_selection.md` for the
feature-noise robustness sweep that shows FP rises predictably as
classifier input fidelity degrades.

---

## Bug 9 — Same one-step lag reproduced in `main.py` and `evaluation/sedm_eval.py` (High)

**Files:** `main.py` (`run_demo`, `run_compare`), `evaluation/sedm_eval.py` (`run_intent`)
**Severity:** High — same failure mode as Bug 8, in two code paths that were not touched by that fix

**Description**

Bug 8 was fixed in `evaluate.py` but the identical one-step lag pattern
existed independently in `main.py`'s `demo`/`compare` loops (present since
before Bug 8 was found) and was *reintroduced* in `evaluation/sedm_eval.py`
when the extended-metrics evaluation (precision/recall/F1/F2/specificity/
proportionality/containment) was added on 2026-04-26:

```python
action, _ = defender.observe(state, features, training=False)  # decided from state_t
next_state, reward, ..., info = env.step(action)                # attacker advances to t+1
is_attack = info["is_attack"]                                    # t+1's ground truth — wrong step
```

Under the extended evaluation's 30%-benign-injection protocol
(`benign_ratio=0.30`), this produced specificity ≈ 0.30 and FPR ≈ 0.70 in
`logs/sedm_eval_results.json` — an apparent near-random classifier that
does not reflect the SEDM's actual per-step behaviour. `evaluate.py`'s own
FP≈0% (Bug 8, fixed) and `sedm_eval.py`'s FPR≈70% (Bug 9, unfixed) coexisted
in the same repository, evaluating the same policy under a similar
protocol, which is itself a strong internal-consistency signal that
something was wrong.

**Fix**

Both files now: (1) decode ground truth from `state`/`info` — the
observation the action was chosen from — before calling `env.step()`, and
(2) `evaluation/sedm_eval.py::run_intent` additionally gained a
`classifier_driven` flag mirroring `evaluate.py`'s fix for Bug 8, so the
extended metrics can be reported both as an oracle upper bound and as a
realistic classifier-driven estimate. Re-running the extended evaluation
after the fix moved specificity from ≈0.30 → 0.999, F1 from ≈0.695 → 1.000
(classifier-driven), and late-stage miss rate from ≈0.30 → ≈0.000 — see
`README.md`'s Extended Metrics table for the corrected figures.

---

## Summary table

| # | File | Issue | Severity | Impact on results |
|---|------|-------|----------|-------------------|
| 1 | `venv/` | Packages not installed | Critical | Blocks all execution |
| 2 | `requirements.txt` | `pydantic` missing | Critical | OpenCanary emulator fails to import |
| 3 | `defender/__init__.py` | Stale DQN import forces torch dependency | High | Import failure without torch |
| 4 | `train.py:162` | `defender.dqn_agent` AttributeError | High | `train.py` crashes |
| 5 | `evaluate.py` step loop | One-step lag corrupts FP/FN metrics | High | FP inflated 35× for STEALTHY; apparent imbalance |
| 6 | `session_tracker.py:69,84` | `datetime.utcnow()` deprecated | Medium | DeprecationWarning |
| 7 | `evaluate.py:291` | MetricsCollector recreated per episode | Medium | Inefficient, no result corruption |
| 8 | `evaluation/metrics.py:374` | `plt.cm.get_cmap()` deprecated | Low | MatplotlibDeprecationWarning |
| 9 | `main.py`, `evaluation/sedm_eval.py` | Same one-step lag as Bug 8, unfixed in these files | High | Specificity 0.30→0.999, F1 0.695→1.000 after fix (extended eval) |
