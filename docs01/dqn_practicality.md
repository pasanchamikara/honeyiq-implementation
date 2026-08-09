# Is DQN (or Another Learning-Based Approach) Practical?

**No — not for dynamic behavior adjustment, not right now.** This isn't a
generic RL-vs-rules position; it's grounded in this specific project's own
history and its own evaluation harness's track record.

## What actually happened when this project tried DQN

The DQN was HoneyIQ's original decision policy and was removed in favor
of the SEDM. Per this project's own thesis discussion chapter
(`thesis/doc0/05_discussion.md` §5.2):

- Detection rate jumped **87.6% → 97.4% in a single episode** while false
  positive rate stayed pinned near **1.0**. That's not learned
  discrimination — it's the agent finding a trivial shortcut ("always
  escalate") that exploits the training distribution.
- Training episodes were almost entirely attack traffic (severe class
  imbalance), so the agent never had to learn to recognize benign
  traffic.
- Training only covered the OPPORTUNISTIC intent, so the policy wouldn't
  generalize to STEALTHY/TARGETED/AGGRESSIVE without separate
  multi-intent training.

Any similarly-trained deep RL agent — Double DQN, PPO, whatever — would
very likely reproduce this failure mode unless the class-imbalance and
single-intent-training problems are fixed first. That's a substantial,
separate body of work (multi-intent training, benign-heavy episodes,
careful reward validation) that hasn't been done, and isn't a byproduct of
anything built in this round.

## Why the evaluation harness's own history makes this worse

`docs/BUGS_AND_FIXES.md` documents **two independent, hard-to-catch
label/timestep-alignment bugs** (Bug 8 and Bug 9) that silently inflated
measured false-positive rates by up to 10x before being caught — and both
lived in code that only had to compute descriptive statistics **after the
fact**.

A reward signal for an RL agent is computed by structurally similar code,
on **every single step**, and it directly shapes what gets learned rather
than just mis-reporting a number afterward. An undetected alignment bug in
a reward path doesn't just mis-report a metric — it teaches the agent
something confidently wrong, and the resulting policy looks coherent while
being subtly broken. That's arguably exactly what happened with the
original DQN's "always escalate" outcome, even without a specifically
identified computation bug in that training loop. Given that a simpler bug
class already evaded review twice in this codebase, a reward-shaped
training loop is not where effort should go next without a genuinely
adversarial validation pass on the reward computation first — and that
pass hasn't been done.

## What this project's own thesis already concluded

`thesis/doc0/05_discussion.md` §5.3, citing Rudin (2019): prefer an
interpretable model until a black-box model demonstrates clear, validated
superiority **and** can be explained post-hoc (e.g. via SHAP). The
project's own stated future plan is staged: SEDM as default now; a DQN
trained alongside using live data later, adopted only if it clears that
bar.

## What was built instead, and why it clears the same bar

- **R4 (cross-session reputation)** — a half-life decay formula an analyst
  can hand-verify with a calculator, driven by observable offense
  history. See [`dynamic_response.md`](dynamic_response.md).
- **`AdaptiveThresholds` (bounded `RATE_THRESHOLD` nudge)** — a plain
  deadband controller, explicitly scoped to alert-fatigue management
  rather than framed as a correctness improvement, because no honest
  correctness signal exists in this codebase to tune against.

Both change behavior dynamically in response to observed traffic, without
introducing a training loop, a reward function that can silently teach
the wrong lesson, or a model that needs after-the-fact explanation to the
people who built it.

## If a learning component is ever wanted

The only thing worth considering is a **small contextual bandit — not a
Q-network** — over a tiny discrete action space (e.g. choosing among 3–5
preset `RATE_THRESHOLD` values), driven by a reward computed from a
**concrete, trustworthy signal**: an analyst's own post-hoc label on a
`DummyHoneypot` audit-log entry (confirmed true/false positive).

That labeling ingestion path **does not exist today** — decisions are
logged (`results/evaluation/opencanary_*_audit.jsonl`,
`opencanary_integration.emulator.honeypot_emulator.DummyHoneypot.get_action_log()`),
but nothing closes the loop from "an operator confirmed this was wrong"
back into a trainable signal. Build that path first if this is ever
wanted. Don't manufacture a proxy signal (R1/R3 trigger rates, composite-
risk distribution shape) in the meantime — that would produce something
that *looks* like learning without being grounded in ground truth, which
is the same trap the original DQN fell into.

## Low-risk hygiene, noted but out of scope here

- `defender/dqn.py` is fully orphaned — nothing imports it.
- `train.py` still constructs a `dqn_config` dict and passes it into
  `Defender()`, where it's silently accepted and ignored.
- `torch` remains in `requirements.txt` solely because `dqn.py` still
  imports it.

None of these affect correctness or behavior; they're just left over from
before the SEDM replacement and are candidates for a future cleanup pass.
