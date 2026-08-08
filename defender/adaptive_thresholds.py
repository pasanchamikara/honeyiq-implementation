"""
Bounded, fully-auditable deadband controller for MatrixPolicy's RATE_THRESHOLD.

What this is NOT: a correctness-learning mechanism. HoneyIQ has no live
ground-truth signal for whether any given R3 ("high attack frequency")
trigger was actually the right call — decisions are logged, but nothing
closes the loop from "an operator confirmed this was wrong" back into this
code. The only signal genuinely available is R3's own trigger frequency.

So AdaptiveThresholds is scoped honestly: it keeps R3's firing rate near an
operator-chosen target, purely as an alert-fatigue safety valve (so a
BLOCK/ALERT queue doesn't lose signal value by firing on nearly every step,
or go quiet and stop being useful). It is a plain deadband controller — a
few lines of arithmetic an analyst can hand-verify — not a learned function.

ESC_LOW_THRESHOLD/ESC_HIGH_THRESHOLD (in matrix_policy.py) are deliberately
NOT covered by this or any adaptive mechanism: they bucket esc_risk, which
comes purely from the static, hand-authored Markov TransitionModel with no
live observational input in this codebase, so there is nothing honest to
adapt them against.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class AdaptiveThresholds:
    """
    Nudges a `threshold` value by `step` whenever R3's observed trigger rate
    over the last `observation_window` decisions drifts outside
    `target_rate ± tolerance`, clamped to `initial_threshold ± bound`.

    Construct with `initial_threshold=defender.matrix_policy.RATE_THRESHOLD`
    (the caller's job — this class has no dependency on matrix_policy, so it
    stays independently testable and reusable).
    """

    initial_threshold:  float = 0.80
    target_rate:         float = 0.10
    tolerance:            float = 0.03
    step:                  float = 0.01
    bound:                 float = 0.10   # max drift from initial_threshold
    observation_window:   int   = 200

    _threshold: float = field(init=False, repr=False)
    _observations: deque = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._threshold = self.initial_threshold
        self._observations = deque(maxlen=self.observation_window)

    def record(self, r3_condition: bool) -> None:
        """Record whether R3's condition held on this decision. Once the
        observation window fills, nudge the threshold if the observed rate
        has drifted outside the target band."""
        self._observations.append(bool(r3_condition))
        if len(self._observations) < self.observation_window:
            return

        observed_rate = sum(self._observations) / len(self._observations)
        lo, hi = self.initial_threshold - self.bound, self.initial_threshold + self.bound

        if observed_rate > self.target_rate + self.tolerance:
            # Firing too often — raise the bar to fire less.
            self._threshold = min(hi, self._threshold + self.step)
        elif observed_rate < self.target_rate - self.tolerance:
            # Firing too rarely — lower the bar to fire more.
            self._threshold = max(lo, self._threshold - self.step)

    @property
    def threshold(self) -> float:
        return self._threshold
