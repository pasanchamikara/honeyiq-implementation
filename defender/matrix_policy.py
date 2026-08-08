"""
Stage-Escalation Decision Matrix (SEDM) — deterministic honeypot policy.

Replaces the DQN with an interpretable decision matrix: escalation risk
(from the intent-specific Markov TransitionModel) buckets into a Low/Medium/
High band, the band and kill-chain stage index into `_SEDM` for a base
action, and R1-R3 override rules (see `_apply_overrides`) adjust it. See
`get_matrix()` for the resulting table and `_composite_risk` for the logged
(non-action-affecting) risk score.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from attacker.attack_types import (
    AttackType,
    KillChainStage,
    AttackerIntent,
    ATTACK_SEVERITY,
    KILL_CHAIN_WEIGHT,
)
from attacker.transition_model import TransitionModel
from defender.adaptive_thresholds import AdaptiveThresholds
from defender.honeypot import HoneypotAction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

ESC_LOW_THRESHOLD  = 0.35   # escalation_risk < this  → LOW band
ESC_HIGH_THRESHOLD = 0.65   # escalation_risk ≥ this  → HIGH band
RATE_THRESHOLD     = 0.80   # escalation_rate > this  → override trigger (R3)
REPUTATION_THRESHOLD = 0.60 # cross-session reputation ≥ this → override trigger (R4)

# NOTE: ESC_LOW_THRESHOLD/ESC_HIGH_THRESHOLD are deliberately NOT adaptive.
# esc_risk comes purely from the static, hand-authored Markov TransitionModel
# — there is no live observational signal in this codebase whose drift would
# indicate those two band cuts are miscalibrated. RATE_THRESHOLD is different
# (it gates on escalation_rate, a real function of observed traffic) and can
# optionally be tuned by AdaptiveThresholds — see that module's docstring for
# what it does and does not claim to do.

# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------

# Rows indexed by KillChainStage int value (0-6)
# Columns indexed by band int: 0=Low, 1=Medium, 2=High

_SEDM: list[list[HoneypotAction]] = [
    #  Low                 Medium               High
    [HoneypotAction.ALLOW, HoneypotAction.LOG,   HoneypotAction.LOG   ],  # RECONNAISSANCE
    [HoneypotAction.LOG,   HoneypotAction.LOG,   HoneypotAction.TROLL ],  # WEAPONIZATION
    [HoneypotAction.LOG,   HoneypotAction.TROLL, HoneypotAction.TROLL ],  # DELIVERY
    [HoneypotAction.TROLL, HoneypotAction.BLOCK, HoneypotAction.BLOCK ],  # EXPLOITATION
    [HoneypotAction.BLOCK, HoneypotAction.BLOCK, HoneypotAction.ALERT ],  # INSTALLATION
    [HoneypotAction.BLOCK, HoneypotAction.ALERT, HoneypotAction.ALERT ],  # COMMAND_AND_CTRL
    [HoneypotAction.ALERT, HoneypotAction.ALERT, HoneypotAction.ALERT ],  # ACTIONS_ON_OBJ
]

# Attack types that trigger R2 (R2_TYPES)
_HIGH_IMPACT_ATTACKS: frozenset[AttackType] = frozenset({
    AttackType.DOS,
    AttackType.WORMS,
})


# ---------------------------------------------------------------------------
# MatrixPolicy
# ---------------------------------------------------------------------------

class MatrixPolicy:
    """
    Stage-Escalation Decision Matrix (SEDM) policy.

    Parameters
    ----------
    default_intent : AttackerIntent
        Fallback intent when none can be decoded from the state vector.
    """

    def __init__(
        self,
        default_intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
        adaptive_thresholds: Optional[AdaptiveThresholds] = None,
    ) -> None:
        self._default_intent = default_intent
        # Cache one TransitionModel per intent to avoid repeated construction
        self._tm_cache: Dict[AttackerIntent, TransitionModel] = {}
        # None (default) = static RATE_THRESHOLD, identical to today's
        # behavior. See AdaptiveThresholds' docstring for what it does.
        self._adaptive = adaptive_thresholds

    # ------------------------------------------------------------------
    # Core decision interface
    # ------------------------------------------------------------------

    def decide_from_state(
        self, state: np.ndarray, reputation: float = 0.0
    ) -> tuple[HoneypotAction, dict]:
        """
        Select a honeypot action directly from the 24-dim environment state.

        Parameters
        ----------
        state : np.ndarray of shape (24,)
            Same layout as CyberSecurityEnv:
              [0:10]   attack_type one-hot
              [10:17]  kill_chain_stage one-hot
              [17]     threat_level
              [18]     attack_count_normalized
              [19]     escalation_rate
              [20:24]  attacker_intent one-hot
        reputation : float
            Cross-session, time-decayed offense score for this source IP
            in [0, 1] (see opencanary_integration/engine/reputation.py).
            Not part of the state vector — no persistent IP identity exists
            in CyberSecurityEnv, so this is an out-of-band, live-pipeline-only
            signal, defaulting to 0.0 (today's exact behavior) elsewhere.

        Returns
        -------
        action : HoneypotAction
        info   : dict with intermediate values (for logging / analysis)
        """
        current_attack   = AttackType(int(np.argmax(state[0:10])))
        current_stage    = KillChainStage(int(np.argmax(state[10:17])))
        escalation_rate  = float(state[19])
        intent           = AttackerIntent(int(np.argmax(state[20:24])))

        return self.decide(
            current_stage, current_attack, escalation_rate, intent,
            reputation=reputation,
        )

    def decide(
        self,
        current_stage:   KillChainStage,
        current_attack:  AttackType,
        escalation_rate: float,
        intent:          Optional[AttackerIntent] = None,
        reputation:      float = 0.0,
    ) -> tuple[HoneypotAction, dict]:
        """
        Select a honeypot action from first principles.

        Parameters
        ----------
        current_stage    : observed kill chain stage for this session
        current_attack   : observed attack type
        escalation_rate  : fraction of recent steps that were attacks [0, 1]
        intent           : inferred attacker intent (affects transition probs)
        reputation       : cross-session offense score for this source IP
                            [0, 1]; defaults to 0.0 (R4 never fires)

        Returns
        -------
        action : HoneypotAction
        info   : dict — intermediate values for explainability / logging
        """
        if intent is None:
            intent = self._default_intent

        esc_risk = self._escalation_risk(current_stage, intent)
        band = self._escalation_band(esc_risk)
        base_action = _SEDM[int(current_stage)][band]
        action, override_applied = self._apply_overrides(
            base_action, current_attack, escalation_rate, reputation
        )
        # Logged for analysis only — does not affect the chosen action.
        composite_risk = self._composite_risk(
            current_stage, esc_risk, current_attack, escalation_rate
        )

        info = {
            "stage":            current_stage.name,
            "attack_type":      current_attack.name,
            "intent":           intent.name,
            "escalation_risk":  round(esc_risk, 4),
            "escalation_band":  ["LOW", "MEDIUM", "HIGH"][band],
            "base_action":      base_action.name,
            "reputation":       round(reputation, 4),
            "override_applied": override_applied,
            "final_action":     action.name,
            "composite_risk":   round(composite_risk, 4),
        }

        log.debug("SEDM: %s", info)

        return action, info

    # ------------------------------------------------------------------
    # Matrix introspection (for thesis visualization)
    # ------------------------------------------------------------------

    @staticmethod
    def get_matrix() -> list[list[str]]:
        """Return the SEDM as a list-of-lists of action names."""
        return [[a.name for a in row] for row in _SEDM]

    @staticmethod
    def get_matrix_actions() -> list[list[HoneypotAction]]:
        """Return the SEDM as a list-of-lists of raw HoneypotAction values."""
        return [list(row) for row in _SEDM]

    @staticmethod
    def get_full_matrix_for_intent(
        intent: AttackerIntent,
    ) -> np.ndarray:
        """
        Build a (7, 7) numeric matrix showing the recommended action integer
        for every (stage, next_stage_prob_bucket) pair.

        This is useful for visualising how the policy adapts to intent.
        Returns a (7,) array — one recommended action per stage — given the
        most likely next-stage probability vector under that intent.
        """
        tm    = TransitionModel(intent=intent)
        out   = np.zeros(7, dtype=int)
        mp    = MatrixPolicy(default_intent=intent)
        for stage in KillChainStage:
            action, _ = mp.decide(
                current_stage   = stage,
                current_attack  = AttackType.EXPLOITS,   # representative
                escalation_rate = 0.5,
                intent          = intent,
            )
            out[int(stage)] = int(action)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_model(self, intent: AttackerIntent) -> TransitionModel:
        if intent not in self._tm_cache:
            self._tm_cache[intent] = TransitionModel(intent=intent)
        return self._tm_cache[intent]

    def _escalation_risk(
        self, stage: KillChainStage, intent: AttackerIntent
    ) -> float:
        """P(next stage > current stage) under the given intent."""
        tm    = self._transition_model(intent)
        probs = tm.get_stage_probabilities(stage)
        return float(probs[int(stage) + 1 :].sum())

    @staticmethod
    def _escalation_band(esc_risk: float) -> int:
        """0 = Low, 1 = Medium, 2 = High."""
        if esc_risk < ESC_LOW_THRESHOLD:
            return 0
        if esc_risk < ESC_HIGH_THRESHOLD:
            return 1
        return 2

    @staticmethod
    def _upgrade(action: HoneypotAction) -> HoneypotAction:
        """Advance action one level in severity, clamped at ALERT."""
        return HoneypotAction(min(int(action) + 1, HoneypotAction.ALERT))

    def _apply_overrides(
        self,
        action:          HoneypotAction,
        current_attack:  AttackType,
        escalation_rate: float,
        reputation:      float = 0.0,
    ) -> tuple[HoneypotAction, str]:
        """
        Apply override rules in sequence.

        Returns (final_action, label_of_first_override_triggered | "none").
        """
        rate_threshold = self._adaptive.threshold if self._adaptive else RATE_THRESHOLD
        # Evaluated before the short-circuits below so AdaptiveThresholds
        # observes R3's counterfactual trigger rate on every decision, not a
        # censored subset where R1/R2/R4 already preempted it.
        r3_condition = escalation_rate > rate_threshold
        if self._adaptive is not None:
            self._adaptive.record(r3_condition)

        # R4 — persistent, cross-session repeat offender (checked first: a
        # flagged source IP stays escalated even on a benign-looking event,
        # fail2ban-style "once flagged, stay flagged")
        if reputation >= REPUTATION_THRESHOLD:
            return self._upgrade(action), "R4_REPEAT_OFFENDER"

        # R1 — normal traffic always allowed
        if current_attack == AttackType.NORMAL:
            return HoneypotAction.ALLOW, "R1_NORMAL_ALLOW"

        # R2 — high-impact spreading attacks
        if current_attack in _HIGH_IMPACT_ATTACKS:
            return self._upgrade(action), "R2_HIGH_IMPACT"

        # R3 — very high attack frequency in recent window
        if r3_condition:
            return self._upgrade(action), "R3_HIGH_RATE"

        return action, "none"

    @staticmethod
    def _composite_risk(
        stage:           KillChainStage,
        esc_risk:        float,
        attack_type:     AttackType,
        escalation_rate: float,
    ) -> float:
        """
        Composite risk score ∈ [0, 1] for logging and analysis.

        Weights:
            35% kill chain stage position
            35% escalation probability
            15% attack type severity
            15% recent attack frequency
        """
        stage_w    = KILL_CHAIN_WEIGHT[int(stage)]
        severity   = ATTACK_SEVERITY[int(attack_type)]
        risk = (
            0.35 * stage_w
            + 0.35 * esc_risk
            + 0.15 * severity
            + 0.15 * escalation_rate
        )
        return float(min(max(risk, 0.0), 1.0))
