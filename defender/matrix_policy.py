"""
Stage-Escalation Decision Matrix (SEDM) — deterministic honeypot policy.

Replaces the DQN with an interpretable, probabilistic decision matrix that
maps a session's current kill chain position and escalation risk to the
optimal honeypot action.

Algorithm
---------
1. **Escalation risk** — query the Markov chain TransitionModel for the
   probability of advancing to a *strictly more dangerous* kill chain stage:

       esc_risk = Σ P(next_stage = s') for all s' > current_stage

2. **Escalation band** — discretise esc_risk into three bands:
       Low    :  esc_risk < ESC_LOW_THRESHOLD   (0.35)
       Medium :  esc_risk < ESC_HIGH_THRESHOLD  (0.65)
       High   :  esc_risk ≥ ESC_HIGH_THRESHOLD

3. **Matrix lookup** — read the base action from the 7×3 SEDM:

       SEDM[current_stage][band] → base_action

       Stage / Band         | Low    | Medium | High
       ---------------------|--------|--------|-------
       RECONNAISSANCE       | ALLOW  | LOG    | LOG
       WEAPONIZATION        | LOG    | LOG    | TROLL
       DELIVERY             | LOG    | TROLL  | TROLL
       EXPLOITATION         | TROLL  | BLOCK  | BLOCK
       INSTALLATION         | BLOCK  | BLOCK  | ALERT
       COMMAND_AND_CTRL     | BLOCK  | ALERT  | ALERT
       ACTIONS_ON_OBJ       | ALERT  | ALERT  | ALERT

4. **Override rules** (applied in order after matrix lookup):

       R1 — Normal traffic:
            AttackType.NORMAL → ALLOW (always, regardless of matrix)

       R2 — Rapidly-spreading / high-impact attack types:
            AttackType ∈ {DOS, WORMS} → upgrade action by one level

       R3 — High attack frequency:
            escalation_rate > RATE_THRESHOLD (0.80) → upgrade action by one level

       Action upgrade order: ALLOW → LOG → TROLL → BLOCK → ALERT

5. **Composite risk score** (logged for analysis; does not change the action):

       risk = 0.35 × stage_weight
            + 0.35 × escalation_risk
            + 0.15 × attack_severity
            + 0.15 × escalation_rate

Intent-awareness
----------------
The TransitionModel is intent-specific; the escalation risk therefore differs
across STEALTHY / AGGRESSIVE / TARGETED / OPPORTUNISTIC intents.  The correct
intent is decoded from the 24-dim state vector (bits [20:24]).
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
from defender.honeypot import HoneypotAction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

ESC_LOW_THRESHOLD  = 0.35   # escalation_risk < this  → LOW band
ESC_HIGH_THRESHOLD = 0.65   # escalation_risk ≥ this  → HIGH band
RATE_THRESHOLD     = 0.80   # escalation_rate > this  → override trigger

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

# Action upgrade sequence (R2 / R3 overrides step one level forward)
_UPGRADE_ORDER: list[HoneypotAction] = [
    HoneypotAction.ALLOW,
    HoneypotAction.LOG,
    HoneypotAction.TROLL,
    HoneypotAction.BLOCK,
    HoneypotAction.ALERT,
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
    ) -> None:
        self._default_intent = default_intent
        # Cache one TransitionModel per intent to avoid repeated construction
        self._tm_cache: Dict[AttackerIntent, TransitionModel] = {}

    # ------------------------------------------------------------------
    # Core decision interface
    # ------------------------------------------------------------------

    def decide_from_state(self, state: np.ndarray) -> tuple[HoneypotAction, dict]:
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

        Returns
        -------
        action : HoneypotAction
        info   : dict with intermediate values (for logging / analysis)
        """
        current_attack   = AttackType(int(np.argmax(state[0:10])))
        current_stage    = KillChainStage(int(np.argmax(state[10:17])))
        escalation_rate  = float(state[19])
        intent           = AttackerIntent(int(np.argmax(state[20:24])))

        return self.decide(current_stage, current_attack, escalation_rate, intent)

    def decide(
        self,
        current_stage:   KillChainStage,
        current_attack:  AttackType,
        escalation_rate: float,
        intent:          Optional[AttackerIntent] = None,
    ) -> tuple[HoneypotAction, dict]:
        """
        Select a honeypot action from first principles.

        Parameters
        ----------
        current_stage    : observed kill chain stage for this session
        current_attack   : observed attack type
        escalation_rate  : fraction of recent steps that were attacks [0, 1]
        intent           : inferred attacker intent (affects transition probs)

        Returns
        -------
        action : HoneypotAction
        info   : dict — intermediate values for explainability / logging
        """
        if intent is None:
            intent = self._default_intent

        # ------------------------------------------------------------------
        # Step 1: Compute escalation risk from intent-specific Markov chain
        # ------------------------------------------------------------------
        esc_risk = self._escalation_risk(current_stage, intent)

        # ------------------------------------------------------------------
        # Step 2: Band classification
        # ------------------------------------------------------------------
        band = self._escalation_band(esc_risk)

        # ------------------------------------------------------------------
        # Step 3: Matrix lookup
        # ------------------------------------------------------------------
        base_action = _SEDM[int(current_stage)][band]

        # ------------------------------------------------------------------
        # Step 4: Override rules
        # ------------------------------------------------------------------
        action, override_applied = self._apply_overrides(
            base_action, current_attack, escalation_rate
        )

        # ------------------------------------------------------------------
        # Step 5: Composite risk score (for logging / analysis only)
        # ------------------------------------------------------------------
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
            "override_applied": override_applied,
            "final_action":     action.name,
            "composite_risk":   round(composite_risk, 4),
        }

        log.debug(
            "SEDM: stage=%-18s band=%-6s esc=%.3f rate=%.3f "
            "base=%-5s override=%s → %s",
            current_stage.name, ["LOW", "MED", "HIGH"][band],
            esc_risk, escalation_rate,
            base_action.name, override_applied, action.name,
        )

        return action, info

    # ------------------------------------------------------------------
    # Matrix introspection (for thesis visualization)
    # ------------------------------------------------------------------

    @staticmethod
    def get_matrix() -> list[list[str]]:
        """Return the SEDM as a list-of-lists of action names."""
        return [[a.name for a in row] for row in _SEDM]

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
        """Advance action one level in severity."""
        idx = _UPGRADE_ORDER.index(action)
        return _UPGRADE_ORDER[min(idx + 1, len(_UPGRADE_ORDER) - 1)]

    def _apply_overrides(
        self,
        action:          HoneypotAction,
        current_attack:  AttackType,
        escalation_rate: float,
    ) -> tuple[HoneypotAction, str]:
        """
        Apply override rules in sequence.

        Returns (final_action, label_of_first_override_triggered | "none").
        """
        # R1 — normal traffic always allowed
        if current_attack == AttackType.NORMAL:
            return HoneypotAction.ALLOW, "R1_NORMAL_ALLOW"

        # R2 — high-impact spreading attacks
        if current_attack in _HIGH_IMPACT_ATTACKS:
            return self._upgrade(action), "R2_HIGH_IMPACT"

        # R3 — very high attack frequency in recent window
        if escalation_rate > RATE_THRESHOLD:
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
