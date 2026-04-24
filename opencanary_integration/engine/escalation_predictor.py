"""
Escalation predictor — wraps TransitionModel to expose the probability
distributions the pipeline needs.
"""

from __future__ import annotations

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from attacker.attack_types import AttackType, KillChainStage, AttackerIntent
from attacker.transition_model import TransitionModel


class EscalationPredictor:
    """
    Wraps a TransitionModel to provide escalation probability vectors.

    Parameters
    ----------
    intent : AttackerIntent
        Intent profile shaping the transition matrices.
    """

    def __init__(self, intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC) -> None:
        self._model  = TransitionModel(intent=intent)
        self._intent = intent

    def next_attack_probs(self, current: AttackType) -> np.ndarray:
        return self._model.get_attack_probabilities(current)

    def next_stage_probs(self, current: KillChainStage) -> np.ndarray:
        return self._model.get_stage_probabilities(current)

    def escalation_risk(self, current_stage: KillChainStage) -> float:
        """P(advancing to a stage strictly beyond current_stage)."""
        probs = self.next_stage_probs(current_stage)
        return float(probs[int(current_stage) + 1:].sum())

    def most_likely_next_stage(self, current: KillChainStage) -> KillChainStage:
        return KillChainStage(int(np.argmax(self.next_stage_probs(current))))

    def most_likely_next_attack(self, current: AttackType) -> AttackType:
        return AttackType(int(np.argmax(self.next_attack_probs(current))))
