"""
AttackerAgent — simulates an attacker progressing through the cyber kill chain,
generating synthetic network flow features for each step.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .attack_types import (
    AttackType,
    KillChainStage,
    AttackerIntent,
    FEATURE_DISTRIBUTIONS,
    FEATURE_NAMES,
    ATTACK_PRIMARY_STAGE,
)
from .transition_model import TransitionModel


class Attacker:
    """
    Simulates an attacker executing a sequence of attacks following a Markov
    chain.  Each step may change the attack type and/or kill chain stage, and
    produces a synthetic set of network flow features.

    Parameters
    ----------
    intent : AttackerIntent
        Determines how transition probabilities are skewed.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
        seed: int | None = None,
    ) -> None:
        self.intent = intent
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.transition_model = TransitionModel(intent=intent, seed=seed)

        # Internal state
        self.current_attack: AttackType = AttackType.NORMAL
        self.current_stage: KillChainStage = KillChainStage.RECONNAISSANCE
        self.attack_count: int = 0
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset attacker to initial state (beginning of kill chain)."""
        self.current_attack = AttackType.NORMAL
        self.current_stage  = KillChainStage.RECONNAISSANCE
        self.attack_count   = 0
        self.step_count     = 0
        # Re-seed RNG for reproducibility if a seed was provided
        if self._seed is not None:
            self.rng = np.random.default_rng(self._seed)

    def step(self) -> Dict[str, Any]:
        """
        Advance the attacker by one time step.

        Samples the next attack type from the transition model, then the next
        kill chain stage, and generates synthetic network features for the
        resulting attack type.

        Returns
        -------
        dict with keys:
            attack_type       : AttackType
            kill_chain_stage  : KillChainStage
            intent            : AttackerIntent
            attack_count      : int   (cumulative non-NORMAL attacks)
            step_count        : int
            features          : dict[str, float]  (15 network features)
            is_attack         : bool  (False iff attack_type == NORMAL)
            next_probabilities: np.ndarray  (shape 10, attack type probs)
            stage_probabilities: np.ndarray (shape 7, stage probs)
        """
        self.step_count += 1

        # Transition attack type
        self.current_attack = self.transition_model.next_attack(
            self.current_attack
        )

        # Transition kill chain stage — anchor to attack's primary stage when
        # the attack type implies a forward jump (prevents stage regressing
        # while attack severity rises, which would be unrealistic)
        primary = KillChainStage(ATTACK_PRIMARY_STAGE[self.current_attack])
        sampled_stage = self.transition_model.next_stage(self.current_stage)
        # Take the max to prevent regression beyond primary stage implied by attack
        self.current_stage = KillChainStage(
            max(int(sampled_stage), int(primary) - 1)
        )

        is_attack = self.current_attack != AttackType.NORMAL
        if is_attack:
            self.attack_count += 1

        features = self._simulate_features(self.current_attack)

        return {
            "attack_type":        self.current_attack,
            "kill_chain_stage":   self.current_stage,
            "intent":             self.intent,
            "attack_count":       self.attack_count,
            "step_count":         self.step_count,
            "features":           features,
            "is_attack":          is_attack,
            "next_probabilities": self.transition_model.get_attack_probabilities(
                self.current_attack
            ),
            "stage_probabilities": self.transition_model.get_stage_probabilities(
                self.current_stage
            ),
        }

    def get_state_info(self) -> Dict[str, Any]:
        """Return current state without advancing."""
        return {
            "attack_type":      self.current_attack,
            "kill_chain_stage": self.current_stage,
            "intent":           self.intent,
            "attack_count":     self.attack_count,
            "step_count":       self.step_count,
        }

    # ------------------------------------------------------------------
    # Feature simulation
    # ------------------------------------------------------------------

    def _simulate_features(self, attack_type: AttackType) -> Dict[str, float]:
        """
        Generate synthetic network flow features for a given attack type,
        sampled from the distributions defined in FEATURE_DISTRIBUTIONS.
        """
        dist_spec = FEATURE_DISTRIBUTIONS[attack_type]
        features: Dict[str, float] = {}

        for name in FEATURE_NAMES:
            spec = dist_spec[name]
            kind = spec[0]

            if kind == "uniform":
                val = float(self.rng.uniform(spec[1], spec[2]))
            elif kind == "lognormal":
                val = float(self.rng.lognormal(spec[1], spec[2]))
            elif kind == "poisson":
                val = float(self.rng.poisson(spec[1]))
            elif kind == "choice":
                val = float(self.rng.choice(spec[1]))
            elif kind == "constant":
                val = float(spec[1])
            else:
                raise ValueError(f"Unknown distribution type: {kind!r}")

            features[name] = max(0.0, val)

        return features
