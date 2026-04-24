"""
DefenderAgent — orchestrates the attack classifier and the SEDM matrix policy.

The classifier identifies attack types from raw network features (unchanged).
The MatrixPolicy replaces the former DQN: it selects honeypot actions based
on the current kill chain stage and escalation probability from the Markov
chain transition model, without any learned parameters or training.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from attacker.attack_types import AttackType, KillChainStage, AttackerIntent
from .classifier import AttackClassifier
from .honeypot import HoneypotAction
from .matrix_policy import MatrixPolicy


class Defender:
    """
    Top-level defender agent combining:

    - An AttackClassifier (RandomForest) for identifying attack types from
      raw UNSW-NB15-style network flow features.
    - A MatrixPolicy (SEDM) that selects honeypot response actions using the
      kill chain stage and Markov-chain escalation probability — no training
      required.

    Parameters
    ----------
    classifier_config : dict | None
        Keyword arguments forwarded to AttackClassifier.__init__.
    train_classifier : bool
        If True, auto-generate training data and fit the classifier when
        initialize_classifier() is called.
    seed : int
        Random seed shared across components.
    default_intent : AttackerIntent
        Fallback intent for the MatrixPolicy when none is decodable from state.
    """

    def __init__(
        self,
        classifier_config: Optional[dict] = None,
        train_classifier:  bool = True,
        seed:              int  = 42,
        default_intent:    AttackerIntent = AttackerIntent.OPPORTUNISTIC,
        # Legacy parameter — accepted but ignored (no DQN config needed)
        dqn_config:        Optional[dict] = None,
    ) -> None:
        clf_cfg = classifier_config or {}

        self.classifier    = AttackClassifier(**clf_cfg)
        self.matrix_policy = MatrixPolicy(default_intent=default_intent)
        self._seed         = seed

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_classifier(
        self,
        n_samples_per_class: int = 600,
    ) -> None:
        """
        Generate synthetic training data from the Attacker's feature
        simulation and fit the RandomForest classifier.
        """
        print(f"[Defender] Training attack classifier "
              f"({n_samples_per_class} samples/class × {AttackType.count()} classes)...")
        self.classifier.fit_from_simulation(
            n_samples_per_class=n_samples_per_class,
            seed=self._seed,
        )
        print("[Defender] Classifier ready.")

    # ------------------------------------------------------------------
    # Inference (one step)
    # ------------------------------------------------------------------

    def observe(
        self,
        state:    np.ndarray,
        features: dict[str, float],
        training: bool = True,  # ignored — SEDM is deterministic
    ) -> tuple[int, AttackType]:
        """
        Given the current environment state vector and raw network features:

        1. Classify the features to obtain a predicted attack type.
        2. Decode kill chain stage, escalation rate, and inferred intent from
           the state vector.
        3. Query the MatrixPolicy for the optimal honeypot action.

        Returns
        -------
        action           : int  (HoneypotAction value)
        predicted_attack : AttackType
        """
        # -- Attack type classification from network features --------------
        if self.classifier.is_fitted:
            predicted_attack = self.classifier.predict(features)
        else:
            predicted_attack = AttackType.NORMAL

        # -- Action selection from SEDM ------------------------------------
        action, _info = self.matrix_policy.decide_from_state(state)
        return int(action), predicted_attack

    def get_attack_probabilities(
        self, features: dict[str, float]
    ) -> np.ndarray:
        """Return classifier probability vector over all attack types."""
        if not self.classifier.is_fitted:
            return np.ones(AttackType.count()) / AttackType.count()
        return self.classifier.predict_proba(features)

    def get_decision_info(
        self,
        state:    np.ndarray,
        features: dict[str, float],
    ) -> dict:
        """
        Return the full SEDM decision breakdown for a given state.

        Useful for explainability, logging, and thesis analysis.
        """
        if self.classifier.is_fitted:
            predicted_attack = self.classifier.predict(features)
        else:
            predicted_attack = AttackType.NORMAL

        action, info = self.matrix_policy.decide_from_state(state)
        info["predicted_attack"] = predicted_attack.name
        return info

    # ------------------------------------------------------------------
    # Learning — no-op (SEDM has no trainable parameters)
    # ------------------------------------------------------------------

    def learn(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> Optional[float]:
        """
        No-op — the MatrixPolicy requires no gradient updates.

        Returns None (compatible with code that checks for a loss value).
        """
        return None

    # ------------------------------------------------------------------
    # Persistence — classifier only (policy has no weights)
    # ------------------------------------------------------------------

    def save(self, model_dir: str = "models/") -> None:
        os.makedirs(model_dir, exist_ok=True)
        self.classifier.save(os.path.join(model_dir, "classifier.joblib"))
        print(f"[Defender] Classifier saved to {model_dir}")

    def load(self, model_dir: str = "models/") -> None:
        clf_path = os.path.join(model_dir, "classifier.joblib")
        if os.path.exists(clf_path):
            self.classifier.load(clf_path)
            print(f"[Defender] Classifier loaded from {clf_path}")
        else:
            print(f"[Defender] Warning: Classifier not found at {clf_path}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Stub property — SEDM has no exploration parameter."""
        return 0.0

    @property
    def steps_done(self) -> int:
        """Stub property — SEDM does not count gradient steps."""
        return 0

    def policy_matrix(self) -> list[list[str]]:
        """Return the underlying SEDM as a list-of-lists of action names."""
        return MatrixPolicy.get_matrix()
