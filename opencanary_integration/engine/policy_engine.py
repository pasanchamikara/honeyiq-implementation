"""
Policy engine — uses the Stage-Escalation Decision Matrix (SEDM) to select
honeypot actions from the 24-dim session state vector.

Replaces the former DQN-based engine.  No model file is required; the matrix
policy is fully deterministic and intent-aware.
"""

from __future__ import annotations

import logging
import sys, os
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from attacker.attack_types import AttackType, AttackerIntent
from defender.honeypot import HoneypotAction
from defender.matrix_policy import MatrixPolicy
from defender.classifier import AttackClassifier

log = logging.getLogger(__name__)


class PolicyEngine:
    """
    Wraps MatrixPolicy for use in the OpenCanary integration pipeline.

    Parameters
    ----------
    model_dir : str
        Directory to load the attack classifier from (optional).
        The matrix policy itself requires no model files.
    default_intent : str
        Fallback attacker intent name when none is encoded in the state.
    """

    def __init__(
        self,
        model_dir:      str = "models/",
        default_intent: str = "OPPORTUNISTIC",
    ) -> None:
        self._policy    = MatrixPolicy(
            default_intent=AttackerIntent[default_intent.upper()]
        )
        self._classifier: Optional[AttackClassifier] = None
        self._load_classifier(model_dir)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_classifier(self, model_dir: str) -> None:
        clf_path = os.path.join(model_dir, "classifier.joblib")
        self._classifier = AttackClassifier()
        if os.path.exists(clf_path):
            self._classifier.load(clf_path)
            log.info("Classifier loaded from %s", clf_path)
        else:
            log.warning(
                "Classifier not found at %s — feature-based attack "
                "classification disabled; logtype mapping will be used.",
                clf_path,
            )

    def reload(self, model_dir: str = "models/") -> None:
        """Reload the classifier from disk."""
        self._load_classifier(model_dir)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decide(
        self,
        state:    np.ndarray,
        features: Optional[dict[str, float]] = None,
    ) -> tuple[HoneypotAction, Optional[AttackType]]:
        """
        Select a honeypot action for the given 24-dim state vector.

        Parameters
        ----------
        state : np.ndarray of shape (24,)
            Built by state_builder.build_state().
        features : dict | None
            Optional UNSW-NB15-style feature dict for classifier inference.

        Returns
        -------
        action               : HoneypotAction selected by SEDM
        classifier_prediction: AttackType | None
        """
        action, info = self._policy.decide_from_state(state)

        log.debug(
            "SEDM decision: stage=%s band=%s esc_risk=%.3f action=%s",
            info["stage"], info["escalation_band"],
            info["escalation_risk"], info["final_action"],
        )

        predicted_attack: Optional[AttackType] = None
        if features and self._classifier and self._classifier.is_fitted:
            try:
                predicted_attack = self._classifier.predict(features)
            except Exception as exc:
                log.warning("Classifier inference failed: %s", exc)

        return action, predicted_attack

    def decision_info(self, state: np.ndarray) -> dict:
        """Return full SEDM breakdown for explainability."""
        _, info = self._policy.decide_from_state(state)
        return info

    @property
    def classifier_fitted(self) -> bool:
        return self._classifier is not None and self._classifier.is_fitted
