"""
DefenderAgent — orchestrates the attack classifier and DQN policy.

The classifier identifies attack types from raw network features.
The DQN selects honeypot actions based on the environment state vector.
Both components can be saved and reloaded independently.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from attacker.attack_types import AttackType
from .dqn import DQNAgent
from .classifier import AttackClassifier
from .honeypot import HoneypotAction


class Defender:
    """
    Top-level defender agent combining:
    - An AttackClassifier (RandomForest) for identifying attack types
      from raw network flow features.
    - A DQNAgent that selects honeypot response actions from the
      environment state vector.

    Parameters
    ----------
    dqn_config : dict | None
        Keyword arguments forwarded to DQNAgent.__init__.
    classifier_config : dict | None
        Keyword arguments forwarded to AttackClassifier.__init__.
    train_classifier : bool
        If True, auto-generate training data and fit the classifier
        when initialize_classifier() is called.
    seed : int
        Random seed shared across components.
    """

    def __init__(
        self,
        dqn_config:        Optional[dict] = None,
        classifier_config: Optional[dict] = None,
        train_classifier:  bool = True,
        seed:              int  = 42,
    ) -> None:
        dqn_cfg = dqn_config or {}
        clf_cfg = classifier_config or {}

        self.dqn_agent  = DQNAgent(**dqn_cfg)
        self.classifier = AttackClassifier(**clf_cfg)
        self._train_classifier = train_classifier
        self._seed = seed

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
        training: bool = True,
    ) -> tuple[int, AttackType]:
        """
        Given the current environment state vector and raw network features:
        1. Classify the features to obtain a predicted attack type.
        2. Select a DQN action using epsilon-greedy (training) or greedy.

        Returns
        -------
        action : int  (HoneypotAction value)
        predicted_attack : AttackType
        """
        if self.classifier.is_fitted:
            predicted_attack = self.classifier.predict(features)
        else:
            predicted_attack = AttackType.NORMAL

        action = self.dqn_agent.select_action(state, training=training)
        return action, predicted_attack

    def get_attack_probabilities(
        self, features: dict[str, float]
    ) -> np.ndarray:
        """Return classifier probability vector over all attack types."""
        if not self.classifier.is_fitted:
            return np.ones(AttackType.count()) / AttackType.count()
        return self.classifier.predict_proba(features)

    # ------------------------------------------------------------------
    # Learning
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
        Store the transition in the replay buffer and trigger a DQN update.

        Returns the training loss, or None if the buffer is not yet warm.
        """
        self.dqn_agent.store_transition(state, action, reward, next_state, done)
        return self.dqn_agent.update()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str = "models/") -> None:
        os.makedirs(model_dir, exist_ok=True)
        self.dqn_agent.save(os.path.join(model_dir, "dqn_agent.pt"))
        self.classifier.save(os.path.join(model_dir, "classifier.joblib"))
        print(f"[Defender] Models saved to {model_dir}")

    def load(self, model_dir: str = "models/") -> None:
        dqn_path = os.path.join(model_dir, "dqn_agent.pt")
        clf_path = os.path.join(model_dir, "classifier.joblib")

        if os.path.exists(dqn_path):
            self.dqn_agent.load(dqn_path)
            print(f"[Defender] DQN loaded from {dqn_path}")
        else:
            print(f"[Defender] Warning: DQN checkpoint not found at {dqn_path}")

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
        return self.dqn_agent.epsilon

    @property
    def steps_done(self) -> int:
        return self.dqn_agent.steps_done

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return raw Q-values for all actions given a state (numpy array)."""
        import torch
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.dqn_agent.device).unsqueeze(0)
        with torch.no_grad():
            q = self.dqn_agent.policy_net(state_t)
        return q.squeeze(0).cpu().numpy()
