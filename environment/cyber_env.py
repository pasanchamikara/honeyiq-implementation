"""
CyberSecurityEnv — a Gymnasium environment that bridges the Attacker and
Defender for reinforcement learning.

State vector (24 features):
    [0:10]  attack_type one-hot        (AttackType, 10 classes)
    [10:17] kill_chain_stage one-hot   (KillChainStage, 7 stages)
    [17]    threat_level               float [0, 1]
    [18]    attack_count_normalized    float [0, 1]
    [19]    escalation_rate            float [0, 1]
    [20:24] attacker_intent one-hot    (AttackerIntent, 4 intents)

Action space (discrete, 5):
    0 = ALLOW   1 = LOG   2 = TROLL   3 = BLOCK   4 = ALERT
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from attacker.attack_types import (
    AttackType,
    KillChainStage,
    AttackerIntent,
    ATTACK_SEVERITY,
)
from attacker.attacker import Attacker
from defender.honeypot import compute_threat_level, compute_reward, HoneypotAction

STATE_DIM  = 24
ACTION_DIM =  5
DEFAULT_EMA_ALPHA = 0.15


def encode_state(
    attack_type:      AttackType,
    kill_chain_stage: KillChainStage,
    threat_level:     float,
    attack_count:     int,
    escalation_rate:  float,
    intent:           AttackerIntent,
) -> np.ndarray:
    """Build the 24-dim state vector shared by the training env and the
    OpenCanary integration's live inference path (see module docstring for
    the layout)."""
    state = np.zeros(STATE_DIM, dtype=np.float32)

    state[int(attack_type)]           = 1.0   # [0:10]
    state[10 + int(kill_chain_stage)] = 1.0   # [10:17]
    state[17] = float(threat_level)
    state[18] = float(min(1.0, attack_count / 100.0))
    state[19] = float(escalation_rate)
    state[20 + int(intent)]           = 1.0   # [20:24]

    return state


class CyberSecurityEnv(gym.Env):
    """
    Gymnasium environment for the attacker-defender cybersecurity game.

    Parameters
    ----------
    attacker_intent : AttackerIntent
        Intent profile of the attacker in this episode.
    max_steps : int
        Maximum number of steps before truncation.
    escalation_window : int
        Sliding window size (in steps) for computing the window-based
        escalation_rate (used when escalation_mode == "window").
    escalation_mode : str
        "window" (default): escalation_rate is the fraction of the last
        `escalation_window` steps that were any attack — a hard cutoff,
        binary per step. "ema": escalation_rate is a severity-weighted
        exponential moving average instead — smoother, and reflects how
        *bad* recent attacks were, not just how many. Both signals are
        always computed and exposed in `info` as `escalation_window_rate`
        and `escalation_ema`, regardless of which one feeds the state
        vector, so switching modes is purely which one `state[19]` sees.
    escalation_ema_alpha : float
        EMA smoothing factor (higher = more weight on the latest step).
    seed : int | None
        Random seed.
    render_mode : str | None
        "human" prints a formatted summary; None disables rendering.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        attacker_intent:      AttackerIntent = AttackerIntent.OPPORTUNISTIC,
        max_steps:            int = 500,
        escalation_window:    int = 20,
        escalation_mode:      str = "window",
        escalation_ema_alpha: float = DEFAULT_EMA_ALPHA,
        seed:                 Optional[int] = None,
        render_mode:          Optional[str] = None,
        benign_ratio:         float = 0.0,
    ) -> None:
        super().__init__()

        if escalation_mode not in ("window", "ema"):
            raise ValueError(
                f"escalation_mode must be 'window' or 'ema', got {escalation_mode!r}"
            )

        self.attacker_intent     = attacker_intent
        self.max_steps           = max_steps
        self.escalation_window   = escalation_window
        self.escalation_mode     = escalation_mode
        self.escalation_ema_alpha = escalation_ema_alpha
        self.render_mode         = render_mode
        self._seed               = seed
        self.benign_ratio        = float(np.clip(benign_ratio, 0.0, 1.0))

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        # Components (initialized in reset)
        self._attacker:          Optional[Attacker] = None
        self._recent_attacks:    deque               = deque(maxlen=escalation_window)
        self._escalation_ema:    float               = 0.0

        # Episode tracking
        self._step_count:        int   = 0
        self._total_attacks:     int   = 0
        self._current_threat:    float = 0.0
        self._last_info:         dict  = {}
        self._current_state:     Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        effective_seed = seed if seed is not None else self._seed

        self._attacker = Attacker(
            intent=self.attacker_intent,
            seed=effective_seed,
        )
        self._recent_attacks = deque(maxlen=self.escalation_window)
        self._escalation_ema = 0.0
        self._step_count   = 0
        self._total_attacks = 0
        self._current_threat = 0.0

        # Build initial state from the attacker's starting position
        initial_state = encode_state(
            attack_type    = self._attacker.current_attack,
            kill_chain_stage = self._attacker.current_stage,
            threat_level   = 0.0,
            attack_count   = 0,
            escalation_rate = 0.0,
            intent         = self.attacker_intent,
        )
        self._current_state = initial_state
        info = {"step": 0, "attack_type": AttackType.NORMAL,
                "kill_chain_stage": KillChainStage.RECONNAISSANCE,
                "threat_level": 0.0, "is_attack": False,
                "features": {}, "escalation_rate": 0.0,
                "escalation_window_rate": 0.0, "escalation_ema": 0.0,
                "attack_count": 0}
        self._last_info = info
        return initial_state, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._attacker is not None, "Call reset() before step()."

        self._step_count += 1

        # Attacker advances (always, to maintain kill-chain continuity)
        atk_info = self._attacker.step()

        # Benign traffic injection: with probability benign_ratio, override
        # this step with NORMAL-class traffic while preserving the attacker's
        # kill-chain position in the state vector.  Simulates legitimate users
        # occasionally contacting the honeypot alongside the actual campaign.
        if self.benign_ratio > 0.0 and self.np_random.random() < self.benign_ratio:
            attack_type      = AttackType.NORMAL
            kill_chain_stage = atk_info["kill_chain_stage"]  # preserve stage
            is_attack        = False
            features         = self._attacker._simulate_features(AttackType.NORMAL)
        else:
            attack_type      = atk_info["attack_type"]
            kill_chain_stage = atk_info["kill_chain_stage"]
            is_attack        = atk_info["is_attack"]
            features         = atk_info["features"]

        # Update sliding window for escalation rate
        self._recent_attacks.append(int(is_attack))
        window_rate = (
            sum(self._recent_attacks) / len(self._recent_attacks)
            if self._recent_attacks else 0.0
        )

        # Severity-weighted EMA — smoother than the hard window cutoff, and
        # reflects how bad recent attacks were, not just how many occurred.
        severity = ATTACK_SEVERITY[int(attack_type)]
        self._escalation_ema = (
            self.escalation_ema_alpha * severity
            + (1 - self.escalation_ema_alpha) * self._escalation_ema
        )
        escalation_rate = (
            self._escalation_ema if self.escalation_mode == "ema" else window_rate
        )

        if is_attack:
            self._total_attacks += 1

        # Threat level
        threat_level = compute_threat_level(
            attack_type      = attack_type,
            kill_chain_stage = kill_chain_stage,
            escalation_rate  = escalation_rate,
            attack_count     = self._total_attacks,
        )
        self._current_threat = threat_level

        # Reward
        reward = compute_reward(
            action           = action,
            threat_level     = threat_level,
            is_attack        = is_attack,
            kill_chain_stage = kill_chain_stage,
            attack_type      = attack_type,
        )

        # Next state
        next_state = encode_state(
            attack_type      = attack_type,
            kill_chain_stage = kill_chain_stage,
            threat_level     = threat_level,
            attack_count     = self._total_attacks,
            escalation_rate  = escalation_rate,
            intent           = self.attacker_intent,
        )
        self._current_state = next_state

        terminated = False
        truncated  = self._step_count >= self.max_steps

        info = {
            "step":             self._step_count,
            "attack_type":      attack_type,
            "kill_chain_stage": kill_chain_stage,
            "threat_level":     threat_level,
            "is_attack":        is_attack,
            "features":         features,
            "escalation_rate":  escalation_rate,
            "escalation_window_rate": window_rate,
            "escalation_ema":   self._escalation_ema,
            "attack_count":     self._total_attacks,
            "action_name":      HoneypotAction(action).name,
            "next_probs":       atk_info["next_probabilities"],
            "stage_probs":      atk_info["stage_probabilities"],
        }
        self._last_info = info

        if self.render_mode == "human":
            self.render()

        return next_state, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        info = self._last_info
        if not info:
            return None

        lines = [
            f"\n{'='*60}",
            f"Step {info.get('step', '?'):>4} | "
            f"Attack: {info.get('attack_type', AttackType.NORMAL).name:<16} "
            f"Stage: {info.get('kill_chain_stage', KillChainStage.RECONNAISSANCE).name}",
            f"         Threat: {info.get('threat_level', 0.0):.3f}  "
            f"Escalation: {info.get('escalation_rate', 0.0):.3f}  "
            f"Action: {info.get('action_name', '?')}",
            f"{'='*60}",
        ]
        output = "\n".join(lines)

        if self.render_mode == "human":
            print(output)
            return None
        return output   # "ansi" mode

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def current_threat(self) -> float:
        return self._current_threat

    @property
    def current_state(self) -> Optional[np.ndarray]:
        return self._current_state
