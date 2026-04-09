"""
Markov chain transition model for attack type and kill chain stage sequences.
Intent-specific modifiers shape the base matrices to reflect different attacker
strategies.
"""

from __future__ import annotations

import numpy as np

from .attack_types import AttackType, KillChainStage, AttackerIntent

N_ATTACKS = AttackType.count()
N_STAGES  = KillChainStage.count()


# ---------------------------------------------------------------------------
# Base transition matrix — attack type to attack type
# ---------------------------------------------------------------------------
# Rows = current attack type, columns = next attack type (row-stochastic)

_BASE_ATTACK_MATRIX = np.array([
    # NOR    REC    ANA    FUZ    EXP    BAC    SHL    GEN    DOS    WRM
    [0.60,  0.28,  0.05,  0.03,  0.02,  0.01,  0.00,  0.01,  0.00,  0.00],  # NORMAL
    [0.05,  0.20,  0.30,  0.18,  0.14,  0.03,  0.02,  0.05,  0.02,  0.01],  # RECONNAISSANCE
    [0.02,  0.05,  0.10,  0.25,  0.35,  0.08,  0.05,  0.06,  0.02,  0.02],  # ANALYSIS
    [0.01,  0.02,  0.04,  0.20,  0.38,  0.10,  0.13,  0.06,  0.04,  0.02],  # FUZZERS
    [0.01,  0.01,  0.02,  0.05,  0.10,  0.30,  0.25,  0.06,  0.09,  0.11],  # EXPLOITS
    [0.00,  0.01,  0.01,  0.03,  0.05,  0.25,  0.10,  0.10,  0.15,  0.30],  # BACKDOORS
    [0.00,  0.01,  0.02,  0.05,  0.20,  0.38,  0.10,  0.05,  0.07,  0.12],  # SHELLCODE
    [0.02,  0.05,  0.05,  0.22,  0.30,  0.10,  0.10,  0.10,  0.04,  0.02],  # GENERIC
    [0.00,  0.01,  0.01,  0.02,  0.04,  0.05,  0.04,  0.08,  0.62,  0.13],  # DOS
    [0.00,  0.01,  0.01,  0.04,  0.05,  0.18,  0.05,  0.05,  0.35,  0.26],  # WORMS
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Base transition matrix — kill chain stage to kill chain stage
# ---------------------------------------------------------------------------
# Mostly forward-progressing; small regression probability models defenders
# pushing back.  Rows = current stage, columns = next stage (row-stochastic).

_BASE_STAGE_MATRIX = np.array([
    # REC    WPN    DLV    EXP    INS    C2     ACT
    [0.30,  0.60,  0.10,  0.00,  0.00,  0.00,  0.00],  # RECONNAISSANCE
    [0.10,  0.20,  0.65,  0.05,  0.00,  0.00,  0.00],  # WEAPONIZATION
    [0.05,  0.10,  0.10,  0.70,  0.05,  0.00,  0.00],  # DELIVERY
    [0.02,  0.03,  0.10,  0.20,  0.60,  0.05,  0.00],  # EXPLOITATION
    [0.01,  0.01,  0.03,  0.08,  0.20,  0.62,  0.05],  # INSTALLATION
    [0.00,  0.01,  0.02,  0.05,  0.07,  0.30,  0.55],  # COMMAND_AND_CTRL
    [0.00,  0.00,  0.01,  0.02,  0.05,  0.22,  0.70],  # ACTIONS_ON_OBJ
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Intent-specific multiplicative modifiers
# ---------------------------------------------------------------------------
# Applied element-wise to the base attack matrix, then rows are renormalized.
# Values > 1.0 increase probability; < 1.0 decrease it.

def _intent_attack_modifier(intent: AttackerIntent) -> np.ndarray:
    """Return a (10, 10) element-wise multiplier for _BASE_ATTACK_MATRIX."""
    mod = np.ones((N_ATTACKS, N_ATTACKS), dtype=np.float64)
    idx = {at: int(at) for at in AttackType}

    if intent == AttackerIntent.STEALTHY:
        # Prefer recon, analysis, backdoors — avoid noisy attacks
        for col in [idx[AttackType.RECONNAISSANCE], idx[AttackType.ANALYSIS],
                    idx[AttackType.BACKDOORS]]:
            mod[:, col] *= 2.0
        for col in [idx[AttackType.DOS], idx[AttackType.WORMS],
                    idx[AttackType.GENERIC], idx[AttackType.FUZZERS]]:
            mod[:, col] *= 0.15
        # Slow down progression — keep self-loops higher
        for at in AttackType:
            mod[int(at), int(at)] *= 1.5

    elif intent == AttackerIntent.AGGRESSIVE:
        # Prefer high-impact attacks; move fast through kill chain
        for col in [idx[AttackType.DOS], idx[AttackType.WORMS],
                    idx[AttackType.EXPLOITS], idx[AttackType.SHELLCODE]]:
            mod[:, col] *= 2.5
        for col in [idx[AttackType.RECONNAISSANCE], idx[AttackType.ANALYSIS],
                    idx[AttackType.NORMAL]]:
            mod[:, col] *= 0.2
        # Reduce self-loops to force faster transitions
        for at in AttackType:
            mod[int(at), int(at)] *= 0.4

    elif intent == AttackerIntent.TARGETED:
        # Focus on exploit chain: exploits → shellcode → backdoors → worms
        for col in [idx[AttackType.EXPLOITS], idx[AttackType.SHELLCODE],
                    idx[AttackType.BACKDOORS]]:
            mod[:, col] *= 2.0
        for col in [idx[AttackType.GENERIC], idx[AttackType.FUZZERS],
                    idx[AttackType.DOS], idx[AttackType.NORMAL]]:
            mod[:, col] *= 0.15
        # Boost worms for lateral movement objective
        mod[idx[AttackType.BACKDOORS], idx[AttackType.WORMS]] *= 3.0

    elif intent == AttackerIntent.OPPORTUNISTIC:
        # Scattered — boosts generic/fuzzers, moderate noise everywhere
        for col in [idx[AttackType.GENERIC], idx[AttackType.FUZZERS]]:
            mod[:, col] *= 2.0
        for at in AttackType:
            if at != AttackType.NORMAL:
                mod[:, int(at)] *= 1.3
        # Reduce normal traffic
        mod[:, idx[AttackType.NORMAL]] *= 0.3

    return mod


def _intent_stage_modifier(intent: AttackerIntent) -> np.ndarray:
    """Return a (7, 7) element-wise multiplier for _BASE_STAGE_MATRIX."""
    mod = np.ones((N_STAGES, N_STAGES), dtype=np.float64)
    idx = {s: int(s) for s in KillChainStage}

    if intent == AttackerIntent.STEALTHY:
        # Slower progression — spend more time at early stages
        for s in KillChainStage:
            mod[int(s), int(s)] *= 2.0
        # Reduce fast-forward jumps
        for s in range(N_STAGES - 1):
            if s + 2 < N_STAGES:
                mod[s, s + 2] *= 0.2

    elif intent == AttackerIntent.AGGRESSIVE:
        # Fast progression — reduce self-loops, boost forward transitions
        for s in KillChainStage:
            mod[int(s), int(s)] *= 0.3
        for s in range(N_STAGES - 1):
            mod[s, s + 1] *= 2.0
            if s + 2 < N_STAGES:
                mod[s, s + 2] *= 1.5
        # Reduce regression
        for s in range(1, N_STAGES):
            mod[s, s - 1] *= 0.2

    elif intent == AttackerIntent.TARGETED:
        # Direct path: skip early stages, stay in EXPLOITATION → INSTALLATION → C2
        for s in [idx[KillChainStage.EXPLOITATION],
                  idx[KillChainStage.INSTALLATION],
                  idx[KillChainStage.COMMAND_AND_CTRL]]:
            mod[s, s] *= 2.0
        for s in range(N_STAGES - 1):
            mod[s, s + 1] *= 1.5
        mod[idx[KillChainStage.RECONNAISSANCE],
            idx[KillChainStage.WEAPONIZATION]] *= 3.0

    elif intent == AttackerIntent.OPPORTUNISTIC:
        # Random walk — slightly favour forward transitions but no strong pull
        for s in range(N_STAGES - 1):
            mod[s, s + 1] *= 1.3
        # Moderate regression probability
        for s in range(1, N_STAGES):
            mod[s, s - 1] *= 1.5

    return mod


def _apply_modifier_and_normalize(base: np.ndarray,
                                   modifier: np.ndarray) -> np.ndarray:
    """Apply element-wise modifier, clip negatives, renormalize rows."""
    result = np.clip(base * modifier, 0.0, None)
    row_sums = result.sum(axis=1, keepdims=True)
    # Guard against all-zero rows (shouldn't happen but be safe)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return result / row_sums


# ---------------------------------------------------------------------------
# TransitionModel
# ---------------------------------------------------------------------------

class TransitionModel:
    """
    Markov chain model for attack type and kill chain stage transitions.

    Uses a base transition matrix modified by the attacker's intent, then
    renormalized to form a valid probability distribution.
    """

    def __init__(
        self,
        intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
        seed: int | None = None,
    ) -> None:
        self.intent = intent
        self.rng = np.random.default_rng(seed)

        atk_mod   = _intent_attack_modifier(intent)
        stage_mod = _intent_stage_modifier(intent)

        self._attack_matrix = _apply_modifier_and_normalize(
            _BASE_ATTACK_MATRIX, atk_mod
        )
        self._stage_matrix = _apply_modifier_and_normalize(
            _BASE_STAGE_MATRIX, stage_mod
        )

    # ------------------------------------------------------------------
    def next_attack(self, current: AttackType) -> AttackType:
        """Sample next attack type given current, using intent-modified matrix."""
        probs = self._attack_matrix[int(current)]
        choice = self.rng.choice(N_ATTACKS, p=probs)
        return AttackType(choice)

    def next_stage(self, current: KillChainStage) -> KillChainStage:
        """Sample next kill chain stage given current."""
        probs = self._stage_matrix[int(current)]
        choice = self.rng.choice(N_STAGES, p=probs)
        return KillChainStage(choice)

    # ------------------------------------------------------------------
    def get_attack_probabilities(self, current: AttackType) -> np.ndarray:
        """Return the full probability row for the given attack type."""
        return self._attack_matrix[int(current)].copy()

    def get_stage_probabilities(self, current: KillChainStage) -> np.ndarray:
        """Return the full probability row for the given kill chain stage."""
        return self._stage_matrix[int(current)].copy()

    def get_attack_matrix(self) -> np.ndarray:
        return self._attack_matrix.copy()

    def get_stage_matrix(self) -> np.ndarray:
        return self._stage_matrix.copy()
