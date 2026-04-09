"""
Honeypot action definitions, threat level computation, and the reward
function that drives the DQN's learning signal.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict

from attacker.attack_types import (
    AttackType,
    KillChainStage,
    ATTACK_SEVERITY,
    KILL_CHAIN_WEIGHT,
)


# ---------------------------------------------------------------------------
# Honeypot actions
# ---------------------------------------------------------------------------

class HoneypotAction(IntEnum):
    ALLOW = 0   # Let traffic through untouched
    LOG   = 1   # Record and monitor the session
    TROLL = 2   # Respond with fake data / tarpit the attacker
    BLOCK = 3   # Drop / firewall the connection
    ALERT = 4   # Trigger a high-priority security alert

    @classmethod
    def names(cls) -> list[str]:
        return [e.name for e in cls]

    @classmethod
    def count(cls) -> int:
        return len(cls)


# ---------------------------------------------------------------------------
# Threat level computation
# ---------------------------------------------------------------------------

# Threat band boundaries
_THRESHOLDS = {
    "critical": 0.75,
    "high":     0.55,
    "medium":   0.35,
    "low":      0.15,
}


def threat_band(threat_level: float) -> str:
    """Classify a continuous threat level into a named band."""
    if threat_level >= _THRESHOLDS["critical"]:
        return "critical"
    elif threat_level >= _THRESHOLDS["high"]:
        return "high"
    elif threat_level >= _THRESHOLDS["medium"]:
        return "medium"
    elif threat_level >= _THRESHOLDS["low"]:
        return "low"
    else:
        return "benign"


def compute_threat_level(
    attack_type: AttackType,
    kill_chain_stage: KillChainStage,
    escalation_rate: float,
    attack_count: int,
) -> float:
    """
    Compute a composite threat level in [0.0, 1.0].

    Formula weights:
        45% — attack severity (how dangerous the attack type is)
        35% — kill chain stage weight (how far into the chain)
        15% — escalation rate (frequency of attacks in recent window)
         5% — cumulative attack count (sustained pressure)

    Parameters
    ----------
    attack_type      : current attack type
    kill_chain_stage : current kill chain stage
    escalation_rate  : fraction of recent steps that contained attacks [0,1]
    attack_count     : total non-NORMAL attacks so far in the episode
    """
    base_severity  = ATTACK_SEVERITY[attack_type]
    stage_weight   = KILL_CHAIN_WEIGHT[kill_chain_stage]
    count_factor   = min(1.0, attack_count / 100.0)

    threat = (
        0.45 * base_severity
        + 0.35 * stage_weight
        + 0.15 * escalation_rate
        + 0.05 * count_factor
    )
    return float(min(max(threat, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

# Base reward matrix: reward[action][band]
_REWARD_MATRIX: Dict[int, Dict[str, float]] = {
    HoneypotAction.ALLOW: {
        "benign":   1.0,
        "low":      0.5,
        "medium":  -1.0,
        "high":    -3.0,
        "critical":-6.0,
    },
    HoneypotAction.LOG: {
        "benign":   0.2,
        "low":      1.5,
        "medium":   2.0,
        "high":     1.0,
        "critical": -1.0,
    },
    HoneypotAction.TROLL: {
        "benign":  -1.0,
        "low":      1.0,
        "medium":   3.0,
        "high":     2.5,
        "critical":  0.5,
    },
    HoneypotAction.BLOCK: {
        "benign":  -2.0,
        "low":     -0.5,
        "medium":   1.5,
        "high":     3.5,
        "critical":  5.0,
    },
    HoneypotAction.ALERT: {
        "benign":  -3.0,
        "low":     -1.0,
        "medium":   0.5,
        "high":     2.0,
        "critical":  6.0,
    },
}


def compute_reward(
    action: int,
    threat_level: float,
    is_attack: bool,
    kill_chain_stage: KillChainStage,
    attack_type: AttackType,
) -> float:
    """
    Compute the reward for taking *action* given the current threat state.

    Design principles:
    - Correct responses yield positive reward; mismatches yield negative.
    - TROLL is optimal for medium threats (gather intel, waste attacker time).
    - BLOCK/ALERT are optimal for high/critical threats.
    - LOG is optimal for low threats (collect intelligence without disruption).
    - ALLOW is only correct for benign traffic.
    - Late kill chain stages apply a 1.5× multiplier on negative rewards.
    - Specific attack-type bonuses for honeypot-specific value.
    """
    band  = threat_band(threat_level)
    action_enum = HoneypotAction(action)
    reward = _REWARD_MATRIX[action_enum][band]

    # Late kill chain stages amplify negative rewards (higher stakes)
    late_stages = {
        KillChainStage.INSTALLATION,
        KillChainStage.COMMAND_AND_CTRL,
        KillChainStage.ACTIONS_ON_OBJ,
    }
    if kill_chain_stage in late_stages and reward < 0:
        reward *= 1.5

    # Honeypot-specific bonuses
    if action_enum == HoneypotAction.TROLL and attack_type in (
        AttackType.BACKDOORS, AttackType.SHELLCODE, AttackType.WORMS
    ):
        reward += 0.8   # Trolling persistent/spreading attacks is especially valuable

    if action_enum == HoneypotAction.BLOCK and attack_type == AttackType.WORMS:
        reward += 1.0   # Containment bonus for worms

    if action_enum == HoneypotAction.LOG and attack_type == AttackType.RECONNAISSANCE:
        reward += 0.5   # Intelligence value of logging early-stage recon

    # Small bonus for correctly allowing normal traffic
    if action_enum == HoneypotAction.ALLOW and not is_attack:
        reward += 0.5

    return float(reward)
