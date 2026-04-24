"""
State vector builder — replicates CyberSecurityEnv._build_state so the
24-dim vector passed to the policy is consistent with training.
"""

from __future__ import annotations

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from attacker.attack_types import AttackType, KillChainStage, AttackerIntent
from defender.honeypot import compute_threat_level
from opencanary_integration.engine.session_tracker import SessionState

STATE_DIM = 24


def build_state(session: SessionState) -> np.ndarray:
    """Build the 24-dim state vector from a SessionState."""
    state = np.zeros(STATE_DIM, dtype=np.float32)

    state[int(session.current_attack)]        = 1.0          # [0:10]
    state[10 + int(session.current_stage)]    = 1.0          # [10:17]
    state[17] = compute_threat_level(
        attack_type      = session.current_attack,
        kill_chain_stage = session.current_stage,
        escalation_rate  = session.escalation_rate,
        attack_count     = session.attack_count,
    )
    state[18] = float(min(1.0, session.attack_count / 100.0)) # [18]
    state[19] = float(session.escalation_rate)                # [19]
    state[20 + int(session.inferred_intent)]  = 1.0          # [20:24]

    return state
