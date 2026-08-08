"""
State vector builder — wraps CyberSecurityEnv.encode_state so the vector
passed to the policy stays consistent with training by construction.
"""

from __future__ import annotations

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from defender.honeypot import compute_threat_level
from environment.cyber_env import encode_state
from opencanary_integration.engine.session_tracker import SessionState


def build_state(session: SessionState, escalation_mode: str = "window") -> np.ndarray:
    """
    Build the 24-dim state vector from a SessionState.

    escalation_mode : "window" (default, matches today's behavior) uses
        `session.escalation_rate` (fraction of the last N events that were
        any attack); "ema" uses `session.escalation_ema` (severity-weighted
        exponential moving average) instead. The same rate feeds both the
        threat-level computation and state[19], so the two stay consistent.
    """
    escalation_rate = (
        session.escalation_ema if escalation_mode == "ema" else session.escalation_rate
    )
    threat_level = compute_threat_level(
        attack_type      = session.current_attack,
        kill_chain_stage = session.current_stage,
        escalation_rate  = escalation_rate,
        attack_count     = session.attack_count,
    )
    return encode_state(
        attack_type      = session.current_attack,
        kill_chain_stage = session.current_stage,
        threat_level     = threat_level,
        attack_count     = session.attack_count,
        escalation_rate  = escalation_rate,
        intent           = session.inferred_intent,
    )
