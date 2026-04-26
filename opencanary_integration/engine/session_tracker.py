"""
Per-IP session state tracker for the OpenCanary integration pipeline.
"""

from __future__ import annotations

import logging
import sys, os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from attacker.attack_types import AttackType, KillChainStage, AttackerIntent
from opencanary_integration.ingest.logtype_map import initial_stage_for

log = logging.getLogger(__name__)


@dataclass
class SessionState:
    src_ip:          str
    current_attack:  AttackType     = AttackType.RECONNAISSANCE
    current_stage:   KillChainStage = KillChainStage.RECONNAISSANCE
    attack_count:    int            = 0
    event_count:     int            = 0
    recent_attacks:  deque          = field(default_factory=lambda: deque(maxlen=20))
    last_seen:       datetime       = field(default_factory=datetime.utcnow)
    inferred_intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC

    @property
    def escalation_rate(self) -> float:
        if not self.recent_attacks:
            return 0.0
        return sum(self.recent_attacks) / len(self.recent_attacks)


class SessionTracker:
    def __init__(self, ttl_seconds: int = 3600, escalation_window: int = 20) -> None:
        self._sessions:   dict[str, SessionState] = {}
        self._ttl         = timedelta(seconds=ttl_seconds)
        self._window_size = escalation_window

    def update(self, src_ip: str, attack_type: AttackType) -> SessionState:
        self._expire_old_sessions()
        session = self._sessions.get(src_ip)

        if session is None:
            session = SessionState(
                src_ip=src_ip,
                current_attack=attack_type,
                current_stage=initial_stage_for(attack_type),
                recent_attacks=deque(maxlen=self._window_size),
            )
            self._sessions[src_ip] = session
        else:
            session.current_attack = attack_type
            implied = initial_stage_for(attack_type)
            if int(implied) > int(session.current_stage):
                session.current_stage = implied

        is_attack = attack_type != AttackType.NORMAL
        session.recent_attacks.append(int(is_attack))
        if is_attack:
            session.attack_count += 1
        session.event_count += 1
        session.last_seen = datetime.now(timezone.utc)
        return session

    def get(self, src_ip: str) -> Optional[SessionState]:
        self._expire_old_sessions()
        return self._sessions.get(src_ip)

    def remove(self, src_ip: str) -> None:
        self._sessions.pop(src_ip, None)

    def all_sessions(self) -> dict[str, SessionState]:
        self._expire_old_sessions()
        return dict(self._sessions)

    def _expire_old_sessions(self) -> None:
        cutoff  = datetime.now(timezone.utc) - self._ttl
        expired = [ip for ip, s in self._sessions.items() if s.last_seen < cutoff]
        for ip in expired:
            del self._sessions[ip]
