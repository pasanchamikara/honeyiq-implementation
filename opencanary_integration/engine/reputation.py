"""
Persistent, cross-session reputation tracking per source IP.

SessionState/SessionTracker's sliding window and EMA are scoped to a single
session and expire with it (SessionTracker's TTL sweep). ReputationTracker
is deliberately separate: it remembers a source IP across sessions, with a
time decay instead of a hard expiry — analogous to fail2ban's escalating
memory of a repeat offender. In-memory only, no external dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class ReputationEntry:
    score:       float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReputationTracker:
    """
    Time-decayed offense score per source IP.

    `record_offense(src_ip, severity)` bumps the score by
    `offense_increment * severity` (severity in [0, 1] — pass
    `ATTACK_SEVERITY[attack_type]`; 0.0 for NORMAL traffic still applies
    decay via the read, just no increment), clamped to `max_score`. Decay
    is a plain half-life formula (`score * 0.5 ** (elapsed / half_life)`),
    applied lazily whenever an entry is read — a human can hand-verify it
    with a calculator, unlike a learned function.
    """

    def __init__(
        self,
        decay_half_life_seconds: float = 6 * 3600,
        offense_increment:       float = 0.25,
        max_score:                float = 1.0,
        stale_after_seconds:      float = 30 * 24 * 3600,
        sweep_interval_seconds:   int   = 300,
    ) -> None:
        self._half_life         = decay_half_life_seconds
        self._increment          = offense_increment
        self._max_score          = max_score
        self._stale_after         = timedelta(seconds=stale_after_seconds)
        self._sweep_interval      = timedelta(seconds=sweep_interval_seconds)
        self._entries: dict[str, ReputationEntry] = {}
        self._last_sweep = datetime.min.replace(tzinfo=timezone.utc)

    def record_offense(self, src_ip: str, severity: float) -> float:
        """Decay the existing entry, add `offense_increment * severity`,
        clamp, store, and return the resulting score."""
        self._sweep_stale()
        entry = self._decayed_entry(src_ip)
        entry.score = min(self._max_score, entry.score + self._increment * severity)
        entry.last_update = datetime.now(timezone.utc)
        self._entries[src_ip] = entry
        return entry.score

    def get(self, src_ip: str) -> float:
        """Return the current decayed score without recording an offense."""
        return self._decayed_entry(src_ip).score

    def reset(self, src_ip: str) -> None:
        """Clear a source IP's reputation entirely (e.g. manual pardon)."""
        self._entries.pop(src_ip, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decayed_entry(self, src_ip: str) -> ReputationEntry:
        entry = self._entries.get(src_ip)
        if entry is None:
            return ReputationEntry(score=0.0)
        elapsed = (datetime.now(timezone.utc) - entry.last_update).total_seconds()
        decayed_score = entry.score * (0.5 ** (elapsed / self._half_life))
        return ReputationEntry(score=decayed_score, last_update=entry.last_update)

    def _sweep_stale(self) -> None:
        """Throttled sweep dropping entries untouched for `stale_after`,
        same throttle pattern as SessionTracker._expire_old_sessions."""
        now = datetime.now(timezone.utc)
        if now - self._last_sweep < self._sweep_interval:
            return
        self._last_sweep = now

        cutoff = now - self._stale_after
        stale = [ip for ip, e in self._entries.items() if e.last_update < cutoff]
        for ip in stale:
            del self._entries[ip]
