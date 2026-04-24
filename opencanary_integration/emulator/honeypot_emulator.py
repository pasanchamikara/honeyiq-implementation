"""
Dummy honeypot behavior service.

Replaces the real OpenCanary config-mutation + reload logic with in-memory
state and console output.  All methods have the same signatures as the real
ConfManager / behavior service so the pipeline can call them transparently.

This module requires NO external dependencies (no FastAPI, uvicorn, grpcio).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emulated config state
# ---------------------------------------------------------------------------

@dataclass
class _IPState:
    action:     str = "ALLOW"
    blocked:    bool = False
    trolled:    bool = False
    logged:     bool = True
    alerted:    bool = False
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_count: int = 0
    notes: list[str] = field(default_factory=list)


class DummyHoneypot:
    """
    In-memory honeypot emulator.  Mirrors the interface of ConfManager so the
    pipeline can call ``apply_action(src_ip, action)`` without a real OpenCanary
    process.

    State is kept in a dict keyed by source IP.  All mutations are logged to
    the console and optionally to a JSON audit file.

    Parameters
    ----------
    audit_file : str | None
        If set, every action is appended as a JSON line to this file.
    verbose : bool
        Print action banners to stdout (True by default).
    """

    # Action → ASCII banner colour prefix
    _BANNERS: dict[str, str] = {
        "ALLOW": "\033[32m[  ALLOW ]\033[0m",   # green
        "LOG":   "\033[34m[  LOG   ]\033[0m",   # blue
        "TROLL": "\033[33m[  TROLL ]\033[0m",   # yellow
        "BLOCK": "\033[31m[  BLOCK ]\033[0m",   # red
        "ALERT": "\033[35m[  ALERT ]\033[0m",   # magenta
    }

    def __init__(
        self,
        audit_file: str | None = None,
        verbose: bool = True,
    ) -> None:
        self._state:      dict[str, _IPState] = defaultdict(_IPState)
        self._audit_file  = audit_file
        self._verbose     = verbose
        self._lock        = asyncio.Lock()
        self._action_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core interface (mirrors ConfManager)
    # ------------------------------------------------------------------

    async def apply_action(
        self,
        src_ip: str,
        action: str,
        attack_type: str = "UNKNOWN",
        stage: str = "UNKNOWN",
        threat_level: float = 0.0,
        event_id: str = "",
    ) -> None:
        """
        Apply a honeypot action for the given source IP.

        In a real deployment this mutates opencanary.conf and triggers a
        reload.  Here it updates in-memory state and emits a log record.
        """
        async with self._lock:
            state = self._state[src_ip]
            state.last_seen   = datetime.now(timezone.utc)
            state.event_count += 1
            state.action      = action

            self._dispatch(src_ip, action, state, attack_type, stage,
                           threat_level, event_id)

    def apply_action_sync(
        self,
        src_ip: str,
        action: str,
        attack_type: str = "UNKNOWN",
        stage: str = "UNKNOWN",
        threat_level: float = 0.0,
        event_id: str = "",
    ) -> None:
        """Synchronous variant for use outside asyncio contexts."""
        state = self._state[src_ip]
        state.last_seen   = datetime.now(timezone.utc)
        state.event_count += 1
        state.action      = action
        self._dispatch(src_ip, action, state, attack_type, stage,
                       threat_level, event_id)

    # ------------------------------------------------------------------
    # Individual action handlers (dummy implementations)
    # ------------------------------------------------------------------

    def _allow(self, src_ip: str, state: _IPState) -> None:
        """[DUMMY] Allow traffic — no-op in emulator (real: remove from blocklist)."""
        state.blocked = False
        state.notes.append("Traffic allowed (blocklist cleared if present)")

    def _log(self, src_ip: str, state: _IPState) -> None:
        """[DUMMY] Enable enhanced logging — real: add file handler to opencanary.conf."""
        state.logged = True
        state.notes.append("Enhanced logging enabled (would add FILE handler)")

    def _troll(self, src_ip: str, state: _IPState) -> None:
        """
        [DUMMY] Engage troll mode — real: sets http.skin='nasty', changes SSH
        version string to an ancient banner, modifies FTP welcome message.
        """
        state.trolled = True
        state.notes.append(
            "Troll mode engaged: "
            "http.skin→'nasty', "
            "ssh.version→'SSH-2.0-OpenSSH_5.1p1 Debian-5', "
            "ftp.banner→'Connection from banned host logged'"
        )

    def _block(self, src_ip: str, state: _IPState) -> None:
        """
        [DUMMY] Block IP — real: appends src_ip to ip.ignorelist in
        opencanary.conf, adds iptables DROP rule, triggers immediate reload.
        """
        state.blocked = True
        state.notes.append(
            f"IP blocked: would append {src_ip} to opencanary.conf ip.ignorelist "
            f"and add iptables DROP rule"
        )

    def _alert(self, src_ip: str, state: _IPState) -> None:
        """
        [DUMMY] Fire alert — real: appends src_ip to ip.ignorelist, writes a
        JSONL alert record to the alert log, triggers immediate reload and
        optionally sends a webhook/email notification.
        """
        state.blocked = True
        state.alerted = True
        state.notes.append(
            f"ALERT fired: {src_ip} blocked + JSONL alert record written + "
            f"webhook notification would be sent"
        )

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_ip_status(self, src_ip: str) -> dict[str, Any]:
        """Return current emulated state for an IP."""
        if src_ip not in self._state:
            return {"src_ip": src_ip, "known": False}
        s = self._state[src_ip]
        return {
            "src_ip":      src_ip,
            "known":       True,
            "action":      s.action,
            "blocked":     s.blocked,
            "trolled":     s.trolled,
            "logged":      s.logged,
            "alerted":     s.alerted,
            "first_seen":  s.first_seen.isoformat(),
            "last_seen":   s.last_seen.isoformat(),
            "event_count": s.event_count,
            "notes":       s.notes[-5:],  # last 5 notes
        }

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Return status for all known IPs."""
        return [self.get_ip_status(ip) for ip in self._state]

    def get_action_log(self) -> list[dict[str, Any]]:
        """Return full action audit log."""
        return list(self._action_log)

    def clear_ip(self, src_ip: str) -> bool:
        """Remove an IP from state (simulate session expiry)."""
        if src_ip in self._state:
            del self._state[src_ip]
            return True
        return False

    # ------------------------------------------------------------------
    # Reload stub (matches ConfManager.schedule_reload signature)
    # ------------------------------------------------------------------

    async def schedule_reload(self, urgent: bool = False) -> None:
        """
        [DUMMY] Simulate an opencanary reload.

        Real: runs ``opencanaryctl restart`` or sends SIGHUP; updates are
        picked up from the modified opencanary.conf.
        """
        delay = 0 if urgent else 0.1
        await asyncio.sleep(delay)
        log.info("[EMULATOR] opencanary reload triggered (urgent=%s) — no-op in emulator",
                 urgent)

    def schedule_reload_sync(self, urgent: bool = False) -> None:
        """Synchronous reload stub."""
        log.info("[EMULATOR] opencanary reload triggered (urgent=%s) — no-op in emulator",
                 urgent)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        src_ip: str,
        action: str,
        state: _IPState,
        attack_type: str,
        stage: str,
        threat_level: float,
        event_id: str,
    ) -> None:
        handler = {
            "ALLOW": self._allow,
            "LOG":   self._log,
            "TROLL": self._troll,
            "BLOCK": self._block,
            "ALERT": self._alert,
        }.get(action, self._log)
        handler(src_ip, state)

        record = {
            "ts":           datetime.now(timezone.utc).isoformat(),
            "event_id":     event_id,
            "src_ip":       src_ip,
            "action":       action,
            "attack_type":  attack_type,
            "stage":        stage,
            "threat_level": round(threat_level, 4),
            "event_count":  state.event_count,
        }
        self._action_log.append(record)

        if self._verbose:
            banner = self._BANNERS.get(action, "[??????]")
            print(
                f"{banner} {src_ip:<18} "
                f"attack={attack_type:<16} stage={stage:<22} "
                f"threat={threat_level:.3f}"
            )
            if state.notes:
                print(f"           └─ {state.notes[-1]}")

        if self._audit_file:
            with open(self._audit_file, "a") as fh:
                fh.write(json.dumps(record) + "\n")

        log.info(
            "Dummy honeypot: %s src=%s attack=%s stage=%s threat=%.3f",
            action, src_ip, attack_type, stage, threat_level,
        )
