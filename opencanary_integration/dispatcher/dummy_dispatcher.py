"""
Dummy dispatcher — replaces REST/gRPC with in-memory DummyHoneypot calls.

Satisfies the same async interface as RESTDispatcher and GRPCDispatcher so
the Pipeline can select it via ``dispatcher_mode = "dummy"``.  No network
connections, no external services required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opencanary_integration.dispatcher.models import DecisionPayload
from opencanary_integration.emulator.honeypot_emulator import DummyHoneypot

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class DummyDispatcher:
    """
    In-process dispatcher backed by DummyHoneypot.

    Instead of sending an HTTP POST or gRPC call, every dispatch call is
    forwarded directly to ``DummyHoneypot.apply_action_sync``.  This lets
    the full Pipeline run without any live services.

    Parameters
    ----------
    audit_file : str | None
        Path for the JSONL audit log written by DummyHoneypot.
    verbose : bool
        Print coloured action banners to stdout.
    """

    def __init__(
        self,
        audit_file: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.honeypot = DummyHoneypot(audit_file=audit_file, verbose=verbose)

    # ------------------------------------------------------------------
    # Lifecycle (no-ops — nothing to open/close)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        log.info("[DummyDispatcher] started (in-process, no network)")

    async def stop(self) -> None:
        log.info("[DummyDispatcher] stopped")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, payload: DecisionPayload) -> bool:
        """
        Apply the decision to the in-memory honeypot.

        Always returns True (the call cannot fail due to network errors).
        Mirrors the real dispatchers' signature so Pipeline works unchanged.
        """
        self.honeypot.apply_action_sync(
            src_ip       = payload.src_ip,
            action       = payload.action,
            attack_type  = payload.attack_type,
            stage        = payload.stage,
            threat_level = payload.threat_level,
            event_id     = payload.event_id,
        )
        urgent = payload.action in ("BLOCK", "ALERT")
        self.honeypot.schedule_reload_sync(urgent=urgent)

        log.info(
            "[DummyDispatcher] dispatched action=%s src=%s stage=%s threat=%.3f",
            payload.action, payload.src_ip, payload.stage, payload.threat_level,
        )
        return True

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_action_log(self) -> list[dict]:
        """Return the full action audit log from the underlying honeypot."""
        return self.honeypot.get_action_log()

    def get_all_sessions(self) -> list[dict]:
        """Return current IP state for all seen source addresses."""
        return self.honeypot.get_all_sessions()

    def get_ip_status(self, src_ip: str) -> dict:
        """Return current emulated state for a single IP."""
        return self.honeypot.get_ip_status(src_ip)

    def reset(self) -> None:
        """Clear all IP state (useful between evaluation runs)."""
        self.honeypot._state.clear()
        self.honeypot._action_log.clear()
