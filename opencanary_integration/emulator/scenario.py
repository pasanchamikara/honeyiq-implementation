"""
EmulatorScenario — end-to-end pipeline demo using only dummy components.

Wires together:
    OpenCanaryEventGenerator  → produces synthetic events
    Pipeline engine           → logtype_map, session_tracker, state_builder,
                                policy_engine, escalation_predictor
    DummyHoneypot             → prints what opencanary would do

No network sockets, no FastAPI, no gRPC, no opencanary binary required.

Usage (CLI):
    python -m opencanary_integration.emulator.scenario
    python -m opencanary_integration.emulator.scenario --scenario kill_chain
    python -m opencanary_integration.emulator.scenario --scenario random --events 20
    python -m opencanary_integration.emulator.scenario --src-ip 10.0.0.1 --scenario ssh_brute
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from opencanary_integration.ingest.models import OpenCanaryEvent
from opencanary_integration.ingest.logtype_map import map_logtype
from opencanary_integration.engine.session_tracker import SessionTracker
from opencanary_integration.engine.escalation_predictor import EscalationPredictor
from opencanary_integration.engine.state_builder import build_state
from opencanary_integration.engine.policy_engine import PolicyEngine
from opencanary_integration.emulator.event_generator import (
    OpenCanaryEventGenerator, SCENARIO_CATEGORY,
)
from opencanary_integration.emulator.honeypot_emulator import DummyHoneypot
from attacker.attack_types import AttackerIntent

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


class EmulatorScenario:
    """
    End-to-end pipeline emulator.

    Parameters
    ----------
    model_dir : str
        Path to a trained model directory (must contain dqn_agent.pt).
    intent : str
        Attacker intent used by the escalation predictor
        (OPPORTUNISTIC | STEALTHY | AGGRESSIVE | TARGETED).
    audit_file : str | None
        If set, every decision is written as a JSON line to this path.
    verbose : bool
        Print per-event action banners.
    """

    def __init__(
        self,
        model_dir: str = "models/",
        intent: str = "OPPORTUNISTIC",
        audit_file: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.generator = OpenCanaryEventGenerator(seed=42)
        self.honeypot  = DummyHoneypot(audit_file=audit_file, verbose=verbose)

        self.session_tracker = SessionTracker(ttl_seconds=300, escalation_window=20)
        self.predictor = EscalationPredictor(
            intent=AttackerIntent[intent.upper()]
        )
        # PolicyEngine now uses MatrixPolicy — no model file required
        self.policy = PolicyEngine(model_dir=model_dir, default_intent=intent)

        self._verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_event(self, event: OpenCanaryEvent) -> dict:
        """
        Process a single OpenCanaryEvent through the full pipeline.

        Returns a decision dict with action, threat_level, stage, etc.
        """
        attack_type = map_logtype(event)
        session     = self.session_tracker.update(event.src_host, attack_type)
        state       = build_state(session)
        threat_level = float(state[17])

        action, _ = self.policy.decide(state)

        stage_probs  = self.predictor.next_stage_probs(session.current_stage).tolist()
        attack_probs = self.predictor.next_attack_probs(session.current_attack).tolist()
        esc_risk     = self.predictor.escalation_risk(session.current_stage)

        decision = {
            "event_id":        str(uuid4()),
            "src_ip":          event.src_host,
            "logtype":         event.logtype,
            "service":         event.service_name,
            "attack_type":     session.current_attack.name,
            "stage":           session.current_stage.name,
            "threat_level":    round(threat_level, 4),
            "escalation_risk": round(esc_risk, 4),
            "action":          action.name,
            "stage_probs":     [round(p, 4) for p in stage_probs],
            "attack_probs":    [round(p, 4) for p in attack_probs],
        }

        # Apply to dummy honeypot
        self.honeypot.apply_action_sync(
            src_ip       = event.src_host,
            action       = action.name,
            attack_type  = session.current_attack.name,
            stage        = session.current_stage.name,
            threat_level = threat_level,
            event_id     = decision["event_id"],
        )
        self.honeypot.schedule_reload_sync(
            urgent = action.name in ("BLOCK", "ALERT")
        )

        return decision

    def run_scenario(
        self,
        scenario: str,
        src_ip: Optional[str] = None,
        n: int = 1,
    ) -> list[dict]:
        """
        Run `n` events from the named scenario against the pipeline.

        Parameters
        ----------
        scenario : str
            e.g. "ssh_brute", "port_scan", "smb_access".
        src_ip : str | None
            Fixed attacker IP; random if None.
        n : int
            Number of events to generate.
        """
        if self._verbose:
            cat = SCENARIO_CATEGORY.get(scenario, "?")
            self._print_header(f"Scenario: {scenario}  ({cat})", n)

        decisions = []
        for _ in range(n):
            event = self.generator.generate(scenario, src_ip=src_ip)
            decisions.append(self.run_event(event))

        if self._verbose:
            self._print_summary(decisions)
        return decisions

    def run_kill_chain(
        self,
        src_ip: Optional[str] = None,
    ) -> list[dict]:
        """
        Run a full kill-chain sequence: recon → analysis → exploit → backdoor → shellcode.
        """
        events = self.generator.generate_kill_chain(src_ip=src_ip)
        if self._verbose:
            self._print_header("Full Kill-Chain Simulation", len(events))

        decisions = [self.run_event(ev) for ev in events]

        if self._verbose:
            self._print_summary(decisions)
        return decisions

    def run_random(
        self,
        n: int = 10,
        src_ip: Optional[str] = None,
    ) -> list[dict]:
        """
        Run `n` random-scenario events, optionally from the same IP.
        """
        import random
        scenarios = self.generator.available_scenarios()
        if self._verbose:
            self._print_header(f"Random Scenario Mix ({n} events)", n)

        decisions = []
        for _ in range(n):
            scenario = random.choice(scenarios)
            event    = self.generator.generate(scenario, src_ip=src_ip)
            decisions.append(self.run_event(event))

        if self._verbose:
            self._print_summary(decisions)
        return decisions

    def get_session_status(self) -> list[dict]:
        """Return current honeypot state for all seen IPs."""
        return self.honeypot.get_all_sessions()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header(title: str, n_events: int) -> None:
        print()
        print("=" * 75)
        print(f"  HoneyIQ OpenCanary Emulator — {title}")
        print(f"  Events: {n_events}")
        print("=" * 75)
        print(
            f"{'ACTION':<8} {'SRC_IP':<18} {'ATTACK_TYPE':<16} "
            f"{'STAGE':<22} {'THREAT':>7}  DETAIL"
        )
        print("-" * 75)

    @staticmethod
    def _print_summary(decisions: list[dict]) -> None:
        print("-" * 75)
        from collections import Counter
        action_counts = Counter(d["action"] for d in decisions)
        avg_threat    = sum(d["threat_level"] for d in decisions) / max(len(decisions), 1)
        avg_esc       = sum(d["escalation_risk"] for d in decisions) / max(len(decisions), 1)

        print(f"  Events processed : {len(decisions)}")
        print(f"  Action breakdown : "
              + "  ".join(f"{k}={v}" for k, v in sorted(action_counts.items())))
        print(f"  Avg threat level : {avg_threat:.3f}")
        print(f"  Avg esc. risk    : {avg_esc:.3f}")
        print("=" * 75)
        print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HoneyIQ OpenCanary emulator — runs the full pipeline without "
                    "a live honeypot."
    )
    p.add_argument("--model-dir", default="models/",
                   help="Path to trained DQN model directory")
    p.add_argument("--intent", default="OPPORTUNISTIC",
                   choices=["STEALTHY", "AGGRESSIVE", "TARGETED", "OPPORTUNISTIC"],
                   help="Attacker intent for escalation predictor")
    p.add_argument("--scenario", default="kill_chain",
                   help=(
                       "Scenario to run. Use 'kill_chain' for a staged sequence, "
                       "'random' for random events, or a specific name like 'ssh_brute'. "
                       f"Available: kill_chain, random, {', '.join(sorted(SCENARIO_CATEGORY))}"
                   ))
    p.add_argument("--events", type=int, default=7,
                   help="Number of events (for 'random' and single-scenario modes)")
    p.add_argument("--src-ip", default=None,
                   help="Fix attacker source IP (random if omitted)")
    p.add_argument("--audit-file", default=None,
                   help="Write JSONL audit log to this file")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-event output")
    return p


def main() -> None:
    args   = _build_parser().parse_args()
    emu    = EmulatorScenario(
        model_dir  = args.model_dir,
        intent     = args.intent,
        audit_file = args.audit_file,
        verbose    = not args.quiet,
    )

    if args.scenario == "kill_chain":
        emu.run_kill_chain(src_ip=args.src_ip)
    elif args.scenario == "random":
        emu.run_random(n=args.events, src_ip=args.src_ip)
    else:
        emu.run_scenario(args.scenario, src_ip=args.src_ip, n=args.events)


if __name__ == "__main__":
    main()
