"""
Thesis-01 evaluation: cross-session reputation and the R4 override.

Simulates a population of source IPs returning to the honeypot across
repeated, separately-expired sessions. Each "offending" visit sends one
EXPLOITS-severity event (severity 0.70); after each visit's session is
force-expired (simulating TTL expiry between real-world visits), the IP
returns and opens its next visit with a RECONNAISSANCE-stage NORMAL-
looking event. We compare the action MatrixPolicy selects for that
opening event under two conditions:

  - "R4 active"  (reputation = session.reputation, current live-pipeline
                  behaviour)
  - "R4 disabled" (reputation forced to 0.0, i.e. the pre-R4 baseline,
                  where R1 unconditionally allows any NORMAL-looking event)

This directly measures how many of a returning offender's innocuous-
looking opening probes get escalated past ALLOW purely because of
accumulated reputation, and at what offense count R4 first overrides R1.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

from attacker.attack_types import AttackType, AttackerIntent, ATTACK_SEVERITY, KillChainStage
from defender.matrix_policy import MatrixPolicy, REPUTATION_THRESHOLD
from opencanary_integration.engine.session_tracker import SessionTracker

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "thesis01_eval")
os.makedirs(OUT_DIR, exist_ok=True)

N_IPS = 30
MAX_VISITS = 8
OFFENSE_ATTACK = AttackType.EXPLOITS   # severity 0.70


def main():
    mp = MatrixPolicy(default_intent=AttackerIntent.OPPORTUNISTIC)
    rows = []

    for ip_idx in range(N_IPS):
        src_ip = f"203.0.113.{ip_idx + 1}"
        tracker = SessionTracker(ttl_seconds=300, escalation_window=20)

        for visit in range(1, MAX_VISITS + 1):
            # The offending event for this visit.
            session = tracker.update(src_ip, OFFENSE_ATTACK)
            reputation_after_offense = session.reputation

            # Force the session to expire (simulate returning after the
            # TTL has lapsed) — but reputation persists, since it lives in
            # the separate ReputationTracker, not the expiring SessionState.
            tracker.remove(src_ip)

            # The IP's *next* visit opens with an innocuous-looking probe.
            opening_session = tracker.update(src_ip, AttackType.NORMAL)
            reputation_at_open = opening_session.reputation

            action_r4_on, info_on = mp.decide(
                current_stage=KillChainStage.RECONNAISSANCE,
                current_attack=AttackType.NORMAL,
                escalation_rate=0.0,
                reputation=reputation_at_open,
            )
            action_r4_off, info_off = mp.decide(
                current_stage=KillChainStage.RECONNAISSANCE,
                current_attack=AttackType.NORMAL,
                escalation_rate=0.0,
                reputation=0.0,
            )

            rows.append({
                "src_ip": src_ip, "offending_visits_so_far": visit,
                "reputation_after_offense": round(reputation_after_offense, 4),
                "reputation_at_next_open": round(reputation_at_open, 4),
                "action_r4_on": action_r4_on.name,
                "override_r4_on": info_on["override_applied"],
                "action_r4_off": action_r4_off.name,
                "override_r4_off": info_off["override_applied"],
                "r4_changed_outcome": action_r4_on != action_r4_off,
            })

            # Undo the extra NORMAL-event side effect on the tracked
            # session/reputation state before the next offending visit —
            # remove the just-created opening session so the next
            # iteration's offense count stays clean.
            tracker.remove(src_ip)

    # Summarize: at what offense count does R4 first override R1, and how
    # consistently across the 30 simulated IPs?
    by_visit = {}
    for r in rows:
        v = r["offending_visits_so_far"]
        by_visit.setdefault(v, {"n": 0, "r4_fired": 0, "reputations": []})
        by_visit[v]["n"] += 1
        by_visit[v]["r4_fired"] += int(r["r4_changed_outcome"])
        by_visit[v]["reputations"].append(r["reputation_at_next_open"])

    summary = []
    for v in sorted(by_visit):
        d = by_visit[v]
        mean_rep = sum(d["reputations"]) / len(d["reputations"])
        summary.append({
            "offending_visits_so_far": v,
            "mean_reputation_at_next_open": round(mean_rep, 4),
            "fraction_r4_overrides_r1": round(d["r4_fired"] / d["n"], 4),
        })
        print(f"visits={v}  mean_reputation={mean_rep:.4f}  "
              f"R4_overrides_R1_fraction={d['r4_fired']/d['n']:.4f}  "
              f"(REPUTATION_THRESHOLD={REPUTATION_THRESHOLD})")

    with open(os.path.join(OUT_DIR, "reputation_detail.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(OUT_DIR, "reputation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detail + summary -> {OUT_DIR}")


if __name__ == "__main__":
    main()
