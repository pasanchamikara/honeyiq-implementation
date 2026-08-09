"""
Generate the 3 new figures for thesis-01 from the eval_scripts/*.json results.
Matches the color palette used in evaluate.py (INTENT_COLORS) for visual
consistency with the existing thesis figures.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "thesis01_eval")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

INTENT_COLORS = {
    "STEALTHY":      "#4CAF50",
    "AGGRESSIVE":    "#F44336",
    "TARGETED":      "#FF9800",
    "OPPORTUNISTIC": "#2196F3",
}


def fig_escalation_mode():
    with open(os.path.join(RESULTS_DIR, "escalation_mode_comparison.json")) as f:
        data = json.load(f)

    intents = ["STEALTHY", "AGGRESSIVE", "TARGETED", "OPPORTUNISTIC"]
    window_rates = [next(d for d in data if d["intent"] == i and d["mode"] == "window")["r3_trigger_rate"] for i in intents]
    ema_rates    = [next(d for d in data if d["intent"] == i and d["mode"] == "ema")["r3_trigger_rate"] for i in intents]

    x = np.arange(len(intents))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, window_rates, width, label="window (default)", color="#607D8B")
    ax.bar(x + width/2, ema_rates, width, label="severity-weighted EMA", color="#26A69A")
    ax.set_xticks(x); ax.set_xticklabels(intents, fontsize=10)
    ax.set_ylabel("R3 (\"high attack rate\") trigger rate")
    ax.set_title("R3 Override Trigger Rate: Window vs. EMA Escalation Tracking\n(30 episodes × 200 steps, continuous attack traffic)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for i, v in enumerate(window_rates):
        ax.text(i - width/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(ema_rates):
        ax.text(i + width/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "escalation_mode_r3_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig_reputation_crossing():
    with open(os.path.join(RESULTS_DIR, "reputation_summary.json")) as f:
        data = json.load(f)

    visits = [d["offending_visits_so_far"] for d in data]
    reps   = [d["mean_reputation_at_next_open"] for d in data]
    frac_r4 = [d["fraction_r4_overrides_r1"] for d in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(visits, reps, "o-", color="#E53935", linewidth=2, markersize=7, label="Mean reputation score")
    ax1.axhline(0.60, color="#424242", linestyle="--", linewidth=1.2, label="REPUTATION_THRESHOLD (0.60)")
    ax1.set_xlabel("Offending visits so far")
    ax1.set_ylabel("Reputation score", color="#E53935")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#E53935")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(visits, frac_r4, "s--", color="#1E88E5", linewidth=2, markersize=6, label="Fraction of IPs where R4 overrides R1")
    ax2.set_ylabel("Fraction of IPs where R4 fires", color="#1E88E5")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#1E88E5")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    ax1.set_title("Cross-Session Reputation Growth and the R4 Override\n(30 simulated source IPs, EXPLOITS-severity offenses)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "reputation_threshold_crossing.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig_adaptive_thresholds():
    with open(os.path.join(RESULTS_DIR, "adaptive_thresholds_trace.json")) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, res in zip(axes, results):
        static_rates = [w["r3_rate"] for w in res["static_windows"]]
        adaptive_rates = [w["r3_rate"] for w in res["adaptive_windows"]]
        w_idx = list(range(len(static_rates)))
        ax.plot(w_idx, static_rates, "-", color="#78909C", linewidth=1.5, label="static RATE_THRESHOLD")
        ax.plot(w_idx, adaptive_rates, "-", color="#43A047", linewidth=1.5, label="AdaptiveThresholds")
        ax.axhline(0.10, color="#424242", linestyle="--", linewidth=1, label="target_rate (0.10)")
        ax.set_title(res["label"].replace("_", " ").title() + f"\n(benign_ratio={res['benign_ratio']})", fontsize=11)
        ax.set_xlabel("Observation window (200 decisions each)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("R3 trigger rate")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("AdaptiveThresholds: R3 Trigger Rate Under Static vs. Adaptive RATE_THRESHOLD", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "adaptive_thresholds_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    fig_escalation_mode()
    fig_reputation_crossing()
    fig_adaptive_thresholds()
