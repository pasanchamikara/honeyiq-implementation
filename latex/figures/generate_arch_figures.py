"""
Generate architecture diagrams for HoneyIQ.
Run from project root: python latex/figures/generate_arch_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__))

# ─── colour palette ───────────────────────────────────────────────────────────
C = {
    "bg":      "#F8F9FA",
    "attacker":"#C0392B",
    "defender":"#2471A3",
    "env":     "#1E8449",
    "metrics": "#7D3C98",
    "neutral": "#2C3E50",
    "arrow":   "#555555",
    "light_a": "#FADBD8",
    "light_d": "#D6EAF8",
    "light_e": "#D5F5E3",
    "light_m": "#E8DAEF",
    "light_n": "#EAECEE",
    "sedm":    "#E67E22",
    "light_s": "#FDEBD0",
    "dqn":     "#2471A3",
    "clf":     "#1A5276",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, facecolor, edgecolor, lw=1.5, radius=0.04, zorder=3):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _label(ax, x, y, text, fontsize=9, color="black",
           bold=False, zorder=5, ha="center", va="center"):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha=ha, va=va, weight=weight, zorder=zorder,
            wrap=True)


def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.4,
           arrowstyle="-|>", mutation_scale=12, zorder=4, label=""):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=arrowstyle, color=color,
                        lw=lw, mutation_scale=mutation_scale),
        zorder=zorder,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.02, my, label, fontsize=7, color=color,
                ha="left", va="center", style="italic", zorder=5)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — System Architecture Overview
# ═════════════════════════════════════════════════════════════════════════════

def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    ax.set_title("HoneyIQ — System Architecture Overview",
                 fontsize=14, weight="bold", pad=12, color=C["neutral"])

    # ── Environment shell ──
    env_rect = FancyBboxPatch((0.3, 0.3), 11.4, 6.2,
                              boxstyle="round,pad=0.05",
                              facecolor="#EBF5FB", edgecolor=C["env"],
                              linewidth=2, zorder=1, linestyle="--")
    ax.add_patch(env_rect)
    ax.text(6, 6.65, "CyberSecurityEnv  (Gymnasium Wrapper)",
            fontsize=10, ha="center", color=C["env"], weight="bold", zorder=5)

    # ── Attacker block ──
    _box(ax, 2.1, 3.5, 2.8, 3.6, C["light_a"], C["attacker"], lw=2)
    _label(ax, 2.1, 5.15, "ATTACKER", fontsize=10, color=C["attacker"], bold=True)
    for i, txt in enumerate(["AttackerAgent", "TransitionModel\n(Markov Chain)",
                              "Kill Chain\n7 Stages", "4 Intent Profiles"]):
        _box(ax, 2.1, 4.6 - i * 0.8, 2.2, 0.55, "white", C["attacker"], lw=1, radius=0.03)
        _label(ax, 2.1, 4.6 - i * 0.8, txt, fontsize=7.5)

    # ── Defender block ──
    _box(ax, 9.9, 3.5, 2.8, 3.6, C["light_d"], C["defender"], lw=2)
    _label(ax, 9.9, 5.15, "DEFENDER", fontsize=10, color=C["defender"], bold=True)
    sub = [("SEDM Policy\n(Primary)", C["sedm"]),
           ("DQN Agent\n(Baseline)", C["dqn"]),
           ("RF Classifier", C["clf"]),
           ("Reward\nFunction", C["defender"])]
    for i, (txt, ec) in enumerate(sub):
        _box(ax, 9.9, 4.6 - i * 0.8, 2.2, 0.55, "white", ec, lw=1, radius=0.03)
        _label(ax, 9.9, 4.6 - i * 0.8, txt, fontsize=7.5)

    # ── State / Observation box ──
    _box(ax, 6.0, 4.7, 2.8, 1.2, C["light_e"], C["env"], lw=1.5)
    _label(ax, 6.0, 5.05, "State Vector  (24-dim)", fontsize=8.5, color=C["env"], bold=True)
    _label(ax, 6.0, 4.7,
           "[0:10] attack_type (one-hot)\n[10:17] kill_chain_stage\n"
           "[17:20] threat, count, esc_rate\n[20:24] intent (one-hot)",
           fontsize=6.5, color="#1A5276")

    # ── Action box ──
    _box(ax, 6.0, 3.1, 2.8, 1.0, C["light_s"], C["sedm"], lw=1.5)
    _label(ax, 6.0, 3.38, "Action Space  (discrete 5)", fontsize=8.5, color=C["sedm"], bold=True)
    _label(ax, 6.0, 3.05, "ALLOW · LOG · TROLL · BLOCK · ALERT", fontsize=7)

    # ── Metrics box ──
    _box(ax, 6.0, 1.6, 3.2, 0.9, C["light_m"], C["metrics"], lw=1.5)
    _label(ax, 6.0, 1.85, "Metrics Collector", fontsize=8.5, color=C["metrics"], bold=True)
    _label(ax, 6.0, 1.55, "Detection Rate · FP Rate · Reward · CSV / PNG Export",
           fontsize=6.8)

    # ── Arrows ──
    # Attacker → state
    _arrow(ax, 3.5, 4.7, 4.6, 4.7, C["attacker"], label="attack event")
    # State → defender
    _arrow(ax, 7.4, 4.7, 8.5, 4.7, C["env"], label="observation")
    # Defender → action
    _arrow(ax, 8.5, 3.1, 7.4, 3.1, C["defender"], label="action")
    # Action → env
    _arrow(ax, 4.6, 3.1, 3.5, 3.1, C["env"], label="reward + next state")
    # Env → metrics
    _arrow(ax, 6.0, 2.6, 6.0, 2.05, C["neutral"], label="")
    ax.text(6.15, 2.32, "step records", fontsize=7, color=C["neutral"],
            ha="left", va="center", style="italic")

    plt.tight_layout()
    path = os.path.join(OUT, "arch_system_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — SEDM Decision Flow
# ═════════════════════════════════════════════════════════════════════════════

def fig_sedm_flow():
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_xlim(0, 9); ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_title("SEDM — Stage-Escalation Decision Matrix\nDecision Flow",
                 fontsize=13, weight="bold", pad=10, color=C["neutral"])

    def step_box(x, y, w, h, title, body, step_num=None, fc="white", ec=C["sedm"]):
        _box(ax, x, y, w, h, fc, ec, lw=2)
        ty = y + h / 2 - 0.22
        if step_num:
            ax.text(x - w / 2 + 0.18, ty, step_num,
                    fontsize=9, weight="bold", color=ec,
                    ha="left", va="center", zorder=6)
        ax.text(x, ty, title, fontsize=9, weight="bold",
                color=ec, ha="center", va="center", zorder=6)
        ax.text(x, y - 0.05, body, fontsize=7.5, ha="center", va="center",
                color=C["neutral"], zorder=6,
                multialignment="center")

    steps = [
        (4.5, 10.0, 7.2, 0.8, "STEP 1 — Compute Escalation Risk",
         "esc_risk = Σ P(next_stage = s')  for all s' > current_stage",
         "①", "#FEF9E7", C["sedm"]),
        (4.5, 8.6,  7.2, 0.8, "STEP 2 — Band Classification",
         "Low: esc_risk < 0.35    Medium: 0.35 – 0.65    High: ≥ 0.65",
         "②", "#FDFEFE", C["sedm"]),
        (4.5, 7.2,  7.2, 0.8, "STEP 3 — Matrix Lookup  (7 stages × 3 bands)",
         "stage × band → base action   (see heatmap figure)",
         "③", "#FDFEFE", C["sedm"]),
        (4.5, 5.8,  7.2, 0.8, "STEP 4 — Override Rules",
         "R1: NORMAL → ALLOW always\n"
         "R2: DOS / WORMS → upgrade action +1\n"
         "R3: esc_rate > 0.80 → upgrade action +1",
         "④", "#FEF9E7", "#E74C3C"),
        (4.5, 4.0,  7.2, 1.0, "STEP 5 — Composite Risk (log only)",
         "risk = 0.35×stage_weight + 0.35×esc_risk\n"
         "     + 0.15×attack_severity + 0.15×esc_rate",
         "⑤", "#FDFEFE", C["metrics"]),
    ]

    for (x, y, w, h, title, body, num, fc, ec) in steps:
        step_box(x, y, w, h, title, body, num, fc, ec)

    # arrows between steps
    for top_y, bot_y in [(9.6, 9.0), (8.2, 7.6), (6.8, 6.2), (5.4, 4.5)]:
        _arrow(ax, 4.5, top_y, 4.5, bot_y, C["sedm"], lw=1.8)

    # Final action output
    _box(ax, 4.5, 2.7, 4.0, 0.7, "#FDFEFE", "#1E8449", lw=2.5)
    _label(ax, 4.5, 2.7, "Final Action  →  ALLOW / LOG / TROLL / BLOCK / ALERT",
           fontsize=9, color="#1E8449", bold=True)
    _arrow(ax, 4.5, 3.5, 4.5, 3.05, "#1E8449", lw=1.8)

    # Upgrade order legend
    ax.text(4.5, 1.85, "Upgrade order:  ALLOW → LOG → TROLL → BLOCK → ALERT",
            fontsize=8, ha="center", color="#E74C3C", style="italic")

    # Input labels
    ax.text(0.45, 10.0, "Inputs:\n• KillChainStage\n• AttackType\n• AttackerIntent\n• esc_rate",
            fontsize=7.5, va="center", color=C["neutral"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB", ec=C["defender"], lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, "arch_sedm_flow.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — DQN Network Architecture
# ═════════════════════════════════════════════════════════════════════════════

def fig_dqn_architecture():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_title("DQN Network Architecture  (Baseline Policy)",
                 fontsize=13, weight="bold", pad=10, color=C["neutral"])

    layers = [
        (1.1,  "Input\nState",   "24",  "#D5F5E3", C["env"]),
        (3.0,  "Linear\n+ LayerNorm\n+ ReLU", "256", C["light_d"], C["defender"]),
        (5.0,  "Linear\n+ LayerNorm\n+ ReLU", "128", C["light_d"], C["defender"]),
        (7.0,  "Linear\n+ LayerNorm\n+ ReLU", "64",  C["light_d"], C["defender"]),
        (9.0,  "Output\nQ-values",  "5",   C["light_s"], C["sedm"]),
    ]

    xs = [l[0] for l in layers]

    for (x, name, size, fc, ec) in layers:
        h = max(0.6, int(size) / 512 * 3.5 + 0.5)
        h = min(h, 3.2)
        _box(ax, x, 2.5, 1.5, h, fc, ec, lw=2)
        ax.text(x, 2.5 + h / 2 + 0.15, name,
                fontsize=8.5, ha="center", va="bottom",
                color=ec, weight="bold")
        ax.text(x, 2.5 - h / 2 - 0.18, f"dim = {size}",
                fontsize=8, ha="center", va="top", color=C["neutral"])

    # arrows
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + 0.75, 2.5, xs[i + 1] - 0.75, 2.5,
               C["arrow"], lw=1.8)

    # Annotations
    ann = [
        (3.0, 0.45, "Kaiming init", C["defender"]),
        (5.0, 0.45, "LayerNorm → stable\nwith small batches", C["defender"]),
        (7.0, 0.45, "Huber loss\nAdam lr=1e-3", C["defender"]),
        (9.0, 0.45, "argmax → action\nε-greedy explore", C["sedm"]),
    ]
    for (x, y, txt, col) in ann:
        ax.text(x, y, txt, fontsize=7, ha="center", va="center",
                color=col, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.8, alpha=0.85))

    # replay buffer note
    ax.text(5.5, 4.65,
            "Experience Replay Buffer  (capacity 15,000)  ·  Batch 64  ·  "
            "Target Network hard-copy every 150 steps",
            fontsize=8, ha="center", color=C["neutral"],
            bbox=dict(boxstyle="round,pad=0.25", fc=C["light_d"], ec=C["defender"], lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, "arch_dqn_network.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Cyber Kill Chain progression
# ═════════════════════════════════════════════════════════════════════════════

def fig_kill_chain():
    stages = [
        "Reconnaissance", "Weaponization", "Delivery",
        "Exploitation", "Installation", "C2", "Actions on Obj",
    ]
    colors = [
        "#85C1E9", "#5DADE2", "#F0B27A", "#E59866",
        "#E74C3C", "#C0392B", "#922B21",
    ]
    weights = [0.10, 0.15, 0.20, 0.40, 0.60, 0.80, 1.00]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Cyber Kill Chain — Stage Progression & SEDM Mapping",
                 fontsize=13, weight="bold", color=C["neutral"])

    # ── Left: chain diagram ──
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_facecolor(C["bg"])

    bw = 1.18
    for i, (stage, col) in enumerate(zip(stages, colors)):
        x = i * (bw + 0.05) + 0.4
        # arrow shape
        if i < len(stages) - 1:
            ax.annotate("",
                        xy=(x + bw + 0.05, 1.5),
                        xytext=(x + bw, 1.5),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=2, mutation_scale=14))
        box = FancyBboxPatch((x, 0.9), bw, 1.2,
                             boxstyle="round,pad=0.04",
                             facecolor=col, edgecolor="white",
                             linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x + bw / 2, 1.7,
                stage.replace(" ", "\n"), fontsize=7.2,
                ha="center", va="center", color="white",
                weight="bold", zorder=4)
        ax.text(x + bw / 2, 1.05, f"w={weights[i]:.2f}",
                fontsize=6.5, ha="center", va="center",
                color="white", zorder=4)

    ax.text(5, 0.3, "← Early stage (low threat weight)              "
            "Late stage (high threat weight) →",
            fontsize=7.5, ha="center", color=C["neutral"], style="italic")

    # ── Right: SEDM band table ──
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_facecolor(C["bg"])

    data = [
        ["Stage", "Low", "Medium", "High"],
        ["Recon", "ALLOW", "LOG", "LOG"],
        ["Weapon", "LOG", "LOG", "TROLL"],
        ["Delivery", "LOG", "TROLL", "TROLL"],
        ["Exploit", "TROLL", "BLOCK", "BLOCK"],
        ["Install", "BLOCK", "BLOCK", "ALERT"],
        ["C2", "BLOCK", "ALERT", "ALERT"],
        ["Act.Obj", "ALERT", "ALERT", "ALERT"],
    ]

    action_colors = {
        "ALLOW": "#82E0AA", "LOG": "#85C1E9", "TROLL": "#F9E79F",
        "BLOCK": "#F0B27A", "ALERT": "#EC7063",
        "Low": "#D5F5E3", "Medium": "#FDFEFE", "High": "#FDEDEC",
        "Stage": "#D6EAF8",
    }

    table = ax2.table(
        cellText=[row for row in data],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.15, 1.55)

    for (r, c), cell in table.get_celld().items():
        txt = data[r][c] if r < len(data) and c < len(data[r]) else ""
        cell.set_facecolor(action_colors.get(txt, "white"))
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_text_props(weight="bold", color=C["neutral"])
        if c == 0:
            cell.set_text_props(weight="bold")

    ax2.set_title("SEDM Lookup Table", fontsize=9,
                  weight="bold", color=C["neutral"], pad=4)

    plt.tight_layout()
    path = os.path.join(OUT, "arch_kill_chain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 — Threat Level Formula breakdown
# ═════════════════════════════════════════════════════════════════════════════

def fig_threat_level():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Threat Level Computation", fontsize=13, weight="bold",
                 color=C["neutral"])

    # ── Left: pie / bar of weights ──
    ax = axes[0]
    ax.set_facecolor(C["bg"])
    labels = ["Attack\nSeverity", "Kill Chain\nStage Weight",
              "Escalation\nRate", "Cumulative\nAttack Count"]
    sizes  = [45, 35, 15, 5]
    palette = ["#E74C3C", "#E67E22", "#3498DB", "#2ECC71"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=palette,
        autopct="%10f%%", startangle=140,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_weight("bold")
        at.set_color("white")
    ax.set_title("Component Weights", fontsize=10, weight="bold",
                 color=C["neutral"], pad=6)

    # ── Right: threat band colour bar ──
    ax2 = axes[1]
    ax2.set_facecolor(C["bg"])
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.3, 5.8)
    ax2.axis("off")
    ax2.set_title("Threat Bands → Action Escalation", fontsize=10,
                  weight="bold", color=C["neutral"], pad=6)

    bands = [
        (0.00, 0.15, "#82E0AA", "benign\n< 0.15",    "ALLOW"),
        (0.15, 0.35, "#85C1E9", "low\n0.15–0.35",    "LOG"),
        (0.35, 0.55, "#F9E79F", "medium\n0.35–0.55", "TROLL"),
        (0.55, 0.75, "#F0B27A", "high\n0.55–0.75",   "BLOCK"),
        (0.75, 1.00, "#EC7063", "critical\n≥ 0.75",  "ALERT"),
    ]
    bar_h = 0.75
    for i, (lo, hi, col, label, action) in enumerate(bands):
        y = i * (bar_h + 0.3) + 0.1
        width = hi - lo
        rect = mpatches.FancyBboxPatch(
            (lo, y), width, bar_h,
            boxstyle="round,pad=0.01",
            facecolor=col, edgecolor="#888888", linewidth=1,
        )
        ax2.add_patch(rect)
        ax2.text(lo + width / 2, y + bar_h / 2, label,
                 ha="center", va="center", fontsize=8, weight="bold")
        ax2.text(lo + width / 2, y - 0.12, f"→ {action}",
                 ha="center", va="top", fontsize=7.5, color=C["neutral"],
                 style="italic")

    ax2.text(0.5, 5.5, "T  =  0.45×severity + 0.35×stage_w\n"
             "       + 0.15×esc_rate + 0.05×count_norm",
             ha="center", va="center", fontsize=9,
             color=C["neutral"],
             bbox=dict(boxstyle="round,pad=0.35", fc="#EBF5FB",
                       ec=C["defender"], lw=1.2))

    plt.tight_layout()
    path = os.path.join(OUT, "arch_threat_level.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6 — Episode Lifecycle / Game Loop
# ═════════════════════════════════════════════════════════════════════════════

def fig_game_loop():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_title("HoneyIQ — Episode Game Loop", fontsize=13,
                 weight="bold", pad=10, color=C["neutral"])

    nodes = [
        # (x, y, w, h, label, fc, ec)
        (5.0, 5.9, 2.4, 0.55, "Episode Start\nenv.reset()", "#D5F5E3", C["env"]),
        (2.0, 4.6, 2.4, 0.65, "Attacker Step\nMarkov transition\n→ attack event", C["light_a"], C["attacker"]),
        (5.0, 4.6, 2.4, 0.65, "Env computes\nstate (24-dim)\nthreat level", C["light_e"], C["env"]),
        (8.0, 4.6, 2.4, 0.65, "Defender\nobserves state\nclassifies attack", C["light_d"], C["defender"]),
        (8.0, 3.1, 2.4, 0.65, "SEDM / DQN\nselects action\n(ALLOW→ALERT)", C["light_s"], C["sedm"]),
        (5.0, 3.1, 2.4, 0.65, "Env applies action\ncomputes reward\nnext state", C["light_e"], C["env"]),
        (2.0, 3.1, 2.4, 0.65, "MetricsCollector\nrecords StepRecord\n(TP/FP/reward)", C["light_m"], C["metrics"]),
        (5.0, 1.8, 2.2, 0.55, "done?\n(max_steps or\ntermination)", "#FDFEFE", C["neutral"]),
        (5.0, 0.7, 2.4, 0.55, "Episode End\nEpisodeRecord\nexport CSV/PNG", C["light_m"], C["metrics"]),
    ]

    for (x, y, w, h, label, fc, ec) in nodes:
        _box(ax, x, y, w, h, fc, ec, lw=1.8)
        _label(ax, x, y, label, fontsize=7.5, color=C["neutral"])

    # arrows
    arrows = [
        (5.0, 5.63, 5.0, 4.93, C["env"]),
        (3.8, 5.9,  3.2, 4.93, C["attacker"]),   # start→attacker
        (3.2, 4.6,  3.8, 4.6,  C["env"]),         # att→env state
        (6.2, 4.6,  6.8, 4.6,  C["env"]),         # env→defender
        (8.0, 4.27, 8.0, 3.43, C["defender"]),    # defender→sedm
        (6.8, 3.1,  6.2, 3.1,  C["sedm"]),        # sedm→env apply
        (3.8, 3.1,  3.2, 3.1,  C["env"]),         # env→metrics
        (5.0, 2.77, 5.0, 2.07, C["env"]),         # env apply→done
    ]
    for (x0, y0, x1, y1, col) in arrows:
        _arrow(ax, x0, y0, x1, y1, col, lw=1.6)

    # done → loop back
    ax.annotate("", xy=(2.0, 4.27), xytext=(2.0, 1.8),
                arrowprops=dict(arrowstyle="-|>", color=C["env"], lw=1.6,
                                mutation_scale=12,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(0.9, 3.05, "no", fontsize=8, color=C["env"], style="italic",
            ha="center")
    ax.text(1.45, 4.1, "next step", fontsize=7, color=C["env"], style="italic")

    # done → episode end
    _arrow(ax, 5.0, 1.53, 5.0, 0.97, C["metrics"], lw=1.6)
    ax.text(5.55, 1.25, "yes", fontsize=8, color=C["metrics"], style="italic")

    # loop back from attacker after episode end — label
    ax.text(5.0, 0.12, "30 evaluation episodes × 200 steps per intent profile",
            fontsize=8, ha="center", color=C["neutral"], style="italic")

    plt.tight_layout()
    path = os.path.join(OUT, "arch_game_loop.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7 — State Vector breakdown
# ═════════════════════════════════════════════════════════════════════════════

def fig_state_vector():
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 24); ax.set_ylim(0, 2.8)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_title("State Vector  (24-dimensional observation)",
                 fontsize=13, weight="bold", pad=8, color=C["neutral"])

    segments = [
        (0,  10, C["light_a"], C["attacker"], "Attack Type\none-hot (10)",  "[0:10]"),
        (10, 17, C["light_e"], C["env"],      "Kill Chain Stage\none-hot (7)", "[10:17]"),
        (17, 18, "#FEF9E7",    C["sedm"],     "Threat\nLevel",               "[17]"),
        (18, 19, "#FDFEFE",    C["neutral"],  "Attack\nCount\nnorm.",         "[18]"),
        (19, 20, "#FDFEFE",    C["neutral"],  "Escal.\nRate",                 "[19]"),
        (20, 24, C["light_d"], C["defender"], "Attacker Intent\none-hot (4)", "[20:24]"),
    ]

    for (lo, hi, fc, ec, label, idx) in segments:
        w = hi - lo
        rect = FancyBboxPatch((lo + 0.05, 0.6), w - 0.1, 1.4,
                              boxstyle="round,pad=0.05",
                              facecolor=fc, edgecolor=ec,
                              linewidth=1.8)
        ax.add_patch(rect)
        ax.text(lo + w / 2, 1.3, label,
                ha="center", va="center", fontsize=7.8,
                color=C["neutral"], weight="bold",
                multialignment="center")
        ax.text(lo + w / 2, 0.38, idx,
                ha="center", va="center", fontsize=7,
                color=ec)

    # dimension ruler
    for i in range(0, 25, 5):
        ax.text(i, 2.3, str(i), ha="center", va="center",
                fontsize=7, color="#888888")
        ax.plot([i, i], [2.1, 2.2], color="#CCCCCC", lw=0.8)

    ax.plot([0, 24], [2.1, 2.1], color="#CCCCCC", lw=0.8)

    plt.tight_layout()
    path = os.path.join(OUT, "arch_state_vector.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating HoneyIQ architecture figures …")
    fig_system_architecture()
    fig_sedm_flow()
    fig_dqn_architecture()
    fig_kill_chain()
    fig_threat_level()
    fig_game_loop()
    fig_state_vector()
    print("Done.")
