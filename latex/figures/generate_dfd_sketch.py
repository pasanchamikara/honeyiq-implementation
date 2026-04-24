"""
Generate DFD (Level-0 + Level-1) and Sketch/Actor diagram for HoneyIQ.
Run from project root:  python latex/figures/generate_dfd_sketch.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import matplotlib.patheffects as pe
import numpy as np

OUT = os.path.join(os.path.dirname(__file__))

# ── palette ──────────────────────────────────────────────────────────────────
P = {
    "bg":       "#FAFAFA",
    "ext":      "#2C3E50",      # external entity fill
    "ext_bg":   "#D5D8DC",
    "proc":     "#1A5276",      # process fill
    "proc_bg":  "#D6EAF8",
    "store":    "#1E8449",      # data store fill
    "store_bg": "#D5F5E3",
    "arrow":    "#555555",
    "label":    "#2C3E50",
    "attacker": "#C0392B",
    "att_bg":   "#FADBD8",
    "sedm":     "#E67E22",
    "sedm_bg":  "#FDEBD0",
    "metrics":  "#7D3C98",
    "met_bg":   "#E8DAEF",
    "actor_bg": "#EBF5FB",
    "actor_bd": "#2471A3",
    "comp_bg":  "#FEF9E7",
    "comp_bd":  "#D4AC0D",
    "user_bg":  "#E9F7EF",
    "user_bd":  "#1E8449",
    "sys_bg":   "#EAF2FF",
    "sys_bd":   "#154360",
}


# ─── low-level drawing helpers ───────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, ec, lw=1.6, radius=0.04, zorder=3, ls="-"):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder, linestyle=ls,
    )
    ax.add_patch(p)


def txt(ax, x, y, s, fs=8.5, color="black", bold=False, zorder=5,
        ha="center", va="center", italic=False):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=zorder,
            weight="bold" if bold else "normal",
            style="italic" if italic else "normal",
            multialignment="center")


def arr(ax, x0, y0, x1, y1, color=P["arrow"], lw=1.5, ms=11,
        label="", label_side="top", connectionstyle="arc3,rad=0.0", zorder=4):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=ms,
                        connectionstyle=connectionstyle),
        zorder=zorder,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        off_x = 0.08 if label_side == "right" else (-0.08 if label_side == "left" else 0)
        off_y = 0.12 if label_side == "top" else (-0.14 if label_side == "bottom" else 0)
        ax.text(mx + off_x, my + off_y, label,
                fontsize=6.8, color=color, ha="center", va="center",
                style="italic", zorder=5)


def darr(ax, x0, y0, x1, y1, color=P["arrow"], lw=1.4, ms=10,
         label="", label_side="top"):
    """Double-headed arrow."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw,
                        mutation_scale=ms),
        zorder=4,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        off_y = 0.12 if label_side == "top" else -0.14
        ax.text(mx, my + off_y, label, fontsize=6.8, color=color,
                ha="center", va="center", style="italic", zorder=5)


def ext_entity(ax, x, y, w, h, label, number=None):
    """External entity: double-bordered rectangle."""
    rbox(ax, x, y, w, h, P["ext_bg"], P["ext"], lw=2.2)
    rbox(ax, x, y, w - 0.12, h - 0.12, "none", P["ext"], lw=0.8)
    if number:
        ax.text(x - w / 2 + 0.14, y + h / 2 - 0.14, number,
                fontsize=7, color=P["ext"], weight="bold", zorder=6)
    txt(ax, x, y, label, fs=8.5, color=P["ext"], bold=True)


def process_circle(ax, x, y, r, label, number, fc=P["proc_bg"], ec=P["proc"]):
    """DFD process: circle with number/name split."""
    circ = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                       linewidth=2, zorder=3)
    ax.add_patch(circ)
    # horizontal divider
    ax.plot([x - r * 0.85, x + r * 0.85], [y + r * 0.25, y + r * 0.25],
            color=ec, lw=0.8, zorder=4)
    txt(ax, x, y + r * 0.62, number, fs=8, color=ec, bold=True)
    txt(ax, x, y - r * 0.18, label, fs=7.5, color=P["label"])


def data_store(ax, x, y, w, h, label, number=None):
    """DFD data store: open-sided rectangle."""
    top_y = y + h / 2
    bot_y = y - h / 2
    left_x = x - w / 2
    right_x = x + w / 2
    ax.plot([left_x, right_x], [top_y, top_y], color=P["store"], lw=2)
    ax.plot([left_x, right_x], [bot_y, bot_y], color=P["store"], lw=2)
    # left vertical only
    ax.plot([left_x, left_x], [bot_y, top_y], color=P["store"], lw=2)
    rect = mpatches.Rectangle(
        (left_x, bot_y), w, h,
        facecolor=P["store_bg"], edgecolor="none", zorder=2,
    )
    ax.add_patch(rect)
    if number:
        txt(ax, left_x + 0.2, y, number, fs=7.5, color=P["store"], bold=True)
    txt(ax, x + 0.15, y, label, fs=8, color=P["store"])


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — DFD Level-0  (Context Diagram)
# ═════════════════════════════════════════════════════════════════════════════

def fig_dfd_level0():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(P["bg"])
    ax.set_facecolor(P["bg"])
    ax.set_title("HoneyIQ — Data Flow Diagram  Level 0  (Context Diagram)",
                 fontsize=13, weight="bold", pad=10, color=P["label"])

    # ── Central system ──
    cx, cy, cr = 5.5, 3.5, 1.55
    circ = plt.Circle((cx, cy), cr,
                       facecolor="#E8F4FD", edgecolor=P["sys_bd"],
                       linewidth=2.5, zorder=3)
    ax.add_patch(circ)
    txt(ax, cx, cy + 0.35, "HoneyIQ", fs=13, color=P["sys_bd"], bold=True)
    txt(ax, cx, cy - 0.15, "Attacker–Defender", fs=9, color=P["sys_bd"])
    txt(ax, cx, cy - 0.52, "Simulation System", fs=9, color=P["sys_bd"])

    # ── External entities ──
    entities = [
        # (x, y, w, h, label, number, arr_from, arr_to, label_out, label_in)
        (1.1, 5.8, 1.7, 0.7,  "Security\nAnalyst",  "E1",
         (1.9, 5.65), (4.1, 4.5),   "config / intents",   "reports / plots"),
        (1.1, 2.0, 1.7, 0.7,  "Network\nTraffic\n(Simulated)", "E2",
         (1.95, 2.1), (4.0, 3.0),  "raw flow features",  "blocked / allowed"),
        (9.9, 5.8, 1.7, 0.7,  "Model\nStore", "E3",
         (6.95, 4.5), (9.05, 5.55), "trained weights",   "checkpoints"),
        (9.9, 2.0, 1.7, 0.7,  "Log /\nReport Store", "E4",
         (6.95, 3.0), (9.05, 2.1),  "metrics & plots",   ""),
        (5.5, 0.5, 1.7, 0.6,  "Researcher /\nEvaluator", "E5",
         (5.5, 1.77), (5.5, 1.05),  "evaluation\nresults", "intent profiles"),
    ]

    for (x, y, w, h, label, num, p0, p1, l_out, l_in) in entities:
        ext_entity(ax, x, y, w, h, label, num)
        # outflow entity→system
        arr(ax, p0[0], p0[1], p1[0], p1[1],
            color=P["ext"], label=l_out, label_side="top")
        # inflow system→entity
        if l_in:
            arr(ax, p1[0] + 0.05, p1[1] - 0.05,
                p0[0] + 0.05, p0[1] - 0.05,
                color=P["proc"], label=l_in, label_side="bottom")

    plt.tight_layout()
    path = os.path.join(OUT, "dfd_level0_context.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — DFD Level-1  (Decomposed)
# ═════════════════════════════════════════════════════════════════════════════

def fig_dfd_level1():
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15); ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(P["bg"])
    ax.set_facecolor(P["bg"])
    ax.set_title(
        "HoneyIQ — Data Flow Diagram  Level 1  (Internal Processes)",
        fontsize=13, weight="bold", pad=10, color=P["label"],
    )

    # ── External entities (left column) ──
    ext_entity(ax, 1.1, 8.5, 1.8, 0.75, "Security\nAnalyst", "E1")
    ext_entity(ax, 1.1, 6.2, 1.8, 0.75, "Network\nTraffic", "E2")
    ext_entity(ax, 1.1, 1.5, 1.8, 0.75, "Researcher /\nEvaluator", "E5")

    # ── External entities (right column) ──
    ext_entity(ax, 13.9, 8.5, 1.8, 0.75, "Model\nStore", "E3")
    ext_entity(ax, 13.9, 1.5, 1.8, 0.75, "Log /\nReport Store", "E4")

    # ── Processes ──
    # P1 — Attack Simulation
    process_circle(ax, 3.5, 8.1, 0.85, "Attack\nSimulation", "P1",
                   fc=P["att_bg"], ec=P["attacker"])
    # P2 — Feature Extraction / Env State
    process_circle(ax, 6.3, 8.1, 0.85, "Env State\nConstruction", "P2",
                   fc=P["proc_bg"], ec=P["proc"])
    # P3 — Attack Classification
    process_circle(ax, 6.3, 5.5, 0.85, "Attack\nClassification\n(RF)", "P3",
                   fc=P["proc_bg"], ec=P["proc"])
    # P4 — Threat Assessment
    process_circle(ax, 9.1, 8.1, 0.85, "Threat\nAssessment", "P4",
                   fc=P["sedm_bg"], ec=P["sedm"])
    # P5 — Policy Decision (SEDM)
    process_circle(ax, 9.1, 5.5, 0.85, "Policy\nDecision\n(SEDM)", "P5",
                   fc=P["sedm_bg"], ec=P["sedm"])
    # P6 — DQN Training
    process_circle(ax, 11.8, 7.2, 0.85, "DQN\nTraining\n(Baseline)", "P6",
                   fc=P["proc_bg"], ec=P["proc"])
    # P7 — Reward Computation
    process_circle(ax, 9.1, 3.0, 0.85, "Reward\nComputation", "P7",
                   fc=P["sedm_bg"], ec=P["sedm"])
    # P8 — Metrics & Export
    process_circle(ax, 6.3, 1.8, 0.85, "Metrics\nCollection\n& Export", "P8",
                   fc=P["met_bg"], ec=P["metrics"])

    # ── Data Stores ──
    data_store(ax, 4.8, 6.4, 2.4, 0.55, "Transition Model\n(Markov chains)", "D1")
    data_store(ax, 7.7, 3.8, 2.4, 0.55, "Replay Buffer\n(15 k transitions)", "D2")
    data_store(ax, 4.8, 3.0, 2.4, 0.55, "Classifier Model\n(RF weights)", "D3")
    data_store(ax, 3.5, 5.1, 2.4, 0.55, "Feature Distrib.\n(UNSW-NB15)", "D4")

    # ── Arrows — external → processes ──
    arr(ax, 2.0, 8.5, 2.65, 8.2, P["ext"], label="intent config", label_side="top")
    arr(ax, 2.0, 6.2, 3.0, 7.55, P["ext"], label="flow features", label_side="top")

    # ── E1 → P1 (intent) ──
    arr(ax, 2.0, 8.55, 2.65, 8.35, P["ext"],
        label="attacker intent", label_side="top")

    # ── Processes → each other ──
    # P1 → P2 (attack event + stage)
    arr(ax, 4.35, 8.1, 5.45, 8.1, P["attacker"],
        label="attack event +\nkill chain stage", label_side="top")
    # P2 → P3 (state vector)
    arr(ax, 6.3, 7.25, 6.3, 6.35, P["proc"],
        label="state (24-dim)", label_side="right")
    # P2 → P4 (state vector)
    arr(ax, 7.15, 8.1, 8.25, 8.1, P["proc"],
        label="state (24-dim)", label_side="top")
    # P3 → P5 (predicted attack type)
    arr(ax, 6.3, 4.65, 6.3 + 0.05, 5.5 - 0.85, P["proc"],
        label="predicted\nattack type", label_side="right")
    # P3 → P5 via side
    arr(ax, 7.15, 5.5, 8.25, 5.5, P["proc"],
        label="pred. attack type", label_side="top")
    # P4 → P5 (threat level + band)
    arr(ax, 9.1, 7.25, 9.1, 6.35, P["sedm"],
        label="threat level\n+ esc. risk", label_side="right")
    # P5 → P7 (action)
    arr(ax, 9.1, 4.65, 9.1, 3.85, P["sedm"],
        label="action\n(ALLOW→ALERT)", label_side="right")
    # P7 → P8 (reward + step record)
    arr(ax, 8.35, 2.78, 7.15, 2.1, P["sedm"],
        label="reward +\nstep record", label_side="top")
    # P5 → P6 (transition for DQN)
    arr(ax, 9.96, 5.9, 10.95, 6.9, P["proc"],
        label="(s,a,r,s') tuple", label_side="top")
    # P7 → D2
    arr(ax, 8.4, 3.0, 8.9, 3.8, P["sedm"],
        label="transition", label_side="left")
    # D2 → P6
    arr(ax, 8.9, 4.1, 10.95, 7.0, P["store"],
        label="mini-batch", label_side="right")

    # ── Data store reads ──
    arr(ax, 3.5, 7.25, 3.5, 5.65, P["attacker"], label="", label_side="right")
    txt(ax, 2.9, 6.45, "stage probs", fs=6.5, color=P["attacker"], italic=True)
    arr(ax, 3.68, 5.1, 4.5, 7.7, P["attacker"],
        label="feat. distrib.", label_side="left")

    # D3 reads/writes
    arr(ax, 5.7, 5.18, 5.7, 3.3, P["store"],
        label="", label_side="right")
    txt(ax, 5.1, 4.25, "load weights", fs=6.5, color=P["store"], italic=True)
    arr(ax, 6.3, 4.65, 5.7 + 0.7, 3.3 + 0.05, P["proc"],
        label="", label_side="right")

    # ── Processes → external ──
    # P6 → E3 (model checkpoint)
    arr(ax, 12.65, 7.2, 13.0, 8.2, P["proc"],
        label="checkpoint\n(.pt file)", label_side="right")
    # P8 → E4 (CSV + PNG)
    arr(ax, 7.15, 1.8, 13.0, 1.5, P["metrics"],
        label="CSV + PNG exports", label_side="top")
    # P8 → E5
    arr(ax, 6.3, 0.95, 2.0, 1.35, P["metrics"],
        label="evaluation report", label_side="top")

    # ── Legend ──
    legend_items = [
        (mpatches.Patch(fc=P["att_bg"], ec=P["attacker"], lw=1.5), "Attacker Process"),
        (mpatches.Patch(fc=P["proc_bg"], ec=P["proc"], lw=1.5),    "Env / Defender Process"),
        (mpatches.Patch(fc=P["sedm_bg"], ec=P["sedm"], lw=1.5),    "SEDM / Reward Process"),
        (mpatches.Patch(fc=P["met_bg"], ec=P["metrics"], lw=1.5),  "Metrics Process"),
        (mpatches.Patch(fc=P["ext_bg"], ec=P["ext"], lw=1.5),      "External Entity"),
        (mpatches.Patch(fc=P["store_bg"], ec=P["store"], lw=1.5),  "Data Store"),
    ]
    ax.legend(
        handles=[h for h, _ in legend_items],
        labels=[l for _, l in legend_items],
        loc="lower right", fontsize=8, framealpha=0.92,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.99, 0.01),
    )

    plt.tight_layout()
    path = os.path.join(OUT, "dfd_level1_decomposed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Sketch Diagram: Actors, Components & End Users
# ═════════════════════════════════════════════════════════════════════════════

def _stick_figure(ax, x, y, size=0.28, color="#2C3E50", label="", label_fs=8):
    """Draw a simple stick-figure person."""
    # head
    head = plt.Circle((x, y + size * 1.45), size * 0.38,
                       facecolor="white", edgecolor=color, linewidth=1.8, zorder=5)
    ax.add_patch(head)
    # body
    ax.plot([x, x], [y + size * 1.05, y + size * 0.3], color=color, lw=2, zorder=5)
    # arms
    ax.plot([x - size * 0.55, x + size * 0.55],
            [y + size * 0.85, y + size * 0.85], color=color, lw=2, zorder=5)
    # legs
    ax.plot([x, x - size * 0.45], [y + size * 0.3, y - size * 0.3],
            color=color, lw=2, zorder=5)
    ax.plot([x, x + size * 0.45], [y + size * 0.3, y - size * 0.3],
            color=color, lw=2, zorder=5)
    if label:
        ax.text(x, y - size * 0.55, label, ha="center", va="top",
                fontsize=label_fs, color=color, weight="bold", zorder=5)


def fig_sketch():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor(P["bg"])
    ax.set_facecolor(P["bg"])
    ax.set_title(
        "HoneyIQ — System Sketch: Actors, Components & End Users",
        fontsize=14, weight="bold", pad=12, color=P["label"],
    )

    # ── boundary box: System ──
    sys_rect = FancyBboxPatch((3.4, 0.5), 9.4, 9.8,
                              boxstyle="round,pad=0.08",
                              facecolor="#EAF4FB", edgecolor=P["sys_bd"],
                              linewidth=2.2, zorder=1, linestyle="--")
    ax.add_patch(sys_rect)
    ax.text(8.1, 10.42, "◈  HoneyIQ System Boundary",
            fontsize=10, ha="center", color=P["sys_bd"],
            weight="bold", zorder=5)

    # ════════════════════════════════
    # END USERS  (left column)
    # ════════════════════════════════
    ax.text(1.35, 10.15, "END USERS", fontsize=9, ha="center",
            color=P["user_bd"], weight="bold")

    users = [
        (1.35, 8.5,  "#154360", "Security\nAnalyst"),
        (1.35, 6.0,  "#145A32", "SOC /\nDefender"),
        (1.35, 3.5,  "#6E2F8A", "Researcher"),
        (1.35, 1.2,  "#784212", "System\nAdmin"),
    ]
    for (x, y, col, label) in users:
        rbox(ax, x, y + 0.25, 2.0, 2.3, P["user_bg"], P["user_bd"], lw=1.5, radius=0.08)
        _stick_figure(ax, x, y + 0.9, size=0.3, color=col)
        ax.text(x, y - 0.22, label, ha="center", va="top",
                fontsize=8.5, color=col, weight="bold")

    # ════════════════════════════════
    # CLI / Entry Point
    # ════════════════════════════════
    rbox(ax, 5.0, 9.0, 2.4, 0.85, "#FDFEFE", P["sys_bd"], lw=2)
    txt(ax, 5.0, 9.15, "CLI  main.py", fs=9, color=P["sys_bd"], bold=True)
    txt(ax, 5.0, 8.78, "demo · compare · train · analyze",
        fs=7.5, color="#555555")

    # ════════════════════════════════
    # Core engine row
    # ════════════════════════════════

    # Attacker Simulator
    rbox(ax, 5.2, 6.8, 2.6, 1.6, P["att_bg"], P["attacker"], lw=2)
    txt(ax, 5.2, 7.45, "Attacker Simulator", fs=9, color=P["attacker"], bold=True)
    for i, line in enumerate(["AttackerAgent",
                               "TransitionModel (Markov)",
                               "Kill Chain  (7 stages)",
                               "4 Intent Profiles"]):
        txt(ax, 5.2, 7.1 - i * 0.28, "• " + line, fs=7, color="#555555")

    # Environment
    rbox(ax, 8.1, 6.8, 2.6, 1.6, "#D5F5E3", P["store"], lw=2)
    txt(ax, 8.1, 7.45, "CyberSecurityEnv", fs=9, color=P["store"], bold=True)
    txt(ax, 8.1, 7.45 - 0.01, "", fs=7)
    for i, line in enumerate(["Gymnasium Wrapper",
                               "State (24-dim)",
                               "Action Space (5)",
                               "Episode lifecycle"]):
        txt(ax, 8.1, 7.1 - i * 0.28, "• " + line, fs=7, color="#555555")

    # Defender
    rbox(ax, 11.0, 6.8, 2.6, 1.6, P["proc_bg"], P["proc"], lw=2)
    txt(ax, 11.0, 7.45, "Defender", fs=9, color=P["proc"], bold=True)
    for i, line in enumerate(["SEDM Policy  ★ primary",
                               "DQN Agent  (baseline)",
                               "RF Classifier",
                               "Reward Function"]):
        txt(ax, 11.0, 7.1 - i * 0.28, "• " + line, fs=7, color="#555555")

    # ════════════════════════════════
    # Sub-component row
    # ════════════════════════════════

    # SEDM
    rbox(ax, 5.2, 4.7, 2.6, 1.45, P["sedm_bg"], P["sedm"], lw=2)
    txt(ax, 5.2, 5.3, "SEDM Policy", fs=9, color=P["sedm"], bold=True)
    for i, line in enumerate(["Esc. Risk calc.",
                               "7×3 Matrix lookup",
                               "Override Rules R1–R3",
                               "Composite risk log"]):
        txt(ax, 5.2, 5.0 - i * 0.27, "• " + line, fs=7, color="#555555")

    # DQN Network
    rbox(ax, 8.1, 4.7, 2.6, 1.45, P["proc_bg"], P["proc"], lw=2)
    txt(ax, 8.1, 5.3, "DQN Network", fs=9, color=P["proc"], bold=True)
    for i, line in enumerate(["FC 256→128→64→5",
                               "LayerNorm + ReLU",
                               "Replay Buffer 15k",
                               "Target Net  (150 steps)"]):
        txt(ax, 8.1, 5.0 - i * 0.27, "• " + line, fs=7, color="#555555")

    # RF Classifier
    rbox(ax, 11.0, 4.7, 2.6, 1.45, P["proc_bg"], P["proc"], lw=2)
    txt(ax, 11.0, 5.3, "RF Classifier", fs=9, color=P["proc"], bold=True)
    for i, line in enumerate(["150 estimators",
                               "max_depth = 20",
                               "Balanced classes",
                               "UNSW-NB15 synthetic"]):
        txt(ax, 11.0, 5.0 - i * 0.27, "• " + line, fs=7, color="#555555")

    # ════════════════════════════════
    # Data Stores row
    # ════════════════════════════════

    stores = [
        (4.7,  3.0, "Transition\nModel (Markov)", P["attacker"]),
        (6.8,  3.0, "Replay Buffer\n(15k tuples)",  P["proc"]),
        (9.0,  3.0, "Classifier\n(.joblib)",          P["proc"]),
        (11.2, 3.0, "DQN Weights\n(.pt file)",        P["proc"]),
    ]
    for (x, y, label, col) in stores:
        # open-ended store shape
        ax.plot([x - 0.9, x + 0.9], [y + 0.38, y + 0.38], color=col, lw=2, zorder=4)
        ax.plot([x - 0.9, x + 0.9], [y - 0.38, y - 0.38], color=col, lw=2, zorder=4)
        ax.plot([x - 0.9, x - 0.9], [y - 0.38, y + 0.38], color=col, lw=2, zorder=4)
        rect2 = mpatches.Rectangle((x - 0.9, y - 0.38), 1.8, 0.76,
                                   facecolor=P["store_bg"], edgecolor="none", zorder=3)
        ax.add_patch(rect2)
        txt(ax, x, y, label, fs=7.5, color=col)

    ax.text(8.1, 2.43, "Data Stores", fontsize=8, ha="center",
            color=P["store"], style="italic")

    # ════════════════════════════════
    # Metrics / Output row
    # ════════════════════════════════
    rbox(ax, 8.1, 1.35, 5.8, 1.0, P["met_bg"], P["metrics"], lw=2)
    txt(ax, 8.1, 1.7, "MetricsCollector  &  Output", fs=9, color=P["metrics"], bold=True)
    txt(ax, 6.0, 1.3, "StepRecord · EpisodeRecord", fs=7.5, color="#555555")
    txt(ax, 10.2, 1.3, "CSV export · PNG plots", fs=7.5, color="#555555")

    # ════════════════════════════════
    # EXTERNAL ACTORS  (right column)
    # ════════════════════════════════
    ax.text(14.65, 10.15, "EXTERNAL", fontsize=9, ha="center",
            color=P["ext"], weight="bold")

    ext_actors = [
        (14.65, 8.4, "#1A5276", "Log /\nReport Store\n(CSV, PNG)"),
        (14.65, 6.0, "#145A32", "Model\nStore\n(.pt, .joblib)"),
        (14.65, 3.6, "#6E2F8A", "UNSW-NB15\nDataset\n(synthetic)"),
        (14.65, 1.4, "#784212", "OpenCanary\nIntegration\n(optional)"),
    ]
    for (x, y, col, label) in ext_actors:
        rbox(ax, x, y, 2.2, 1.3, P["ext_bg"], P["ext"], lw=1.8, radius=0.06)
        # inner border
        rbox(ax, x, y, 2.0, 1.1, "none", P["ext"], lw=0.7, radius=0.04)
        txt(ax, x, y, label, fs=7.8, color=col, bold=True)

    # ════════════════════════════════
    # Arrows — users → CLI
    # ════════════════════════════════
    for uy in [8.5, 6.0, 3.5, 1.2]:
        arr(ax, 2.35, uy + 0.25, 3.8, 9.0,
            P["user_bd"], lw=1.3, ms=9,
            connectionstyle=f"arc3,rad={0.0}")

    # ════════════════════════════════
    # Arrows — internal flows
    # ════════════════════════════════
    # CLI → Attacker
    arr(ax, 5.0, 8.58, 5.2, 7.6, P["sys_bd"], lw=1.4)
    # CLI → Env
    arr(ax, 5.6, 8.58, 8.1, 7.6, P["sys_bd"], lw=1.4)
    # CLI → Defender
    arr(ax, 6.2, 8.58, 11.0, 7.6, P["sys_bd"], lw=1.4)

    # Attacker → Env
    arr(ax, 6.5, 6.8, 6.8, 6.8, P["attacker"],
        label="attack event", label_side="top")
    # Env ↔ Defender
    arr(ax, 9.4, 7.1, 9.7, 7.1, P["store"],
        label="state (24-dim)", label_side="top")
    arr(ax, 9.7, 6.5, 9.4, 6.5, P["proc"],
        label="action", label_side="bottom")

    # Defender → SEDM / DQN / RF
    arr(ax, 10.0, 6.0, 11.0, 6.15, P["proc"], lw=1.2)
    arr(ax, 9.3, 6.0,  8.1, 6.15, P["proc"], lw=1.2)
    arr(ax, 8.7, 6.0,  5.2, 6.15, P["proc"], lw=1.2)

    # Sub-components → Env/Defender loop
    arr(ax, 5.2, 3.97, 5.2, 4.0 - 0.03, P["sedm"], lw=1.1)
    arr(ax, 8.1, 3.97, 8.1, 4.0 - 0.03, P["proc"], lw=1.1)
    arr(ax, 11.0, 3.97, 11.0, 4.0 - 0.03, P["proc"], lw=1.1)

    # Reward → Metrics
    arr(ax, 9.1, 6.0, 8.1, 1.85, P["sedm"], lw=1.2,
        label="reward + records", label_side="right",
        connectionstyle="arc3,rad=0.3")

    # Metrics → Log Store
    arr(ax, 11.0, 1.35, 13.55, 8.1, P["metrics"], lw=1.3,
        label="CSV/PNG", label_side="right",
        connectionstyle="arc3,rad=-0.2")
    # DQN → Model Store
    arr(ax, 12.3, 7.2, 13.55, 6.0, P["proc"], lw=1.3,
        label=".pt checkpoint", label_side="right")
    # Model Store → Defender (load)
    arr(ax, 13.55, 5.6, 12.3, 7.0, P["store"], lw=1.1,
        label="load weights", label_side="left",
        connectionstyle="arc3,rad=0.2")
    # UNSW-NB15 → RF
    arr(ax, 13.55, 3.6, 12.3, 4.7, P["store"],
        label="train data", label_side="top")

    # ════════════════════════════════
    # Legend
    # ════════════════════════════════
    legend_items = [
        (mpatches.Patch(fc=P["user_bg"],  ec=P["user_bd"],  lw=1.4), "End User"),
        (mpatches.Patch(fc=P["att_bg"],   ec=P["attacker"], lw=1.4), "Attacker Component"),
        (mpatches.Patch(fc="#D5F5E3",     ec=P["store"],    lw=1.4), "Environment"),
        (mpatches.Patch(fc=P["proc_bg"],  ec=P["proc"],     lw=1.4), "Defender Component"),
        (mpatches.Patch(fc=P["sedm_bg"],  ec=P["sedm"],     lw=1.4), "SEDM / Policy"),
        (mpatches.Patch(fc=P["met_bg"],   ec=P["metrics"],  lw=1.4), "Metrics / Output"),
        (mpatches.Patch(fc=P["ext_bg"],   ec=P["ext"],      lw=1.4), "External System"),
        (mpatches.Patch(fc=P["store_bg"], ec=P["store"],    lw=1.4), "Data Store"),
    ]
    ax.legend(
        handles=[h for h, _ in legend_items],
        labels=[l for _, l in legend_items],
        loc="lower left", fontsize=8, framealpha=0.94,
        edgecolor="#CCCCCC",
        ncol=2,
        bbox_to_anchor=(0.0, 0.0),
    )

    plt.tight_layout()
    path = os.path.join(OUT, "sketch_actors_components.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating DFD + Sketch diagrams …")
    fig_dfd_level0()
    fig_dfd_level1()
    fig_sketch()
    print("Done.")
