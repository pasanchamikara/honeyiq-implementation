"""
Generate architecture and mechanism diagrams for honeyiq-implementation.
Outputs 5 PNG images to the assets/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Shared style helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "blue":      "#58a6ff",
    "green":     "#3fb950",
    "orange":    "#f0883e",
    "red":       "#f85149",
    "purple":    "#bc8cff",
    "yellow":    "#e3b341",
    "teal":      "#39c5cf",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
    "attacker":  "#f85149",
    "defender":  "#3fb950",
    "env":       "#58a6ff",
    "dqn":       "#bc8cff",
}

def styled_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=PALETTE["bg"])
    return fig

def box(ax, x, y, w, h, color, alpha=0.18, radius=0.04, lw=1.5, label=None,
        ec=None):
    ec = ec or color
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=color, alpha=alpha,
                           edgecolor=ec, linewidth=lw,
                           transform=ax.transData, zorder=3)
    ax.add_patch(rect)
    return rect

def label(ax, x, y, text, color=None, size=9, bold=False, ha="center", va="center",
          zorder=5):
    color = color or PALETTE["text"]
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, zorder=zorder, wrap=False)

def arrow(ax, x1, y1, x2, y2, color=PALETTE["muted"], lw=1.4, style="->",
          rad=0.0, zorder=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=zorder)

def title_text(ax, text, y=0.97, size=14):
    ax.text(0.5, y, text, transform=ax.transAxes,
            color=PALETTE["text"], fontsize=size, fontweight="bold",
            ha="center", va="top", zorder=10)

def subtitle(ax, text, y=0.92, size=8.5):
    ax.text(0.5, y, text, transform=ax.transAxes,
            color=PALETTE["muted"], fontsize=size,
            ha="center", va="top", zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Overall Model Structure
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_structure():
    fig = styled_fig(14, 10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.set_axis_off()
    ax.set_facecolor(PALETTE["bg"])

    title_text(ax, "HoneyIQ — Overall Model Structure", y=0.985, size=15)
    subtitle(ax, "cyber_env.py  ·  attacker/  ·  defender/  ·  evaluation/", y=0.96, size=9)

    # ── Environment (outer shell) ──────────────────────────────────────────
    box(ax, 0.4, 0.55, 13.2, 8.7, PALETTE["env"], alpha=0.06, radius=0.12, lw=2)
    label(ax, 0.95, 9.05, "CyberSecurityEnv  (gymnasium.Env)", color=PALETTE["env"],
          size=9.5, bold=True, ha="left")

    # ── Attacker block ────────────────────────────────────────────────────
    AX, AY, AW, AH = 0.7, 5.3, 5.2, 3.6
    box(ax, AX, AY, AW, AH, PALETTE["attacker"], alpha=0.12, radius=0.07, lw=1.8)
    label(ax, AX+AW/2, AY+AH-0.28, "ATTACKER", color=PALETTE["attacker"],
          size=11, bold=True)

    # sub-boxes inside attacker
    sub_a = [
        ("TransitionModel\n(Markov chain)", 0.9, 7.3, 2.1, 1.1),
        ("Attacker.step()\nfeature sim", 0.9, 5.8, 2.1, 1.1),
        ("AttackerIntent\n(4 profiles)", 3.3, 7.3, 2.2, 1.1),
        ("KillChainStage\n(7 stages)", 3.3, 5.8, 2.2, 1.1),
    ]
    for txt, x, y, w, h in sub_a:
        box(ax, x, y, w, h, PALETTE["orange"], alpha=0.22, radius=0.05)
        label(ax, x+w/2, y+h/2, txt, size=8.2, color=PALETTE["text"])

    # ── Defender block ────────────────────────────────────────────────────
    DX, DY, DW, DH = 7.1, 5.3, 5.9, 3.6
    box(ax, DX, DY, DW, DH, PALETTE["defender"], alpha=0.12, radius=0.07, lw=1.8)
    label(ax, DX+DW/2, DY+DH-0.28, "DEFENDER", color=PALETTE["defender"],
          size=11, bold=True)

    sub_d = [
        ("AttackClassifier\n(RandomForest)", 7.3, 7.3, 2.3, 1.1),
        ("DQNAgent\n(policy_net + target_net)", 7.3, 5.8, 2.3, 1.1),
        ("ReplayBuffer\n(capacity 10k)", 9.9, 7.3, 2.7, 1.1),
        ("HoneypotAction\n(5 discrete actions)", 9.9, 5.8, 2.7, 1.1),
    ]
    for txt, x, y, w, h in sub_d:
        box(ax, x, y, w, h, PALETTE["purple"], alpha=0.22, radius=0.05)
        label(ax, x+w/2, y+h/2, txt, size=8.2, color=PALETTE["text"])

    # ── State vector ──────────────────────────────────────────────────────
    SX, SY, SW, SH = 0.7, 3.0, 5.4, 1.9
    box(ax, SX, SY, SW, SH, PALETTE["env"], alpha=0.18, radius=0.07, lw=1.5)
    label(ax, SX+SW/2, SY+SH-0.28, "State Vector  (dim=24)", color=PALETTE["env"],
          size=9.5, bold=True)

    segs = [
        ("[0:10]\nAttackType\none-hot", PALETTE["red"]),
        ("[10:17]\nKillChain\none-hot", PALETTE["orange"]),
        ("[17-19]\nthreat /\ncount / esc", PALETTE["yellow"]),
        ("[20:24]\nIntent\none-hot", PALETTE["purple"]),
    ]
    sw = SW / len(segs) - 0.1
    for i, (txt, c) in enumerate(segs):
        sx = SX + 0.05 + i * (sw + 0.1)
        box(ax, sx, SY+0.25, sw, SH-0.55, c, alpha=0.28, radius=0.04)
        label(ax, sx+sw/2, SY+0.25+(SH-0.55)/2, txt, size=7.5)

    # ── Reward / Metrics ──────────────────────────────────────────────────
    RX, RY, RW, RH = 6.4, 3.0, 6.6, 1.9
    box(ax, RX, RY, RW, RH, PALETTE["green"], alpha=0.13, radius=0.07, lw=1.5)
    label(ax, RX+RW/2, RY+RH-0.28, "Reward & Evaluation", color=PALETTE["green"],
          size=9.5, bold=True)

    rsubs = [
        ("compute_threat_level()\n45% severity · 35% stage\n15% esc · 5% count", PALETTE["yellow"]),
        ("compute_reward()\nReward matrix\n5 actions × 5 bands", PALETTE["green"]),
        ("EpisodeMetrics\ndetection rate\nfalse-positive", PALETTE["teal"]),
    ]
    rw = RW / len(rsubs) - 0.12
    for i, (txt, c) in enumerate(rsubs):
        rx = RX + 0.06 + i * (rw + 0.12)
        box(ax, rx, RY+0.25, rw, RH-0.55, c, alpha=0.25, radius=0.04)
        label(ax, rx+rw/2, RY+0.25+(RH-0.55)/2, txt, size=7.5)

    # ── Training loop (bottom) ────────────────────────────────────────────
    TX, TY, TW, TH = 0.7, 0.65, 12.6, 2.0
    box(ax, TX, TY, TW, TH, PALETTE["blue"], alpha=0.10, radius=0.07, lw=1.5)
    label(ax, TX+TW/2, TY+TH-0.28, "Training Loop  (train.py)", color=PALETTE["blue"],
          size=9.5, bold=True)

    train_steps = [
        "env.reset()",
        "attacker.step()\nfeatures, attack_type,\nkill_chain_stage",
        "defender.observe()\nclassify attack\nselect action (ε-greedy)",
        "env.step(action)\ncompute reward\nbuild next_state",
        "defender.learn()\nstore transition\nDQN backprop",
        "target_net sync\nevery 100 steps",
    ]
    tw = TW / len(train_steps) - 0.12
    for i, txt in enumerate(train_steps):
        tx = TX + 0.06 + i * (tw + 0.12)
        c = [PALETTE["env"], PALETTE["attacker"], PALETTE["defender"],
             PALETTE["env"], PALETTE["purple"], PALETTE["dqn"]][i]
        box(ax, tx, TY+0.25, tw, TH-0.55, c, alpha=0.28, radius=0.04)
        label(ax, tx+tw/2, TY+0.25+(TH-0.55)/2, txt, size=7.5)
        if i < len(train_steps) - 1:
            arrow(ax, tx+tw, TY+0.25+(TH-0.55)/2,
                       tx+tw+0.12, TY+0.25+(TH-0.55)/2, color=PALETTE["muted"])

    # Arrows connecting major blocks
    arrow(ax, AX+AW, 7.1, DX, 7.1, color=PALETTE["attacker"], lw=2)
    label(ax, (AX+AW+DX)/2, 7.25, "network features", size=7.5,
          color=PALETTE["muted"])

    arrow(ax, 3.3, AY, 3.3, SY+SH, color=PALETTE["muted"], lw=1.5)
    arrow(ax, 10.0, DY, 10.0, RY+RH, color=PALETTE["muted"], lw=1.5)

    fig.savefig(os.path.join(OUT, "01_model_structure.png"), dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  saved 01_model_structure.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Defender Mechanism
# ─────────────────────────────────────────────────────────────────────────────

def fig_defender():
    fig = styled_fig(13, 9)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13); ax.set_ylim(0, 9)
    ax.set_axis_off()
    ax.set_facecolor(PALETTE["bg"])

    title_text(ax, "Defender Mechanism", size=15)
    subtitle(ax, "defender/defender.py  ·  classifier.py  ·  dqn.py  ·  honeypot.py", size=9)

    # ── Input ─────────────────────────────────────────────────────────────
    box(ax, 0.3, 3.5, 2.5, 2.0, PALETTE["env"], alpha=0.20, radius=0.06)
    label(ax, 1.55, 4.8, "Input", color=PALETTE["env"], size=9, bold=True)
    for i, t in enumerate(["state (dim=24)", "features dict (15)"]):
        label(ax, 1.55, 4.4 - i*0.35, t, size=8.2)

    arrow(ax, 2.8, 4.5, 3.5, 6.4, color=PALETTE["env"], lw=1.8)
    arrow(ax, 2.8, 4.5, 3.5, 3.8, color=PALETTE["env"], lw=1.8)

    # ── Classifier path ───────────────────────────────────────────────────
    box(ax, 3.5, 5.8, 3.2, 1.5, PALETTE["yellow"], alpha=0.20, radius=0.06, lw=1.6)
    label(ax, 5.1, 7.1, "AttackClassifier", color=PALETTE["yellow"], size=9.5, bold=True)
    label(ax, 5.1, 6.72, "RandomForest (150 trees)", size=8.2)
    label(ax, 5.1, 6.42, "max_depth=20, balanced", size=8.2)
    label(ax, 5.1, 6.12, "StandardScaler → predict_proba()", size=8.2)

    box(ax, 3.5, 7.6, 3.2, 0.9, PALETTE["yellow"], alpha=0.10, radius=0.04)
    label(ax, 5.1, 8.05, "Training", color=PALETTE["yellow"], size=8.5, bold=True)
    label(ax, 5.1, 7.82, "600 samples/class × 10 classes", size=7.8)

    arrow(ax, 5.1, 7.6, 5.1, 7.3, color=PALETTE["yellow"])

    arrow(ax, 6.7, 6.55, 7.5, 6.55, color=PALETTE["yellow"], lw=1.8)
    label(ax, 7.1, 6.7, "AttackType", size=7.5, color=PALETTE["muted"])

    # ── DQN path ─────────────────────────────────────────────────────────
    box(ax, 3.5, 2.9, 3.2, 2.5, PALETTE["purple"], alpha=0.20, radius=0.06, lw=1.6)
    label(ax, 5.1, 5.1, "DQNAgent", color=PALETTE["purple"], size=9.5, bold=True)
    label(ax, 5.1, 4.75, "policy_net  (24→256→128→64→5)", size=8.2)
    label(ax, 5.1, 4.45, "target_net  (hard-copy every 100 steps)", size=8.2)
    label(ax, 5.1, 4.15, "ε-greedy   ε: 1.0→0.05 (×0.995)", size=8.2)
    label(ax, 5.1, 3.85, "Huber loss + Adam(lr=1e-3)", size=8.2)
    label(ax, 5.1, 3.55, "grad clip max_norm=10.0", size=8.2)
    label(ax, 5.1, 3.2, "γ=0.99", size=8.2)

    arrow(ax, 6.7, 4.15, 7.5, 4.15, color=PALETTE["purple"], lw=1.8)
    label(ax, 7.1, 4.35, "action (int)", size=7.5, color=PALETTE["muted"])

    # ── Replay Buffer ─────────────────────────────────────────────────────
    box(ax, 3.5, 0.5, 3.2, 2.0, PALETTE["blue"], alpha=0.18, radius=0.06, lw=1.6)
    label(ax, 5.1, 2.2, "ReplayBuffer", color=PALETTE["blue"], size=9.5, bold=True)
    label(ax, 5.1, 1.9, "deque(maxlen=10,000)", size=8.2)
    label(ax, 5.1, 1.6, "stores Transition namedtuple", size=8.2)
    label(ax, 5.1, 1.3, "(s, a, r, s', done)", size=8.2)
    label(ax, 5.1, 1.0, "sample batch_size=64", size=8.2)
    label(ax, 5.1, 0.75, "random uniform sampling", size=8.2)

    # ── Actions ──────────────────────────────────────────────────────────
    box(ax, 7.5, 5.2, 2.8, 2.1, PALETTE["green"], alpha=0.20, radius=0.06, lw=1.6)
    label(ax, 8.9, 7.0, "HoneypotAction", color=PALETTE["green"], size=9.5, bold=True)
    actions = ["0: ALLOW", "1: LOG", "2: TROLL", "3: BLOCK", "4: ALERT"]
    ac = [PALETTE["muted"], PALETTE["blue"], PALETTE["yellow"],
          PALETTE["orange"], PALETTE["red"]]
    for i, (t, c) in enumerate(zip(actions, ac)):
        label(ax, 8.9, 6.65 - i*0.30, t, size=8.2, color=c)

    # ── Reward matrix ─────────────────────────────────────────────────────
    box(ax, 7.5, 2.3, 2.8, 2.5, PALETTE["green"], alpha=0.14, radius=0.06, lw=1.6)
    label(ax, 8.9, 4.55, "Reward Matrix", color=PALETTE["green"], size=9.5, bold=True)
    bands = ["benign", "low", "medium", "high", "critical"]
    bc = [PALETTE["muted"], PALETTE["blue"], PALETTE["yellow"],
          PALETTE["orange"], PALETTE["red"]]
    for i, (b, c) in enumerate(zip(bands, bc)):
        label(ax, 8.9, 4.25 - i*0.35, b, size=8, color=c)

    label(ax, 8.9, 2.55, "BLOCK/ALERT → high/critical", size=7.5, color=PALETTE["muted"])
    label(ax, 8.9, 2.35, "TROLL → medium", size=7.5, color=PALETTE["muted"])
    label(ax, 8.9, 2.15, "LOG → low, ALLOW → benign", size=7.5, color=PALETTE["muted"])

    arrow(ax, 10.3, 6.25, 10.9, 6.25, color=PALETTE["green"], lw=1.6)
    arrow(ax, 10.3, 3.55, 10.9, 3.55, color=PALETTE["green"], lw=1.6)

    # ── Output ───────────────────────────────────────────────────────────
    box(ax, 10.9, 4.9, 1.8, 1.7, PALETTE["defender"], alpha=0.22, radius=0.06, lw=1.8)
    label(ax, 11.8, 5.5, "Output", color=PALETTE["defender"], size=9.5, bold=True)
    label(ax, 11.8, 5.15, "action (int)", size=8.2)
    label(ax, 11.8, 4.88, "predicted_attack", size=8.2)

    fig.savefig(os.path.join(OUT, "02_defender_mechanism.png"), dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  saved 02_defender_mechanism.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attacker Mechanism
# ─────────────────────────────────────────────────────────────────────────────

def fig_attacker():
    fig = styled_fig(13, 9)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13); ax.set_ylim(0, 9)
    ax.set_axis_off()
    ax.set_facecolor(PALETTE["bg"])

    title_text(ax, "Attacker Mechanism", size=15)
    subtitle(ax, "attacker/attacker.py  ·  attack_types.py  ·  transition_model.py", size=9)

    # ── Intent ───────────────────────────────────────────────────────────
    box(ax, 0.3, 6.5, 2.8, 2.1, PALETTE["orange"], alpha=0.20, radius=0.06, lw=1.6)
    label(ax, 1.7, 8.3, "AttackerIntent", color=PALETTE["orange"], size=9.5, bold=True)
    intents = [("STEALTHY", PALETTE["muted"]), ("AGGRESSIVE", PALETTE["red"]),
               ("TARGETED", PALETTE["yellow"]), ("OPPORTUNISTIC", PALETTE["orange"])]
    for i, (t, c) in enumerate(intents):
        label(ax, 1.7, 8.0 - i*0.35, t, size=8.2, color=c)

    # ── Kill chain ────────────────────────────────────────────────────────
    stages = ["RECON", "WEAPON", "DELIVERY", "EXPLOIT", "INSTALL", "C2", "ACTIONS"]
    sc = [PALETTE["muted"], PALETTE["blue"], PALETTE["teal"],
          PALETTE["yellow"], PALETTE["orange"], PALETTE["red"], "#ff0000"]

    box(ax, 3.4, 7.3, 9.0, 1.1, PALETTE["bg"], alpha=0.0, radius=0.04, lw=0)
    label(ax, 7.9, 8.55, "Kill Chain Stages", color=PALETTE["muted"], size=8.5, bold=True)
    sw = 9.0 / len(stages) - 0.05
    for i, (s, c) in enumerate(zip(stages, sc)):
        sx = 3.4 + i * (sw + 0.05)
        box(ax, sx, 7.3, sw, 1.0, c, alpha=0.25, radius=0.04, ec=c)
        label(ax, sx + sw/2, 7.80, s, size=7.5, color=c)
        label(ax, sx + sw/2, 7.45, str(i), size=8, color=PALETTE["muted"])
        if i < len(stages) - 1:
            arrow(ax, sx+sw, 7.8, sx+sw+0.05, 7.8, color=c, lw=1.2)

    # ── TransitionModel ──────────────────────────────────────────────────
    box(ax, 0.3, 3.8, 5.5, 3.2, PALETTE["orange"], alpha=0.12, radius=0.07, lw=1.6)
    label(ax, 3.05, 6.75, "TransitionModel  (Markov chain)", color=PALETTE["orange"],
          size=9.5, bold=True)

    # mini matrix hint
    box(ax, 0.5, 4.0, 2.4, 2.5, PALETTE["bg"], alpha=0.4, radius=0.03)
    label(ax, 1.7, 6.3, "Attack Transition", color=PALETTE["yellow"], size=8, bold=True)
    rows = ["NORMAL", "RECON", "ANALYSIS", "FUZZERS", "EXPLOITS"]
    for i, r in enumerate(rows):
        label(ax, 0.75, 6.05 - i*0.37, r, size=7, ha="left", color=PALETTE["muted"])
        vals = [0.60, 0.28, 0.05, 0.03, 0.02][::-1]
        for j, v in enumerate(vals):
            c = plt.cm.YlOrRd(v)
            rect = mpatches.Rectangle((1.55 + j*0.27, 5.9 - i*0.37), 0.25, 0.28,
                                        color=c, alpha=0.8, zorder=3)
            ax.add_patch(rect)

    box(ax, 3.1, 4.0, 2.4, 2.5, PALETTE["bg"], alpha=0.4, radius=0.03)
    label(ax, 4.3, 6.3, "Stage Transition", color=PALETTE["teal"], size=8, bold=True)
    stage_rows = ["RECON", "WEAPON", "DELIVERY", "EXPLOIT", "INSTALL"]
    for i, r in enumerate(stage_rows):
        label(ax, 3.25, 6.05 - i*0.37, r, size=7, ha="left", color=PALETTE["muted"])
        vals2 = [0.30, 0.60, 0.10, 0.00, 0.00]
        for j, v in enumerate(vals2):
            c = plt.cm.Blues(v + 0.1)
            rect = mpatches.Rectangle((4.15 + j*0.27, 5.9 - i*0.37), 0.25, 0.28,
                                        color=c, alpha=0.8, zorder=3)
            ax.add_patch(rect)

    label(ax, 3.05, 4.15, "Intent modifier applied → row-normalize", size=7.8,
          color=PALETTE["muted"])

    # ── AttackType enum ───────────────────────────────────────────────────
    box(ax, 6.1, 3.8, 6.6, 3.2, PALETTE["red"], alpha=0.12, radius=0.07, lw=1.6)
    label(ax, 9.4, 6.75, "AttackType  (10 classes)", color=PALETTE["red"],
          size=9.5, bold=True)

    attacks = [
        ("NORMAL",         "0", PALETTE["muted"],   "0.00"),
        ("RECONNAISSANCE", "1", PALETTE["blue"],     "0.20"),
        ("ANALYSIS",       "2", PALETTE["blue"],     "0.25"),
        ("FUZZERS",        "3", PALETTE["teal"],     "0.35"),
        ("EXPLOITS",       "4", PALETTE["yellow"],   "0.70"),
        ("BACKDOORS",      "5", PALETTE["yellow"],   "0.80"),
        ("SHELLCODE",      "6", PALETTE["orange"],   "0.75"),
        ("GENERIC",        "7", PALETTE["orange"],   "0.40"),
        ("DOS",            "8", PALETTE["red"],      "0.85"),
        ("WORMS",          "9", "#ff2222",           "0.90"),
    ]
    col_w = 3.1
    for i, (name, idx, c, sev) in enumerate(attacks):
        row = i % 5
        col = i // 5
        x = 6.25 + col * col_w
        y = 6.35 - row * 0.5
        box(ax, x, y-0.18, col_w - 0.15, 0.38, c, alpha=0.22, radius=0.03)
        label(ax, x + 0.2, y + 0.01, f"{idx}: {name}", size=7.8, ha="left", color=c)
        label(ax, x + col_w - 0.3, y + 0.01, f"sev={sev}", size=7.2, ha="right",
              color=PALETTE["muted"])

    # ── Feature simulation ────────────────────────────────────────────────
    box(ax, 0.3, 0.4, 5.5, 3.0, PALETTE["teal"], alpha=0.12, radius=0.07, lw=1.6)
    label(ax, 3.05, 3.15, "Feature Simulation  (15 UNSW-NB15 features)",
          color=PALETTE["teal"], size=9.5, bold=True)

    feats = [
        ("dur", "uniform"), ("sbytes", "lognormal"), ("dbytes", "lognormal"),
        ("sttl", "choice"), ("dttl", "choice"), ("sloss", "poisson"),
        ("dloss", "poisson"), ("sload", "uniform"), ("dload", "uniform"),
        ("spkts", "poisson"), ("dpkts", "poisson"), ("swin", "choice"),
        ("dwin", "choice"), ("ct_srv_src", "poisson"), ("ct_dst_ltm", "poisson"),
    ]
    dist_colors = {"uniform": PALETTE["blue"], "lognormal": PALETTE["green"],
                   "poisson": PALETTE["purple"], "choice": PALETTE["yellow"]}
    fw = 5.3 / 5 - 0.08
    fh = 2.3 / 3 - 0.08
    for i, (name, dist) in enumerate(feats):
        col = i % 5
        row = i // 5
        fx = 0.38 + col * (fw + 0.08)
        fy = 2.6 - row * (fh + 0.08)
        c = dist_colors[dist]
        box(ax, fx, fy, fw, fh, c, alpha=0.28, radius=0.03)
        label(ax, fx+fw/2, fy+fh/2+0.05, name, size=7.5, color=PALETTE["text"])
        label(ax, fx+fw/2, fy+fh/2-0.12, dist, size=6.5, color=c)

    legend_elems = [mpatches.Patch(color=dist_colors[d], alpha=0.7, label=d)
                    for d in dist_colors]
    ax.legend(handles=legend_elems, loc="lower right",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=7.5, framealpha=0.9)

    # ── step() output ─────────────────────────────────────────────────────
    box(ax, 6.1, 0.4, 6.6, 3.0, PALETTE["attacker"], alpha=0.12, radius=0.07, lw=1.6)
    label(ax, 9.4, 3.15, "Attacker.step()  output dict",
          color=PALETTE["attacker"], size=9.5, bold=True)
    keys = [
        "attack_type        : AttackType",
        "kill_chain_stage   : KillChainStage",
        "intent             : AttackerIntent",
        "attack_count       : int",
        "step_count         : int",
        "features           : dict[str, float]  (15 feats)",
        "is_attack          : bool",
        "next_probabilities : np.ndarray (10,)",
        "stage_probabilities: np.ndarray (7,)",
    ]
    for i, k in enumerate(keys):
        label(ax, 6.25, 2.75 - i*0.28, k, size=7.5, ha="left",
              color=PALETTE["text"] if i < 7 else PALETTE["muted"])

    # Arrows
    arrow(ax, 0.3+2.8/2, 6.5, 0.3+2.8/2, 5.75, color=PALETTE["orange"], lw=1.5)
    arrow(ax, 5.8, 5.4, 6.1, 5.4, color=PALETTE["orange"], lw=1.5)
    arrow(ax, 3.05, 3.8, 3.05, 3.4, color=PALETTE["teal"], lw=1.5)
    arrow(ax, 9.4, 3.8, 9.4, 3.4, color=PALETTE["red"], lw=1.5)

    fig.savefig(os.path.join(OUT, "03_attacker_mechanism.png"), dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  saved 03_attacker_mechanism.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attacker–Defender Interaction
# ─────────────────────────────────────────────────────────────────────────────

def fig_interaction():
    fig = styled_fig(14, 9)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.set_axis_off()
    ax.set_facecolor(PALETTE["bg"])

    title_text(ax, "Attacker ↔ Defender Interaction  (CyberSecurityEnv)", size=15)
    subtitle(ax, "One episode step — gymnasium compatible", size=9)

    # ── Column headers ────────────────────────────────────────────────────
    cols = {"Attacker": 1.6, "Environment": 6.0, "Defender": 11.5}
    col_c = {"Attacker": PALETTE["attacker"], "Environment": PALETTE["env"],
             "Defender": PALETTE["defender"]}
    for name, cx in cols.items():
        box(ax, cx - 1.5, 8.0, 3.0, 0.65, col_c[name], alpha=0.18, radius=0.05, lw=1.5)
        label(ax, cx, 8.32, name, color=col_c[name], size=11, bold=True)

    # ── Vertical swimlane lines ───────────────────────────────────────────
    for cx in [3.2, 9.0]:
        ax.plot([cx, cx], [0.3, 7.9], color=PALETTE["border"], lw=1, ls="--", zorder=1)

    # ── Steps sequence ────────────────────────────────────────────────────
    steps = [
        # (y, left_cx, right_cx, left_label, msg, right_label, color, direction)
        (7.2, 1.6, 6.0, "attacker.step()",
         "attack_type, kill_chain_stage\nfeatures, is_attack",
         "receives atk_info", PALETTE["attacker"], "→"),

        (6.1, 6.0, 6.0, "",
         "compute_threat_level()\n45%·sev + 35%·stage + 15%·esc + 5%·count",
         "", PALETTE["yellow"], "self"),

        (5.2, 6.0, 11.5, "send state (dim=24)",
         "state_vector: [attack_oh | stage_oh\n| threat | count | esc | intent_oh]",
         "receive state", PALETTE["env"], "→"),

        (4.2, 11.5, 6.0, "defender.observe()",
         "classify features → AttackType\nDQN ε-greedy → action (0-4)",
         "receive action", PALETTE["defender"], "←"),

        (3.2, 6.0, 6.0, "",
         "compute_reward(action, threat_level, …)\nReward matrix + late-stage penalty\n+ attack-type bonuses",
         "", PALETTE["green"], "self"),

        (2.1, 6.0, 11.5, "send (next_state, reward)",
         "next_state, reward, terminated\ntruncated, info dict",
         "receive reward", PALETTE["env"], "→"),

        (1.1, 11.5, 11.5, "",
         "defender.learn()\nstore_transition → replay_buffer\nDQN.update() → backprop",
         "", PALETTE["purple"], "self"),
    ]

    for (y, lcx, rcx, llabel, msg, rlabel, c, direction) in steps:
        box(ax, 3.3, y-0.28, 5.6, 0.68, c, alpha=0.14, radius=0.04, lw=1.2)
        label(ax, 6.1, y, msg, size=7.8, color=c)

        if direction == "→":
            arrow(ax, lcx, y, rcx - 0.2, y, color=c, lw=1.8)
            label(ax, lcx - 0.1, y + 0.22, llabel, size=7.2, color=PALETTE["muted"],
                  ha="center")
            label(ax, rcx + 0.1, y + 0.22, rlabel, size=7.2, color=PALETTE["muted"],
                  ha="center")
        elif direction == "←":
            arrow(ax, lcx, y, rcx + 0.2, y, color=c, lw=1.8)
            label(ax, lcx + 0.1, y + 0.22, llabel, size=7.2, color=PALETTE["muted"],
                  ha="center")
            label(ax, rcx - 0.1, y + 0.22, rlabel, size=7.2, color=PALETTE["muted"],
                  ha="center")
        elif direction == "self":
            ax.annotate("", xy=(6.1, y-0.22), xytext=(6.1, y+0.22),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.4,
                                        connectionstyle="arc3,rad=0.5"), zorder=4)

    # ── Step labels on left margin ────────────────────────────────────────
    step_labels = ["① Attacker advances", "② Threat computed",
                   "③ State built & sent", "④ Defender acts",
                   "⑤ Reward computed", "⑥ Transition returned",
                   "⑦ DQN learns"]
    ys = [7.2, 6.1, 5.2, 4.2, 3.2, 2.1, 1.1]
    for sl, sy in zip(step_labels, ys):
        label(ax, 13.7, sy, sl, size=7.5, color=PALETTE["muted"], ha="right")

    # ── Episode boundary note ─────────────────────────────────────────────
    box(ax, 0.2, 0.15, 13.6, 0.5, PALETTE["blue"], alpha=0.08, radius=0.04, lw=1)
    label(ax, 7.0, 0.4,
          "Episode ends when: truncated (step ≥ max_steps=500)  —  "
          "terminated=False (no absorbing state)",
          size=7.8, color=PALETTE["muted"])

    fig.savefig(os.path.join(OUT, "04_interaction.png"), dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  saved 04_interaction.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Deep Q-Learning Architecture
# ─────────────────────────────────────────────────────────────────────────────

def fig_dqn():
    fig = styled_fig(14, 9.5)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.5)
    ax.set_axis_off()
    ax.set_facecolor(PALETTE["bg"])

    title_text(ax, "Deep Q-Network (DQN) Architecture", size=15)
    subtitle(ax, "defender/dqn.py  —  DQNNetwork · DQNAgent · ReplayBuffer", size=9)

    # ── Neural network layers ─────────────────────────────────────────────
    layers = [
        ("Input\nState\ndim=24",      24, PALETTE["env"],     0.5),
        ("Linear\n→256\nLayerNorm\nReLU", 256, PALETTE["purple"], 2.3),
        ("Linear\n→128\nLayerNorm\nReLU", 128, PALETTE["purple"], 4.2),
        ("Linear\n→64\nLayerNorm\nReLU",  64,  PALETTE["purple"], 6.1),
        ("Linear\n→5\nQ-values",      5,   PALETTE["green"],  8.0),
    ]

    prev_x = None
    for name, neurons, c, lx in layers:
        n_dots = min(neurons, 8)
        dot_gap = 3.5 / (n_dots + 1)
        lw_box = 2.0 if name.startswith("Input") or "Q-values" in name else 1.5

        box(ax, lx, 2.3, 1.5, 4.9, c, alpha=0.15, radius=0.06, lw=lw_box)
        label(ax, lx + 0.75, 5.7, name, size=8, color=c, bold=True)

        for i in range(n_dots):
            cy = 2.5 + (i + 1) * dot_gap + 0.2
            circle = plt.Circle((lx + 0.75, cy), 0.13, color=c, alpha=0.7, zorder=4)
            ax.add_patch(circle)
        if neurons > n_dots:
            label(ax, lx + 0.75, 2.5, f"({neurons} units)", size=7,
                  color=PALETTE["muted"])

        if prev_x is not None:
            ax.plot([prev_x + 1.5, lx], [4.75, 4.75],
                    color=PALETTE["muted"], lw=1, alpha=0.4, zorder=2)
            arrow(ax, prev_x + 1.5, 4.75, lx, 4.75, color=PALETTE["muted"], lw=1.2)

        prev_x = lx

    # ── Weight init note ──────────────────────────────────────────────────
    label(ax, 7.0, 2.1, "Kaiming uniform init for all Linear layers  ·  bias=0",
          size=8, color=PALETTE["muted"])

    # ── Policy vs Target ──────────────────────────────────────────────────
    box(ax, 0.3, 7.5, 6.0, 1.7, PALETTE["purple"], alpha=0.12, radius=0.06, lw=1.6)
    label(ax, 3.3, 9.0, "Policy Network  (policy_net)", color=PALETTE["purple"],
          size=9.5, bold=True)
    label(ax, 3.3, 8.65, "selects actions via ε-greedy during training", size=8.2)
    label(ax, 3.3, 8.35, "updated every step via backprop", size=8.2)
    label(ax, 3.3, 8.05, "grad clip: max_norm=10.0", size=8.2)
    label(ax, 3.3, 7.75, "optimizer: Adam  lr=1e-3", size=8.2)

    box(ax, 6.6, 7.5, 7.1, 1.7, PALETTE["blue"], alpha=0.12, radius=0.06, lw=1.6)
    label(ax, 10.15, 9.0, "Target Network  (target_net)", color=PALETTE["blue"],
          size=9.5, bold=True)
    label(ax, 10.15, 8.65, "provides stable TD bootstrap targets", size=8.2)
    label(ax, 10.15, 8.35, "hard copy from policy_net every 100 steps", size=8.2)
    label(ax, 10.15, 8.05, "kept in eval mode (no grad)", size=8.2)
    label(ax, 10.15, 7.75, "target = r + γ · max_a Q_target(s', a)   γ=0.99", size=8.2)

    arrow(ax, 6.3, 8.35, 6.6, 8.35, color=PALETTE["teal"], lw=2)
    label(ax, 6.45, 8.6, "sync\n100 steps", size=7.5, color=PALETTE["teal"], ha="center")

    # ── ε-greedy ──────────────────────────────────────────────────────────
    box(ax, 0.3, 5.7, 4.2, 1.5, PALETTE["orange"], alpha=0.14, radius=0.06, lw=1.5)
    label(ax, 2.4, 7.0, "ε-greedy Exploration", color=PALETTE["orange"],
          size=9.5, bold=True)
    label(ax, 2.4, 6.7, "start=1.0  end=0.05  decay=0.995", size=8.2)
    label(ax, 2.4, 6.4, "ε × 0.995 each update step", size=8.2)
    label(ax, 2.4, 6.1, "random action with prob ε, greedy otherwise", size=8.2)

    # ── Replay buffer ─────────────────────────────────────────────────────
    box(ax, 4.7, 5.7, 4.5, 1.5, PALETTE["blue"], alpha=0.14, radius=0.06, lw=1.5)
    label(ax, 6.95, 7.0, "Experience Replay", color=PALETTE["blue"],
          size=9.5, bold=True)
    label(ax, 6.95, 6.7, "deque capacity = 10,000 transitions", size=8.2)
    label(ax, 6.95, 6.4, "uniform random sample — batch size 64", size=8.2)
    label(ax, 6.95, 6.1, "breaks temporal correlation", size=8.2)

    # ── Loss ─────────────────────────────────────────────────────────────
    box(ax, 9.4, 5.7, 4.3, 1.5, PALETTE["green"], alpha=0.14, radius=0.06, lw=1.5)
    label(ax, 11.55, 7.0, "Loss Function", color=PALETTE["green"],
          size=9.5, bold=True)
    label(ax, 11.55, 6.7, "Huber (SmoothL1) loss", size=8.2)
    label(ax, 11.55, 6.4, "L = SmoothL1( Q_policy(s,a), Q_target )", size=8.2)
    label(ax, 11.55, 6.1, "robust to outlier rewards vs MSE", size=8.2)

    # ── Hyperparams table ─────────────────────────────────────────────────
    box(ax, 0.3, 0.2, 13.4, 1.5, PALETTE["bg"], alpha=0.0, radius=0.04, lw=0)
    box(ax, 0.3, 0.2, 13.4, 1.5, PALETTE["border"], alpha=0.15, radius=0.05, lw=1)
    label(ax, 7.0, 1.52, "Hyperparameters", color=PALETTE["muted"], size=8.5, bold=True)
    params = [
        ("state_dim", "24"), ("action_dim", "5"), ("hidden_dims", "[256,128,64]"),
        ("lr", "1e-3"), ("γ (gamma)", "0.99"), ("ε start→end", "1.0→0.05"),
        ("ε decay", "0.995"), ("batch_size", "64"), ("target_update", "100 steps"),
        ("buffer_cap", "10,000"), ("loss", "SmoothL1"), ("optimizer", "Adam"),
        ("grad_clip", "10.0"), ("device", "auto (CUDA/CPU)"),
    ]
    pw = 13.0 / 7
    for i, (k, v) in enumerate(params):
        col = i % 7
        row = i // 7
        px = 0.45 + col * pw
        py = 1.1 - row * 0.5
        box(ax, px, py - 0.18, pw - 0.1, 0.42, PALETTE["panel"],
            alpha=0.6, radius=0.03, lw=0.8, ec=PALETTE["border"])
        label(ax, px + 0.15, py + 0.04, k, size=7.2, ha="left",
              color=PALETTE["muted"])
        label(ax, px + pw - 0.2, py + 0.04, v, size=7.5, ha="right",
              color=PALETTE["text"])

    fig.savefig(os.path.join(OUT, "05_dqn_architecture.png"), dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  saved 05_dqn_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating diagrams...")
    fig_model_structure()
    fig_defender()
    fig_attacker()
    fig_interaction()
    fig_dqn()
    print("Done. All images written to:", OUT)
