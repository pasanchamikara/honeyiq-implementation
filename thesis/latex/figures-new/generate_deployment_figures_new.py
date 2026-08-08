"""
Generate deployment-architecture diagrams for the Oracle Cloud + certbot
deployment discussion (thesis/questions/deployment_defense_questions.md).
Run from project root: python thesis/latex/figures-new/generate_deployment_figures_new.py

Style matches thesis/latex/figures/generate_arch_figures.py so these can be
folded into the existing figure set later.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(__file__)

C = {
    "bg":      "#F8F9FA",
    "neutral": "#2C3E50",
    "arrow":   "#555555",
    "cloud":   "#2471A3",
    "light_cloud": "#D6EAF8",
    "honeypot": "#C0392B",
    "light_honeypot": "#FADBD8",
    "tls":     "#1E8449",
    "light_tls": "#D5F5E3",
    "ca":      "#7D3C98",
    "light_ca": "#E8DAEF",
    "external": "#7B241C",
    "light_external": "#F2D7D5",
    "warn":    "#B9770E",
}


def _box(ax, x, y, w, h, facecolor, edgecolor, lw=1.5, radius=0.04, zorder=3, linestyle="solid"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=zorder, linestyle=linestyle,
    )
    ax.add_patch(box)
    return box


def _label(ax, x, y, text, fontsize=9, color="black", bold=False, zorder=5, ha="center", va="center"):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va, weight=weight, zorder=zorder, wrap=True)


def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.4, arrowstyle="-|>", mutation_scale=12, zorder=4, label="", label_offset=(0.02, 0)):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=lw, mutation_scale=mutation_scale),
        zorder=zorder,
    )
    if label:
        mx, my = (x0 + x1) / 2 + label_offset[0], (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=7, color=color, ha="left", va="center", style="italic", zorder=5)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 - Proposed Oracle Cloud Deployment Architecture
# ═════════════════════════════════════════════════════════════════════════════

def fig_oracle_deployment():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    ax.set_title("HoneyIQ - Proposed Oracle Cloud Deployment Architecture",
                 fontsize=13.5, weight="bold", pad=12, color=C["neutral"])
    ax.text(6, 7.05,
            "Conceptual sketch, not a diagram of a verified, already-provisioned environment",
            fontsize=8.5, ha="center", color=C["arrow"], style="italic", zorder=5)

    # Internet / attacker source
    _box(ax, 1.4, 6.6, 2.0, 0.8, C["light_external"], C["external"], lw=1.5)
    _label(ax, 1.4, 6.6, "Internet\n(scanners / attackers)", fontsize=8, color=C["external"], bold=True)

    # Let's Encrypt CA
    _box(ax, 10.6, 6.6, 2.2, 0.8, C["light_ca"], C["ca"], lw=1.5)
    _label(ax, 10.6, 6.6, "Let's Encrypt CA\n(ACME endpoint)", fontsize=8, color=C["ca"], bold=True)

    # VCN boundary
    vcn = FancyBboxPatch((0.5, 0.4), 11.0, 5.6, boxstyle="round,pad=0.05",
                          facecolor="#EBF5FB", edgecolor=C["cloud"], linewidth=2,
                          zorder=1, linestyle="--")
    ax.add_patch(vcn)
    ax.text(6, 5.75, "Cloud Virtual Network", fontsize=10,
            ha="center", color=C["cloud"], weight="bold", zorder=5)

    # Security list / firewall
    _box(ax, 6.0, 5.15, 6.5, 0.5, "white", C["warn"], lw=1.2, radius=0.03)
    _label(ax, 6.0, 5.15, "Firewall rules: allow 80/443 (ACME + honeypot listener) only, deny else",
           fontsize=7.5, color=C["warn"], bold=True)

    # Public subnet - honeypot instance
    _box(ax, 3.0, 3.3, 4.2, 2.6, C["light_honeypot"], C["honeypot"], lw=2)
    _label(ax, 3.0, 4.35, "Honeypot Instance (public subnet)", fontsize=9, color=C["honeypot"], bold=True)
    for i, txt in enumerate(["OpenCanary listeners\n(HTTP/HTTPS, FTP, SSH, ...)",
                              "nginx TLS termination\n(certbot-managed cert)",
                              "Certbot renew cron"]):
        _box(ax, 3.0, 3.75 - i * 0.75, 3.7, 0.55, "white", C["honeypot"], lw=1, radius=0.03)
        _label(ax, 3.0, 3.75 - i * 0.75, txt, fontsize=7.2)

    # Private subnet - defender / evaluation instance
    _box(ax, 8.7, 3.3, 4.2, 2.6, C["light_cloud"], C["cloud"], lw=2)
    _label(ax, 8.7, 4.35, "Policy Engine Instance\n(private subnet, no public IP)",
           fontsize=8.3, color=C["cloud"], bold=True)
    for i, txt in enumerate(["opencanary_integration\n(ingest + policy_engine)",
                              "SEDM policy decision",
                              "Metrics + audit log storage"]):
        _box(ax, 8.7, 3.7 - i * 0.75, 3.7, 0.55, "white", C["cloud"], lw=1, radius=0.03)
        _label(ax, 8.7, 3.7 - i * 0.75, txt, fontsize=7.2)

    # Storage / results
    _box(ax, 6.0, 1.0, 3.6, 0.8, C["light_tls"], C["tls"], lw=1.5)
    _label(ax, 6.0, 1.0, "Object Storage bucket\n(session logs, evaluation exports)",
           fontsize=7.8, color=C["tls"], bold=True)

    # Arrows
    _arrow(ax, 2.2, 6.2, 3.0, 4.65, color=C["external"], label="probe traffic")
    _arrow(ax, 3.0, 3.6, 3.0, 2.75, color=C["honeypot"])
    _arrow(ax, 5.1, 3.3, 6.6, 3.3, color=C["arrow"], label="session events\n(internal only)")
    _arrow(ax, 3.0, 1.95, 5.4, 1.2, color=C["tls"], label="logs")
    _arrow(ax, 8.7, 1.95, 6.6, 1.2, color=C["tls"], label="metrics")
    _arrow(ax, 9.4, 6.2, 3.6, 3.55, color=C["ca"], lw=1.3, arrowstyle="<|-|>",
           label="ACME challenge\n(DNS-01 preferred)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "arch_oracle_deployment_new.png"), dpi=200,
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 - Certbot / ACME TLS Provisioning Flow
# ═════════════════════════════════════════════════════════════════════════════

def fig_certbot_flow():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    ax.set_title("Certbot / ACME TLS Provisioning Flow for the Honeypot Listener",
                 fontsize=13.5, weight="bold", pad=12, color=C["neutral"])

    steps = [
        ("1. certbot certonly\n--dns-01", C["cloud"], C["light_cloud"]),
        ("2. DNS TXT record\npublished (ACME\nchallenge)", C["ca"], C["light_ca"]),
        ("3. Let's Encrypt\nvalidates DNS-01\nchallenge", C["ca"], C["light_ca"]),
        ("4. Certificate issued\n(90-day / 6-day\nshort-lived option)", C["tls"], C["light_tls"]),
        ("5. nginx reload\n+ cert installed on\nhoneypot instance", C["honeypot"], C["light_honeypot"]),
    ]

    n = len(steps)
    xs = [1.3 + i * 2.35 for i in range(n)]
    y = 3.4
    for x, (txt, ec, fc) in zip(xs, steps):
        _box(ax, x, y, 2.0, 1.5, fc, ec, lw=1.8)
        _label(ax, x, y, txt, fontsize=8, color=ec, bold=True)

    for i in range(n - 1):
        _arrow(ax, xs[i] + 1.0, y, xs[i + 1] - 1.0, y, color=C["arrow"])

    # renewal loop
    _arrow(ax, xs[-1], y - 0.9, xs[0], y - 0.9, color=C["warn"], lw=1.3,
           arrowstyle="-|>", mutation_scale=12)
    ax.plot([xs[-1], xs[-1]], [y - 0.75, y - 0.9], color=C["warn"], lw=1.3)
    ax.plot([xs[0], xs[0]], [y - 0.9, y - 0.75], color=C["warn"], lw=1.3)
    _label(ax, 6.0, y - 1.15, "certbot renew (cron, before expiry), no inbound port 80 required with DNS-01",
           fontsize=8, color=C["warn"], bold=True)

    _label(ax, 6.0, 0.7,
           "DNS-01 challenge avoids exposing port 80 on the honeypot instance itself,\n"
           "keeping the ACME control path separate from the monitored attack surface.",
           fontsize=8.2, color=C["neutral"])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "arch_certbot_tls_flow_new.png"), dpi=200,
                facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    fig_oracle_deployment()
    fig_certbot_flow()
    print("Saved:")
    print(" -", os.path.join(OUT, "arch_oracle_deployment_new.png"))
    print(" -", os.path.join(OUT, "arch_certbot_tls_flow_new.png"))
