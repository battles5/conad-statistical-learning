"""
Diagrammi aggiuntivi: la scelta per task (flowchart) e le architetture dei
modelli profondi (RNN, transformer, GAN), in stile IFAB.

    python genera_figure_3.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).parent
NAVY, CYAN, LIGHT = "#002060", "#00ADCF", "#8FD2E9"
SLATE, GREY, RED = "#414F69", "#545454", "#DC4C4C"
GOLD, GREEN = "#E0A93B", "#2E8B6F"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "text.color": SLATE,
})


def save(fig, name):
    fig.savefig(HERE / name, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("scritto", name)


def box(ax, x, y, w, h, testo, fill, fg="white", fs=15, weight="bold"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=NAVY, lw=1.8, zorder=3))
    ax.text(x, y, testo, ha="center", va="center", color=fg,
            fontsize=fs, fontweight=weight, zorder=4)


def arrow(ax, p1, p2, label=None, color=SLATE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=20,
                                 color=color, lw=2.0, shrinkA=12, shrinkB=12, zorder=2))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=12.5, color=NAVY, style="italic",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#c9d2dd"))


# ---------------------------------------------------------------------------
# 1) La scelta PER TASK: diagramma di flusso
# ---------------------------------------------------------------------------
def scelta_per_task():
    fig, ax = plt.subplots(figsize=(11.4, 7.4))
    Q1 = (6.0, 8.4); Q2 = (4.0, 6.0); Q3 = (4.0, 3.6)
    INF = (9.6, 7.0); NS = (9.6, 4.6)
    REG = (1.7, 1.4); CLA = (4.4, 1.4); FOR = (7.0, 1.4)

    arrow(ax, Q1, Q2, "prevedere")
    arrow(ax, Q1, INF, "capire")
    arrow(ax, Q2, Q3, "sì → supervisionato")
    arrow(ax, Q2, NS, "no")
    arrow(ax, Q3, REG, "numero")
    arrow(ax, Q3, CLA, "categoria")
    arrow(ax, Q3, FOR, "tempo")

    box(ax, *Q1, 4.2, 1.0, "1 · Qual è l'obiettivo?", NAVY)
    box(ax, *Q2, 3.4, 1.1, "2 · I dati contengono\nla risposta $Y$?", NAVY)
    box(ax, *Q3, 3.4, 1.0, "3 · Che natura ha $Y$?", NAVY)
    box(ax, *INF, 4.0, 1.1, "Inferenza / modeling:\nmodelli interpretabili", CYAN, fg=NAVY)
    box(ax, *NS, 4.0, 1.1, "Non supervisionato:\nclustering, PCA", LIGHT, fg=NAVY)
    box(ax, *REG, 2.6, 1.2, "Regressione\n(Y numerica)", GREEN)
    box(ax, *CLA, 2.6, 1.2, "Classificazione\n(Y categoria)", GOLD)
    box(ax, *FOR, 2.6, 1.2, "Forecast\n(Y nel tempo)", CYAN, fg=NAVY)

    ax.set_xlim(0, 12); ax.set_ylim(0.5, 9.2); ax.axis("off")
    save(fig, "scelta-per-task.svg")


# ---------------------------------------------------------------------------
# 2) Architettura RNN: la stessa cella ripetuta nel tempo
# ---------------------------------------------------------------------------
def architettura_rnn():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    xs = [2.0, 5.0, 8.0]
    etich = [r"$t-1$", r"$t$", r"$t+1$"]
    for i, x in enumerate(xs):
        box(ax, x, 1.0, 1.5, 0.8, f"$x_{{{['t-1','t','t+1'][i]}}}$", LIGHT, fg=NAVY, fs=16)
        box(ax, x, 3.0, 1.7, 0.9, f"$h_{{{['t-1','t','t+1'][i]}}}$", CYAN, fg=NAVY, fs=16)
        box(ax, x, 5.0, 1.5, 0.8, f"$y_{{{['t-1','t','t+1'][i]}}}$", NAVY, fs=16)
        arrow(ax, (x, 1.45), (x, 2.5))
        arrow(ax, (x, 3.45), (x, 4.6))
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 0.85, 3.0), (b - 0.85, 3.0), "memoria")
    ax.text(5.0, 5.95, "La stessa cella si ripete nel tempo: lo stato $h_t$ porta avanti la memoria",
            ha="center", fontsize=13.5, color=SLATE)
    ax.set_xlim(0.6, 9.4); ax.set_ylim(0.3, 6.3); ax.axis("off")
    save(fig, "architettura-rnn.svg")


# ---------------------------------------------------------------------------
# 3) Architettura transformer: blocco con auto-attenzione
# ---------------------------------------------------------------------------
def architettura_transformer():
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    cx = 0.5
    livelli = [
        (0.07, "Input + posizione", LIGHT, NAVY),
        (0.30, "Auto-attenzione\n(ogni token guarda gli altri)", CYAN, NAVY),
        (0.52, "Somma e normalizza", "#dbe7f0", NAVY),
        (0.70, "Rete feed-forward", CYAN, NAVY),
        (0.88, "Somma e normalizza", "#dbe7f0", NAVY),
    ]
    for y, txt, fill, fg in livelli:
        box(ax, cx, y, 0.74, 0.13, txt, fill, fg=fg, fs=14)
    for y1, y2 in zip([l[0] for l in livelli[:-1]], [l[0] for l in livelli[1:]]):
        arrow(ax, (cx, y1 + 0.065), (cx, y2 - 0.065))
    arrow(ax, (cx, 0.945), (cx, 1.0))
    ax.text(cx, 1.02, "Output", ha="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.annotate("× N\nblocchi", xy=(0.92, 0.59), fontsize=13, color=SLATE,
                style="italic", ha="center")
    ax.set_xlim(0.05, 1.06); ax.set_ylim(0.0, 1.08); ax.axis("off")
    save(fig, "architettura-transformer.svg")


# ---------------------------------------------------------------------------
# 4) Architettura GAN: due reti in competizione
# ---------------------------------------------------------------------------
def architettura_gan():
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    # riga di generazione (in alto) e dato reale (in basso)
    box(ax, 1.25, 4.2, 2.1, 1.05, "Rumore", LIGHT, fg=NAVY, fs=18)
    box(ax, 4.25, 4.2, 2.5, 1.25, "Generatore", CYAN, fg=NAVY, fs=19)
    box(ax, 7.35, 4.2, 2.3, 1.05, "Dato finto", "#dbe7f0", fg=NAVY, fs=18)
    box(ax, 7.35, 1.6, 2.3, 1.05, "Dati reali", LIGHT, fg=NAVY, fs=18)
    box(ax, 10.4, 2.9, 3.0, 1.3, "Discriminatore", NAVY, fs=19)
    # flusso principale
    arrow(ax, (2.3, 4.2), (3.0, 4.2))
    arrow(ax, (5.5, 4.2), (6.2, 4.2))
    arrow(ax, (8.5, 4.2), (9.15, 3.45))
    arrow(ax, (8.5, 1.6), (9.15, 2.35))
    # verdetto del discriminatore
    arrow(ax, (10.4, 2.25), (10.4, 1.75))
    ax.text(10.4, 1.35, "vero o finto?", fontsize=17, color=NAVY,
            ha="center", va="center", style="italic", fontweight="bold")
    # arco di retroazione: l'errore aggiorna il generatore
    ax.add_patch(FancyArrowPatch((10.4, 3.55), (4.25, 4.83),
                                 connectionstyle="arc3,rad=-0.32",
                                 arrowstyle="-|>", mutation_scale=22,
                                 color=RED, lw=2.4, ls=(0, (6, 3)),
                                 shrinkA=10, shrinkB=10, zorder=2))
    ax.text(6.9, 5.78, "errore: aggiorna il generatore", fontsize=16,
            color=RED, ha="center", va="center", style="italic", fontweight="bold")
    ax.set_xlim(0.0, 12.1); ax.set_ylim(0.7, 6.2); ax.axis("off")
    save(fig, "architettura-gan.svg")


if __name__ == "__main__":
    scelta_per_task()
    architettura_rnn()
    architettura_transformer()
    architettura_gan()
    print("Diagrammi aggiuntivi generati.")
