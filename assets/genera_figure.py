"""
Genera le figure del deck del mattino, in stile IFAB (palette navy/ciano).
Output: file SVG nella cartella assets/. Riproducibile (seed fisso).

    python genera_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, FancyArrowPatch

HERE = Path(__file__).parent

NAVY = "#002060"
CYAN = "#00ADCF"
LIGHT = "#8FD2E9"
SLATE = "#414F69"
GREY = "#545454"
RED = "#DC4C4C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "text.color": SLATE,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
})


def save(fig, name):
    fig.savefig(HERE / name, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("scritto", name)


# ---------------------------------------------------------------------------
# 1) Mappa del campo: AI superset di ML (= apprendimento statistico) superset di DL
# ---------------------------------------------------------------------------
def mappa_ai_ml():
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.add_patch(Circle((0.5, 0.5), 0.48, facecolor=LIGHT, edgecolor=SLATE, lw=1.6, alpha=0.35))
    ax.add_patch(Circle((0.5, 0.42), 0.33, facecolor=CYAN, edgecolor=NAVY, lw=1.6, alpha=0.55))
    ax.add_patch(Circle((0.5, 0.34), 0.17, facecolor=NAVY, edgecolor=NAVY, lw=1.6))

    ax.text(0.5, 0.93, "Intelligenza artificiale", ha="center", va="center",
            fontsize=19.5, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.69, "Machine learning\n= apprendimento statistico", ha="center", va="center",
            fontsize=16.9, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.34, "Deep\nlearning", ha="center", va="center",
            fontsize=15.6, fontweight="bold", color="white")
    ax.text(0.18, 0.14, "AI simbolica,\nsistemi a regole", ha="center", va="center",
            fontsize=13.7, color=SLATE, style="italic")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_aspect("equal")
    save(fig, "mappa-ai-ml.svg")


# ---------------------------------------------------------------------------
# 2) Y = f(X) + epsilon: relazione vera, dati, errore irriducibile
# ---------------------------------------------------------------------------
def y_fx():
    rng = np.random.default_rng(7)
    f = lambda x: 8 + 1.1 * x + 4.5 * np.sin(0.7 * x)
    x = np.linspace(0, 12, 38)
    y = f(x) + rng.normal(0, 1.8, x.size)
    xs = np.linspace(0, 12, 300)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(xs, f(xs), color=NAVY, lw=3, label="f(X): la relazione vera", zorder=2)
    ax.scatter(x, y, color=CYAN, s=34, edgecolor="white", lw=0.6, zorder=3, label="Dati osservati")
    for xi, yi in zip(x[5::8], y[5::8]):
        ax.plot([xi, xi], [f(xi), yi], color=RED, lw=1.3, ls=":", zorder=2)
    xi, yi = x[5], y[5]
    ax.annotate(r"$\varepsilon$", (xi, (f(xi) + yi) / 2), color=RED,
                fontsize=20.8, fontweight="bold", xytext=(8, 0), textcoords="offset points")

    ax.set_xlabel("X  (predittore)", fontsize=15.6)
    ax.set_ylabel("Y  (risposta)", fontsize=15.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.96, edgecolor="#c9d2dd", facecolor="white", fontsize=14.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "y-fx-illustrazione.svg")


# ---------------------------------------------------------------------------
# 3) Piano flessibilita' vs interpretabilita' (stile ISL fig. 2.7)
# ---------------------------------------------------------------------------
def flessibilita_interpretabilita():
    # (nome, x, y, dx, dy, allineamento orizzontale dell'etichetta)
    metodi = [
        ("Regressione lineare", 0.12, 0.92, 10, 6, "left"),
        ("Regressione logistica, lasso", 0.20, 0.82, 10, 0, "left"),
        ("Ridge", 0.30, 0.73, 10, -2, "left"),
        ("GAM, spline", 0.44, 0.60, -10, -4, "right"),
        ("Alberi di decisione", 0.50, 0.69, 10, 6, "left"),
        ("SVM", 0.68, 0.40, -10, 6, "right"),
        ("Bagging, random forest, boosting", 0.80, 0.28, -10, 6, "right"),
        ("Reti neurali profonde", 0.93, 0.13, -10, -2, "right"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for nome, fx, it, dx, dy, ha in metodi:
        ax.scatter(fx, it, s=130, color=CYAN, edgecolor=NAVY, lw=1.5, zorder=3)
        ax.annotate(nome, (fx, it), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=13.7, color=NAVY, fontweight="bold")

    ax.annotate("", xy=(1.05, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.6))
    ax.annotate("", xy=(0, 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.6))
    ax.text(0.52, -0.07, "Flessibilità  (bassa verso alta)", ha="center", fontsize=15.6, color=SLATE)
    ax.text(-0.05, 0.52, "Interpretabilità  (bassa verso alta)", va="center", rotation=90,
            fontsize=15.6, color=SLATE)

    ax.set_xlim(-0.02, 1.15); ax.set_ylim(-0.02, 1.1); ax.axis("off")
    save(fig, "flessibilita-interpretabilita.svg")


# ---------------------------------------------------------------------------
# 4) Geometria di ridge e lasso: perche' il lasso azzera i coefficienti
# ---------------------------------------------------------------------------
def ridge_lasso():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))
    bhat = (1.9, 1.5)
    s = 1.0

    def contorni(ax):
        for r in (0.45, 0.9, 1.4, 1.95):
            ax.add_patch(Ellipse(bhat, width=2.0 * r, height=1.3 * r,
                                 fill=False, edgecolor=CYAN, lw=1.4, alpha=0.9))
        ax.scatter(*bhat, color=NAVY, s=45, zorder=5)
        ax.annotate(r"$\hat\beta$ (minimi quadrati)", bhat, xytext=(6, 8),
                    textcoords="offset points", fontsize=13.0, color=NAVY)
        ax.axhline(0, color=GREY, lw=1); ax.axvline(0, color=GREY, lw=1)
        ax.set_xlim(-0.8, 3.2); ax.set_ylim(-0.8, 2.8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.text(3.15, -0.18, r"$\beta_1$", fontsize=15.6, color=SLATE)
        ax.text(-0.2, 2.75, r"$\beta_2$", fontsize=15.6, color=SLATE)

    # lasso: vincolo a rombo (L1), soluzione su un vertice -> un coefficiente a zero
    ax = axes[0]
    ax.add_patch(Polygon([(s, 0), (0, s), (-s, 0), (0, -s)],
                         facecolor=NAVY, edgecolor=NAVY, alpha=0.18, lw=1.6))
    contorni(ax)
    ax.scatter(0, s, color=RED, s=70, zorder=6)
    ax.annotate("Soluzione:\n" + r"$\beta_1=0$", (0, s), xytext=(-70, 6),
                textcoords="offset points", fontsize=13.0, color=RED, fontweight="bold")
    ax.set_title("Lasso  (vincolo L1)", color=NAVY, fontsize=16.9, fontweight="bold")

    # ridge: vincolo a cerchio (L2), soluzione mai esattamente su un asse
    ax = axes[1]
    ax.add_patch(Circle((0, 0), s, facecolor=NAVY, edgecolor=NAVY, alpha=0.18, lw=1.6))
    contorni(ax)
    ax.scatter(0.62, 0.78, color=RED, s=70, zorder=6)
    ax.annotate("Soluzione:\nentrambi piccoli,\nnessuno a zero", (0.62, 0.78),
                xytext=(-95, -38), textcoords="offset points", fontsize=13.0,
                color=RED, fontweight="bold")
    ax.set_title("Ridge  (vincolo L2)", color=NAVY, fontsize=16.9, fontweight="bold")

    save(fig, "ridge-lasso.svg")


# ---------------------------------------------------------------------------
# 5) Genealogia dei modelli profondi: e' sempre statistical learning
# ---------------------------------------------------------------------------
def genealogia():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    def box(x, y, testo, fill, fg, w=2.3, h=0.95):
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=fill, edgecolor=NAVY, lw=1.6, zorder=3))
        ax.text(x, y, testo, ha="center", va="center", fontsize=13.7,
                color=fg, fontweight="bold", zorder=4)

    def freccia(p1, p2):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                     color=SLATE, lw=1.6, shrinkA=14, shrinkB=14, zorder=2))

    P = (1.3, 3.0); M = (4.0, 3.0)
    CNN = (6.7, 4.7); RNN = (6.7, 3.0); GAN = (6.7, 1.3)
    TR = (9.4, 3.0); LLM = (11.8, 3.0)

    for a, b in [(P, M), (M, CNN), (M, RNN), (M, GAN), (RNN, TR), (TR, LLM)]:
        freccia(a, b)

    box(*P, "Percettrone", CYAN, NAVY)
    box(*M, "Rete profonda\n(MLP)", CYAN, NAVY)
    box(*CNN, "CNN\nimmagini", LIGHT, NAVY)
    box(*RNN, "RNN, LSTM\nsequenze", LIGHT, NAVY)
    box(*GAN, "GAN\ngenerazione", LIGHT, NAVY)
    box(*TR, "Transformer\nattenzione", NAVY, "white", w=2.5)
    box(*LLM, "LLM", NAVY, "white", w=1.7)

    ax.set_xlim(0.2, 13.0); ax.set_ylim(0.4, 5.6); ax.axis("off")
    save(fig, "genealogia-modelli.svg")


# ---------------------------------------------------------------------------
# 6) Linea del tempo dello statistical learning (storia da ISL cap. 1)
# ---------------------------------------------------------------------------
def linea_tempo():
    tappe = [
        ("1805", "Minimi quadrati\n(regressione lineare)", 0.03, -1),
        ("1936", "Analisi\ndiscriminante", 0.19, 1),
        ("Anni '40", "Regressione\nlogistica", 0.33, -1),
        ("Anni '70", "Modelli lineari\ngeneralizzati", 0.49, 1),
        ("Anni '80", "Alberi (CART), GAM,\nreti neurali", 0.66, -1),
        ("Anni '90", "Support vector\nmachines", 0.81, 1),
        ("Oggi", "Deep learning,\nLLM", 0.96, -1),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    ax.annotate("", xy=(1.03, 0), xytext=(-0.03, 0),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=3.8, mutation_scale=26))
    for anno, testo, x, lato in tappe:
        ax.plot([x], [0], "o", color=CYAN, markersize=23,
                markeredgecolor=NAVY, markeredgewidth=2.4, zorder=4)
        ax.plot([x, x], [0, 0.19 * lato], color=SLATE, lw=1.8, zorder=2)
        ax.text(x, 0.25 * lato, anno, ha="center",
                va="bottom" if lato > 0 else "top", fontsize=22,
                fontweight="bold", color=NAVY)
        ax.text(x, 0.58 * lato, testo, ha="center",
                va="bottom" if lato > 0 else "top", fontsize=18, color=SLATE)
    ax.set_xlim(-0.07, 1.10); ax.set_ylim(-0.96, 0.96); ax.axis("off")
    save(fig, "linea-tempo.svg")


if __name__ == "__main__":
    mappa_ai_ml()
    y_fx()
    flessibilita_interpretabilita()
    ridge_lasso()
    genealogia()
    linea_tempo()
    print("Tutte le figure generate.")
