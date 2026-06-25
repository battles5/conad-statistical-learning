"""
Figure pedagogiche del corso, in stile ISL / Torelli ma con palette IFAB.
Riproducono i grafici chiave del libro (Introduction to Statistical Learning)
e delle dispense Torelli, generandoli da simulazioni autentiche.

    python genera_figure_2.py

Dipendenze: numpy, matplotlib, scikit-learn. Riproducibile (seed fissi).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
NAVY, CYAN, LIGHT = "#002060", "#00ADCF", "#8FD2E9"
SLATE, GREY, RED = "#414F69", "#545454", "#DC4C4C"
GOLD = "#E0A93B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "text.color": SLATE, "axes.edgecolor": SLATE, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE, "axes.titlecolor": NAVY,
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
    "legend.fontsize": 13, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
})


def save(fig, name):
    fig.savefig(HERE / name, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("scritto", name)


# funzione vera comune agli esempi simulati (x in [0,1])
def f_vera(x):
    return 0.35 + 0.45 * np.sin(2.2 * np.pi * x) * (x + 0.25)


def poly_fit_pred(xtr, ytr, xte, grado):
    c = np.polyfit(xtr, ytr, grado)
    return np.polyval(c, xte)


# ---------------------------------------------------------------------------
# 1) Scatola nera di Breiman: natura che lega X a Y
# ---------------------------------------------------------------------------
def nature_blackbox():
    # layout verticale, tutto centrato sull'asse x=0.5; etichette su due righe
    # centrate (simbolo sopra, descrizione sotto) per stare sull'asse di frecce e box
    fig, ax = plt.subplots(figsize=(5.4, 6.3))
    cx = 0.5
    ax.text(cx, 0.91, "$X$\n(predittori)", ha="center", va="center",
            fontsize=18, color=NAVY, fontweight="bold", linespacing=1.35)
    ax.annotate("", xy=(cx, 0.64), xytext=(cx, 0.82),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=2.8))
    ax.add_patch(FancyBboxPatch((cx - 0.27, 0.42), 0.54, 0.20,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=NAVY, edgecolor=NAVY))
    ax.text(cx, 0.52, "natura\n$f$ sconosciuta", ha="center", va="center",
            color="white", fontsize=19, fontweight="bold", linespacing=1.35)
    ax.annotate("", xy=(cx, 0.22), xytext=(cx, 0.40),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=2.8))
    ax.text(cx, 0.13, "$Y$\n(risposta)", ha="center", va="center",
            fontsize=18, color=NAVY, fontweight="bold", linespacing=1.35)
    ax.text(cx, 0.0, "osserviamo solo gli ingressi e le uscite,\nnon il meccanismo",
            ha="center", va="center", fontsize=13.5, color=GREY, style="italic", linespacing=1.3)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0); ax.axis("off")
    save(fig, "nature-blackbox.svg")


# ---------------------------------------------------------------------------
# 2) Dati simulati: training (grigi) e test (blu) attorno alla vera f
# ---------------------------------------------------------------------------
def dati_train_test():
    rng = np.random.default_rng(11)
    n = 80
    x = rng.uniform(0, 1, n)
    y = f_vera(x) + rng.normal(0, 0.09, n)
    is_test = rng.random(n) < 0.30
    xs = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, f_vera(xs), color=NAVY, lw=2.6, zorder=2, label="relazione vera $f$")
    ax.scatter(x[~is_test], y[~is_test], s=42, color="#b9bfca", edgecolor="white",
               lw=0.5, zorder=3, label="training set")
    ax.scatter(x[is_test], y[is_test], s=46, color=CYAN, edgecolor=NAVY,
               lw=0.7, zorder=4, label="test set")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True, framealpha=0.96, edgecolor="#c9d2dd", facecolor="white", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "dati-train-test.svg")


# ---------------------------------------------------------------------------
# 3) Tre modelli a flessibilita' crescente sullo stesso training set
# ---------------------------------------------------------------------------
def flessibilita_fit():
    rng = np.random.default_rng(11)
    n = 80
    x = rng.uniform(0, 1, n)
    y = f_vera(x) + rng.normal(0, 0.09, n)
    xs = np.linspace(0.01, 0.99, 400)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.scatter(x, y, s=38, color="#b9bfca", edgecolor="white", lw=0.5, zorder=2)
    ax.plot(xs, poly_fit_pred(x, y, xs, 1), color=GOLD, lw=2.8,
            label="rigido (lineare)", zorder=3)
    ax.plot(xs, poly_fit_pred(x, y, xs, 5), color=CYAN, lw=2.8,
            label="equilibrato", zorder=4)
    ax.plot(xs, poly_fit_pred(x, y, xs, 18), color=RED, lw=2.0,
            label="troppo flessibile", zorder=5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_ylim(y.min() - 0.12, y.max() + 0.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True, framealpha=0.96, edgecolor="#c9d2dd", facecolor="white", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "flessibilita-fit.svg")


# ---------------------------------------------------------------------------
# 4) Errore su training e test al variare della flessibilita' (ISL fig. 2.9)
# ---------------------------------------------------------------------------
def errore_train_test():
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.default_rng(5)
    # flessibilita' crescente = K decrescente (smoother stabile, stile ISL/Torelli)
    ks = np.array([45, 32, 24, 18, 13, 10, 7, 5, 4, 3, 2, 1])
    sigma, reps, n = 0.09, 150, 110
    xte = np.linspace(0.02, 0.98, 300)
    fte = f_vera(xte)
    tr = np.zeros((reps, ks.size)); te = np.zeros((reps, ks.size))
    for r in range(reps):
        x = rng.uniform(0, 1, n)
        y = f_vera(x) + rng.normal(0, sigma, n)
        yte = fte + rng.normal(0, sigma, xte.size)
        for j, k in enumerate(ks):
            mdl = KNeighborsRegressor(n_neighbors=k).fit(x.reshape(-1, 1), y)
            tr[r, j] = np.mean((y - mdl.predict(x.reshape(-1, 1))) ** 2)
            te[r, j] = np.mean((yte - mdl.predict(xte.reshape(-1, 1))) ** 2)
    tr_m, te_m = tr.mean(0), te.mean(0)
    flex = np.arange(ks.size)  # asse equispaziato: a destra piu' flessibile

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(sigma ** 2, color=GREY, ls="--", lw=1.6, label="errore irriducibile")
    ax.plot(flex, tr_m, "o-", color="#9aa3b2", lw=2.4, ms=6, label="errore di training")
    ax.plot(flex, te_m, "o-", color=RED, lw=2.6, ms=6, label="errore di test")
    jmin = int(np.argmin(te_m))
    ax.scatter([flex[jmin]], [te_m[jmin]], s=150, facecolor="none",
               edgecolor=NAVY, lw=2.2, zorder=6)
    ax.annotate("punto di equilibrio", (flex[jmin], te_m[jmin]),
                xytext=(10, 30), textcoords="offset points", color=NAVY,
                fontsize=13.7, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel("flessibilità del modello  (più a destra = più flessibile)", fontsize=14.3)
    ax.set_ylabel("errore quadratico medio", fontsize=14.3)
    ax.set_ylim(0, te_m.max() * 1.12)
    ax.set_xticks([])
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, edgecolor="#c9d2dd", facecolor="white", fontsize=13.0)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "errore-train-test.svg")


# ---------------------------------------------------------------------------
# 5) Scomposizione dell'errore: bias^2 + varianza + irriducibile (ISL fig. 2.12)
# ---------------------------------------------------------------------------
def scomposizione_bias_var():
    rng = np.random.default_rng(9)
    gradi = np.arange(1, 14)
    sigma = 0.09
    reps = 400
    n = 60
    x0 = np.linspace(0.05, 0.95, 60)
    f0 = f_vera(x0)
    preds = np.zeros((gradi.size, reps, x0.size))
    for j, g in enumerate(gradi):
        for r in range(reps):
            x = rng.uniform(0, 1, n)
            y = f_vera(x) + rng.normal(0, sigma, n)
            c = np.polyfit(x, y, g)
            preds[j, r] = np.polyval(c, x0)
    bias2 = ((preds.mean(1) - f0) ** 2).mean(1)
    var = preds.var(1).mean(1)
    irr = np.full(gradi.size, sigma ** 2)
    tot = bias2 + var + irr

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(gradi, bias2, "o-", color=NAVY, lw=2.4, ms=5, label="distorsione$^2$ (bias)")
    ax.plot(gradi, var, "o-", color=CYAN, lw=2.4, ms=5, label="varianza")
    ax.plot(gradi, irr, "--", color=GREY, lw=1.8, label="errore irriducibile")
    ax.plot(gradi, tot, "o-", color=RED, lw=2.8, ms=5, label="errore di test totale")
    ax.set_xlabel("flessibilità del modello", fontsize=14.3)
    ax.set_ylabel("contributo all'errore", fontsize=14.3)
    ax.set_ylim(0, np.percentile(tot, 92) * 1.15)
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, edgecolor="#c9d2dd", facecolor="white", fontsize=13.0)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "scomposizione-bias-var.svg")


# ---------------------------------------------------------------------------
# 6) Il bersaglio: le quattro combinazioni di bias e varianza
# ---------------------------------------------------------------------------
def bersaglio_bias_varianza():
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.6))
    casi = [
        ("bias basso, varianza bassa", (0, 0), 0.07, axes[0, 0]),
        ("bias basso, varianza alta", (0, 0), 0.22, axes[0, 1]),
        ("bias alto, varianza bassa", (0.42, 0.30), 0.07, axes[1, 0]),
        ("bias alto, varianza alta", (0.42, 0.30), 0.22, axes[1, 1]),
    ]
    for titolo, centro, sd, ax in casi:
        for r, col in [(1.0, LIGHT), (0.66, "#cfe7f2"), (0.33, CYAN)]:
            ax.add_patch(plt.Circle((0, 0), r, facecolor=col, edgecolor="white",
                                    lw=1.2, zorder=1))
        ax.add_patch(plt.Circle((0, 0), 0.10, facecolor=NAVY, zorder=2))
        px = rng.normal(centro[0], sd, 18)
        py = rng.normal(centro[1], sd, 18)
        ax.scatter(px, py, s=34, color=RED, edgecolor="white", lw=0.6, zorder=3)
        ax.set_title(titolo, fontsize=15.0, fontweight="bold")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal"); ax.axis("off")
    save(fig, "bersaglio-bias-varianza.svg")


# ---------------------------------------------------------------------------
# 7) Regressione logistica: la curva a S della probabilita' (ISL fig. 4.2)
# ---------------------------------------------------------------------------
def logistica_sigmoide():
    rng = np.random.default_rng(4)
    n = 400
    balance = rng.uniform(0, 2700, n)
    p = 1 / (1 + np.exp(-(-9.5 + 0.0055 * balance)))
    y = (rng.random(n) < p).astype(int)
    m = LogisticRegression().fit(balance.reshape(-1, 1), y)
    xs = np.linspace(0, 2700, 300)
    ps = m.predict_proba(xs.reshape(-1, 1))[:, 1]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, ps, color=NAVY, lw=3, zorder=3)
    ax.scatter(balance[y == 1], np.full((y == 1).sum(), 1.02), s=14, color=RED,
               marker="|", alpha=0.6)
    ax.scatter(balance[y == 0], np.full((y == 0).sum(), -0.02), s=14, color=CYAN,
               marker="|", alpha=0.6)
    ax.axhline(0.5, color=GREY, ls=":", lw=1.4)
    ax.text(120, 0.54, "soglia 0,5", color=GREY, fontsize=13.0)
    ax.set_xlabel("saldo sul conto (balance)", fontsize=14.3)
    ax.set_ylabel("probabilità stimata di insolvenza", fontsize=14.3)
    ax.set_ylim(-0.08, 1.10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "logistica-sigmoide.svg")


# ---------------------------------------------------------------------------
# dati 2D per la classificazione (confine non lineare, stile ISL)
# ---------------------------------------------------------------------------
def _dati_2classi(rng, n=260):
    # confine di Bayes ondulato e regolare, con sovrapposizione (stile ISL)
    X = rng.uniform(0, 1, (n, 2))
    conf = 0.5 + 0.20 * np.sin(2.2 * np.pi * X[:, 0])
    z = 7.0 * (X[:, 1] - conf)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# 8) Confini di decisione KNN: K=1 (sovradattato) vs K=100 (troppo liscio)
# ---------------------------------------------------------------------------
def confine_knn():
    rng = np.random.default_rng(7)
    X, y = _dati_2classi(rng)
    xx, yy = np.meshgrid(np.linspace(-0.05, 1.05, 300), np.linspace(-0.05, 1.05, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(["#d7eef6", "#f7d6d6"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8))
    for ax, k in zip(axes, [1, 100]):
        m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        Z = m.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.7, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors=[NAVY], linewidths=1.8)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=22, color=CYAN, edgecolor="white", lw=0.4)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, color=RED, edgecolor="white", lw=0.4)
        ax.set_title(f"KNN con K = {k}", fontsize=16.9, fontweight="bold")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("$X_1$"); ax.set_ylabel("$X_2$")
    axes[0].text(0.5, -0.13, "troppo flessibile: confine frastagliato",
                 transform=axes[0].transAxes, ha="center", fontsize=13.0, color=RED)
    axes[1].text(0.5, -0.13, "troppo rigido: confine quasi lineare",
                 transform=axes[1].transAxes, ha="center", fontsize=13.0, color=GREY)
    save(fig, "confine-knn.svg")


# ---------------------------------------------------------------------------
# 9) Errore di classificazione train/test al variare di 1/K (ISL fig. 2.17)
# ---------------------------------------------------------------------------
def knn_errore():
    rng = np.random.default_rng(7)
    X, y = _dati_2classi(rng)
    Xte, yte = _dati_2classi(np.random.default_rng(21))
    Ks = np.array([1, 2, 3, 5, 8, 12, 18, 30, 50, 90, 150])
    etr, ete = [], []
    for k in Ks:
        m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        etr.append(np.mean(m.predict(X) != y))
        ete.append(np.mean(m.predict(Xte) != yte))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(1 / Ks, etr, "o-", color="#9aa3b2", lw=2.2, ms=6, label="errore di training")
    ax.plot(1 / Ks, ete, "o-", color=RED, lw=2.6, ms=6, label="errore di test")
    ax.set_xscale("log")
    ax.set_xlabel("flessibilità  →  $1/K$  (scala logaritmica)", fontsize=14.3)
    ax.set_ylabel("tasso di errore", fontsize=14.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.96, edgecolor="#c9d2dd", facecolor="white", fontsize=13.0)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "knn-errore.svg")


# ---------------------------------------------------------------------------
# 10) Un albero di decisione (orientato al churn Conad)
# ---------------------------------------------------------------------------
def albero_decisione():
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    def nodo(x, y, testo, foglia=False, esito=None):
        col = {"alto": RED, "medio": GOLD, "basso": "#2e8b6f"}.get(esito, NAVY) if foglia else CYAN
        w, h = (1.7, 0.74) if foglia else (2.5, 0.74)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor=col, edgecolor=NAVY, lw=1.4, zorder=3))
        ax.text(x, y, testo, ha="center", va="center", color="white",
                fontsize=12.3, fontweight="bold", zorder=4)

    def ramo(p1, p2, etichetta):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.5), zorder=1)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.06, etichetta, ha="center", fontsize=12.3,
                color=SLATE, style="italic")

    R = (4.4, 4.4); A = (2.2, 2.9); B = (6.6, 2.9)
    AA = (1.0, 1.2); AB = (3.4, 1.2); BA = (5.4, 1.2); BB = (7.8, 1.2)
    for a, b, e in [(R, A, "sì"), (R, B, "no"), (A, AA, "sì"), (A, AB, "no"),
                    (B, BA, "sì"), (B, BB, "no")]:
        ramo(a, b, e)
    nodo(*R, "recency > 30 giorni?")
    nodo(*A, "visite/mese < 3?")
    nodo(*B, "reclami > 0?")
    nodo(*AA, "churn\nalto", foglia=True, esito="alto")
    nodo(*AB, "churn\nmedio", foglia=True, esito="medio")
    nodo(*BA, "churn\nmedio", foglia=True, esito="medio")
    nodo(*BB, "churn\nbasso", foglia=True, esito="basso")
    ax.set_xlim(0, 8.8); ax.set_ylim(0.5, 5.0); ax.axis("off")
    save(fig, "albero-decisione.svg")


# ---------------------------------------------------------------------------
# 11) Oltre la linearita': salario in funzione dell'eta' (stile Wage, ISL cap.7)
# ---------------------------------------------------------------------------
def polinomi_spline():
    rng = np.random.default_rng(8)
    n = 240
    eta = rng.uniform(18, 80, n)
    vera = 20 + 2.6 * (eta - 18) - 0.032 * (eta - 18) ** 2
    wage = vera + rng.normal(0, 12, n)
    xs = np.linspace(18, 80, 300)
    cl = np.polyfit(eta, wage, 1)
    cp = np.polyfit(eta, wage, 4)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.scatter(eta, wage, s=20, color="#c4cad4", edgecolor="white", lw=0.3, zorder=2)
    ax.plot(xs, np.polyval(cl, xs), color=GOLD, lw=2.6, label="lineare", zorder=3)
    ax.plot(xs, np.polyval(cp, xs), color=NAVY, lw=3, label="polinomio (grado 4)", zorder=4)
    ax.set_xlabel("età", fontsize=14.3); ax.set_ylabel("salario (migliaia)", fontsize=14.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.96, edgecolor="#c9d2dd", facecolor="white", fontsize=13.0)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "polinomi-spline.svg")


if __name__ == "__main__":
    nature_blackbox()
    dati_train_test()
    flessibilita_fit()
    errore_train_test()
    scomposizione_bias_var()
    bersaglio_bias_varianza()
    logistica_sigmoide()
    confine_knn()
    knn_errore()
    albero_decisione()
    polinomi_spline()
    print("Figure pedagogiche generate.")
