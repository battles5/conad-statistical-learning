"""
Genera i notebook Colab del pomeriggio (NB0-NB4) in formato .ipynb.

Stile: celle markdown didattiche in italiano + codice commentato; ogni notebook
ha almeno una "manopola" (una variabile evidenziata da cambiare) e celle "bonus"
per i piu' esperti. Dati caricati via URL dal repo (auto-contenuti per Colab).

    python genera_notebooks.py

Dipendenza: nbformat (incluso con jupyter).
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
REPO_RAW = "https://raw.githubusercontent.com/battles5/conad-statistical-learning/main"
COLAB = "https://colab.research.google.com/github/battles5/conad-statistical-learning/blob/main/notebooks"

URL_AUTO = f"{REPO_RAW}/data/Auto.csv"
URL_DEFAULT = f"{REPO_RAW}/data/Default.csv"
URL_CONAD = f"{REPO_RAW}/data/conad_retail.csv"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def badge(nb_file, titolo, sottotitolo):
    return md(f"""
# {titolo}

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}/{nb_file})

**{sottotitolo}**

corso *Apprendimento statistico* per CONAD Nord Ovest · Orso Peruzzi (IFAB)

> come si esegue una cella: clicca dentro e premi **Shift + Invio**. esegui le celle in ordine, dall'alto verso il basso.
""")


def scrivi(nb_file, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    out = HERE / nb_file
    nbf.write(nb, out)
    print("scritto", out.name)


# ===========================================================================
# NB0 - Setup e warm-up
# ===========================================================================
def nb0():
    c = [
        badge("NB0_setup.ipynb", "NB0 · Setup e warm-up",
              "primo contatto con Colab: import, un dataset, primo split, prima previsione"),
        md("""
## 1. Gli strumenti

importiamo le librerie che useremo tutto il pomeriggio:

- `numpy` e `pandas` per i dati;
- `matplotlib` per i grafici;
- `scikit-learn` per i modelli.
"""),
        code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
print("tutto pronto, versione pandas:", pd.__version__)
"""),
        md("""
## 2. Carichiamo un dataset

usiamo il dataset **Auto** di ISL: consumo e caratteristiche di automobili. lo leggiamo direttamente da un URL, senza scaricare nulla.
"""),
        code(f"""
URL = "{URL_AUTO}"
auto = pd.read_csv(URL)
auto.head()
"""),
        md("""
la colonna `horsepower` (cavalli) ha qualche valore mancante segnato con `?`. la rendiamo numerica e togliamo le righe incomplete.
"""),
        code("""
auto["horsepower"] = pd.to_numeric(auto["horsepower"], errors="coerce")
auto = auto.dropna(subset=["horsepower", "mpg"]).reset_index(drop=True)
print("righe utilizzabili:", len(auto))
"""),
        md("""
## 3. Uno sguardo ai dati

`describe()` riassume ogni colonna numerica (media, minimo, quartili, massimo).
"""),
        code("""
auto[["mpg", "horsepower", "weight", "year"]].describe().round(1)
"""),
        md("""
disegniamo la relazione che studieremo nel prossimo notebook: consumo (`mpg`) contro cavalli (`horsepower`).
"""),
        code("""
plt.figure(figsize=(7, 5))
plt.scatter(auto["horsepower"], auto["mpg"], alpha=0.5, color="#00ADCF")
plt.xlabel("horsepower (cavalli)")
plt.ylabel("mpg (miglia per gallone)")
plt.title("Auto: consumo vs potenza")
plt.show()
"""),
        md("""
## 4. Training set e test set

il principio chiave del mattino: si allena su una parte dei dati e si valuta su dati **mai visti**. `train_test_split` fa esattamente questo.

> **manopola**: cambia `TEST_SIZE` (per esempio 0.2, 0.5) e riesegui. quante righe finiscono nei due insiemi?
"""),
        code("""
# >>> MANOPOLA: quota di dati riservata al test set <<<
TEST_SIZE = 0.30

train, test = train_test_split(auto, test_size=TEST_SIZE, random_state=0)
print("righe di training:", len(train))
print("righe di test:    ", len(test))
"""),
        md("""
## 5. La previsione piu' banale: la media

prima di ogni modello, chiediamoci: quanto sbaglia un "modello" che prevede sempre la **media** del consumo? e' la nostra linea di base (baseline) da battere.
"""),
        code("""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

media_mpg = train["mpg"].mean()                 # impariamo la media SOLO dal training
pred = np.full(len(test), media_mpg)            # prevediamo sempre quella media
print(f"previsione costante: {media_mpg:.1f} mpg")
print(f"errore medio assoluto (MAE) sul test: {mean_absolute_error(test['mpg'], pred):.2f} mpg")
print(f"radice dell'MSE (RMSE) sul test:      {root_mean_squared_error(test['mpg'], pred):.2f} mpg")
"""),
        md("""
nel prossimo notebook vedremo di quanto un vero modello batte questa baseline, e cosa succede se lo rendiamo troppo flessibile.

---

### Cella bonus (per i piu' esperti)

prova a guardare la relazione tra il peso (`weight`) e il consumo: e' piu' forte o piu' debole di quella con i cavalli?
"""),
        code("""
# BONUS
plt.figure(figsize=(7, 4))
plt.scatter(auto["weight"], auto["mpg"], alpha=0.4, color="#414F69")
plt.xlabel("weight (peso)"); plt.ylabel("mpg"); plt.title("Auto: consumo vs peso")
plt.show()
print("correlazione mpg-horsepower:", round(auto['mpg'].corr(auto['horsepower']), 2))
print("correlazione mpg-weight:    ", round(auto['mpg'].corr(auto['weight']), 2))
"""),
    ]
    scrivi("NB0_setup.ipynb", c)


# ===========================================================================
# NB1 - Regressione e bias-varianza visibile (Auto)
# ===========================================================================
def nb1():
    c = [
        badge("NB1_regressione_biasvarianza.ipynb", "NB1 · Regressione e bias-varianza, resi visibili",
              "fit lineare e polinomiale su Auto, la manopola del grado, la U del test error"),
        md("## 1. Dati e setup"),
        code(f"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

auto = pd.read_csv("{URL_AUTO}")
auto["horsepower"] = pd.to_numeric(auto["horsepower"], errors="coerce")
auto = auto.dropna(subset=["horsepower", "mpg"]).reset_index(drop=True)

X = auto[["horsepower"]].values
y = auto["mpg"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=0)
print("training:", len(Xtr), " test:", len(Xte))
"""),
        md("""
## 2. La baseline lineare

una retta: `mpg = b0 + b1 * horsepower`. semplice e interpretabile.
"""),
        code("""
from sklearn.linear_model import LinearRegression

lin = LinearRegression().fit(Xtr, ytr)
mse_tr = mean_squared_error(ytr, lin.predict(Xtr))
mse_te = mean_squared_error(yte, lin.predict(Xte))
print(f"MSE training: {mse_tr:.1f}")
print(f"MSE test:     {mse_te:.1f}")

griglia = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.figure(figsize=(7, 5))
plt.scatter(Xtr, ytr, alpha=0.4, color="#00ADCF", label="training")
plt.plot(griglia, lin.predict(griglia), color="#002060", lw=2.5, label="retta")
plt.xlabel("horsepower"); plt.ylabel("mpg"); plt.legend(); plt.title("Fit lineare")
plt.show()
"""),
        md("""
la retta non coglie la curvatura dei dati: e' un caso di **distorsione (bias) alta**, il modello e' troppo rigido.

## 3. Saliamo in flessibilita': i polinomi

aggiungiamo `horsepower^2`, `horsepower^3`, ... la manopola e' il **grado** del polinomio.

> **manopola**: cambia `GRADO` (prova 1, 2, 5, 10, 15) e osserva come cambia la curva e i due MSE.
"""),
        code("""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# >>> MANOPOLA: grado del polinomio <<<
GRADO = 2

modello = make_pipeline(PolynomialFeatures(GRADO), LinearRegression()).fit(Xtr, ytr)
mse_tr = mean_squared_error(ytr, modello.predict(Xtr))
mse_te = mean_squared_error(yte, modello.predict(Xte))
print(f"grado {GRADO}  ->  MSE training: {mse_tr:.1f}   MSE test: {mse_te:.1f}")

plt.figure(figsize=(7, 5))
plt.scatter(Xtr, ytr, alpha=0.4, color="#00ADCF", label="training")
plt.plot(griglia, modello.predict(griglia), color="#DC4C4C", lw=2.5, label=f"polinomio grado {GRADO}")
plt.xlabel("horsepower"); plt.ylabel("mpg"); plt.legend(); plt.title(f"Fit polinomiale (grado {GRADO})")
plt.ylim(5, 50)
plt.show()
"""),
        md("""
> **indovina prima di eseguire**: cosa succede al **MSE di training** e al **MSE di test** se metti `GRADO = 15`? quale dei due peggiora?

provalo, poi continua.

## 4. La U del test error

invece di provare i gradi a mano, li proviamo tutti in un ciclo e disegniamo le due curve.
"""),
        code("""
gradi = range(1, 16)
err_tr, err_te = [], []
for g in gradi:
    m = make_pipeline(PolynomialFeatures(g), LinearRegression()).fit(Xtr, ytr)
    err_tr.append(mean_squared_error(ytr, m.predict(Xtr)))
    err_te.append(mean_squared_error(yte, m.predict(Xte)))

plt.figure(figsize=(7.5, 5))
plt.plot(list(gradi), err_tr, "o-", color="#9aa3b2", label="MSE training")
plt.plot(list(gradi), err_te, "o-", color="#DC4C4C", label="MSE test")
plt.xlabel("grado del polinomio (flessibilità)"); plt.ylabel("MSE")
plt.title("Training error che scende, test error a U")
plt.ylim(15, 40); plt.legend(); plt.show()

migliore = list(gradi)[int(np.argmin(err_te))]
print("grado con MSE di test minimo:", migliore)
"""),
        md("""
ecco il trade-off bias-varianza con i tuoi occhi: il training error scende sempre, ma il test error prima cala e poi risale. il modello migliore sta nel minimo della U, non al grado piu' alto.

---

### Cella bonus: la cross-validation

invece di un solo split, la k-fold cross-validation media l'errore su piu' divisioni: una stima piu' stabile del test error.
"""),
        code("""
# BONUS
from sklearn.model_selection import cross_val_score

cv_mse = []
for g in gradi:
    m = make_pipeline(PolynomialFeatures(g), LinearRegression())
    punteggi = cross_val_score(m, X, y, cv=10, scoring="neg_mean_squared_error")
    cv_mse.append(-punteggi.mean())

plt.figure(figsize=(7.5, 4.5))
plt.plot(list(gradi), cv_mse, "o-", color="#002060")
plt.xlabel("grado"); plt.ylabel("MSE in 10-fold CV"); plt.title("Cross-validation: scelta del grado")
plt.ylim(15, 40); plt.show()
print("grado scelto dalla CV:", list(gradi)[int(np.argmin(cv_mse))])
"""),
    ]
    scrivi("NB1_regressione_biasvarianza.ipynb", c)


# ===========================================================================
# NB2 - Classificazione, regolarizzazione e CV (Default)
# ===========================================================================
def nb2():
    c = [
        badge("NB2_classificazione_regolarizzazione.ipynb",
              "NB2 · Classificazione, regolarizzazione e cross-validation",
              "logistica su Default, la soglia, ridge e lasso, la scelta di lambda"),
        md("""
## 1. Dati: chi sara' insolvente?

dataset **Default** di ISL: per 10.000 clienti, se sono andati in insolvenza (`default`), se sono studenti (`student`), il saldo della carta (`balance`) e il reddito (`income`).
"""),
        code(f"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("{URL_DEFAULT}")
df["default"] = (df["default"] == "Yes").astype(int)
df["student"] = (df["student"] == "Yes").astype(int)
print("tasso di insolvenza:", round(df["default"].mean(), 3))
df.head()
"""),
        md("""
## 2. Regressione logistica

prevediamo la probabilita' di insolvenza dal saldo, dal reddito e dallo stato di studente.
"""),
        code("""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report

X = df[["balance", "income", "student"]].values
y = df["default"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

logit = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
prob = logit.predict_proba(Xte)[:, 1]
print(classification_report(yte, (prob > 0.5).astype(int), target_names=["non insolvente", "insolvente"]))
"""),
        md("""
## 3. La soglia e' una scelta, non un dato

di default si classifica "insolvente" se la probabilita' supera 0,5. ma la soglia si puo' spostare: abbassarla trova piu' insolventi (piu' veri positivi) ma con piu' falsi allarmi.

> **manopola**: cambia `SOGLIA` (prova 0.2, 0.3, 0.5) e guarda come cambia la matrice di confusione.
"""),
        code("""
# >>> MANOPOLA: soglia di decisione <<<
SOGLIA = 0.5

pred = (prob > SOGLIA).astype(int)
cm = confusion_matrix(yte, pred)
print("matrice di confusione (righe = vero, colonne = previsto):")
print(pd.DataFrame(cm, index=["vero: no", "vero: sì"], columns=["prev: no", "prev: sì"]))
veri_pos = cm[1, 1]; falsi_neg = cm[1, 0]
print(f"\\ninsolventi individuati: {veri_pos} su {veri_pos + falsi_neg}")
"""),
        md("""
## 4. Regolarizzazione: ridge e lasso

con molti predittori conviene **penalizzare** i coefficienti. in `LogisticRegression` la forza della penalita' e' `C = 1/lambda`: **C piccolo = penalita' forte** (coefficienti piu' piccoli).

> **manopola**: cambia `C` (prova 0.001, 0.01, 1, 100) e osserva come cambiano i coefficienti.
"""),
        code("""
# >>> MANOPOLA: forza della regolarizzazione (C = 1/lambda) <<<
C = 1.0

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

ridge = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr_s, ytr)
nomi = ["balance", "income", "student"]
print(f"C = {C}  (penalità L2 / ridge)")
for n, b in zip(nomi, ridge.coef_[0]):
    print(f"  coef {n:>8}: {b:+.3f}")
"""),
        md("""
il **lasso** (penalita' L1) puo' azzerare del tutto i coefficienti meno utili, facendo selezione delle variabili.
"""),
        code("""
lasso = LogisticRegression(penalty="l1", C=0.05, solver="liblinear", max_iter=2000).fit(Xtr_s, ytr)
print("coefficienti con il lasso (C piccolo):")
for n, b in zip(nomi, lasso.coef_[0]):
    stato = "  <- azzerato" if abs(b) < 1e-6 else ""
    print(f"  {n:>8}: {b:+.3f}{stato}")
"""),
        md("""
## 5. Scegliere lambda con la cross-validation

non scegliamo `C` a occhio: lo scegliamo con la k-fold cross-validation, che stima l'errore su dati nuovi.
"""),
        code("""
from sklearn.linear_model import LogisticRegressionCV

cv = LogisticRegressionCV(Cs=np.logspace(-3, 2, 20), cv=5, max_iter=2000).fit(Xtr_s, ytr)
print("C scelto dalla cross-validation:", round(cv.C_[0], 4))
print("accuratezza sul test:", round(cv.score(Xte_s, yte), 3))
"""),
        md("""
---

### Cella bonus: il "percorso" del lasso

al crescere della penalita', i coefficienti del lasso si restringono e a uno a uno vanno a zero.
"""),
        code("""
# BONUS
Cs = np.logspace(-3, 1, 30)
percorso = []
for c in Cs:
    m = LogisticRegression(penalty="l1", C=c, solver="liblinear", max_iter=2000).fit(Xtr_s, ytr)
    percorso.append(m.coef_[0])
percorso = np.array(percorso)

plt.figure(figsize=(7.5, 5))
for j, n in enumerate(nomi):
    plt.plot(Cs, percorso[:, j], "o-", label=n)
plt.xscale("log"); plt.xlabel("C = 1/lambda (log)"); plt.ylabel("coefficiente")
plt.title("Percorso del lasso: i coefficienti si restringono"); plt.legend(); plt.axhline(0, color="grey", lw=1)
plt.show()
"""),
    ]
    scrivi("NB2_classificazione_regolarizzazione.ipynb", c)


# ===========================================================================
# NB3 - Non-linearita', ensemble e caso retail (Conad)
# ===========================================================================
def nb3():
    c = [
        badge("NB3_ensemble_caso_retail.ipynb",
              "NB3 · Non-linearità, ensemble e il caso retail Conad",
              "albero, random forest, boosting e SVM su un dataset retail sintetico"),
        md("""
## 1. Il dataset retail (sintetico, stile Conad)

10.000 clienti di una carta fedelta'. due compiti sugli **stessi dati**:

- **churn** (classificazione): il cliente e' a rischio abbandono? (0/1);
- **spesa prevista a 12 mesi** (regressione): quanto spenderà? (euro).
"""),
        code(f"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

df = pd.read_csv("{URL_CONAD}")
print("righe:", len(df), " colonne:", df.shape[1])
df.head()
"""),
        code("""
# prepariamo le variabili: togliamo l'id e codifichiamo le categoriche
cat = ["formato_pdv", "canale", "reparto_preferito"]
X = pd.get_dummies(df.drop(columns=["cliente_id", "churn", "spesa_prevista_12m"]), columns=cat)
y_churn = df["churn"]
y_spesa = df["spesa_prevista_12m"]
print("numero di feature dopo la codifica:", X.shape[1])
"""),
        md("""
## 2. Compito A, churn: un albero che overfitta

l'albero di decisione e' intuitivo, ma se lo lasciamo crescere troppo impara il rumore: ottimo sul training, peggiore sul test.

> **manopola**: cambia `PROFONDITA` (prova 2, 4, 8, 20) e confronta AUC di training e di test.
"""),
        code("""
from sklearn.tree import DecisionTreeClassifier

Xtr, Xte, ytr, yte = train_test_split(X, y_churn, test_size=0.30, random_state=0, stratify=y_churn)

# >>> MANOPOLA: profondità massima dell'albero <<<
PROFONDITA = 4

albero = DecisionTreeClassifier(max_depth=PROFONDITA, random_state=0).fit(Xtr, ytr)
auc_tr = roc_auc_score(ytr, albero.predict_proba(Xtr)[:, 1])
auc_te = roc_auc_score(yte, albero.predict_proba(Xte)[:, 1])
print(f"profondità {PROFONDITA}  ->  AUC training: {auc_tr:.3f}   AUC test: {auc_te:.3f}")
print("differenza training - test (più è grande, più overfitta):", round(auc_tr - auc_te, 3))
"""),
        md("""
prova `PROFONDITA = 20`: l'AUC di training si avvicina a 1 ma quella di test cala. e' overfitting.

## 3. La random forest stabilizza

tanti alberi diversi mediati insieme: meno varianza, di solito molto piu' accurata. e ci dice quali variabili contano.
"""),
        code("""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
print("AUC test random forest:", round(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]), 3))

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(7.5, 4.5))
imp[::-1].plot(kind="barh", color="#00ADCF")
plt.title("Random forest: importanza delle variabili (churn)"); plt.tight_layout(); plt.show()
"""),
        md("""
le variabili in cima (recency, visite, distanza, reclami) sono i veri segnali del churn; in fondo trovi le feature di rumore.

## 4. Gradient boosting

alberi piccoli in sequenza, ognuno corregge gli errori del precedente: spesso il piu' accurato.
"""),
        code("""
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
print("AUC test gradient boosting:", round(roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]), 3))
"""),
        md("""
## 5. SVM: il confine in 2D

per vedere il confine di decisione usiamo solo due variabili (recency e visite) e una SVM con kernel non lineare.
"""),
        code("""
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

due = ["recency_giorni", "n_visite_mese"]
X2 = df[due].values
X2tr, X2te, y2tr, y2te = train_test_split(X2, y_churn, test_size=0.30, random_state=0, stratify=y_churn)
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)).fit(X2tr, y2tr)

xx, yy = np.meshgrid(np.linspace(0, 120, 300), np.linspace(0, 20, 300))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(7.5, 5))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
plt.scatter(X2te[:, 0], X2te[:, 1], c=y2te, cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("recency (giorni dall'ultimo acquisto)"); plt.ylabel("visite al mese")
plt.title("SVM: confine di decisione per il churn"); plt.show()
"""),
        md("""
## 6. Compito B, previsione vendite: lineare vs boosting

stesso dataset, target continuo. qui si vede che i modelli flessibili battono il lineare grazie alle non-linearita'.
"""),
        code("""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

Xtr, Xte, str_, ste = train_test_split(X, y_spesa, test_size=0.30, random_state=0)
lin = LinearRegression().fit(Xtr, str_)
gbr = GradientBoostingRegressor(random_state=0).fit(Xtr, str_)
print("R2 test lineare:          ", round(r2_score(ste, lin.predict(Xte)), 3))
print("R2 test gradient boosting:", round(r2_score(ste, gbr.predict(Xte)), 3))
"""),
        md("""
**quale modello per quale obiettivo?** se conta spiegare, parti dal lineare interpretabile; se conta solo prevedere bene, gli ensemble di solito vincono. e' di nuovo il trade-off flessibilita' / interpretabilita'.

---

### Cella bonus: permutation importance

l'importanza basata sull'impurita' favorisce le variabili continue. la *permutation importance* e' piu' onesta: mescola una colonna alla volta e misura quanto peggiora il modello. le feature di rumore vanno vicino a zero.
"""),
        code("""
# BONUS
from sklearn.inspection import permutation_importance

Xtr, Xte, ytr, yte = train_test_split(X, y_churn, test_size=0.30, random_state=0, stratify=y_churn)
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(10)
imp[::-1].plot(kind="barh", figsize=(7.5, 4.5), color="#002060")
plt.title("Permutation importance (churn)"); plt.tight_layout(); plt.show()
"""),
    ]
    scrivi("NB3_ensemble_caso_retail.ipynb", c)


# ===========================================================================
# NB4 - Demo reti neurali
# ===========================================================================
def nb4():
    c = [
        badge("NB4_reti_neurali.ipynb", "NB4 · Demo reti neurali",
              "una piccola rete che richiama il mattino: è ancora apprendimento statistico"),
        md("""
## 1. Una rete neurale e' ancora statistical learning

usiamo gli stessi dati Default e una piccola rete (un MLP, *multi-layer perceptron*). valgono gli stessi concetti del mattino: training/test, overfitting, regolarizzazione.
"""),
        code(f"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_csv("{URL_DEFAULT}")
df["default"] = (df["default"] == "Yes").astype(int)
df["student"] = (df["student"] == "Yes").astype(int)
X = df[["balance", "income", "student"]].values
y = df["default"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
"""),
        md("""
## 2. Una piccola rete

due strati nascosti. la "manopola" qui e' `ALPHA`, la forza della regolarizzazione (come il lambda di prima).

> **manopola**: prova `ALPHA = 0.0001` (poca regolarizzazione) e `ALPHA = 1.0` (tanta). guarda la differenza tra AUC di training e di test.
"""),
        code("""
from sklearn.neural_network import MLPClassifier

# >>> MANOPOLA: regolarizzazione della rete <<<
ALPHA = 0.0001

rete = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32, 16), alpha=ALPHA, max_iter=400, random_state=0),
).fit(Xtr, ytr)

auc_tr = roc_auc_score(ytr, rete.predict_proba(Xtr)[:, 1])
auc_te = roc_auc_score(yte, rete.predict_proba(Xte)[:, 1])
print(f"alpha = {ALPHA}")
print(f"AUC training: {auc_tr:.3f}   AUC test: {auc_te:.3f}")
print("differenza (overfitting se grande):", round(auc_tr - auc_te, 3))
"""),
        md("""
## 3. Confronto con la baseline lineare

la rete batte davvero la semplice logistica? su questi dati spesso no: piu' flessibilita' non e' sempre meglio.
"""),
        code("""
from sklearn.linear_model import LogisticRegression

logit = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
print("AUC test logistica:", round(roc_auc_score(yte, logit.predict_proba(Xte)[:, 1]), 3))
print("AUC test rete:     ", round(auc_te, 3))
"""),
        md("""
stesso identico schema del mattino: train/test, overfitting, regolarizzazione. la rete e' solo una `f` piu' flessibile, non un mondo a parte.

---

### Cella bonus: la stessa rete in Keras (TensorFlow)

Colab ha gia' TensorFlow installato. questa cella mostra la stessa idea con Keras; se esegui in locale senza TensorFlow, stampa solo un avviso.
"""),
        code("""
# BONUS (gira su Colab, dove TensorFlow è preinstallato)
try:
    import tensorflow as tf
    from tensorflow import keras

    scaler = StandardScaler().fit(Xtr)
    modello = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    modello.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    modello.fit(scaler.transform(Xtr), ytr, epochs=10, batch_size=64, verbose=0)
    auc = modello.evaluate(scaler.transform(Xte), yte, verbose=0)[1]
    print("AUC test (Keras):", round(float(auc), 3))
except ImportError:
    print("TensorFlow non disponibile qui: esegui questa cella su Colab.")
"""),
        md("""
## 4. La big picture

la stessa idea, scalata, porta a:

- **CNN** per le immagini;
- **RNN** e **LSTM** per le sequenze;
- **GAN** per generare dati nuovi;
- **transformer** e da lì i **grandi modelli linguistici (LLM)**.

un unico continuum, dai minimi quadrati di stamattina fino agli LLM. fine del corso, e grazie.
"""),
    ]
    scrivi("NB4_reti_neurali.ipynb", c)


if __name__ == "__main__":
    nb0(); nb1(); nb2(); nb3(); nb4()
    print("Notebook generati.")
