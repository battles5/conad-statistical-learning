"""
Generatore del dataset retail sintetico stile Conad Nord Ovest.

Un solo dataset cliente-livello (carta fedelta'), pensato per due compiti sugli
stessi dati nel notebook NB3 del pomeriggio:
  - churn / fidelizzazione  -> classificazione (target: churn 0/1);
  - previsione vendite       -> regressione    (target: spesa_prevista_12m, euro).

Proprieta' progettate apposta per la didattica:
  - segnale forte ma non perfetto, cosi' bias-varianza e overfitting si vedono;
  - collinearita' voluta (spesa ~ visite x scontrino) per dare senso a ridge;
  - feature plausibili ma irrilevanti, cosi' lasso le azzera e la random forest
    le mette in fondo all'importanza;
  - interazione non lineare recency x frequenza, utile per la demo SVM in 2D e
    per far battere alberi/boosting al modello lineare;
  - i due target sono legati: chi e' a rischio churn riduce la spesa futura.

Uso:
    python genera_dataset.py
Produce: conad_retail.csv (nella stessa cartella).

Riproducibile: seed fisso. Nessuna dipendenza oltre numpy e pandas.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N = 10_000
SEED = 42
rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


# ---------------------------------------------------------------------------
# 1) Fattori latenti del cliente (non osservati, servono solo a generare)
#    valore     -> capacita' di spesa;
#    engagement -> attaccamento all'insegna (visite, app, recency bassa);
#    prezzo     -> sensibilita' al prezzo (promo, volantino, marca propria).
# ---------------------------------------------------------------------------
z_valore = rng.standard_normal(N)
z_engage = rng.standard_normal(N)
z_prezzo = rng.standard_normal(N)
# i clienti piu' fedeli tendono a spendere un po' di piu': leggera correlazione
z_valore = 0.85 * z_valore + 0.15 * z_engage

# ---------------------------------------------------------------------------
# 2) Feature demografiche e di relazione
# ---------------------------------------------------------------------------
eta = np.clip(rng.normal(48, 15, N), 18, 90).round().astype(int)

# piu' anziani e piu' engaged -> piu' anni di carta fedelta'
anni_fedelta = np.clip(
    0.18 * (eta - 25) + 2.5 * z_engage + rng.normal(0, 2, N), 0, 30
).round().astype(int)

# distanza dal punto vendita: molti vicini, coda lunga di lontani
distanza_pdv_km = np.clip(rng.exponential(3.5, N), 0.2, 40).round(1)

# formato del punto vendita prevalente (insegne tipiche Conad)
formato_pdv = rng.choice(
    ["Conad City", "Conad", "Conad Superstore"],
    size=N, p=[0.32, 0.45, 0.23],
)
# i formati grandi alzano lo scontrino medio
eff_formato_scontrino = np.select(
    [formato_pdv == "Conad City", formato_pdv == "Conad", formato_pdv == "Conad Superstore"],
    [-6.0, 0.0, 9.0],
)

# uso dell'app: piu' frequente nei giovani ed engaged
p_app = sigmoid(0.9 * z_engage - 0.03 * (eta - 45))
uso_app = (rng.random(N) < p_app).astype(int)

# canale preferito
p_online_base = sigmoid(1.1 * z_engage - 0.04 * (eta - 40) + 0.8 * uso_app - 1.6)
canale = np.where(
    rng.random(N) < p_online_base * 0.45, "online",
    np.where(rng.random(N) < p_online_base, "misto", "negozio"),
)

# ---------------------------------------------------------------------------
# 3) Comportamento d'acquisto
# ---------------------------------------------------------------------------
# frequenza mensile: cresce con engagement, cala con la distanza
lam_visite = np.exp(1.45 + 0.45 * z_engage - 0.02 * distanza_pdv_km)
n_visite_mese = np.clip(rng.poisson(lam_visite), 1, 45)

# scontrino medio in euro: guidato dal valore e dal formato
scontrino_medio = np.clip(
    26 + 13 * z_valore + eff_formato_scontrino + rng.normal(0, 5, N), 5, 160
).round(2)

# spesa media mensile ~ visite x scontrino, con rumore moltiplicativo.
# collinearita' VOLUTA con visite e scontrino: utile per mostrare ridge.
spesa_media_mensile = np.clip(
    n_visite_mese * scontrino_medio * np.exp(rng.normal(0, 0.10, N)), 5, None
).round(2)

# ampiezza del paniere (numero di categorie merceologiche distinte)
n_categorie = np.clip(
    rng.poisson(np.exp(1.2 + 0.35 * z_valore + 0.25 * zscore(n_visite_mese))), 1, 20
).astype(int)

# recency: giorni dall'ultimo acquisto. DRIVER CHIAVE del churn.
recency_giorni = np.clip(
    np.exp(2.6 - 0.75 * z_engage + rng.normal(0, 0.45, N)), 1, 365
).round().astype(int)

# quota di spesa in promozione
pct_promozioni = np.clip(sigmoid(0.9 * z_prezzo - 0.6) + rng.normal(0, 0.05, N), 0, 1).round(3)

# reclami negli ultimi 12 mesi (pochi, code rare)
reclami_12m = np.clip(rng.poisson(np.exp(-1.6 + 0.35 * z_prezzo)), 0, 12).astype(int)

# ---------------------------------------------------------------------------
# 4) Feature food retail / GDO
# ---------------------------------------------------------------------------
# quota di spesa sui freschi (ortofrutta, macelleria, pescheria, gastronomia):
# piu' alta nei formati grandi e nei clienti di valore
mu_freschi = sigmoid(
    0.4 + 0.35 * z_valore
    + np.where(formato_pdv == "Conad Superstore", 0.3, 0.0)
    - np.where(formato_pdv == "Conad City", 0.25, 0.0)
)
quota_freschi = np.clip(mu_freschi + rng.normal(0, 0.07, N), 0.02, 0.95).round(3)

# quota di spesa sulla marca del distributore (MDD / private label):
# piu' alta nei clienti sensibili al prezzo
quota_mdd = np.clip(sigmoid(-0.4 + 0.85 * z_prezzo) + rng.normal(0, 0.06, N), 0.0, 0.9).round(3)

# sensibilita' al volantino (0..1), legata alla sensibilita' al prezzo
sensibilita_volantino = np.clip(
    sigmoid(0.95 * z_prezzo + 0.4 * zscore(pct_promozioni)) + rng.normal(0, 0.05, N), 0, 1
).round(3)

# reparto preferito: la probabilita' dipende da quota freschi e sensibilita' prezzo
reparti = np.array([
    "ortofrutta", "macelleria", "pescheria", "gastronomia",
    "dispensa", "surgelati", "cura_casa", "cura_persona", "bio_benessere",
])
score_reparti = np.stack([
    1.2 * zscore(quota_freschi),                      # ortofrutta
    0.8 * zscore(quota_freschi) + 0.3 * z_valore,     # macelleria
    0.6 * zscore(quota_freschi) + 0.4 * z_valore,     # pescheria
    0.5 * zscore(quota_freschi) + 0.5 * z_valore,     # gastronomia
    0.9 * zscore(quota_mdd) - 0.3 * z_valore,         # dispensa
    0.2 * z_prezzo + rng.normal(0, 0.5, N),           # surgelati
    0.5 * zscore(quota_mdd) + rng.normal(0, 0.5, N),  # cura_casa
    0.3 * rng.normal(0, 1, N),                        # cura_persona
    0.7 * z_valore - 0.4 * z_prezzo,                  # bio_benessere
], axis=1)
# softmax + estrazione categorica
score_reparti = score_reparti + rng.gumbel(0, 1, size=score_reparti.shape)
reparto_preferito = reparti[score_reparti.argmax(axis=1)]

# ---------------------------------------------------------------------------
# 5) Feature plausibili ma IRRILEVANTI (rumore controllato).
#    Servono per mostrare che lasso le azzera e la random forest le ignora.
# ---------------------------------------------------------------------------
idx_zona_meteo = rng.normal(15, 5, N).round(1)            # finto indice meteo zona
giorno_iscrizione_carta = rng.integers(1, 29, N)          # giorno del mese, irrilevante
punteggio_sondaggio = rng.integers(0, 11, N)              # punteggio sondaggio generico

# ---------------------------------------------------------------------------
# 6) TARGET 1 - churn (classificazione)
#    Logit costruito sulle feature OSSERVATE (cosi' e' imparabile dai modelli),
#    con una interazione non lineare recency x frequenza (per SVM/alberi).
# ---------------------------------------------------------------------------
z_rec = zscore(recency_giorni)
z_vis = zscore(n_visite_mese)
logit_churn = (
    -1.15
    + 1.30 * z_rec
    - 0.95 * z_vis
    + 0.55 * zscore(distanza_pdv_km)
    + 0.60 * zscore(reclami_12m)
    - 0.45 * uso_app
    - 0.30 * zscore(anni_fedelta)
    - 0.35 * zscore(quota_mdd)
    + 0.50 * z_rec * np.maximum(-z_vis, 0)   # interazione: recency alta + poche visite
)
p_churn = sigmoid(logit_churn)
churn = (rng.random(N) < p_churn).astype(int)

# ---------------------------------------------------------------------------
# 7) TARGET 2 - spesa_prevista_12m (regressione, euro)
#    Vero forecast multi-driver costruito in scala logaritmica: la spesa futura
#    e' ancorata alla storia (frequenza e scontrino) ma si sposta con ampiezza
#    panieri, fedelta', uso app, quota freschi, ed e' depressa da recency alta e
#    rischio churn. Una interazione frequenza x scontrino e il termine non lineare
#    sul churn fanno si' che alberi/boosting battano il modello lineare in modo
#    visibile, e l'importanza delle variabili si distribuisce su piu' feature.
# ---------------------------------------------------------------------------
# i termini non lineari sono centrati a media zero, cosi' spostano la spesa
# relativa dei clienti senza alzare il livello medio (mediana attesa ~1600 euro)
sqrt_cat = np.sqrt(n_categorie)
oltre_45 = (recency_giorni > 45).astype(float)
# NB: visite, scontrino e n_categorie sono correlate (stessi fattori latenti):
# coefficienti contenuti, altrimenti la varianza della somma esplode e la coda
# diventa irrealistica.
mu_spesa = (
    np.log(1600)
    + 0.24 * zscore(n_visite_mese)
    + 0.22 * zscore(scontrino_medio)
    + 0.15 * zscore(n_categorie)
    + 0.12 * zscore(anni_fedelta)
    + 0.08 * (uso_app - uso_app.mean())
    + 0.45 * (quota_freschi - 0.5)                  # i clienti orientati al fresco spendono di piu'
    - 0.16 * z_rec                                  # recency alta -> spesa futura piu' bassa
    - 0.70 * (p_churn - p_churn.mean())             # i rischio-churn spendono meno (non lineare)
    - 0.30 * (oltre_45 - oltre_45.mean())           # soglia: clienti che si stanno perdendo, calo netto
    + 0.14 * (sqrt_cat - sqrt_cat.mean())           # rendimenti decrescenti sull'ampiezza panieri
)
spesa_prevista_12m = np.clip(
    np.exp(mu_spesa + rng.normal(0, 0.16, N)), 50, 18000
).round(2)

# ---------------------------------------------------------------------------
# 8) Assemblaggio del DataFrame
# ---------------------------------------------------------------------------
cliente_id = np.array([f"CN{idx:06d}" for idx in range(1, N + 1)])

df = pd.DataFrame({
    "cliente_id": cliente_id,
    "eta": eta,
    "anni_fedelta": anni_fedelta,
    "distanza_pdv_km": distanza_pdv_km,
    "formato_pdv": formato_pdv,
    "canale": canale,
    "uso_app": uso_app,
    "n_visite_mese": n_visite_mese,
    "scontrino_medio": scontrino_medio,
    "spesa_media_mensile": spesa_media_mensile,
    "n_categorie": n_categorie,
    "recency_giorni": recency_giorni,
    "pct_promozioni": pct_promozioni,
    "reclami_12m": reclami_12m,
    "quota_freschi": quota_freschi,
    "quota_mdd": quota_mdd,
    "sensibilita_volantino": sensibilita_volantino,
    "reparto_preferito": reparto_preferito,
    "idx_zona_meteo": idx_zona_meteo,
    "giorno_iscrizione_carta": giorno_iscrizione_carta,
    "punteggio_sondaggio": punteggio_sondaggio,
    "churn": churn,
    "spesa_prevista_12m": spesa_prevista_12m,
})

out = Path(__file__).parent / "conad_retail.csv"
df.to_csv(out, index=False)
print(f"Scritto {out}  ({len(df)} righe, {df.shape[1]} colonne)")
print(f"Tasso di churn: {churn.mean():.1%}")
print(f"Spesa prevista 12m - mediana: {np.median(spesa_prevista_12m):,.0f} euro")
