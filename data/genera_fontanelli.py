# -*- coding: utf-8 -*-
"""
Generatore del dataset sintetico per il caso "fontanelli" (erogatori d'acqua) - v2.

Scenario didattico (NB5): un'insegna del Nord Ovest vuole stimare i litri d'acqua che i
fontanelli erogheranno se installati nei suoi negozi, ma ne ha installati solo in 3 (i
pilota). Gli altri 500 sono da stimare.

Copertura geografica: 6 regioni del Nord Ovest, dalla Valle d'Aosta alla Sardegna.
I 3 negozi pilota sono uno in Liguria, uno a Prato (Toscana) e uno nel Lazio.

Il file contiene la verita' sintetica (`litri_erogati_sett`) per TUTTI i negozi, cosi'
nel notebook possiamo "svelare" il vero totale e giudicare i diversi approcci. La colonna
`misurato` vale 1 solo per i 3 negozi pilota.

Variabili:
  - regione: una delle 6 regioni del Nord Ovest (geografia, da trattare);
  - tipologia: prossimita' o attrazione;
  - visite_giorno: afflusso medio giornaliero (le attrazioni ne hanno molte di piu');
  - superficie_mq: dimensione del punto vendita;
  - presenza_area_ristoro: 1 se c'e' bar/ristoro (piu' sosta -> piu' consumo d'acqua);
  - temp_media_zona: lettura locale del clima (proxy RUMOROSO);
  - giorni_caldi_anno: giorni caldi all'anno della zona (clima piu' affidabile, da regione);
  - acqua_bottiglia_litri_sett: consumi di acqua in bottiglia (proxy della domanda).

Uso:
    python genera_fontanelli.py
Produce: fontanelli_negozi.csv (503 righe: 3 pilota + 500 da stimare).
Riproducibile: seed fisso. Solo numpy e pandas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
N_TARGET = 500
N_PILOTA = 3
N = N_TARGET + N_PILOTA
SEED = 7
rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 6 regioni del Nord Ovest e loro peso (densita' di negozi indicativa)
REGIONI = ["Valle d'Aosta", "Piemonte", "Liguria", "Toscana", "Lazio", "Sardegna"]
PESO = [0.05, 0.13, 0.18, 0.34, 0.16, 0.14]
# clima medio indicativo per regione (gradi C): guida giorni caldi ed erogato
TEMP_BASE = {
    "Valle d'Aosta": 12.5, "Piemonte": 14.5, "Liguria": 16.5,
    "Toscana": 16.0, "Lazio": 17.5, "Sardegna": 18.5,
}


def genera():
    # ------------------------------------------------------------------
    # 1) geografia e tipologia (con i 3 pilota forzati)
    # ------------------------------------------------------------------
    regione = rng.choice(REGIONI, size=N, p=PESO).astype(object)
    tipologia = rng.choice(["prossimita", "attrazione"], size=N, p=[0.72, 0.28]).astype(object)

    # i 3 pilota: uno a Prato (Toscana), uno in Liguria, uno nel Lazio
    regione[0], tipologia[0] = "Toscana", "attrazione"     # Prato
    regione[1], tipologia[1] = "Liguria", "prossimita"     # es. Savona
    regione[2], tipologia[2] = "Lazio", "prossimita"       # es. Latina
    attr = (tipologia == "attrazione")

    # ------------------------------------------------------------------
    # 2) clima: dalla regione al numero di giorni caldi (driver pulito);
    #    la "temperatura media" misurata in zona e' invece RUMOROSA (proxy debole)
    # ------------------------------------------------------------------
    t_base = np.array([TEMP_BASE[r] for r in regione])
    giorni_caldi_anno = np.clip((t_base - 11.0) * 9.0 + rng.normal(0, 6, N), 5, 120).round().astype(int)
    temp_media_zona = (t_base + rng.normal(0, 2.0, N)).round(1)

    # ------------------------------------------------------------------
    # 3) afflusso (le attrazioni ne hanno molte di piu', media alzata) e dimensione
    # ------------------------------------------------------------------
    base_v = np.where(attr, 1700.0, 380.0)
    visite_giorno = np.clip(rng.normal(base_v, base_v * 0.25), 60, None).round().astype(int)

    superficie_mq = np.clip(
        np.where(attr, rng.normal(2400, 700, N), rng.normal(520, 180, N)), 120, None
    ).round().astype(int)

    # area ristoro: piu' probabile nei negozi grandi / attrazione
    p_ristoro = sigmoid(-1.2 + 1.7 * attr + 0.0006 * (superficie_mq - 500))
    presenza_area_ristoro = (rng.random(N) < p_ristoro).astype(int)

    # ------------------------------------------------------------------
    # 4) consumi di acqua in bottiglia (proxy della domanda, correlato a visite e caldo)
    # ------------------------------------------------------------------
    acqua_bottiglia_litri_sett = np.clip(
        visite_giorno * 7 * (0.012 + 0.0016 * (t_base - 13))
        * np.exp(rng.normal(0, 0.22, N)),
        5, None,
    ).round(1)

    # ------------------------------------------------------------------
    # 5) TARGET vero: litri erogati a settimana.
    #    driver = afflusso x tasso; il tasso cresce col caldo (giorni_caldi),
    #    con l'area ristoro (piu' sosta) e un po' con la tipologia attrazione.
    #    rumore moltiplicativo = errore irriducibile.
    # ------------------------------------------------------------------
    tasso = (
        0.024
        + 0.00040 * (giorni_caldi_anno - 30)
        + 0.006 * attr
        + 0.010 * presenza_area_ristoro
    )
    tasso = np.clip(tasso, 0.004, None)
    litri_erogati_sett = np.clip(
        visite_giorno * 7 * tasso * np.exp(rng.normal(0, 0.16, N)),
        0, None,
    ).round(1)

    df = pd.DataFrame({
        "negozio_id": [f"NEG{i + 1:03d}" for i in range(N)],
        "regione": regione,
        "tipologia": tipologia,
        "visite_giorno": visite_giorno,
        "superficie_mq": superficie_mq,
        "presenza_area_ristoro": presenza_area_ristoro,
        "temp_media_zona": temp_media_zona,
        "giorni_caldi_anno": giorni_caldi_anno,
        "acqua_bottiglia_litri_sett": acqua_bottiglia_litri_sett,
        "litri_erogati_sett": litri_erogati_sett,
    })

    # i 3 pilota sono gli indici 0, 1, 2 (gia' in cima)
    df["misurato"] = 0
    df.loc[[0, 1, 2], "misurato"] = 1

    out = HERE / "fontanelli_negozi.csv"
    df.to_csv(out, index=False)
    print("scritto", out.name, "-", len(df), "negozi,", int(df["misurato"].sum()), "misurati")
    print("regioni:", df["regione"].value_counts().to_dict())
    print("visite medie - attrazione:", round(df.loc[df.tipologia == 'attrazione', 'visite_giorno'].mean()),
          "| prossimita:", round(df.loc[df.tipologia == 'prossimita', 'visite_giorno'].mean()))
    print("totale vero litri/sett (500 da stimare):",
          round(df.loc[df.misurato == 0, "litri_erogati_sett"].sum(), 1))


if __name__ == "__main__":
    genera()
