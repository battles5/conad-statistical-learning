# -*- coding: utf-8 -*-
"""
Generatore del dataset sintetico per il caso "fontanelli" (erogatori d'acqua).

Scenario didattico (NB5): un'insegna vuole stimare i litri d'acqua che i fontanelli
erogheranno se installati nei suoi negozi, ma ne ha installati solo in 3 (i pilota).
Gli altri 500 sono da stimare.

Il file contiene la verita' sintetica (`litri_erogati_sett`) per TUTTI i negozi, cosi'
nel notebook possiamo "svelare" il vero totale e giudicare i diversi approcci. La
colonna `misurato` vale 1 solo per i 3 negozi pilota: e' l'unico dato che, nella realta',
l'azienda possiede davvero.

Variabili (richieste dal cliente):
  - tipologia: prossimita' o attrazione (tipo di punto vendita);
  - visite_giorno: frequenza di visita (afflusso medio giornaliero);
  - acqua_bottiglia_litri_sett: consumi di acqua in bottiglia (proxy della domanda);
  - temp_media_zona: posizione geografica resa come clima medio della zona.

Uso:
    python genera_fontanelli.py
Produce: fontanelli_negozi.csv (503 righe: 3 pilota + 500 da stimare).

Riproducibile: seed fisso. Nessuna dipendenza oltre numpy e pandas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
N_TARGET = 500       # negozi da stimare
N_PILOTA = 3         # negozi gia' dotati di fontanello (misurati)
N = N_TARGET + N_PILOTA
SEED = 7
rng = np.random.default_rng(SEED)


def genera():
    # ------------------------------------------------------------------
    # 1) tipologia del punto vendita
    #    attrazione = grande, molto frequentato; prossimita' = di quartiere
    # ------------------------------------------------------------------
    tipologia = rng.choice(["prossimita", "attrazione"], size=N, p=[0.72, 0.28])
    attr = (tipologia == "attrazione")

    # ------------------------------------------------------------------
    # 2) frequenza di visita (afflusso giornaliero)
    # ------------------------------------------------------------------
    base = np.where(attr, 1150.0, 380.0)
    visite_giorno = np.clip(rng.normal(base, base * 0.25), 60, None).round().astype(int)

    # ------------------------------------------------------------------
    # 3) posizione geografica -> clima medio della zona (gradi C)
    #    zone piu' calde -> piu' uso del fontanello
    # ------------------------------------------------------------------
    temp_media_zona = np.clip(rng.normal(17.0, 2.3, N), 12, 24).round(1)

    # ------------------------------------------------------------------
    # 4) consumi di acqua in bottiglia (litri/settimana):
    #    correlati alle visite e al caldo, con rumore. e' un PROXY della
    #    domanda d'acqua, non la causa diretta dell'erogato.
    # ------------------------------------------------------------------
    acqua_bottiglia_litri_sett = np.clip(
        visite_giorno * 7 * (0.012 + 0.0018 * (temp_media_zona - 14))
        * np.exp(rng.normal(0, 0.22, N)),
        5, None,
    ).round(1)

    # ------------------------------------------------------------------
    # 5) TARGET vero: litri erogati dal fontanello a settimana.
    #    driver principali: afflusso x clima; piccolo effetto della tipologia
    #    (nei negozi "attrazione" si sosta di piu'); rumore moltiplicativo
    #    = errore irriducibile.
    # ------------------------------------------------------------------
    tasso_per_visita = (
        0.022
        + 0.0045 * (temp_media_zona - 14)   # piu' caldo -> piu' acqua
        + np.where(attr, 0.006, 0.0)         # attrazione -> sosta -> piu' acqua
    )
    litri_erogati_sett = np.clip(
        visite_giorno * 7 * tasso_per_visita * np.exp(rng.normal(0, 0.18, N)),
        0, None,
    ).round(1)

    df = pd.DataFrame({
        "negozio_id": [f"NEG{i + 1:03d}" for i in range(N)],
        "tipologia": tipologia,
        "visite_giorno": visite_giorno,
        "acqua_bottiglia_litri_sett": acqua_bottiglia_litri_sett,
        "temp_media_zona": temp_media_zona,
        "litri_erogati_sett": litri_erogati_sett,
    })

    # ------------------------------------------------------------------
    # 6) i 3 negozi pilota (gli unici "misurati"): scelti diversi tra loro
    #    (un'attrazione e due prossimita', clima vario), come capita davvero
    # ------------------------------------------------------------------
    idx_attr = df.index[df["tipologia"] == "attrazione"]
    idx_pross = df.index[df["tipologia"] == "prossimita"]
    pilota = [
        int(rng.choice(idx_attr)),
        int(rng.choice(idx_pross)),
        int(rng.choice(idx_pross)),
    ]
    df["misurato"] = 0
    df.loc[pilota, "misurato"] = 1

    # mettiamo i 3 pilota in cima, per chiarezza
    df = pd.concat([df[df["misurato"] == 1], df[df["misurato"] == 0]]).reset_index(drop=True)

    out = HERE / "fontanelli_negozi.csv"
    df.to_csv(out, index=False)
    print("scritto", out.name, "-", len(df), "negozi,", int(df["misurato"].sum()), "misurati")
    print("totale vero litri/sett (tutti):", round(df["litri_erogati_sett"].sum(), 1))


if __name__ == "__main__":
    genera()
