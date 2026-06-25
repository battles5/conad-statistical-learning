# Dataset retail sintetico stile Conad Nord Ovest

Dataset **sintetico** (generato, non reale) a livello cliente / carta fedelta',
costruito apposta per il corso. Un solo file regge due compiti sugli stessi dati:

- **churn / fidelizzazione** -> classificazione (target `churn`);
- **previsione vendite** -> regressione (target `spesa_prevista_12m`).

Nessun dato personale o reale: ogni riga e' un cliente fittizio.

## File

- `conad_retail.csv` (10.000 righe, 23 colonne);
- `genera_dataset.py` script riproducibile che lo genera (seed fisso, solo numpy e pandas).

## Come caricarlo (Colab)

```python
import pandas as pd
URL = "https://raw.githubusercontent.com/battles5/conad-statistical-learning/main/data/conad_retail.csv"
df = pd.read_csv(URL)
df.head()
```

## Dizionario dei dati

| colonna | tipo | unita' / valori | descrizione |
|---|---|---|---|
| `cliente_id` | testo | `CN000001`... | identificativo cliente (non e' una feature) |
| `eta` | intero | 18..90 anni | eta' del titolare carta |
| `anni_fedelta` | intero | 0..30 anni | anzianita' della carta fedelta' |
| `distanza_pdv_km` | decimale | km | distanza dal punto vendita abituale |
| `formato_pdv` | categoria | Conad City / Conad / Conad Superstore | formato del punto vendita prevalente |
| `canale` | categoria | negozio / misto / online | canale d'acquisto preferito |
| `uso_app` | binario | 0 / 1 | usa l'app dell'insegna |
| `n_visite_mese` | intero | visite/mese | frequenza di visita |
| `scontrino_medio` | decimale | euro | valore medio dello scontrino |
| `spesa_media_mensile` | decimale | euro/mese | spesa media mensile (circa visite x scontrino) |
| `n_categorie` | intero | 1..20 | numero di categorie merceologiche distinte acquistate |
| `recency_giorni` | intero | giorni | giorni dall'ultimo acquisto |
| `pct_promozioni` | decimale | 0..1 | quota di spesa fatta in promozione |
| `reclami_12m` | intero | conteggio | reclami negli ultimi 12 mesi |
| `quota_freschi` | decimale | 0..1 | quota di spesa sui freschi (ortofrutta, macelleria, pescheria, gastronomia) |
| `quota_mdd` | decimale | 0..1 | quota di spesa sulla marca del distributore (private label) |
| `sensibilita_volantino` | decimale | 0..1 | reattivita' alle promozioni del volantino |
| `reparto_preferito` | categoria | ortofrutta, macelleria, pescheria, gastronomia, dispensa, surgelati, cura_casa, cura_persona, bio_benessere | reparto con spesa prevalente |
| `idx_zona_meteo` | decimale | indice | indice meteo medio della zona (**rumore**: irrilevante per i target) |
| `giorno_iscrizione_carta` | intero | 1..28 | giorno del mese di sottoscrizione carta (**rumore**) |
| `punteggio_sondaggio` | intero | 0..10 | punteggio a un sondaggio generico (**rumore**) |
| `churn` | binario | 0 / 1 | **target classificazione**: 1 = cliente perso/inattivo |
| `spesa_prevista_12m` | decimale | euro | **target regressione**: spesa attesa nei prossimi 12 mesi |

Le tre colonne marcate **rumore** sono plausibili ma volutamente prive di
segnale: servono a mostrare che lasso le porta a zero e che la random forest
le mette in fondo all'importanza (meglio ancora con la permutation importance).

## Proprieta' didattiche (verificate)

Con un train/test split 70/30:

| compito | modello | metrica |
|---|---|---|
| churn (classificazione) | regressione logistica | AUC ~0,89 |
| churn (classificazione) | random forest | AUC ~0,89 |
| vendite (regressione) | regressione lineare | R2 ~0,87, MAE ~630 euro |
| vendite (regressione) | gradient boosting | R2 ~0,95, MAE ~360 euro |

Segnale forte ma non perfetto (si vedono overfitting e trade-off bias-varianza);
collinearita' voluta tra `spesa_media_mensile`, `n_visite_mese` e `scontrino_medio`
(utile per ridge); driver del churn guidato soprattutto da `recency_giorni`,
`n_visite_mese`, `distanza_pdv_km` e `reclami_12m`, con una interazione non lineare
recency x frequenza utile per la demo SVM in 2D; sulle vendite il boosting batte il
lineare grazie a effetti non lineari (rischio churn, soglia di recency, rendimenti
decrescenti sull'ampiezza panieri).

## Rigenerare il dataset

```bash
python genera_dataset.py
```

Il seed e' fisso (`SEED = 42`): la rigenerazione produce lo stesso file.
