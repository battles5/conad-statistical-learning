# Corso CONAD Nord Ovest — Statistical Learning

**Docente:** Orso Peruzzi (Data Scientist, IFAB) · **Data:** venerdì, in presenza a Pistoia · **Durata:** 8h (4h frontale + 4h hands-on)

**Tesi del corso:** lo *statistical learning* non è "AI tradizionale" di serie B. È machine learning a tutti gli effetti, ed è la base concettuale da cui discendono anche i modelli più avanzati (CNN, RNN, LSTM, GAN, transformer, LLM).

---

## Decisioni di progetto (bloccate)

| Tema | Scelta |
|---|---|
| Lingua | Tutto in italiano |
| Riferimento | *Introduction to Statistical Learning* (ISL), edizione Python; mappatura terminologica IT dall'edizione Piccin (R) + dispense Torelli |
| Architettura | Sito **Quarto / RevealJS** su **GitHub Pages** (spina dorsale narrativa) + **notebook Colab** collegati con badge "Apri in Colab" (superficie eseguibile) |
| Repo | `battles5/conad-statistical-learning` — **pubblico** |
| Modalità pomeriggio | Code-along guidato |
| Livello partecipanti | Misto → progetto per il livello più basso, con bonus per i più esperti |
| Dataset | Canonici ISL (Auto, Default, Carseats…) + dataset retail **sintetico** stile Conad (churn + previsione vendite) |
| Esecuzione pomeriggio | **Colab** come primario; notebook resi auto-contenuti (dati via URL) con fallback eseguibile senza login se serve |
| Materiale partecipanti | solo **link al sito** pubblico |
| Upgrade "v2" (se avanza tempo) | codice eseguibile *dentro* il sito via Pyodide/`quarto-live`, senza Colab |

---

## Identità visiva IFAB (dal template ufficiale)

Approccio: **ricreo lo stile IFAB come tema RevealJS** nel sito (match ~90%, non il .pptx identico). Font web: **fallback simile ad Aptos** (Inter).

| Elemento | Valore |
|---|---|
| Navy primario | `#002060` |
| Ciano accento / link | `#00ADCF` |
| Azzurro chiaro | `#8FD2E9` |
| Ardesia | `#414F69` |
| Grigio | `#545454` |
| Rosso (accento minore) | `#DC4C4C` |
| Font | Aptos / Aptos Display → web: Inter |
| Logo | `assets-ifab/ifab-logo-color.svg` (+ versione bianca) |
| Slide titolo | fondo navy con texture bokeh (`assets-ifab/title-bg-bokeh.jpeg`), logo bianco |
| Formule | vero LaTeX via MathJax |

---

## Scaletta — Mattina (4h frontale)

> Obiettivo: costruire l'intuizione, non la matematica pesante. Ogni blocco chiude con un'idea memorabile.

### Blocco 0 · Apertura / la provocazione (~20 min)
- "AI tradizionale": da dove viene il termine e perché è fuorviante.
- Mappa del campo: AI ⊃ Machine Learning ≡ Apprendimento Statistico; dove si colloca il Deep Learning.
- La promessa della giornata e il filo rosso.

### Blocco 1 · Dai modelli statistici allo statistical learning (~50 min) *(ISL 1-2)*
- Esempi aziendali/assicurativi (frame di Torelli): cosa chiedevamo alla statistica.
- `Y = f(X) + ε` — cos'è *f*, perché vogliamo stimarla.
- **Predizione vs interpretazione** (i due obiettivi).
- **Modeling vs learning**: la distinzione concettuale chiave.
- Supervisionato vs non supervisionato; regressione vs classificazione; parametrico vs non parametrico.

### Blocco 2 · Il cuore: accuratezza e trade-off bias-varianza (~55 min) *(ISL 2, 5)*
- Misurare la qualità: MSE, training error vs test error.
- La **U** del test error, overfitting / underfitting.
- **La slide-madre: trade-off distorsione (bias) – varianza.**
- Flessibilità vs interpretabilità.
- Resampling / cross-validation come bussola (anticipo del pomeriggio).

### Blocco 3 · Le famiglie di modelli — panoramica ragionata (~55 min) *(ISL 3,4,6,7,8,9)*
- Regressione lineare / logistica: la baseline interpretabile.
- Regolarizzazione: **ridge** e **lasso** (perché, e la sparsità).
- Oltre la linearità: polinomi, spline, **GAM**.
- Metodi ad albero: albero singolo → bagging → **random forest** → **boosting**.
- **SVM**: margine massimo e kernel.
- Tutto posizionato sul piano flessibilità ↔ interpretabilità.

### Blocco 4 · Reti neurali e la big picture (~40 min)
- Dal percettrone alla rete profonda: *è ancora statistical learning*.
- Mappa genealogica: CNN (immagini), RNN/LSTM (sequenze), GAN (generazione), transformer → LLM.
- Chiusura della tesi: un continuum, niente "serie B".
- Ponte al pomeriggio.

---

## Scaletta — Pomeriggio (4h code-along, Colab)

> Ogni notebook: celle markdown didattiche + codice commentato + almeno una "manopola" da girare con effetto visibile. Progettati per il livello più basso, con celle "bonus" per i più esperti.

- **NB0 · Setup & warm-up** (~20 min) — Colab, import, un dataset, `train_test_split`, prima previsione.
- **NB1 · Regressione & bias-varianza *visibile*** (~60 min) — fit lineare → polinomiale, manopola sul grado, curve train/test, la U. *(Auto)*
- **NB2 · Classificazione, regolarizzazione & CV** (~55 min) — logistica su *Default* → ridge/lasso → k-fold CV, scelta di λ.
- **NB3 · Non-linearità, ensemble & caso retail** (~60 min) — albero → random forest → boosting (+ cenno SVM) sul dataset retail Conad sintetico.
- **NB4 · Demo reti neurali** (~30 min) — piccola rete Keras che richiama il mattino; confronto coi modelli precedenti.
- **Chiusura** (~15 min) — recap, come rieseguire a casa, link.

---

## Note editoriali (preferenze Orso)

- niente em-dash o en-dash; iniziale minuscola nelle voci di elenco; punto e virgola tra le voci;
- ogni slide: credito a piè di pagina con nome "Orso Peruzzi" + capitolo/sezione corrente;
- in fondo al sito/deck: sezione referenze con bibliografia arricchita (ISL Python, ISL ed. Piccin, Torelli, ESL, paper chiave).

## Dettaglio notebook pomeriggio (Colab, livello misto)

> codice pre-scritto e commentato; le "manopole" sono singole variabili evidenziate da cambiare; celle "bonus" per i piu' esperti.

- **NB0 — setup & warm-up:** cos'e' Colab e come si eseguono le celle; import (numpy, pandas, matplotlib, scikit-learn); caricare un dataset; sguardo ai dati (head, describe); `train_test_split`; manopola su `test_size`; baseline banale (media).
- **NB1 — regressione & bias-varianza visibile (Auto):** scatter mpg vs horsepower; fit lineare e MSE train/test; fit polinomiale con manopola sul grado (1..15); ciclo sui gradi e grafico della U train/test; "indovina prima di eseguire"; bonus k-fold.
- **NB2 — classificazione, regolarizzazione & CV (Default):** logistica; matrice di confusione e soglia (manopola); standardizzazione; ridge e lasso con manopola su lambda e coefficienti che si restringono; cross-validation per scegliere lambda; bonus path del lasso.
- **NB3 — non-linearita', ensemble & caso retail (dataset Conad sintetico):** due compiti sugli stessi dati: churn/fidelizzazione (classificazione) e previsione vendite (regressione); albero con manopola sulla profondita' (overfitting visibile); random forest e importanza delle variabili; gradient boosting; cenno SVM su 2D per vedere il confine; quale modello per quale obiettivo.
- **NB4 — demo reti neurali (no code-along):** piccola rete (MLP/Keras); mostrare che e' ancora statistical learning (stesso train/test, overfitting, CV); confronto coi modelli; big picture CNN/RNN/LSTM/transformer/LLM.
- **Chiusura:** recap, come rieseguire a casa, Q&A.

## Stato avanzamento
- [x] Strategia e architettura
- [x] Decisioni di progetto bloccate
- [x] Dettaglio slide mattina
- [x] Dettaglio notebook pomeriggio
- [x] Stile IFAB validato (mockup)
- [x] **Pre-progettazione completata**
- [ ] Fase di build: dataset retail sintetico
- [ ] Fase di build: scaffold repo + sito Quarto
- [ ] Fase di build: slide mattina
- [ ] Fase di build: notebook Colab
- [ ] Fase di build: test e pubblicazione
