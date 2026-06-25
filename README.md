# Apprendimento statistico · CONAD Nord Ovest

Sito del corso (8 ore: 4 frontali + 4 hands-on), spina dorsale narrativa in **Quarto / RevealJS**, superficie eseguibile nei **notebook Colab** collegati.

Tesi del corso: l'apprendimento statistico è machine learning a tutti gli effetti, ed è la base
concettuale da cui discendono anche i modelli più avanzati (CNN, RNN, LSTM, GAN, transformer, LLM).

## ▶ Apri le slide

### [👉 Apri le slide del corso](https://battles5.github.io/conad-statistical-learning/)

`https://battles5.github.io/conad-statistical-learning/`

## I notebook del pomeriggio (Google Colab)

Si aprono direttamente in Colab, senza installare nulla:

- [NB0 · Setup e warm-up](https://colab.research.google.com/github/battles5/conad-statistical-learning/blob/main/notebooks/NB0_setup.ipynb)
- [NB1 · Regressione e bias-varianza](https://colab.research.google.com/github/battles5/conad-statistical-learning/blob/main/notebooks/NB1_regressione_biasvarianza.ipynb)
- [NB2 · Classificazione, regolarizzazione e CV](https://colab.research.google.com/github/battles5/conad-statistical-learning/blob/main/notebooks/NB2_classificazione_regolarizzazione.ipynb)
- [NB3 · Ensemble e caso retail Conad](https://colab.research.google.com/github/battles5/conad-statistical-learning/blob/main/notebooks/NB3_ensemble_caso_retail.ipynb)
- [NB4 · Demo reti neurali](https://colab.research.google.com/github/battles5/conad-statistical-learning/blob/main/notebooks/NB4_reti_neurali.ipynb)

## Cosa contiene

- **deck del mattino** (`index.qmd`, ~65 slide): dalla provocazione sull'"AI tradizionale" al trade-off
  bias-varianza, le famiglie di modelli e le reti neurali; formule in LaTeX, figure in stile ISL/Torelli,
  quiz a rivelazione, esempi ISL (Advertising, Income, Wage, Default) e frame di Torelli;
- **notebook del pomeriggio** (`notebooks/NB0-NB4`): code-along guidato su Colab, con manopole e celle bonus;
- **dati** (`data/`): dataset canonici ISL (Auto, Default, Carseats) più un dataset retail sintetico stile
  Conad (churn e previsione vendite), caricati nei notebook via URL.

## Struttura

```
conad-statistical-learning/
├── _quarto.yml                 configurazione RevealJS + tema IFAB (rende solo index.qmd)
├── index.qmd                   deck del corso (~65 slide)
├── requirements.txt            stack Python per i notebook
├── theme/ifab.scss             tema IFAB (palette navy/ciano, fasce, footer, divisori)
├── assets/
│   ├── ifab-deco.html          decorazioni iniettate (fasce, logo, footer per sezione)
│   ├── genera_figure.py        genera mappa, timeline, y=f(x), flessibilità, ridge/lasso, genealogia
│   ├── genera_figure_2.py      genera le figure pedagogiche ISL/Torelli (train/test, bias-varianza, KNN, ...)
│   └── *.svg                   18 figure on-brand
├── data/
│   ├── genera_dataset.py       genera il dataset retail sintetico (seed fisso)
│   ├── conad_retail.csv        dataset retail sintetico (10.000 clienti)
│   ├── Auto.csv, Default.csv, Carseats.csv, Advertising.csv, Boston.csv, Hitters.csv   dataset canonici ISL
│   └── README.md               dizionario dei dati
├── notebooks/
│   └── NB0..NB4 .ipynb         notebook Colab del pomeriggio (editati direttamente)
├── docs/00_progettazione_corso.md   documento di progettazione (non pubblicato)
└── .github/workflows/publish.yml    render e deploy su GitHub Pages
```

## Anteprima in locale

Serve [Quarto](https://quarto.org).

```bash
quarto preview        # anteprima con live reload
quarto render         # build statica in _site/
```

Per rigenerare figure, dataset o notebook:

```bash
python assets/genera_figure.py
python assets/genera_figure_2.py
python data/genera_dataset.py
```

## Pubblicazione

Push su `main`: il workflow renderizza e pubblica sul branch `gh-pages`.
In GitHub, impostare Pages con sorgente il branch `gh-pages`.

## Identità visiva IFAB

- navy primario `#002060`; ciano accento `#00ADCF`; azzurro chiaro `#8FD2E9`; ardesia `#414F69`; grigio `#545454`; rosso `#DC4C4C`;
- font web Inter (fallback di Aptos);
- slide titolo e divisori di blocco su fondo navy bokeh con logo bianco;
- slide di contenuto con fascia navy bokeh in alto, striscia ciano in basso, footer con docente e sezione;
- formule in vero LaTeX via MathJax.

## Crediti

Materiale didattico costruito sulle spalle di *An Introduction to Statistical Learning* (ISL, edizione Python)
e delle dispense di N. Torelli (Università di Trieste). Dataset retail interamente sintetico.
