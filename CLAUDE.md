# CLAUDE.md - Corso CONAD Nord Ovest, Apprendimento statistico

Contesto di progetto per Claude Code. Questo file viene letto in automatico all'avvio.
Tutta la fase di progettazione e' chiusa: le decisioni qui sotto NON vanno rimesse in discussione.

## Cos'e' il progetto

Sito del corso (8 ore: 4 frontali + 4 hands-on) tenuto da Orso Peruzzi (Data Scientist, IFAB) per CONAD Nord Ovest, in presenza a Pistoia. Tema: apprendimento statistico come machine learning a tutti gli effetti, base concettuale anche dei modelli profondi (CNN, RNN, LSTM, GAN, transformer, LLM).

Documento di progettazione completo (scaletta mattina e pomeriggio, decisioni, identita' visiva): `docs/00_progettazione_corso.md`. Leggilo per primo.

## Architettura (bloccata)

- spina dorsale narrativa: sito Quarto / RevealJS pubblicato su GitHub Pages;
- superficie eseguibile: notebook Colab collegati con badge "Apri in Colab", auto-contenuti (dati via URL);
- repo: `battles5/conad-statistical-learning`, pubblico;
- URL finale: `https://battles5.github.io/conad-statistical-learning/`;
- lingua: tutto in italiano;
- riferimento: Introduction to Statistical Learning (ISL), edizione Python; mappatura terminologica IT da edizione Piccin (R) e dispense Torelli;
- dataset: canonici ISL (Auto, Default, Carseats) piu' dataset retail sintetico stile Conad (churn e previsione vendite);
- pomeriggio: code-along guidato su Colab, livello misto (progettato per il livello piu' basso, con celle bonus per i piu' esperti).

## Regole editoriali (preferenze Orso, vincolanti)

- niente em-dash e niente en-dash, mai;
- nelle voci di elenco: iniziale minuscola e punto e virgola tra le voci;
- footer di ogni slide: `Orso Peruzzi · Apprendimento statistico (CONAD Nord Ovest) · sezione corrente` (sostituire "sezione corrente" con la sezione effettiva);
- in fondo al sito: sezione referenze con bibliografia arricchita (ISL Python, ISL ed. Piccin, Torelli, ESL, paper chiave);
- formule in vero LaTeX via MathJax.

## Identita' visiva IFAB

Ricreata come tema RevealJS (match ~90% del template ufficiale, non il .pptx identico).

- navy primario `#002060`; ciano accento e link `#00ADCF`; azzurro chiaro `#8FD2E9`; ardesia `#414F69`; grigio `#545454`; rosso `#DC4C4C`;
- font web Inter (fallback di Aptos);
- slide titolo: fondo navy con texture bokeh (`assets/title-bg-bokeh.jpeg`), logo bianco centrato;
- slide di contenuto: fascia navy bokeh in alto con logo bianco a destra, striscia ciano in basso, footer con il credito del docente;
- logo bianco per fondi scuri `assets/ifab-logo-white.svg`; logo navy per fondi chiari `assets/ifab-logo-navy.svg`.

## Struttura del repo

```
_quarto.yml                 configurazione RevealJS + tema IFAB
index.qmd                   deck del corso (per ora 2 slide di prova validate)
requirements.txt            stack Python per i notebook (jupyter, numpy, pandas, matplotlib, scikit-learn)
theme/ifab.scss             tema IFAB (palette, fasce, footer, slide titolo)
assets/
  ifab-logo-white.svg       logo per fondo navy
  ifab-logo-navy.svg        logo per fondo bianco
  title-bg-bokeh.jpeg       sfondo bokeh
  ifab-deco.html            decorazioni iniettate via include-after-body (fasce + logo + toggle JS)
  bias-variance-ucurve.svg  figura curva a U (generata con matplotlib)
docs/00_progettazione_corso.md   documento di progettazione completo
.github/workflows/publish.yml    render e deploy automatico su gh-pages
```

Note tecniche sul tema: il footer e' globale in `_quarto.yml` e si sovrascrive per slide con `## Titolo {footer="..."}`. La slide titolo usa `title-slide-attributes` per lo sfondo bokeh. Le fasce e il footer sono nascosti sulla slide titolo da un piccolo toggle JS in `assets/ifab-deco.html` (classe `ifab-on-title` sul body).

## Comandi

```bash
quarto preview     # anteprima live in locale
quarto render      # build statica in _site/
```

Pubblicazione: push su `main`, il workflow renderizza e pubblica su `gh-pages`; in GitHub impostare Pages con sorgente il branch `gh-pages`.

## Stato di avanzamento

Fatto:
- pre-progettazione completa (scaletta, decisioni, stile validato);
- scaffold repo + tema IFAB SCSS;
- 2 slide di prova validate (titolo navy bokeh + contenuto bias-varianza con curva a U).

Da fare (nell'ordine):
- dataset retail sintetico stile Conad (churn + previsione vendite);
- slide del mattino (blocchi 0-4, vedi `docs/00_progettazione_corso.md`);
- notebook Colab NB0-NB4 con badge "Apri in Colab";
- sezione referenze;
- test e pubblicazione su GitHub Pages.
