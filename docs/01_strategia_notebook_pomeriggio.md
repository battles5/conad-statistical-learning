# Strategia notebook del pomeriggio (ISL-faithful)

Pre-progettazione validata con il docente. Documento operativo per la costruzione dei
notebook Colab NB0-NB4. Tutte le decisioni qui sotto sono confermate.

## Tesi e vincoli

- code-along guidato di 4h su Colab, livello misto (progettato per il livello piu' basso,
  con celle bonus per i piu' esperti);
- ogni notebook ancorato a *An Introduction to Statistical Learning* (ISL, edizione Python);
- materiale sorgente: `raw_materials/13_islp` (22 dataset canonici, capitoli in PDF, figure
  ufficiali del libro, lab ufficiali `intro-stat-learning/ISLP_labs`);
- regole editoriali: niente em-dash ne en-dash; voci di elenco in minuscolo con punto e
  virgola; italiano ovunque; formule LaTeX.

## Decisioni confermate

1. stack **ibrido**: scikit-learn come motore del code-along; cella bonus `statsmodels`
   per la tabella di sintesi OLS/logit (filo "predizione vs interpretazione"); pacchetto
   `ISLP` opzionale, mai obbligatorio (nessun `pip install` bloccante);
2. **NB2** usa Hitters per ridge/lasso (19 predittori, sparsita' visibile) oltre a Default
   per logistica e soglia; **NB3** doppio binario: meccanica su dataset ISL, poi transfer
   sul caso retail Conad sintetico;
3. **riprodurre dal vivo** le figure iconiche del libro (la U bias-varianza, i polinomi su
   Auto, il path del lasso, l'albero, l'importanza delle variabili);
4. i notebook si **editano direttamente** come `.ipynb`; il generatore `genera_notebooks.py`
   e' rimosso (fonte unica = i notebook stessi);
5. i 5 notebook restano raggiungibili dal README con i badge "Apri in Colab".

I notebook possono essere rifatti ex novo dove la prima versione non e' all'altezza.

## Mappatura notebook -> ISL

| NB | capitolo / lab ISL | dataset | figura del libro riprodotta | manopola | bonus |
|---|---|---|---|---|---|
| NB0 setup (~20m) | Ch2 statlearn | Auto | scatter mpg~horsepower | `test_size` | weight vs mpg |
| NB1 regressione e U (~60m) | Ch3 + Ch5 | Auto (+Advertising bonus) | polinomi Fig 3.8; U test error Fig 2.9-2.12 | grado polinomio | k-fold CV; summary OLS |
| NB2 classificazione, regolarizz., CV (~55m) | Ch4 + Ch6 + Ch5 | Default; Hitters | path lasso Fig 6.6; soglia/ROC | soglia, poi C=1/lambda | path completo; summary logit |
| NB3 ensemble e caso retail (~60m) | Ch8 + Ch9 | Boston/Carseats -> conad_retail | albero Fig 8.1; importanza variabili | profondita' albero | permutation importance |
| NB4 reti neurali (~30m) | Ch10 | Hitters o Default | confronto flessibilita' | `alpha` | cenno torch/Keras; big picture |

## Piano dati

- gia' in `data/`: Auto, Default, Carseats, conad_retail (sintetico);
- da aggiungere (copiati da `raw_materials/13_islp/ALL CSV FILES - 2nd Edition/`):
  Advertising, Boston, Hitters;
- tutti serviti via URL raw dal repo, come gli altri (notebook auto-contenuti per Colab);
- `data/README.md` aggiornato con i nuovi dataset e i crediti ISL.

## Pattern didattico per notebook

- intestazione con badge Colab e istruzioni "Shift+Invio";
- celle markdown brevi in italiano, una idea per cella;
- una sola **manopola** evidenziata (`>>> MANOPOLA <<<`) con effetto visibile;
- almeno un momento "indovina prima di eseguire";
- cella **bonus** per i piu' esperti a fine notebook;
- riproduzione dal vivo della figura iconica del capitolo, con riferimento (es. "cfr. ISL Fig 2.9");
- nessun output salvato (notebook puliti per Colab).

## Figure nelle slide (raccomandazione)

Non sostituire in blocco le figure del deck con quelle del libro. Motivi: le SVG on-brand
(navy/ciano, etichette in italiano) garantiscono coerenza visiva IFAB; le figure del libro
hanno etichette in inglese e stile accademico diverso. Approccio: il deck tiene le SVG
on-brand come linguaggio primario; i notebook portano la versione "autentica" riprodotta dal
vivo dai dati reali; dove una SVG riproduce una figura iconica si aggiunge il riferimento
(es. "cfr. ISL Fig 2.9"). Le foto di contesto (scaffali, scontrino, volantino, NASDAQ) non
sono figure del libro e restano.

## Sequenza di build

1. setup: rimosso il generatore, aggiunti i dataset, aggiornati README e CLAUDE; *(fatto)*
2. NB0 e NB1 come slice verticale per validare il pattern;
3. NB2, NB3, NB4 a seguire;
4. QA: esecuzione end-to-end in locale di ogni notebook, zero errori;
5. rifinitura: regole editoriali, badge e link nel README, dizionario dati, crediti;
6. pulizia output e commit.
