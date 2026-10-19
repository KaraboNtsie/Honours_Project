# Sesotho NLP Corpus Pipeline

A semi-automated pipeline for collecting, verifying, and curating Sesotho-language text, plus evaluations that test whether the curated data is actually useful.

Sesotho is low-resource, and the text that does exist online is noisy: it is mixed with English, and it comes in two orthographic standards (Lesotho, **LS**, and South African, **SAS**). This repo tackles both problems and then checks the result with an intrinsic test (language model perplexity) and a downstream test (NER).

Honours research project, Computer Science, Rhodes University.


## How it fits together

```
 BUILD THE CURATED CORPUS (sesotho_clean.csv, labelled LS / SAS by source)
   semi_automated_pipeline.py   Mokhosi Zenodo news (LS), Nalibali stories (LS), LS and SAS websites, Vukuzenzele CSVs and gov.za PDFs (SAS)
   exam_pipeline.py             matric exam PDFs (Sesotho HL and FAL), all labelled SAS
            │
            ▼
   orthographic_classifier.py   learns LS vs SAS spelling from the labelled corpus → ortho_classifier.pkl
            │
            ▼
   automated_pipeline.py        crawls seed sites unsupervised, filters with GlotLID, labels with the classifier → sesotho_corpus.jsonl

 EVALUATE THE CURATED CORPUS
   lmp.py                       char 5-gram LM perplexity, curated vs raw, tested on FLORES-200
   NER_domain.py / NER_zabantu.py   NER after continued MLM training, six-way comparison
```

Two different ways of getting data on purpose. The semi-automated route gives you labels you trust (the label comes from where the text was published). The fully automated route scales, but has to guess the label, which is what the classifier is for.

## Repo contents

| File | What it does |
|---|---|
| `semi_automated_pipeline.py` | Builds `sesotho_clean.csv`. Pulls text from curated sources. **LS:** the Mokhosi news dataset on Zenodo, Nalibali `st-ls` stories (discovered from category pages), and Lesotho government, news and literacy sites. **SAS:** Vukuzenzele aligned CSVs from GitHub, three gov.za Sesotho PDFs, and SA government and media pages. Applies quality filters (LaTeX artefacts, English code-switching, alpha ratio) and GlotLID verification, labels each sentence by source, dedupes, and merges into any existing `sesotho_clean.csv`. |
| `exam_pipeline.py` | Adds SAS sentences from matric Sesotho Home Language and First Additional Language exam PDFs (PyMuPDF). Skips cover and instruction pages and English-heavy pages, applies the same quality and GlotLID filters, and merges into `sesotho_clean.csv`. |
| `orthographic_classifier.py` | Trains and evaluates the LS vs SAS spelling classifier. Compares a rule-based baseline (Makutoane 2022 markers) with a character n-gram + logistic regression model. Saves `ortho_classifier.pkl`. |
| `automated_pipeline.py` | End to end unsupervised collection. Discovers URLs (sitemaps, RSS, focused crawl), extracts text with trafilatura, keeps Sesotho sentences using GlotLID, labels orthography with the trained classifier, filters, and writes `sesotho_corpus.jsonl`. |
| `lmp.py` | Research question 3. Trains two character-level 5-gram language models (curated vs raw corpus, same size) and compares perplexity on FLORES-200 Sesotho devtest, with a Wilcoxon signed-rank test. |
| `NER_domain.py` | Runs all six NER experiments in one go (AfroXLMR-mini and Zabantu-XLM-RoBERTa, each with no adaptation, MLM on the curated corpus, MLM on NCHLT), then six paired Wilcoxon tests. Written to run on Kaggle. |
| `NER_zabantu.py` | The resumable version for the Zabantu half. Runs only D, E, F, loads the AfroXLMR results (A, B, C) from an earlier run (`ner_results_three_way.json`), and adds three cross-model tests (A vs D, B vs E, C vs F). Saves each finished run straight away and deletes model files to save disk. |

### The six NER runs

| Run | Base model | MLM adaptation corpus |
|---|---|---|
| A | AfroXLMR-mini | none |
| B | AfroXLMR-mini | curated corpus |
| C | AfroXLMR-mini | NCHLT text |
| D | Zabantu-XLM-RoBERTa | none |
| E | Zabantu-XLM-RoBERTa | curated corpus |
| F | Zabantu-XLM-RoBERTa | NCHLT text |

`NER_zabantu.py` only produces valid paired tests if the NER file and seed (42) match the earlier AfroXLMR run (948 test sentences). It checks the test set size and stops if they differ. Each finished run is cached in `run_results/`, so rerunning the script skips completed runs.

## Setup

Python 3.10+ recommended.

```bash
pip install pandas numpy scikit-learn scipy fasttext trafilatura huggingface_hub requests beautifulsoup4 pymupdf
# only needed for the NER scripts
pip install torch transformers datasets accelerate
```

<!-- TODO: pin versions in a requirements.txt -->

## Data you need

| File | Used by | Notes |
|---|---|---|
| Exam paper PDFs | `exam_pipeline.py` | Two local folders, one for HL and one for FAL. Set `HL_FOLDER` and `FAL_FOLDER` at the top of the script (currently hard-coded Windows paths). |
| `sot_Latn_devtest.jsonl` | `lmp.py` | FLORES-200 Sesotho devtest, one JSON object per line with a `text` field. Downloaded from HuggingFace. |
| `Dataset.NCHLT-II.st.NER.Full.txt` | NER scripts | NCHLT Sesotho NER data (CoNLL style). |
| `CORP.NCHLT.st.CLEAN.2.0.txt` | NER scripts | NCHLT Sesotho text corpus for MLM adaptation. |
| `ner_results_three_way.json` | `NER_zabantu.py` | Per-sentence scores for runs A, B, C from the earlier AfroXLMR run. |

The Mokhosi dataset (Zenodo), Vukuzenzele CSVs (GitHub) and gov.za PDFs are downloaded by `semi_automated_pipeline.py` itself, so they need internet access but no manual download. `sesotho_clean.csv` is produced by the two collection scripts (columns: `text`, `label`, `lid_confidence`, `source`, `source_url`). GlotLID (`cis-lmu/glotlid`) is downloaded from HuggingFace on first run.

## Running it

Run in this order. Later steps depend on files made by earlier ones.

```bash
# 1. Build the curated corpus (both scripts merge into sesotho_clean.csv, any order)
python semi_automated_pipeline.py
python exam_pipeline.py

# 2. Train the orthographic classifier (writes ortho_classifier.pkl and classifier_results.json)
python orthographic_classifier.py

# 3. Run the unsupervised collection pipeline (needs ortho_classifier.pkl, writes sesotho_corpus.jsonl)
python automated_pipeline.py

# 4. Evaluate the curated corpus
python lmp.py              # writes perplexity_results.json
python NER_domain.py       # all six runs, writes ner_results_six_way.json (GPU strongly recommended)
# or, if you already have the AfroXLMR results:
python NER_zabantu.py      # runs D, E, F only, writes ner_results_six_way.json
```

Both NER scripts write to the same output filename, so run one or the other, or rename the output. They look for inputs under `/kaggle/working` and `/kaggle/input`. To run elsewhere, edit `find_input` and the output paths near the top. Expect long runtimes: each run is an MLM training plus an NER fine-tune.

Both collection scripts report progress toward a target of 2,500 sentences per variant (LS and SAS).

## Output format

Each line of `sesotho_corpus.jsonl` is one sentence:

```json
{
  "text": "...",
  "ortho_label": "LS",
  "ortho_confidence": 0.93,
  "lid_confidence": 0.98,
  "source_url": "https://...",
  "token_count": 12
}
```
## Acknowledgements and references

- Makutoane (2022), LS vs SAS orthographic markers
- GlotLID (`cis-lmu/glotlid`) for language identification
- FLORES-200 for the perplexity test set
- NCHLT Sesotho corpora (NWU) for NER data and MLM text
- AfroXLMR-mini (`Davlan/afro-xlmr-mini`) and Zabantu-XLM-RoBERTa (`dsfsi/zabantu-xlm-roberta`)
- Vukuzenzele and Mokhosi Sesotho datasets, Nalibali stories


