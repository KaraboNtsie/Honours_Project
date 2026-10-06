import re
import unicodedata
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load AfriXNLI Sesotho benchmark
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading AfriXNLI Sesotho benchmark...")
print("  Source: masakhane/afrixnli — 1,050 Sesotho NLI pairs")
print("  Labels: 0=entailment, 1=neutral, 2=contradiction")
print("█"*60)

dataset = load_dataset("masakhane/afrixnli", "sot")
print(f"  ✓ Dev split:  {len(dataset['validation'])} examples")
print(f"  ✓ Test split: {len(dataset['test'])} examples")
print(f"\n  Sample:")
for ex in list(dataset['test'])[:2]:
    print(f"    Premise:    {ex['premise'][:60]}")
    print(f"    Hypothesis: {ex['hypothesis'][:60]}")
    print(f"    Label:      {ex['label']}")
    print()

# combine premise and hypothesis as input feature
def make_input(examples):
    return [f"{p} [SEP] {h}"
            for p, h in zip(examples['premise'], examples['hypothesis'])]

test_X = make_input(dataset['test'])
test_y = dataset['test']['label']

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — Load your curated corpus as training data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Loading training corpora...")
print("█"*60)

curated_df = pd.read_csv("sesotho_clean.csv", encoding="utf-8")
print(f"  ✓ Curated corpus: {len(curated_df)} sentences")
print(f"  {curated_df['label'].value_counts().to_dict()}")

# ── Build raw (unfiltered) baseline corpus ────────────────────────────────────
# Raw corpus = same sources but WITHOUT GlotLID filter and quality filters
# Since we don't have the raw data saved, we simulate by:
# 1. Taking your curated corpus
# 2. Adding back noise: randomly duplicate sentences, add some English fragments
# NOTE: If you saved raw data during collection, load that instead

# Better approach: use a publicly available raw Sesotho corpus for comparison
# The Vukuzenzele raw GitHub files before filtering
import requests, io

print("\n  Loading raw (unfiltered) comparison corpus...")
raw_sentences = []
VUKUZENZELE_API = (
    "https://api.github.com/repos/dsfsi/vukuzenzele-nlp"
    "/contents/data/simple_align_output"
)
try:
    r = requests.get(VUKUZENZELE_API, timeout=30)
    items = r.json()
    csv_files = [f for f in items
                 if isinstance(f, dict) and f["name"].endswith(".csv")][:5]
    for f in csv_files:
        r2 = requests.get(f["download_url"], timeout=30)
        df_raw = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip")
        if len(df_raw.columns) >= 2:
            lines = df_raw.iloc[:, 1].dropna().astype(str).tolist()
            raw_sentences.extend(lines)
    print(f"  ✓ Raw corpus (Vukuzenzele, no filtering): {len(raw_sentences)} sentences")
except Exception as e:
    print(f"  ⚠ Could not load raw corpus: {e}")
    # fallback: use curated without quality filter as proxy
    raw_sentences = curated_df["text"].tolist()

raw_df = pd.DataFrame({"text": raw_sentences, "label": "SAS"})

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — Train on CURATED corpus, evaluate on AfriXNLI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Model A — trained on curated corpus...")
print("█"*60)

# For NLI we train a model to classify premise-hypothesis pairs
# We use the AfriXNLI dev set to create pseudo training examples
# augmented with our corpus sentences as the language model base

# Approach: use dev set for few-shot training + curated corpus for
# language adaptation, test on held-out test set
dev_X = make_input(dataset['validation'])
dev_y = dataset['validation']['label']

# Model A: TF-IDF fitted on curated corpus vocabulary
curated_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000,
        sublinear_tf=True,
        # vocabulary built from curated corpus
    )),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
])

# fit vectorizer on curated corpus, then train classifier on dev labels
curated_model.named_steps["tfidf"].fit(
    curated_df["text"].tolist() + dev_X
)
curated_model.fit(dev_X, dev_y)

curated_preds = curated_model.predict(test_X)
curated_f1 = f1_score(test_y, curated_preds, average="macro")

print(f"\n  Model A (curated corpus vocabulary):")
print(classification_report(test_y, curated_preds,
      target_names=["Entailment", "Neutral", "Contradiction"]))
print(f"  Macro F1: {round(curated_f1, 4)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Train on RAW corpus, evaluate on AfriXNLI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Model B — trained on raw/unfiltered corpus...")
print("█"*60)

raw_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
])

raw_model.named_steps["tfidf"].fit(
    raw_df["text"].tolist() + dev_X
)
raw_model.fit(dev_X, dev_y)

raw_preds = raw_model.predict(test_X)
raw_f1 = f1_score(test_y, raw_preds, average="macro")

print(f"\n  Model B (raw corpus vocabulary):")
print(classification_report(test_y, raw_preds,
      target_names=["Entailment", "Neutral", "Contradiction"]))
print(f"  Macro F1: {round(raw_f1, 4)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Statistical significance test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Statistical significance (Wilcoxon signed-rank)...")
print("█"*60)

# per-sample correctness scores
curated_correct = (np.array(curated_preds) == np.array(test_y)).astype(float)
raw_correct     = (np.array(raw_preds)     == np.array(test_y)).astype(float)

stat, p_value = wilcoxon(curated_correct, raw_correct)

print(f"\n  Wilcoxon signed-rank test:")
print(f"  Statistic: {round(stat, 4)}")
print(f"  p-value:   {round(p_value, 4)}")
print(f"  Significant at p < 0.05: {'YES' if p_value < 0.05 else 'NO'}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 6 — Summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 6: Results summary")
print("█"*60)

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Downstream Evaluation: AfriXNLI (Sesotho)          │
  │  Task: Natural Language Inference (3-class)         │
  │  Test set: {len(test_y)} examples                          │
  ├─────────────────────────────────────────────────────┤
  │  Model A (curated corpus)   Macro F1 = {round(curated_f1, 4)}       │
  │  Model B (raw corpus)       Macro F1 = {round(raw_f1, 4)}       │
  │  Improvement:               {round((curated_f1 - raw_f1)*100, 2)}%                │
  │  Statistically significant: {'YES (p=' + str(round(p_value,4)) + ')' if p_value < 0.05 else 'NO (p=' + str(round(p_value,4)) + ')'}  │
  └─────────────────────────────────────────────────────┘
""")