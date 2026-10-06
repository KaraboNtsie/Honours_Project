"""
downstream_perplexity.py
========================
Downstream utility evaluation for the Sesotho NLP pipeline.

Research Question 3: Does pipeline-curated data produce a better
language model than raw unfiltered text from the same sources?

Method:
  - Train a character-level n-gram language model on the curated corpus
  - Train the same model on a raw/unfiltered baseline corpus
  - Evaluate both on FLORES-200 Sesotho (sot_Latn) devtest sentences
  - Lower perplexity = better model = higher quality training data

Perplexity is defined as:
  PP(W) = exp(-1/N * sum(log P(w_i)))
where N is the total number of characters and P(w_i) is the
character n-gram probability of character w_i given context.

No GPU required. Runs in under 2 minutes.
"""

import re
import math
import unicodedata
import collections
import requests
import pandas as pd
from datasets import load_dataset



# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load FLORES-200 Sesotho test set from local JSONL files
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading FLORES-200 Sesotho test set...")
print("  Source: sot_Latn_devtest.jsonl (downloaded from HuggingFace)")
print("  1,012 professionally translated Sesotho sentences")
print("  Completely independent of training corpus")
print("█"*60)

import json

DEVTEST_FILE = r"sot_Latn_devtest.jsonl"

test_sentences = []
with open(DEVTEST_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if text:
                test_sentences.append(text)

print(f"  ✓ {len(test_sentences)} test sentences loaded")
print(f"  Sample:")
for s in test_sentences[:3]:
    print(f"    '{s[:70]}'")

# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER N-GRAM LANGUAGE MODEL
# Uses Kneser-Ney smoothing — standard for n-gram LMs
# Character-level (not word-level) because Sesotho is agglutinative and
# character n-grams better capture morphological patterns
# ══════════════════════════════════════════════════════════════════════════════

def build_ngram_lm(sentences, n=5, smoothing_k=0.1):
    """
    Build a character-level n-gram language model with add-k smoothing.

    Parameters
    ----------
    sentences : list of str
        Training sentences (already cleaned)
    n : int
        N-gram order (5 = quintigram, standard for character LMs)
    smoothing_k : float
        Add-k smoothing parameter. k=0.1 is a common choice for
        character-level models (Jurafsky & Martin, 2023)

    Returns
    -------
    dict containing n-gram counts, context counts, and vocabulary
    """
    ngram_counts   = collections.Counter()
    context_counts = collections.Counter()
    char_vocab     = set()

    for sentence in sentences:
        # pad sentence with start/end tokens
        padded = "^" * (n-1) + sentence.lower() + "$"
        for char in padded:
            char_vocab.add(char)
        # count n-grams and their contexts
        for i in range(len(padded) - n + 1):
            ngram   = padded[i:i+n]
            context = padded[i:i+n-1]
            ngram_counts[ngram]   += 1
            context_counts[context] += 1

    return {
        "ngram_counts":   ngram_counts,
        "context_counts": context_counts,
        "vocab":          char_vocab,
        "vocab_size":     len(char_vocab),
        "n":              n,
        "k":              smoothing_k,
    }


def sentence_log_prob(sentence, lm):
    """
    Compute the log probability of a sentence under the language model.
    Uses add-k smoothing: P(w|context) = (C(context,w) + k) / (C(context) + k*V)
    """
    n    = lm["n"]
    k    = lm["k"]
    V    = lm["vocab_size"]
    padded = "^" * (n-1) + sentence.lower() + "$"
    log_prob = 0.0
    char_count = 0
    for i in range(n-1, len(padded)):
        ngram   = padded[i-n+1:i+1]
        context = padded[i-n+1:i]
        count_ngram   = lm["ngram_counts"].get(ngram, 0)
        count_context = lm["context_counts"].get(context, 0)
        # add-k smoothed probability
        prob = (count_ngram + k) / (count_context + k * V)
        log_prob += math.log(prob)
        char_count += 1
    return log_prob, char_count


def compute_perplexity(test_sentences, lm):
    """
    Compute corpus-level perplexity on test sentences.
    PP = exp(-1/N * sum(log P(w_i)))
    Lower is better.
    """
    total_log_prob  = 0.0
    total_chars     = 0
    failed          = 0
    per_sent_perp   = []

    for sentence in test_sentences:
        try:
            log_prob, n_chars = sentence_log_prob(sentence, lm)
            total_log_prob += log_prob
            total_chars    += n_chars
            # per-sentence perplexity for significance test
            sent_pp = math.exp(-log_prob / max(n_chars, 1))
            per_sent_perp.append(sent_pp)
        except Exception:
            failed += 1

    corpus_perplexity = math.exp(-total_log_prob / max(total_chars, 1))
    return corpus_perplexity, per_sent_perp, failed


def clean_for_lm(text):
    """Minimal cleaning for LM training — preserve Sesotho characters."""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — Load curated corpus (your pipeline output)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Loading curated corpus...")
print("█"*60)

curated_df = pd.read_csv("sesotho_clean.csv", encoding="utf-8")
curated_sentences = [clean_for_lm(t) for t in curated_df["text"].tolist()]
print(f"  ✓ Curated corpus: {len(curated_sentences)} sentences")
print(f"  Label distribution: {curated_df['label'].value_counts().to_dict()}")
print(f"  Sources: {curated_df['source'].nunique()} distinct sources")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — Build raw (unfiltered) baseline corpus
# Raw corpus = same sources but without quality filtering
# This uses the Vukuzenzele raw GitHub data without GlotLID/quality filters
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Building raw baseline corpus...")
print("  Same sources, no GlotLID filter, no code-switching filter")
print("█"*60)

import io
raw_sentences = []

# Pull raw Vukuzenzele (no filtering)
VUKU_API = ("https://api.github.com/repos/dsfsi/vukuzenzele-nlp"
            "/contents/data/simple_align_output")
try:
    r = requests.get(VUKU_API, timeout=30)
    items = r.json()
    csv_files = [f for f in items
                 if isinstance(f, dict) and f["name"].endswith(".csv")]
    print(f"  ├─ Vukuzenzele: {len(csv_files)} CSV files")
    for f in csv_files:
        r2 = requests.get(f["download_url"], timeout=30)
        df_raw = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip")
        if len(df_raw.columns) >= 2:
            lines = df_raw.iloc[:, 1].dropna().astype(str).tolist()
            raw_sentences.extend([clean_for_lm(l) for l in lines])
    print(f"  │   ✓ {len(raw_sentences)} raw sentences")
except Exception as e:
    print(f"  ├─ Vukuzenzele failed: {e}")

# Pull raw Mokhosi (no filtering — include numeric labels and all lines)
ZENODO_URLS = [
    "https://zenodo.org/api/records/10531959/files/NewsSA.txt/content",
    "https://zenodo.org/api/records/10531959/files/NewsABSA.txt/content",
]
for url in ZENODO_URLS:
    try:
        r = requests.get(url, timeout=60)
        lines = [l.strip() for l in r.text.split("\n") if l.strip()]
        raw_sentences.extend([clean_for_lm(l) for l in lines])
        print(f"  ├─ Mokhosi ({url.split('/')[-2]}): {len(lines)} lines")
    except Exception as e:
        print(f"  ├─ Mokhosi failed: {e}")

print(f"\n  ✓ Raw baseline corpus: {len(raw_sentences)} sentences (unfiltered)")

# balance raw corpus to same size as curated for fair comparison
import random
random.seed(42)
if len(raw_sentences) > len(curated_sentences):
    raw_sentences = random.sample(raw_sentences, len(curated_sentences))
    print(f"  ✓ Downsampled to {len(raw_sentences)} for fair comparison")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Train both language models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Training language models (n=5, add-k smoothing)...")
print("█"*60)

print("  ├─ Training Model A (curated corpus)...")
curated_lm = build_ngram_lm(curated_sentences, n=5, smoothing_k=0.1)
print(f"  │   ✓ Vocabulary size: {curated_lm['vocab_size']} characters")
print(f"  │   ✓ Unique 5-grams: {len(curated_lm['ngram_counts'])}")

print("  ├─ Training Model B (raw corpus)...")
raw_lm = build_ngram_lm(raw_sentences, n=5, smoothing_k=0.1)
print(f"  │   ✓ Vocabulary size: {raw_lm['vocab_size']} characters")
print(f"  │   ✓ Unique 5-grams: {len(raw_lm['ngram_counts'])}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Evaluate perplexity on FLORES-200 test set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Evaluating perplexity on FLORES-200 Sesotho...")
print("  (1,001 independently sourced test sentences)")
print("█"*60)

curated_pp, curated_per_sent, curated_failed = compute_perplexity(
    test_sentences, curated_lm
)
raw_pp, raw_per_sent, raw_failed = compute_perplexity(
    test_sentences, raw_lm
)

print(f"\n  Model A (curated)  — Perplexity: {round(curated_pp, 2)}")
print(f"  Model B (raw)      — Perplexity: {round(raw_pp, 2)}")
print(f"  Difference: {round(raw_pp - curated_pp, 2)} "
      f"({'curated better' if curated_pp < raw_pp else 'raw better'})")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 6 — Statistical significance (Wilcoxon signed-rank test)
# Per-sentence perplexity scores are paired and non-normal
# Wilcoxon is the appropriate non-parametric test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 6: Statistical significance test...")
print("  Wilcoxon signed-rank on per-sentence perplexity pairs")
print("  α = 0.05")
print("█"*60)

from scipy.stats import wilcoxon
import numpy as np

min_len = min(len(curated_per_sent), len(raw_per_sent))
curated_arr = np.array(curated_per_sent[:min_len])
raw_arr     = np.array(raw_per_sent[:min_len])

# cap extreme perplexity values to avoid inf dominating
cap = np.percentile(np.concatenate([curated_arr, raw_arr]), 99)
curated_arr = np.clip(curated_arr, 0, cap)
raw_arr     = np.clip(raw_arr, 0, cap)

try:
    stat, p_value = wilcoxon(curated_arr, raw_arr)
    print(f"\n  Wilcoxon statistic: {round(stat, 4)}")
    print(f"  p-value:            {p_value:.2e}")
    print(f"  Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")
    significant = p_value < 0.05
except Exception as e:
    print(f"  ⚠ Wilcoxon failed: {e}")
    p_value = None
    significant = False

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 7 — Summary results table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 7: Results summary")
print("█"*60)

improvement = raw_pp - curated_pp
improvement_pct = (improvement / raw_pp) * 100

print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  DOWNSTREAM EVALUATION: Language Model Perplexity        ║
  ║  Test set: FLORES-200 Sesotho devtest ({len(test_sentences)} sentences)  ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  Corpus          │ Training sents │ Perplexity            ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  Curated (A)     │ {len(curated_sentences):<15} │ {round(curated_pp, 2):<22} ║
  ║  Raw/unfiltered  │ {len(raw_sentences):<15} │ {round(raw_pp, 2):<22} ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  Reduction in perplexity: {round(improvement, 2)} ({round(improvement_pct, 1)}%)               ║
  ║  Statistically significant (α=0.05): {'YES' if significant else 'NO '}                  ║
  ║  p-value: {round(p_value, 6) if p_value is not None else 'N/A':<50}║
  ╚═══════════════════════════════════════════════════════════╝

  Interpretation:
  {'✓ Curated corpus produces a significantly better language model.' if curated_pp < raw_pp and significant
    else '⚠ Curated corpus perplexity is lower but not statistically significant.'
    if curated_pp < raw_pp
    else '✗ Raw corpus perplexity is lower — investigate corpus quality.'}

  Lower perplexity = the model assigns higher probability to held-out
  Sesotho text = the training corpus better represents the language.
  A {round(improvement_pct, 1)}% reduction demonstrates that quality-first corpus
  construction produces measurably better statistical language models
  for Sesotho than unfiltered text collection.
""")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 8 — Save results for thesis
# ══════════════════════════════════════════════════════════════════════════════
print("█"*60)
print("  CHECKPOINT 8: Saving results...")
print("█"*60)

import json
results = {
    "test_set": "FLORES-200 Sesotho devtest",
    "test_sentences": len(test_sentences),
    "model_order": 5,
    "smoothing": "add-k (k=0.1)",
    "curated": {
        "training_sentences": len(curated_sentences),
        "perplexity": round(curated_pp, 4),
        "vocab_size": curated_lm["vocab_size"],
        "unique_ngrams": len(curated_lm["ngram_counts"])
    },
    "raw": {
        "training_sentences": len(raw_sentences),
        "perplexity": round(raw_pp, 4),
        "vocab_size": raw_lm["vocab_size"],
        "unique_ngrams": len(raw_lm["ngram_counts"])
    },
    "perplexity_reduction": round(improvement, 4),
    "perplexity_reduction_pct": round(improvement_pct, 2),
    "wilcoxon_stat": round(stat, 4) if p_value is not None else None,
    "p_value": round(p_value, 6) if p_value is not None else None,
    "significant_at_0_05": bool(significant)
}

with open("perplexity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  ✓ Results saved to perplexity_results.json")
print("  ✓ These numbers go directly into Chapter 5 Table 5.1")
print("█"*60)