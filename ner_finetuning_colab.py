"""
Sesotho NER comparison using the NCHLT corpus and curated text.

Two matched experiments are run:
  1. Baseline: fine-tune AfroXLMR directly on the labeled NCHLT corpus.
  2. Curated-adapted: continue masked-language-model training on the
     unannotated text in sesotho_clean.csv, then fine-tune on NCHLT.

The curated CSV has no token-level NER labels, so it is used for domain
adaptation only. Both models are evaluated on the same held-out NCHLT test
split.

Files needed (set full paths in the Config section below if not in cwd):
  Dataset.NCHLT-II.st.NER.Full.txt
  sesotho_clean.csv

Results are saved to ner_results_domain_adaptation.json.
"""

import json
import os
import random
import subprocess
import sys


def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip("transformers>=4.40", "datasets", "seqeval", "accelerate")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────
# Set these to absolute paths if the files are not in the current directory.
BASE_MODEL  = "Davlan/afro-xlmr-mini"
NER_FILE    = os.path.join("NCHLT Sesotho Named Entity Annotated Corpus",
                           "Dataset.NCHLT-II.st.NER.Full.txt")
CURATED_CSV = "sesotho_clean.csv"
OUTPUT_JSON = "ner_results_domain_adaptation.json"
OUTPUT_DIR  = "ner_output"          # local folder, created automatically

# Reduce BATCH_SIZE or MLM_EPOCHS if you hit OOM on your GPU.
MLM_EPOCHS  = 2
NER_EPOCHS  = 5
BATCH_SIZE  = 8                     # conservative default for a desktop GPU
MAX_LEN     = 128
MLM_LR      = 5e-5
NER_LR      = 2e-5
SEED        = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
print(f"Device: {DEVICE}  fp16={USE_FP16}")

# ── Parse NWU NCHLT NER corpus (CoNLL-style) ─────────────────────────────────
def parse_conll(path):
    """Return list of (token_list, label_list) from a CoNLL-style file."""
    sentences, tokens, labels = [], [], []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens[:], labels[:]))
                    tokens, labels = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                # Skip malformed lines / headers silently
                continue
            tokens.append(parts[0])
            labels.append("O" if parts[-1] == "OUT" else parts[-1])
    if tokens:
        sentences.append((tokens, labels))
    if not sentences:
        raise ValueError(f"No NER sentences found in {path!r}")
    return sentences


all_sentences  = parse_conll(NER_FILE)
random.shuffle(all_sentences)
n              = len(all_sentences)
train_count    = int(0.8 * n)
val_count      = int(0.1 * n)
train_sentences      = all_sentences[:train_count]
validation_sentences = all_sentences[train_count : train_count + val_count]
test_sentences       = all_sentences[train_count + val_count :]

all_labels = sorted({lab for _, labs in all_sentences for lab in labs})
label2id   = {l: i for i, l in enumerate(all_labels)}
id2label   = {i: l for l, i in label2id.items()}
num_labels = len(all_labels)

print(f"NCHLT  train={len(train_sentences)}  val={len(validation_sentences)}  test={len(test_sentences)}")
print(f"Labels ({num_labels}): {all_labels}")

# ── Tokenise + align labels ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def tokenise_and_align(sentence_pairs):
    encoded = []
    for tokens, labels in sentence_pairs:
        item = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        word_ids  = item.word_ids()
        label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None or wid == prev_word:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[labels[wid]])
            prev_word = wid
        item["labels"] = label_ids
        encoded.append(dict(item))   # BatchEncoding -> plain dict for safety
    return encoded


class NERDataset(TorchDataset):
    def __init__(self, encoded):
        self.data = encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}


train_dataset      = NERDataset(tokenise_and_align(train_sentences))
validation_dataset = NERDataset(tokenise_and_align(validation_sentences))
test_dataset       = NERDataset(tokenise_and_align(test_sentences))
data_collator      = DataCollatorForTokenClassification(tokenizer)

# ── Metrics ───────────────────────────────────────────────────────────────────
def sequences_from_predictions(predictions, label_ids):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_ids = np.argmax(predictions, axis=-1)
    true_seqs, pred_seqs = [], []
    for pred_row, lab_row in zip(pred_ids, label_ids):
        t, p = [], []
        for pi, li in zip(pred_row, lab_row):
            if li == -100:
                continue
            t.append(id2label[int(li)])
            p.append(id2label[int(pi)])
        true_seqs.append(t)
        pred_seqs.append(p)
    return true_seqs, pred_seqs


def compute_metrics(eval_pred):
    true_seqs, pred_seqs = sequences_from_predictions(
        eval_pred.predictions, eval_pred.label_ids
    )
    return {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall":    recall_score(true_seqs, pred_seqs),
        "f1":        f1_score(true_seqs, pred_seqs),
    }

# ── Load curated text ─────────────────────────────────────────────────────────
def load_curated_text(path):
    df  = pd.read_csv(path)
    col = next((c for c in ("sentence", "text", "Sentence", "Text") if c in df.columns), None)
    if col is None:
        raise ValueError(
            f"Could not find a text column in {path!r}. "
            f"Columns present: {list(df.columns)}"
        )
    texts = df[col].dropna().astype(str).str.strip().tolist()
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError(f"No non-empty text found in {path!r}")
    print(f"Curated sentences loaded: {len(texts)}")
    return texts

# ── MLM domain adaptation ─────────────────────────────────────────────────────
def adapt_to_curated_text(texts):
    print("\n── MLM domain adaptation ──────────────────────────────────────")
    mlm_dataset = HFDataset.from_dict({"text": texts})

    def tokenize_mlm(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )

    tokenized = mlm_dataset.map(tokenize_mlm, batched=True, remove_columns=["text"])
    mlm_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(DEVICE)

    mlm_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "mlm"),
        num_train_epochs=MLM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=MLM_LR,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        fp16=USE_FP16,
        seed=SEED,
        report_to="none",
    )
    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            processing_class=tokenizer, mlm=True, mlm_probability=0.15
        ),
    )
    mlm_trainer.train()

    adapted_path = os.path.join(OUTPUT_DIR, "mlm_adapted")
    mlm_model.save_pretrained(adapted_path)
    tokenizer.save_pretrained(adapted_path)
    del mlm_model, mlm_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Adapted model saved to {adapted_path!r}")
    return adapted_path

# ── NER fine-tuning ───────────────────────────────────────────────────────────
def run_ner_experiment(run_name, model_source):
    print(f"\n── NER experiment: {run_name} ──────────────────────────────────")
    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    ner_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, run_name),
        num_train_epochs=NER_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=NER_LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=USE_FP16,
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=ner_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    prediction = trainer.predict(test_dataset)
    true_seqs, pred_seqs = sequences_from_predictions(
        prediction.predictions, prediction.label_ids
    )
    metrics = {
        "precision":             precision_score(true_seqs, pred_seqs),
        "recall":                recall_score(true_seqs, pred_seqs),
        "f1":                    f1_score(true_seqs, pred_seqs),
        "classification_report": classification_report(
                                     true_seqs, pred_seqs, output_dict=True
                                 ),
    }
    print(
        f"  {run_name}:  P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
    )
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

# ── Run everything ────────────────────────────────────────────────────────────
#curated_texts      = load_curated_text(CURATED_CSV)
adapted_model_path = "ner_output\\mlm_adapted"
baseline_metrics   = run_ner_experiment("baseline", BASE_MODEL)
adapted_metrics    = run_ner_experiment("curated_adapted", adapted_model_path)

results = {
    "baseline":        baseline_metrics,
    "curated_adapted": adapted_metrics,
    "delta_curated_minus_baseline": {
        m: adapted_metrics[m] - baseline_metrics[m]
        for m in ("precision", "recall", "f1")
    },
    "evaluation": {
        "dataset":              "NCHLT Sesotho Named Entity Annotated Corpus",
        "curated_csv_used_for": "unsupervised MLM domain adaptation only",
        "test_sentences":       len(test_sentences),
    },
    "config": {
        "base_model":           BASE_MODEL,
        "mlm_epochs":           MLM_EPOCHS,
        "ner_epochs":           NER_EPOCHS,
        "batch_size":           BATCH_SIZE,
        "max_len":              MAX_LEN,
        "mlm_lr":               MLM_LR,
        "ner_lr":               NER_LR,
        "seed":                 SEED,
        "train_sentences":      len(train_sentences),
        "validation_sentences": len(validation_sentences),
        "test_sentences":       len(test_sentences),
    },
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print(f"\nResults saved to {OUTPUT_JSON!r}")
print("Done.")