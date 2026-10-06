import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
)
from seqeval.metrics import f1_score, classification_report
from scipy.stats import wilcoxon
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_MODEL   = "Davlan/afro-xlmr-base"   # pretrained on 17 langs incl. Sesotho
NER_DATASET  = "nwu-ctext/sesotho_ner_corpus"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
MLM_EPOCHS   = 3     # further pretraining epochs
NER_EPOCHS   = 5     # NER fine-tuning epochs
BATCH_SIZE   = 16
MAX_LEN      = 128

print(f"Device: {DEVICE}")
print(f"Base model: {BASE_MODEL}")

import zipfile, os

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load NWU Sesotho NER corpus from local file
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading NWU Sesotho NER corpus...")
print("  Source: NCHLT Sesotho Named Entity Annotated Corpus")
print("  Citation: Eiselen & Setaka (2016), SADiLaR")
print("█"*60)

# point this at the extracted txt file
NER_FILE = r"NCHLT Sesotho Named Entity Annotated Corpus\Dataset.NCHLT-II.st.NER.Full.txt"

# parse tab-separated CoNLL format
# format: word\tTAG per line, blank lines between sentences
tokens_list = []
tags_list   = []
current_tokens = []
current_tags   = []

with open(NER_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line == "":
            if current_tokens:
                tokens_list.append(current_tokens)
                tags_list.append(current_tags)
                current_tokens = []
                current_tags   = []
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_tags.append(parts[1].strip())

# flush last sentence
if current_tokens:
    tokens_list.append(current_tokens)
    tags_list.append(current_tags)

print(f"  ✓ {len(tokens_list)} sentences loaded")
print(f"  Sample tokens: {tokens_list[0][:5]}")
print(f"  Sample tags:   {tags_list[0][:5]}")

# build label set — matches the script:
# OUT, B-PERS, I-PERS, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
all_tags = sorted(set(tag for sent in tags_list for tag in sent))
print(f"  ✓ Labels: {all_tags}")

label2id = {l: i for i, l in enumerate(all_tags)}
id2label = {i: l for i, l in enumerate(all_tags)}
label_list = all_tags

# convert string tags to integers
int_tags = [[label2id[t] for t in sent] for sent in tags_list]

# build HuggingFace Dataset object
from datasets import Dataset
dataset = Dataset.from_dict({
    "id":       [str(i) for i in range(len(tokens_list))],
    "tokens":   tokens_list,
    "ner_tags": int_tags,
})

# 80/20 train/test split
split = dataset.train_test_split(test_size=0.2, seed=42)
train_ner = split["train"]
test_ner  = split["test"]

print(f"  ✓ Train: {len(train_ner)} | Test: {len(test_ner)}")
print(f"  ✓ Ready — continuing with rest of pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — Load tokenizer
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Loading tokenizer...")
print("█"*60)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LEN,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)  # subword tokens get -100
            prev_word_id = word_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

train_ner_tok = train_ner.map(tokenize_and_align_labels, batched=True)
test_ner_tok  = test_ner.map(tokenize_and_align_labels, batched=True)

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in labels]
    true_preds  = [[id2label[p] for p, l in zip(pred, label) if l != -100]
                   for pred, label in zip(preds, labels)]
    return {"f1": f1_score(true_labels, true_preds)}

data_collator = DataCollatorForTokenClassification(tokenizer)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: domain adaptive pretraining then NER fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(corpus_sentences, label, output_dir):
    print(f"\n  Running experiment: {label}")
    print(f"  Corpus size: {len(corpus_sentences)} sentences")

    # ── Step 1: Domain adaptive pretraining (MLM) ────────────────────────────
    print(f"  ├─ Step 1: Domain adaptive pretraining (MLM, {MLM_EPOCHS} epochs)...")

    from datasets import Dataset as HFDataset
    mlm_dataset = HFDataset.from_dict({"text": corpus_sentences})

    def tokenize_mlm(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )

    mlm_tok = mlm_dataset.map(tokenize_mlm, batched=True,
                               remove_columns=["text"])
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    mlm_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(DEVICE)

    mlm_args = TrainingArguments(
        output_dir=f"{output_dir}/mlm",
        num_train_epochs=MLM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        fp16=(DEVICE == "cuda"),
        seed=SEED,
    )

    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_args,
        train_dataset=mlm_tok,
        data_collator=mlm_collator,
    )
    mlm_trainer.train()
    print(f"  │   ✓ MLM pretraining complete")

    # save adapted model
    mlm_model.save_pretrained(f"{output_dir}/mlm_adapted")
    tokenizer.save_pretrained(f"{output_dir}/mlm_adapted")

    # ── Step 2: NER fine-tuning ───────────────────────────────────────────────
    print(f"  ├─ Step 2: NER fine-tuning ({NER_EPOCHS} epochs)...")

    ner_model = AutoModelForTokenClassification.from_pretrained(
        f"{output_dir}/mlm_adapted",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    ner_args = TrainingArguments(
        output_dir=f"{output_dir}/ner",
        num_train_epochs=NER_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=50,
        fp16=(DEVICE == "cuda"),
        seed=SEED,
    )

    ner_trainer = Trainer(
        model=ner_model,
        args=ner_args,
        train_dataset=train_ner_tok,
        eval_dataset=test_ner_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    ner_trainer.train()
    results = ner_trainer.evaluate()
    print(f"  │   ✓ NER F1: {round(results['eval_f1'], 4)}")

    # get per-sentence predictions for significance test
    preds_output = ner_trainer.predict(test_ner_tok)
    preds = np.argmax(preds_output.predictions, axis=2)
    labels_arr = preds_output.label_ids

    per_sent_f1 = []
    for pred_row, label_row in zip(preds, labels_arr):
        true = [id2label[l] for l in label_row if l != -100]
        pred = [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
        if true:
            sent_f1 = f1_score([true], [pred], zero_division=0)
            per_sent_f1.append(sent_f1)

    return results["eval_f1"], per_sent_f1

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — Build corpora
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Loading corpora...")
print("█"*60)

import io, requests

# Curated corpus
curated_df = pd.read_csv("sesotho_clean.csv", encoding="utf-8")
curated_sentences = curated_df["text"].tolist()
print(f"  ✓ Curated: {len(curated_sentences)} sentences")

# Raw corpus — same sources no filtering
raw_sentences = []
VUKU_API = ("https://api.github.com/repos/dsfsi/vukuzenzele-nlp"
            "/contents/data/simple_align_output")
try:
    r = requests.get(VUKU_API, timeout=30)
    for f in [x for x in r.json() if isinstance(x, dict)
              and x["name"].endswith(".csv")]:
        r2 = requests.get(f["download_url"], timeout=30)
        df_r = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip")
        if len(df_r.columns) >= 2:
            raw_sentences.extend(df_r.iloc[:, 1].dropna().astype(str).tolist())
except Exception as e:
    print(f"  ⚠ Vukuzenzele: {e}")

for url in [
    "https://zenodo.org/api/records/10531959/files/NewsSA.txt/content",
    "https://zenodo.org/api/records/10531959/files/NewsABSA.txt/content",
]:
    try:
        r = requests.get(url, timeout=60)
        raw_sentences.extend([l.strip() for l in r.text.split("\n") if l.strip()])
    except Exception:
        pass

# balance to same size
import random
random.seed(SEED)
if len(raw_sentences) > len(curated_sentences):
    raw_sentences = random.sample(raw_sentences, len(curated_sentences))
print(f"  ✓ Raw (balanced): {len(raw_sentences)} sentences")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Run both experiments
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Running experiments...")
print("  A: AfroXLMR + curated corpus → NER")
print("  B: AfroXLMR + raw corpus → NER")
print("█"*60)

f1_curated, per_sent_curated = run_experiment(
    curated_sentences, "Model A (curated)", "output_curated"
)
f1_raw, per_sent_raw = run_experiment(
    raw_sentences, "Model B (raw)", "output_raw"
)

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Statistical significance
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Statistical significance (Wilcoxon)...")
print("█"*60)

min_len = min(len(per_sent_curated), len(per_sent_raw))
stat, p_value = wilcoxon(per_sent_curated[:min_len], per_sent_raw[:min_len])
significant = p_value < 0.05
print(f"  Wilcoxon stat: {round(stat, 4)}")
print(f"  p-value:       {round(p_value, 6)}")
print(f"  Significant:   {'YES' if significant else 'NO'}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 6 — Results
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 6: Results")
print("█"*60)

improvement = f1_curated - f1_raw
print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  DOWNSTREAM EVALUATION: NER (NWU Sesotho NER Corpus)    ║
  ║  Model base: AfroXLMR-base (pretrained incl. Sesotho)   ║
  ║  Test set: {len(test_ner)} NER sentences (20% holdout)          ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Model A (curated corpus)   NER Macro F1 = {round(f1_curated,4)}      ║
  ║  Model B (raw corpus)       NER Macro F1 = {round(f1_raw,4)}      ║
  ║  Improvement: {round(improvement,4)} F1 points                         ║
  ║  Significant (p<0.05): {'YES' if significant else 'NO '}                            ║
  ║  p-value: {round(p_value,6)}                                   ║
  ╚══════════════════════════════════════════════════════════╝
""")

with open("ner_results.json", "w") as f:
    json.dump({
        "task": "NER",
        "benchmark": "NWU Sesotho NER Corpus",
        "base_model": BASE_MODEL,
        "test_sentences": len(test_ner),
        "curated_f1": round(f1_curated, 4),
        "raw_f1": round(f1_raw, 4),
        "improvement": round(improvement, 4),
        "wilcoxon_stat": round(stat, 4),
        "p_value": round(p_value, 6),
        "significant": bool(significant),
    }, f, indent=2)
print("  ✓ Saved to ner_results.json")