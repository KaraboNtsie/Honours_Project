"""
NER Domain Adaptation — Three-Way Comparison
=============================================
Run A : AfroXLMR-mini  →  NER fine-tuning (no MLM adaptation)
Run B : AfroXLMR-mini  →  MLM on curated corpus  →  NER fine-tuning
Run C : AfroXLMR-mini  →  MLM on NCHLT text corpus  →  NER fine-tuning

Wilcoxon signed-rank tests:
  A vs B  — does curated corpus adaptation help vs baseline?
  A vs C  — does NCHLT corpus adaptation help vs baseline?
  B vs C  — is curated corpus adaptation better than NCHLT?

Files needed (edit paths below):
  NER_FILE        — Dataset.NCHLT-II.st.NER.Full.txt
  CURATED_CSV     — sesotho_clean.csv  (your pipeline output)
  NCHLT_TEXT_FILE — CORP.NCHLT.st.CLEAN.2.0.txt
  ADAPTED_MODEL_B — path to already-trained MLM model for Run B
                    (set to None to re-run MLM; point to saved dir to skip)

Results saved to ner_results_three_way.json
"""

import os, json, random, subprocess, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score as seq_f1

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("transformers>=4.40", "datasets", "seqeval", "scipy", "accelerate")

# ── 1. Config ─────────────────────────────────────────────────────────────────
BASE_MODEL   = "Davlan/afro-xlmr-mini"

# Edit these paths to match your machine
NER_FILE        = os.path.join("NCHLT Sesotho Named Entity Annotated Corpus",
                                "Dataset.NCHLT-II.st.NER.Full.txt")
CURATED_CSV     = "sesotho_clean.csv"
NCHLT_TEXT_FILE = "CORP.NCHLT.st.CLEAN.2.0.txt"

# Point this to your already-trained Run B MLM model to skip re-training it.
# e.g. "ner_output\\mlm_adapted"  or  None to train from scratch.
ADAPTED_MODEL_B = os.path.join("ner_output", "mlm_adapted")

OUTPUT_DIR  = "ner_output_three_way"
OUTPUT_JSON = "ner_results_three_way.json"

NER_EPOCHS  = 5
MLM_EPOCHS  = 2
BATCH_SIZE  = 8
MAX_LEN     = 128
LR          = 2e-5
SEED        = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
print(f"Device: {DEVICE}")

# ── 2. Parse NWU NER corpus ───────────────────────────────────────────────────
def parse_conll(path):
    sentences, toks, labs = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                if toks:
                    sentences.append((toks, labs))
                    toks, labs = [], []
            else:
                parts = line.split()
                toks.append(parts[0])
                # remap OUT -> O so seqeval scores correctly
                labs.append("O" if parts[-1] == "OUT" else parts[-1])
    if toks:
        sentences.append((toks, labs))
    return sentences

all_sents = parse_conll(NER_FILE)
random.shuffle(all_sents)

n         = len(all_sents)
n_train   = int(0.8 * n)
n_val     = int(0.1 * n)
train_sents = all_sents[:n_train]
val_sents   = all_sents[n_train:n_train + n_val]
test_sents  = all_sents[n_train + n_val:]
print(f"NER corpus  train={len(train_sents)}  val={len(val_sents)}  test={len(test_sents)}")

all_labels = sorted({lab for _, labs in all_sents for lab in labs})
label2id   = {l: i for i, l in enumerate(all_labels)}
id2label   = {i: l for l, i in label2id.items()}
NUM_LABELS = len(all_labels)
print(f"Labels ({NUM_LABELS}): {all_labels}")

# ── 3. MLM dataset helper ─────────────────────────────────────────────────────
class MLMDataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def load_text_lines(path, max_lines=50_000):
    """Load plain-text sentences from a .txt file, one per line."""
    lines = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # skip blank lines, XML-ish tags, and licence header
            if not line or line.startswith("<") or line.startswith("_"):
                continue
            if 5 <= len(line.split()) <= 100:
                lines.append(line)
            if len(lines) >= max_lines:
                break
    print(f"  Loaded {len(lines)} sentences from {path}")
    return lines


def load_csv_sentences(csv_path, max_lines=50_000):
    """Load sentences from a CSV (curated corpus)."""
    df  = pd.read_csv(csv_path)
    col = next((c for c in ("sentence", "text", "Sentence", "Text")
                if c in df.columns), df.columns[0])
    texts = df[col].dropna().astype(str).tolist()
    texts = [t for t in texts if 5 <= len(t.split()) <= 100]
    print(f"  Loaded {len(texts)} sentences from {csv_path}")
    return texts[:max_lines]


def run_mlm(corpus_sentences, save_dir, base_model=BASE_MODEL):
    """Continue MLM pretraining on corpus_sentences; save to save_dir."""
    if os.path.isdir(save_dir) and any(Path(save_dir).iterdir()):
        print(f"  MLM model found at {save_dir} — skipping MLM training.")
        return save_dir

    print(f"  Starting MLM pretraining on {len(corpus_sentences)} sentences...")
    tok = AutoTokenizer.from_pretrained(base_model)

    encodings = tok(
        corpus_sentences,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors=None,
    )

    dataset  = MLMDataset(encodings)
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)

    model = AutoModelForMaskedLM.from_pretrained(base_model).to(DEVICE)

    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=MLM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        fp16=USE_FP16,
        logging_steps=200,
        save_strategy="epoch",
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print(f"  MLM model saved to {save_dir}")

    del model
    torch.cuda.empty_cache()
    return save_dir

# ── 4. Tokenise + align NER labels ───────────────────────────────────────────
def tokenise_and_align(sentence_pairs, tok, max_len=MAX_LEN):
    encoded = []
    for tokens, labels in sentence_pairs:
        enc = tok(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        word_ids  = enc.word_ids()
        label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                label_ids.append(label2id[labels[wid]])
            else:
                label_ids.append(-100)
            prev_word = wid
        enc["labels"] = label_ids
        encoded.append(dict(enc))
    return encoded

# ── 5. NER Dataset wrapper ────────────────────────────────────────────────────
class NERDataset(TorchDataset):
    def __init__(self, encoded):
        self.data = encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}

# ── 6. Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(preds, label_ids):
        t, p = [], []
        for p_id, l_id in zip(pred_row, label_row):
            if l_id == -100:
                continue
            t.append(id2label[l_id])
            p.append(id2label[p_id])
        true_seqs.append(t)
        pred_seqs.append(p)
    return {"f1": seq_f1(true_seqs, pred_seqs)}


def per_sentence_f1(model, encoded_test, tok):
    model.eval()
    scores   = []
    collator = DataCollatorForTokenClassification(tok, padding=True)
    with torch.no_grad():
        for item in encoded_test:
            batch = collator([{k: torch.tensor(v) for k, v in item.items()}])
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            preds = out.logits.argmax(-1)[0].cpu().numpy()
            labs  = batch["labels"][0].cpu().numpy()
            t, p  = [], []
            for pi, li in zip(preds, labs):
                if li == -100:
                    continue
                t.append(id2label[li])
                p.append(id2label[pi])
            scores.append(seq_f1([t], [p]))
    return np.array(scores)

# ── 7. NER fine-tuning helper ─────────────────────────────────────────────────
def run_ner(run_name, model_path):
    """Fine-tune NER from model_path; return (overall_f1, per_sent_f1_list)."""
    print(f"\n{'='*60}")
    print(f"NER Run {run_name}  (base: {model_path})")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_path)

    enc_train = tokenise_and_align(train_sents, tok)
    enc_val   = tokenise_and_align(val_sents,   tok)
    enc_test  = tokenise_and_align(test_sents,  tok)

    ds_train = NERDataset(enc_train)
    ds_val   = NERDataset(enc_val)

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    out_dir = os.path.join(OUTPUT_DIR, f"ner_{run_name}")
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=NER_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=USE_FP16,
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    collator = DataCollatorForTokenClassification(tok)
    trainer  = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_ds    = NERDataset(enc_test)
    results    = trainer.evaluate(test_ds)
    overall_f1 = results.get("eval_f1", float("nan"))
    print(f"  Overall test F1 = {overall_f1:.4f}")

    sent_f1s = per_sentence_f1(model, enc_test, tok)
    print(f"  Mean per-sentence F1 = {sent_f1s.mean():.4f}")

    del model
    torch.cuda.empty_cache()

    return overall_f1, sent_f1s.tolist()

# ── 8. MLM phase ──────────────────────────────────────────────────────────────
print("\n── MLM Phase ────────────────────────────────────────────────")

# Run B: curated corpus (reuse saved model if it exists)
mlm_dir_B = ADAPTED_MODEL_B if ADAPTED_MODEL_B else os.path.join(OUTPUT_DIR, "mlm_curated")
if ADAPTED_MODEL_B and os.path.isdir(ADAPTED_MODEL_B):
    print(f"Run B: reusing existing MLM model at {ADAPTED_MODEL_B}")
    mlm_dir_B = ADAPTED_MODEL_B
else:
    curated_sents = load_csv_sentences(CURATED_CSV)
    mlm_dir_B = run_mlm(curated_sents, os.path.join(OUTPUT_DIR, "mlm_curated"))

# Run C: NCHLT text corpus (always trains fresh)
nchlt_sents = load_text_lines(NCHLT_TEXT_FILE)
mlm_dir_C   = run_mlm(nchlt_sents, os.path.join(OUTPUT_DIR, "mlm_nchlt"))

# ── 9. NER phase ──────────────────────────────────────────────────────────────
print("\n── NER Phase ────────────────────────────────────────────────")

f1_A, scores_A = run_ner("A_baseline",       BASE_MODEL)
f1_B, scores_B = run_ner("B_curated_adapt",  mlm_dir_B)
f1_C, scores_C = run_ner("C_nchlt_adapt",    mlm_dir_C)

# ── 10. Wilcoxon tests ────────────────────────────────────────────────────────
arr_A = np.array(scores_A)
arr_B = np.array(scores_B)
arr_C = np.array(scores_C)

stat_AB, p_AB = wilcoxon(arr_A, arr_B, alternative="two-sided")
stat_AC, p_AC = wilcoxon(arr_A, arr_C, alternative="two-sided")
stat_BC, p_BC = wilcoxon(arr_B, arr_C, alternative="two-sided")

print("\n── Wilcoxon Results ──────────────────────────────────────────")
print(f"  A (baseline)      mean F1 : {arr_A.mean():.4f}")
print(f"  B (curated adapt) mean F1 : {arr_B.mean():.4f}")
print(f"  C (NCHLT adapt)   mean F1 : {arr_C.mean():.4f}")
print(f"  A vs B  stat={stat_AB:.1f}  p={p_AB:.4f}")
print(f"  A vs C  stat={stat_AC:.1f}  p={p_AC:.4f}")
print(f"  B vs C  stat={stat_BC:.1f}  p={p_BC:.4f}")

# ── 11. Save results ──────────────────────────────────────────────────────────
output = {
    "run_A_baseline": {
        "overall_f1": float(f1_A),
        "mean_per_sent_f1": float(arr_A.mean()),
        "per_sent_f1": scores_A,
    },
    "run_B_curated_adapt": {
        "overall_f1": float(f1_B),
        "mean_per_sent_f1": float(arr_B.mean()),
        "per_sent_f1": scores_B,
    },
    "run_C_nchlt_adapt": {
        "overall_f1": float(f1_C),
        "mean_per_sent_f1": float(arr_C.mean()),
        "per_sent_f1": scores_C,
    },
    "wilcoxon_A_vs_B": {"statistic": float(stat_AB), "p_value": float(p_AB)},
    "wilcoxon_A_vs_C": {"statistic": float(stat_AC), "p_value": float(p_AC)},
    "wilcoxon_B_vs_C": {"statistic": float(stat_BC), "p_value": float(p_BC)},
    "config": {
        "base_model": BASE_MODEL,
        "ner_epochs": NER_EPOCHS,
        "mlm_epochs": MLM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "lr": LR,
        "seed": SEED,
        "nchlt_text_sentences": len(nchlt_sents),
    },
}

with open(OUTPUT_JSON, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"\nResults saved to {OUTPUT_JSON}")
print("Done.")