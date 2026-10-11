"""
NER Domain Adaptation — Zabantu runs (D, E, F) + v1 AfroXLMR results (A, B, C)
=============================================================================
Run D : Zabantu-XLM-RoBERTa →  NER fine-tuning (no MLM adaptation)
Run E : Zabantu-XLM-RoBERTa →  MLM on curated corpus  →  NER fine-tuning
Run F : Zabantu-XLM-RoBERTa →  MLM on NCHLT text corpus  →  NER fine-tuning

Runs A, B, C (AfroXLMR-mini) are NOT retrained. Their per-sentence scores are
loaded from your v1 results file (ner_results_three_way.json). Same NER file +
same seed (42) = same 948 test sentences, so the paired tests stay valid.

Disk-friendly:
  - each finished run is saved to run_results/<run>.json straight away
  - rerunning the script skips any run that already has a saved result
  - MLM and NER model folders are deleted as soon as a run is finished

Results saved to /kaggle/working/ner_results_six_way.json
"""

import os, json, random, subprocess, sys, glob, shutil

# Kaggle's "GPU T4 x2" gives two GPUs. Use only one so the effective batch
# size stays 8, matching your v1 Colab run. Must be set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
def seq_f1(true_seqs, pred_seqs):
    """Token-level entity F1 (replaces seqeval, no build deps)."""
    tp, fp, fn = 0, 0, 0
    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        for t, p in zip(true_seq, pred_seq):
            if t != "O":
                if t == p:
                    tp += 1
                else:
                    fn += 1
            if p != "O" and t != p:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("transformers", "datasets", "scipy", "accelerate")

# ── 1. Config ─────────────────────────────────────────────────────────────────
BASE_MODEL    = "Davlan/afro-xlmr-mini"
ZABANTU_MODEL = "dsfsi/zabantu-xlm-roberta"

# Kaggle: finds each file by name, whether it's in your cloned repo
# (/kaggle/working/...) or an uploaded dataset (/kaggle/input/...).
def find_input(filename):
    hits = []
    for root in ("/kaggle/working", "/kaggle/input"):
        hits += glob.glob(f"{root}/**/{filename}", recursive=True)
    if not hits:
        raise FileNotFoundError(
            f"Could not find {filename} under /kaggle/working or /kaggle/input. "
            "Check it's in your cloned repo or added via Add Input.")
    hits.sort(key=len)   # prefer the shortest (least nested) path
    print(f"  Found {filename} -> {hits[0]}")
    return hits[0]

NER_FILE        = find_input("Dataset.NCHLT-II.st.NER.Full.txt")
CURATED_CSV     = find_input("sesotho_clean.csv")
NCHLT_TEXT_FILE = find_input("CORP.NCHLT.st.CLEAN.2.0.txt")

# v1 AfroXLMR results (Runs A, B, C). Put ner_results_three_way.json in your
# repo or upload it with Add Input.
V1_RESULTS_JSON = find_input("ner_results_three_way.json")

# Kaggle: only /kaggle/working is writable (and saved as notebook output).
OUTPUT_DIR   = "/kaggle/working/ner_output_zabantu"     # temporary model files
RUN_RESULTS  = "/kaggle/working/run_results"            # one small JSON per finished run
OUTPUT_JSON  = "/kaggle/working/ner_results_six_way.json"
os.makedirs(RUN_RESULTS, exist_ok=True)

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

    print(f"  Starting MLM pretraining ({base_model}) on {len(corpus_sentences)} sentences...")
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
        save_strategy="no",          # final model is saved below; no checkpoints
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
        save_total_limit=1,
        save_only_model=True,        # skip optimizer state: ~3x smaller checkpoints
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

    del model, trainer
    torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)   # NER checkpoints no longer needed

    return overall_f1, sent_f1s.tolist()

# ── 8. Helpers: disk report + per-run result cache ───────────────────────────
def disk_report(label=""):
    total, used, free = shutil.disk_usage("/kaggle/working")
    print(f"  [disk{(' ' + label) if label else ''}] free: {free / 1e9:.1f} GB")


def result_path(run_name):
    return os.path.join(RUN_RESULTS, f"{run_name}.json")


def do_run(run_name, base_model, corpus_sentences=None, mlm_name=None):
    """MLM (optional) -> NER -> save result -> delete model files.
    Skips everything if this run already has a saved result."""
    rp = result_path(run_name)
    if os.path.exists(rp):
        print(f"\n{run_name}: result already saved at {rp} — skipping.")
        with open(rp) as fh:
            r = json.load(fh)
        return r["overall_f1"], r["per_sent_f1"]

    disk_report(f"before {run_name}")
    model_path, mlm_dir = base_model, None
    if corpus_sentences is not None:
        mlm_dir = os.path.join(OUTPUT_DIR, mlm_name)
        model_path = run_mlm(corpus_sentences, mlm_dir, base_model=base_model)

    f1, scores = run_ner(run_name, model_path)

    with open(rp, "w") as fh:
        json.dump({"overall_f1": float(f1), "per_sent_f1": scores}, fh)
    print(f"  Saved {run_name} result to {rp}")

    if mlm_dir:
        shutil.rmtree(mlm_dir, ignore_errors=True)   # adapted model no longer needed
    disk_report(f"after {run_name}")
    return f1, scores

# ── 9. Load v1 AfroXLMR results (A, B, C) ────────────────────────────────────
print("\n── Loading v1 AfroXLMR results ─────────────────────────────")
with open(V1_RESULTS_JSON) as fh:
    v1 = json.load(fh)
V1_KEYS = {"A": "run_A_baseline", "B": "run_B_curated_adapt", "C": "run_C_nchlt_adapt"}
f1s, scores = {}, {}
for k, key in V1_KEYS.items():
    f1s[k]    = v1[key]["overall_f1"]
    scores[k] = v1[key]["per_sent_f1"]
    print(f"  {k}: overall F1 = {f1s[k]:.4f}  ({len(scores[k])} test sentences)")

if len(scores["A"]) != len(test_sents):
    raise ValueError(
        f"v1 results have {len(scores['A'])} test sentences but this split has "
        f"{len(test_sents)}. The NER file or seed differs from v1, so the paired "
        "A/B/C vs D/E/F comparisons would be invalid.")

# ── 10. Zabantu runs (D, E, F) ───────────────────────────────────────────────
print("\n── Loading corpora ──────────────────────────────────────────")
curated_sents = load_csv_sentences(CURATED_CSV)
nchlt_sents   = load_text_lines(NCHLT_TEXT_FILE)

print("\n── Zabantu runs ─────────────────────────────────────────────")
f1s["D"], scores["D"] = do_run("D_zabantu_baseline", ZABANTU_MODEL)
f1s["E"], scores["E"] = do_run("E_zabantu_curated",  ZABANTU_MODEL,
                               curated_sents, "mlm_zabantu_curated")
f1s["F"], scores["F"] = do_run("F_zabantu_nchlt",    ZABANTU_MODEL,
                               nchlt_sents, "mlm_zabantu_nchlt")

# ── 11. Wilcoxon tests ────────────────────────────────────────────────────────
arr = {k: np.array(v) for k, v in scores.items()}

PAIRS = [
    ("A", "B"), ("A", "C"), ("B", "C"),   # within AfroXLMR (v1)
    ("D", "E"), ("D", "F"), ("E", "F"),   # within Zabantu
    ("A", "D"), ("B", "E"), ("C", "F"),   # AfroXLMR vs Zabantu, same corpus setting
]
wil = {}
for x, y in PAIRS:
    stat, p = wilcoxon(arr[x], arr[y], alternative="two-sided")
    wil[f"{x}_vs_{y}"] = (float(stat), float(p))

NAMES = {
    "A": "AfroXLMR baseline",      "B": "AfroXLMR + curated MLM", "C": "AfroXLMR + NCHLT MLM",
    "D": "Zabantu baseline",       "E": "Zabantu + curated MLM",  "F": "Zabantu + NCHLT MLM",
}
print("\n── Results ───────────────────────────────────────────────────")
for k in "ABCDEF":
    print(f"  {k} ({NAMES[k]:24s})  overall F1 = {f1s[k]:.4f}   mean per-sent F1 = {arr[k].mean():.4f}")
print()
for key, (stat, p) in wil.items():
    sig = "significant" if p < 0.05 else "not significant"
    print(f"  {key.replace('_', ' '):8s}  stat={stat:.1f}  p={p:.4f}  ({sig})")

# ── 12. Save combined results ─────────────────────────────────────────────────
RUN_KEYS = {
    "A": ("run_A_baseline",         BASE_MODEL,    None),
    "B": ("run_B_curated_adapt",    BASE_MODEL,    "curated"),
    "C": ("run_C_nchlt_adapt",      BASE_MODEL,    "nchlt"),
    "D": ("run_D_zabantu_baseline", ZABANTU_MODEL, None),
    "E": ("run_E_zabantu_curated",  ZABANTU_MODEL, "curated"),
    "F": ("run_F_zabantu_nchlt",    ZABANTU_MODEL, "nchlt"),
}
output = {}
for k, (key, model, corpus) in RUN_KEYS.items():
    e = {"model": model}
    if corpus:
        e["mlm_corpus"] = corpus
    e.update({
        "overall_f1": float(f1s[k]),
        "mean_per_sent_f1": float(arr[k].mean()),
        "per_sent_f1": list(scores[k]),
        "source": "v1 Colab run" if k in "ABC" else "this run (Kaggle)",
    })
    output[key] = e
for key, (stat, p) in wil.items():
    output[f"wilcoxon_{key}"] = {"statistic": stat, "p_value": p}

output["config"] = {
    "base_model_abc": BASE_MODEL,
    "base_model_def": ZABANTU_MODEL,
    "ner_epochs": NER_EPOCHS,
    "mlm_epochs": MLM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_len": MAX_LEN,
    "lr": LR,
    "seed": SEED,
    "test_sentences": len(test_sents),
    "curated_sentences": len(curated_sents),
    "nchlt_text_sentences": len(nchlt_sents),
}

with open(OUTPUT_JSON, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"\nResults saved to {OUTPUT_JSON}")
print("Done.")
