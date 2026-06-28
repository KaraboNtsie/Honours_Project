import pandas as pd
import requests
import io
import unicodedata
import re
import fasttext
from huggingface_hub import hf_hub_download

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 0 — Environment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 0: Checking environment...")
print("█"*60)
try:
    import fasttext, pandas, sklearn
    print("  ✓ All libraries imported successfully")
except ImportError as e:
    print(f"  ✗ Missing library: {e}"); exit()

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load GlotLID
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading GlotLID model...")
print("█"*60)
try:
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    model = fasttext.load_model(model_path)
    print("  ✓ GlotLID loaded")
except Exception as e:
    print(f"  ✗ Failed: {e}"); exit()

def verify_sesotho(sentences, confidence_threshold=0.8):
    verified = []
    for sentence in sentences:
        clean = sentence.replace("\n", " ").strip()
        if not clean or len(clean.split()) < 3:
            continue
        labels, scores = model.predict(clean, k=1)
        lang = labels[0].replace("__label__", "")
        score = float(scores[0])
        if lang == "sot_Latn" and score >= confidence_threshold:
            verified.append((clean, round(score, 4)))
    return verified

def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — LS data: Mokhosi SN dataset (Zenodo)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Downloading LS data — Mokhosi SN dataset (Zenodo)")
print("█"*60)

ls_sentences = []
try:
    url = "https://zenodo.org/api/records/10531959/files/NewsSA.txt/content"
    print(f"  ├─ Fetching NewsSA.txt...")
    r = requests.get(url, timeout=60)
    lines = r.text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]
    print(f"  │   ✓ {len(lines)} raw lines")
    print(f"  │   Sample lines:")
    for line in lines[:5]:
        print(f"  │     '{line}'")
    ls_sentences = lines
    print(f"  ✓ Total LS candidates: {len(ls_sentences)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — SAS data: Vukuzenzele simple_align_output (GitHub)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Downloading SAS data — Vukuzenzele simple_align_output")
print("█"*60)

sas_sentences = []

try:
    api_url = "https://api.github.com/repos/dsfsi/vukuzenzele-nlp/contents/data/simple_align_output"
    r = requests.get(api_url, timeout=30)
    items = r.json()

    if isinstance(items, dict) and items.get("message") == "Not Found":
        raise Exception("simple_align_output not found — trying opt_aligned_out")

    csv_files = [f for f in items if f["type"] == "file" and f["name"].endswith(".csv")]
    print(f"  ✓ Found {len(csv_files)} CSV files in simple_align_output")

    for f in csv_files[:10]:  # first 10 files is plenty
        print(f"\n  ├─ {f['name']}")
        try:
            r2 = requests.get(f["download_url"], timeout=30)
            df_raw = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip")
            print(f"  │   Columns: {list(df_raw.columns)}")
            print(f"  │   Rows: {len(df_raw)}")
            print(f"  │   Sample:")
            print(df_raw.head(2).to_string())

            # find Sesotho column
            for col in df_raw.columns:
                if any(x in col.lower() for x in ["sot", "sesotho", "st", "target"]):
                    lines = df_raw[col].dropna().astype(str).tolist()
                    print(f"  │   ✓ Using column '{col}': {len(lines)} sentences")
                    sas_sentences.extend(lines)
                    break
            else:
                # no match — show all and take second column
                # (aligned CSVs are usually English | Sesotho)
                if len(df_raw.columns) >= 2:
                    col = df_raw.columns[1]
                    lines = df_raw[col].dropna().astype(str).tolist()
                    print(f"  │   ⚠ No Sesotho column found — taking column[1] '{col}'")
                    sas_sentences.extend(lines)

        except Exception as e2:
            print(f"  │   ⚠ Failed: {e2}")

    print(f"\n  ✓ Total SAS candidates: {len(sas_sentences)}")

except Exception as e:
    print(f"  ✗ simple_align_output failed: {e}")

    # fallback: try opt_aligned_out
    print("  ⚠ Trying opt_aligned_out instead...")
    try:
        api_url = "https://api.github.com/repos/dsfsi/vukuzenzele-nlp/contents/data/opt_aligned_out"
        r = requests.get(api_url, timeout=30)
        items = r.json()
        csv_files = [f for f in items if f["type"] == "file" and f["name"].endswith(".csv")]
        print(f"  ✓ Found {len(csv_files)} CSV files in opt_aligned_out")

        for f in csv_files[:10]:
            print(f"\n  ├─ {f['name']}")
            r2 = requests.get(f["download_url"], timeout=30)
            df_raw = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip")
            print(f"  │   Columns: {list(df_raw.columns)}")
            print(df_raw.head(2).to_string())

            for col in df_raw.columns:
                if any(x in col.lower() for x in ["sot", "sesotho", "st", "target"]):
                    lines = df_raw[col].dropna().astype(str).tolist()
                    sas_sentences.extend(lines)
                    print(f"  │   ✓ Column '{col}': {len(lines)} sentences")
                    break
            else:
                if len(df_raw.columns) >= 2:
                    col = df_raw.columns[1]
                    lines = df_raw[col].dropna().astype(str).tolist()
                    sas_sentences.extend(lines)
                    print(f"  │   ⚠ Using column[1] '{col}': {len(lines)} sentences")

        print(f"\n  ✓ Total SAS candidates: {len(sas_sentences)}")

    except Exception as e3:
        print(f"  ✗ opt_aligned_out also failed: {e3}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — GlotLID verification
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: GlotLID verification (threshold: 0.8)...")
print("█"*60)

print(f"\n  Verifying {len(ls_sentences)} LS candidates...")
ls_verified = verify_sesotho([clean_text(s) for s in ls_sentences], confidence_threshold=0.8)
print(f"  ✓ {len(ls_verified)} LS sentences passed")

print(f"\n  Verifying {len(sas_sentences)} SAS candidates...")
sas_verified = verify_sesotho([clean_text(s) for s in sas_sentences], confidence_threshold=0.8)
print(f"  ✓ {len(sas_verified)} SAS sentences passed")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Build and save labelled dataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Building labelled dataset...")
print("█"*60)

all_data = []

for text, conf in ls_verified[:200]:
    all_data.append({
        "text": text,
        "label": "LS",
        "lid_confidence": conf,
        "source": "Mokhosi_SN_Zenodo"
    })

for text, conf in sas_verified[:200]:
    all_data.append({
        "text": text,
        "label": "SAS",
        "lid_confidence": conf,
        "source": "Vukuzenzele_DSFSI"
    })

if not all_data:
    print("  ✗ No data collected. Check errors above.")
    exit()

df = pd.DataFrame(all_data)
df = df.drop_duplicates(subset="text")
df = df[df["text"].str.split().str.len() >= 3]
df.to_csv("sesotho_clean.csv", index=False, encoding="utf-8")

print(f"  ✓ Total sentences saved: {len(df)}")
print(f"\n  Label distribution:")
print(df["label"].value_counts().to_string(header=False))
print(f"\n  Average LID confidence: {df['lid_confidence'].mean():.4f}")
print(f"\n  Sample LS:")
for t in df[df["label"]=="LS"]["text"].head(3).tolist():
    print(f"    '{t}'")
print(f"\n  Sample SAS:")
for t in df[df["label"]=="SAS"]["text"].head(3).tolist():
    print(f"    '{t}'")

print("\n" + "█"*60)
print("  PIPELINE STAGES 1+2 COMPLETE")
print("  Next: run orthographic_classifier.py")
print("█"*60)