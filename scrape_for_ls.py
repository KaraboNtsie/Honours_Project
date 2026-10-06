import re
import time
import unicodedata
import requests
import pandas as pd
import fasttext
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load GlotLID
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading GlotLID...")
print("█"*60)
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)
print("  ✓ GlotLID loaded")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
ENGLISH_MARKERS = {
    'the','and','for','are','but','not','you','all','can','was',
    'one','our','out','had','her','his','him','its','may','did',
    'how','who','will','that','this','they','from','been','have',
    'with','into','also','when','there','their','what','which',
    'would','about','could','other','were','your','than','then',
    'them','some','time','very','just','over','such','well','only',
    'come','here','know','take','even','back','good','much','more',
}

def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text):
    raw = re.split(r"[.!?\n]", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 5]

def is_code_switched(text, max_english=2):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    hits = [w for w in words if w in ENGLISH_MARKERS]
    return len(hits) > max_english

def is_quality(text):
    if not text or len(text.split()) < 5:
        return False
    if is_code_switched(text):
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False
    return True

def verify_sesotho(sentence, threshold=0.7):
    clean = sentence.replace("\n", " ").strip()
    labels, scores = model.predict(clean, k=1)
    lang = labels[0].replace("__label__", "")
    score = float(scores[0])
    return lang == "sot_Latn" and score >= threshold, round(score, 4)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

all_data = []

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — LS: OPUS JW300 Sesotho (Lesotho orthography)
# Uses opustools to download and extract LS Sesotho sentences
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Downloading LS data from OPUS JW300...")
print("  JW300 Sesotho uses LS orthography — confirmed by marker analysis")
print("█"*60)

try:
    from opustools import OpusRead

    print("  ├─ Downloading JW300 Sesotho-English parallel corpus...")
    print("  │   This downloads ~5MB — takes 1-2 minutes...")

    # write Sesotho sentences to a temp file
    opus_reader = OpusRead(
        corpus_name="JW300",
        source="st",
        target="en",
        write=["opus_st.txt", "opus_en.txt"],
        download_dir="./opus_data",
        suppress_prompts=True,
        leave_non_alignments_out=True,
        write_mode="normal"
    )
    opus_reader.printPairs()

    # read the Sesotho sentences
    with open("opus_st.txt", "r", encoding="utf-8") as f:
        opus_lines = [l.strip() for l in f.readlines() if l.strip()]

    print(f"  │   ✓ {len(opus_lines)} raw lines from JW300")

    ls_opus_count = 0
    for line in opus_lines:
        line = clean_text(line)
        if not is_quality(line):
            continue
        ok, conf = verify_sesotho(line)
        if ok:
            all_data.append({
                "text": line,
                "label": "LS",
                "lid_confidence": conf,
                "source": "JW300_OPUS_LS",
                "source_url": "https://opus.nlpl.eu/JW300"
            })
            ls_opus_count += 1

    print(f"  ✓ LS from JW300: {ls_opus_count} verified sentences")

except ImportError:
    print("  ✗ opustools not installed")
    print("  Run: pip install opustools")
    print("  Then rerun this script")

except Exception as e:
    print(f"  ✗ OPUS failed: {e}")
    print("  Falling back to Mokhosi full dataset...")

    # fallback — get full Mokhosi dataset properly
    # the issue was numeric labels mixed in — filter those out
    ZENODO_URLS = [
        "https://zenodo.org/api/records/10531959/files/NewsSA.txt/content",
        "https://zenodo.org/api/records/10531959/files/NewsABSA.txt/content",
    ]
    mokhosi_count = 0
    for url in ZENODO_URLS:
        try:
            r = requests.get(url, timeout=60)
            lines = [l.strip() for l in r.text.split("\n") if l.strip()]
            # filter out numeric-only lines (sentiment labels)
            lines = [l for l in lines if not re.match(r'^[\d\s\.,\-]+$', l)]
            print(f"  │   {url.split('/')[-2]}: {len(lines)} non-numeric lines")
            for line in lines:
                line = clean_text(line)
                if not is_quality(line):
                    continue
                ok, conf = verify_sesotho(line)
                if ok:
                    all_data.append({
                        "text": line, "label": "LS",
                        "lid_confidence": conf,
                        "source": "Mokhosi_SN_Zenodo",
                        "source_url": url
                    })
                    mokhosi_count += 1
        except Exception as e2:
            print(f"  │   ✗ {e2}")
    print(f"  ✓ Mokhosi fallback: {mokhosi_count} sentences")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — LS: Lesotho government budget speeches
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Lesotho government sources (LS)...")
print("█"*60)

LS_GOV_URLS = [
    "https://www.finance.gov.ls",
    "https://www.gov.ls/news/",
    "https://www.gov.ls/speeches/",
    "https://www.gov.ls/statements/",
    "https://www.lndc.org.ls",
    "https://www.centralbank.org.ls",
]

for url in LS_GOV_URLS:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  │   ⚠ {url}: HTTP {r.status_code}")
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        raw = " ".join(p.get_text() for p in soup.find_all("p"))
        cleaned = clean_text(raw)
        sentences = split_sentences(cleaned)
        added = 0
        for s in sentences:
            if not is_quality(s):
                continue
            ok, conf = verify_sesotho(s)
            if ok:
                all_data.append({
                    "text": s, "label": "LS",
                    "lid_confidence": conf,
                    "source": "Lesotho_Gov_LS",
                    "source_url": url
                })
                added += 1
        print(f"  ├─ {url.split('/')[2]}: {added} sentences")
        time.sleep(1)
    except Exception as e:
        print(f"  ├─ {url.split('/')[2]}: ✗ {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Merge with existing corpus
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Merging with existing corpus...")
print("█"*60)

if not all_data:
    print("  ✗ No data collected — check errors above")
    exit()

new_df = pd.DataFrame(all_data)
new_df = new_df.drop_duplicates(subset="text")
new_df = new_df[new_df["text"].str.split().str.len() >= 5]
print(f"  New sentences: {len(new_df)}")
print(new_df["label"].value_counts().to_string(header=False))

try:
    existing_df = pd.read_csv("sesotho_clean.csv", encoding="utf-8")
    print(f"\n  Existing corpus: {len(existing_df)} sentences")
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="text")
except FileNotFoundError:
    combined = new_df

combined.to_csv("sesotho_clean.csv", index=False, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 6 — Final report
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 6: Final corpus report")
print("█"*60)
print(f"\n  Total: {len(combined)} sentences")
print(f"\n  Distribution:")
print(combined["label"].value_counts().to_string(header=False))
print(f"\n  Sources:")
print(combined.groupby(["label", "source"]).size().to_string())
print(f"\n  Gap to 2500 per variant:")
counts = combined["label"].value_counts()
for variant in ["SAS", "LS"]:
    count = counts.get(variant, 0)
    gap = max(0, 2500 - count)
    bar = "█" * min(int(count/2500*20), 20) + "░" * max(0, 20-int(count/2500*20))
    print(f"  {variant}: [{bar}] {count}/2500  (need {gap} more)")
print(f"\n  ✓ Saved to sesotho_clean.csv")
print("█"*60)