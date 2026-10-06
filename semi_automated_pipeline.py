import time
import re
import unicodedata
import io
import requests
import pandas as pd
import fasttext
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 0 — Environment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 0: Checking environment...")
print("█"*60)
try:
    import fasttext, pandas, sklearn, bs4
    print("  ✓ All libraries imported")
except ImportError as e:
    print(f"  ✗ Missing: {e}"); exit()

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load GlotLID
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading GlotLID...")
print("█"*60)
try:
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    model = fasttext.load_model(model_path)
    print("  ✓ GlotLID loaded")
except Exception as e:
    print(f"  ✗ Failed: {e}"); exit()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — cleaning, filtering, verification
# ══════════════════════════════════════════════════════════════════════════════

# Unambiguously English words (3+ chars) that cannot be Sesotho morphemes.
# Single chars like 'a', 'o', 'e', 'i' are valid Sesotho and excluded.
ENGLISH_MARKERS = {
    'the','and','for','are','but','not','you','all','can','was',
    'one','our','out','had','her','his','him','its','may','did',
    'how','who','will','that','this','they','from','been','have',
    'with','into','also','when','there','their','what','which',
    'would','about','could','other','were','your','than','then',
    'them','some','time','very','just','over','such','well','only',
    'come','here','know','take','even','back','good','much','more',
    'does','said','each','made','most','after','both','through',
    'where','being','during','between','should','these','those',
    'however','therefore','although','because','whether','without',
}

def has_latex_artifact(text):
    """Catches $T$, $X$ LaTeX anonymisation placeholders."""
    return bool(re.search(r'\$[A-Za-z]\$', text))

def is_code_switched(text, max_english=2):
    """
    Flags sentences with more than max_english unambiguous English words.
    Only matches words of 3+ characters to avoid Sesotho morphemes.
    """
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    hits = [w for w in words if w in ENGLISH_MARKERS]
    return len(hits) > max_english

def clean_text(text):
    """NFC normalise, strip URLs, collapse whitespace."""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"&[a-z]+;", "", text)       # HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text):
    """Split on terminal punctuation, keep segments of 5+ words."""
    raw = re.split(r"[.!?\n]", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 5]

def is_quality(text):
    """All quality gates in one call. Returns True if sentence is usable."""
    if not text or len(text.split()) < 5:
        return False
    if has_latex_artifact(text):
        return False
    if is_code_switched(text):
        return False
    # reject if mostly digits/punctuation
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False
    return True

def verify_sesotho(sentence, threshold=0.7):
    """Returns (is_sesotho: bool, confidence: float)."""
    clean = sentence.replace("\n", " ").strip()
    labels, scores = model.predict(clean, k=1)
    lang = labels[0].replace("__label__", "")
    score = float(scores[0])
    return lang == "sot_Latn" and score >= threshold, round(score, 4)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def fetch_and_collect(url, label, source_name, selector=None, skip_patterns=None):
    """
    Fetch a URL, extract text, run quality + LID filters.
    Returns list of dicts ready for DataFrame.
    """
    results = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  │   ⚠ HTTP {r.status_code}")
            return results
        soup = BeautifulSoup(r.text, "html.parser")

        if selector:
            divs = soup.find_all("div", class_=selector)
            paragraphs = []
            for div in divs:
                for p in div.find_all("p"):
                    paragraphs.append(p.get_text())
        else:
            paragraphs = [p.get_text() for p in soup.find_all("p")]

        if skip_patterns:
            paragraphs = [p for p in paragraphs
                          if not any(sp in p.lower() for sp in skip_patterns)]

        raw_text = " ".join(paragraphs)
        cleaned = clean_text(raw_text)
        sentences = split_sentences(cleaned)

        for sentence in sentences:
            sentence = clean_text(sentence)
            if not is_quality(sentence):
                continue
            ok, conf = verify_sesotho(sentence)
            if ok:
                results.append({
                    "text": sentence,
                    "label": label,
                    "lid_confidence": conf,
                    "source": source_name,
                    "source_url": url
                })
        print(f"  │   ✓ {len(sentences)} candidates → {len(results)} verified")
    except Exception as e:
        print(f"  │   ✗ {e}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SEED SOURCES
# Structured as (url, label, source_name, selector, skip_patterns)
# Add new sources here — one line per source.
# ══════════════════════════════════════════════════════════════════════════════

# ── LS sources ────────────────────────────────────────────────────────────────
LS_STATIC_SOURCES = [
    # Lesotho government
    ("https://www.gov.ls",                    "LS", "gov_ls",          None, None),
    ("https://www.gov.ls/news/",              "LS", "gov_ls_news",     None, None),
    # Lesotho news
    ("https://www.lestimes.com",              "LS", "lestimes",        None, None),
    ("https://www.thepost.co.ls",             "LS", "thepost_ls",      None, None),
    ("https://www.moafrica.co.ls",            "LS", "moafrica_ls",     None, None),
    # Literacy / educational
    ("https://www.leihlolabasotho.co.ls",     "LS", "leihlolabasotho", None, None),
    ("https://www.lmps.org.ls",               "LS", "lmps_ls",        None, None),
]

# ── SAS sources ───────────────────────────────────────────────────────────────
SAS_STATIC_SOURCES = [
    # SA government
    ("https://www.gov.za/st",                      "SAS", "govza_sesotho",  None, None),
    ("https://www.sabc.co.za/sabc/sesotho/",       "SAS", "sabc_sesotho",   None, None),
    ("https://www.sowetanlive.co.za/sesotho/",     "SAS", "sowetan_sesotho",None, None),
    ("https://www.vukuzenzele.gov.za",             "SAS", "vukuzenzele",    None, None),
    # SA department pages with known Sesotho content
    ("https://www.dbe.gov.za/sesotho",             "SAS", "dbe_sesotho",    None, None),
    ("https://www.health.gov.za/sesotho",          "SAS", "health_sesotho", None, None),
    ("https://www.saps.gov.za/sesotho",            "SAS", "saps_sesotho",   None, None),
    ("https://www.justice.gov.za/sesotho",         "SAS", "justice_sesotho",None, None),
]

# ── Nalibali LS stories — discovered dynamically ──────────────────────────────
NALIBALI_BASE = "https://nalibali.org"
NALIBALI_LANG = "st-ls"
NALIBALI_CATEGORIES = [
    f"{NALIBALI_BASE}/story_categories/traditional-tales-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
    f"{NALIBALI_BASE}/story_categories/stories-with-animals-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
    f"{NALIBALI_BASE}/story_categories/funny-stories-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
    f"{NALIBALI_BASE}/story_categories/feel-good-stories-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
    f"{NALIBALI_BASE}/story_categories/fantasy-stories-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
    f"{NALIBALI_BASE}/story_categories/stories-with-life-lessons-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
    f"{NALIBALI_BASE}/story_categories/stories-based-on-real-life-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
    f"{NALIBALI_BASE}/story_categories/latest-stories-{NALIBALI_LANG}/?lang={NALIBALI_LANG}",
]
NALIBALI_SKIP = [
    "click here", "follow us", "author:", "translator:", "illustrator:",
    "this story is also available", "explore more", "donate", "quick links",
    "contact us", "privacy", "terms", "©", "web design",
    "eba mahlahahlaha", "o ikutlwa jwang", "na o nahana", "moral",
]

# ── Zenodo SN dataset — direct download ──────────────────────────────────────
ZENODO_URLS = [
    "https://zenodo.org/api/records/10531959/files/NewsSA.txt/content",
    "https://zenodo.org/api/records/10531959/files/NewsABSA.txt/content",
]

# ── Vukuzenzele GitHub — CSV download ────────────────────────────────────────
VUKUZENZELE_API = (
    "https://api.github.com/repos/dsfsi/vukuzenzele-nlp"
    "/contents/data/simple_align_output"
)

# ── gov.za PDFs ───────────────────────────────────────────────────────────────
GOV_ZA_PDFS = [
    "https://www.dwypd.gov.za/wp-content/uploads/2024/02/Sesotho.pdf",
    "https://www.nwpg.gov.za/wp-content/uploads/2024/02/SOPA-2024-SESOTHO-VERSION-.pdf",
    "https://www.sars.gov.za/wp-content/uploads/Docs/Translations/MediaReleases/2024/Media-release-SARS-media-statement-on-constitutional-court-judgement-updated-on-16-April-2024-Sesotho.pdf",
]

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — LS collection
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Collecting LS data...")
print("█"*60)

all_data = []

# ── 2a: Zenodo SN dataset ─────────────────────────────────────────────────────
print("\n  ├─ Zenodo SN dataset (Mokhosi et al.)...")
for url in ZENODO_URLS:
    try:
        r = requests.get(url, timeout=60)
        lines = [l.strip() for l in r.text.split("\n") if l.strip()]
        print(f"  │   Raw lines: {len(lines)}")
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
        print(f"  │   ✓ {sum(1 for d in all_data if d['source']=='Mokhosi_SN_Zenodo')} verified so far")
    except Exception as e:
        print(f"  │   ✗ Zenodo failed: {e}")

# ── 2b: Nalibali stories ──────────────────────────────────────────────────────
print("\n  ├─ Nalibali stories...")
nalibali_urls = set()
for cat_url in NALIBALI_CATEGORIES:
    try:
        r = requests.get(cat_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (f"/stories/" in href and f"lang={NALIBALI_LANG}" in href
                    and "/story_categories/" not in href
                    and "/story-library/" not in href):
                full = href if href.startswith("http") else NALIBALI_BASE + href
                nalibali_urls.add(full)
        time.sleep(1)
    except Exception as e:
        print(f"  │   ⚠ Category failed: {e}")
print(f"  │   Found {len(nalibali_urls)} story URLs")

for i, url in enumerate(sorted(nalibali_urls), 1):
    slug = url.split("/stories/")[1].split("/")[0]
    print(f"  │   [{i}/{len(nalibali_urls)}] {slug}")
    results = fetch_and_collect(
        url, "LS", "Nalibali_LS",
        selector="elementor-widget-text-editor",
        skip_patterns=NALIBALI_SKIP
    )
    all_data.extend(results)
    time.sleep(1.5)

# ── 2c: Static LS websites ────────────────────────────────────────────────────
print("\n  ├─ Static LS websites...")
for url, label, source, selector, skip in LS_STATIC_SOURCES:
    print(f"  │   {source}: {url[:60]}")
    results = fetch_and_collect(url, label, source, selector, skip)
    all_data.extend(results)
    time.sleep(1)

ls_count = sum(1 for d in all_data if d["label"] == "LS")
print(f"\n  ✓ LS total so far: {ls_count}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — SAS collection
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Collecting SAS data...")
print("█"*60)

# ── 3a: Vukuzenzele GitHub CSV ────────────────────────────────────────────────
print("\n  ├─ Vukuzenzele DSFSI (GitHub)...")
try:
    r = requests.get(VUKUZENZELE_API, timeout=30)
    items = r.json()
    csv_files = [f for f in items if isinstance(f, dict) and f["name"].endswith(".csv")]
    print(f"  │   Found {len(csv_files)} CSV files")

    for f in csv_files:
        try:
            r2 = requests.get(f["download_url"], timeout=30)
            df_raw = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip")
            # find Sesotho column
            sesotho_col = None
            for col in df_raw.columns:
                if any(x in col.lower() for x in ["sot", "sesotho", "st", "target"]):
                    sesotho_col = col
                    break
            if sesotho_col is None and len(df_raw.columns) >= 2:
                sesotho_col = df_raw.columns[1]  # second col in aligned CSV

            if sesotho_col:
                lines = df_raw[sesotho_col].dropna().astype(str).tolist()
                added = 0
                for line in lines:
                    line = clean_text(line)
                    if not is_quality(line):
                        continue
                    ok, conf = verify_sesotho(line)
                    if ok:
                        all_data.append({
                            "text": line, "label": "SAS",
                            "lid_confidence": conf,
                            "source": "Vukuzenzele_DSFSI",
                            "source_url": f["download_url"]
                        })
                        added += 1
                if added:
                    print(f"  │   ✓ {f['name']}: {added} sentences")
        except Exception as e2:
            print(f"  │   ⚠ {f['name']}: {e2}")
except Exception as e:
    print(f"  │   ✗ Vukuzenzele failed: {e}")

# ── 3b: gov.za PDFs ───────────────────────────────────────────────────────────
print("\n  ├─ gov.za PDFs (PyMuPDF)...")
try:
    import fitz
    for url in GOV_ZA_PDFS:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            doc = fitz.open(stream=r.content, filetype="pdf")
            added = 0
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                for sentence in split_sentences(clean_text(text)):
                    if not is_quality(sentence):
                        continue
                    ok, conf = verify_sesotho(sentence)
                    if ok:
                        all_data.append({
                            "text": sentence, "label": "SAS",
                            "lid_confidence": conf,
                            "source": "govza_PDF",
                            "source_url": url
                        })
                        added += 1
            print(f"  │   ✓ {url.split('/')[-1][:50]}: {added} sentences")
        except Exception as e2:
            print(f"  │   ⚠ PDF failed: {e2}")
except ImportError:
    print("  │   ⚠ PyMuPDF not installed — run: pip install pymupdf")

# ── 3c: Static SAS websites ───────────────────────────────────────────────────
print("\n  ├─ Static SAS websites...")
for url, label, source, selector, skip in SAS_STATIC_SOURCES:
    print(f"  │   {source}: {url[:60]}")
    results = fetch_and_collect(url, label, source, selector, skip)
    all_data.extend(results)
    time.sleep(1)

sas_count = sum(1 for d in all_data if d["label"] == "SAS")
print(f"\n  ✓ SAS total so far: {sas_count}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Clean, deduplicate, merge with existing corpus
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Cleaning and merging corpus...")
print("█"*60)

if not all_data:
    print("  ✗ No data collected. Check network and source URLs above.")
    exit()

new_df = pd.DataFrame(all_data)

# final quality pass on the full dataframe
new_df = new_df[new_df["text"].apply(is_quality)]
new_df = new_df.drop_duplicates(subset="text")

print(f"  ✓ New sentences after quality filter + dedup: {len(new_df)}")
print(f"  New label distribution:")
print(new_df["label"].value_counts().to_string(header=False))

# merge with existing corpus if it exists
try:
    existing_df = pd.read_csv("sesotho_clean.csv", encoding="utf-8")
    print(f"\n  ✓ Existing corpus loaded: {len(existing_df)} sentences")
    print(f"  Existing distribution:")
    print(existing_df["label"].value_counts().to_string(header=False))

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="text")

    # apply quality filter to the existing corpus too (removes old LaTeX artifacts)
    combined = combined[combined["text"].apply(is_quality)]

except FileNotFoundError:
    print("  ⚠ No existing sesotho_clean.csv — creating fresh")
    combined = new_df

combined.to_csv("sesotho_clean.csv", index=False, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Final report
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Final corpus report")
print("█"*60)

print(f"\n  Total sentences: {len(combined)}")
print(f"\n  Label distribution:")
print(combined["label"].value_counts().to_string(header=False))
print(f"\n  Average LID confidence: {combined['lid_confidence'].mean():.4f}")

print(f"\n  Sources breakdown:")
print(combined.groupby(["label","source"]).size().to_string())

print(f"\n  Gap to target (2500 per variant):")
counts = combined["label"].value_counts()
for variant in ["SAS", "LS"]:
    count = counts.get(variant, 0)
    gap = max(0, 2500 - count)
    bar = "█" * min(int(count/2500*20), 20) + "░" * max(0, 20 - int(count/2500*20))
    print(f"  {variant}: [{bar}] {count}/2500  (need {gap} more)")

print(f"\n  Sample SAS:")
for t in combined[combined["label"]=="SAS"]["text"].head(3).tolist():
    print(f"    '{t[:70]}'")
print(f"\n  Sample LS:")
for t in combined[combined["label"]=="LS"]["text"].head(3).tolist():
    print(f"    '{t[:70]}'")

print(f"\n  ✓ Saved to sesotho_clean.csv")
print("\n" + "█"*60)
print("  DONE — run orthographic_classifier.py to retrain on new corpus")
print("█"*60)