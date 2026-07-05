import trafilatura
import requests
import pandas as pd
import unicodedata
import re
import fasttext
from huggingface_hub import hf_hub_download
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time
import io

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 0 — Environment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 0: Checking environment...")
print("█"*60)
try:
    import fasttext, pandas, sklearn, trafilatura
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

def verify_sesotho(sentences, confidence_threshold=0.7):
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

def split_sentences(text):
    raw = re.split(r"[.!?\n]", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 3]

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — LS data: Mokhosi SN dataset (Zenodo) + leihlolabasotho
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Collecting LS data...")
print("█"*60)

ls_sentences = []

# Source 1: Mokhosi SN dataset
try:
    print("  ├─ Fetching Mokhosi SN dataset (Zenodo)...")
    url = "https://zenodo.org/api/records/10531959/files/NewsSA.txt/content"
    r = requests.get(url, timeout=60)
    lines = [l.strip() for l in r.text.split("\n") if l.strip()]
    print(f"  │   ✓ {len(lines)} raw lines from NewsSA.txt")
    ls_sentences.extend(lines)

    # also get NewsABSA.txt — more headlines
    url2 = "https://zenodo.org/api/records/10531959/files/NewsABSA.txt/content"
    r2 = requests.get(url2, timeout=60)
    lines2 = [l.strip() for l in r2.text.split("\n") if l.strip()]
    print(f"  │   ✓ {len(lines2)} raw lines from NewsABSA.txt")
    ls_sentences.extend(lines2)
except Exception as e:
    print(f"  │   ✗ Zenodo failed: {e}")

# Source 2: Trafilatura on Lesotho news sites
print("\n  ├─ Fetching LS news sites with Trafilatura...")
ls_urls = [
    "https://www.leihlolabasotho.co.ls",
    "https://www.lmps.org.ls",
    "https://www.gov.ls/news/",
]
for url in ls_urls:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False,
                                       include_tables=False, no_fallback=False)
            if text:
                sentences = split_sentences(text)
                ls_sentences.extend(sentences)
                print(f"  │   ✓ {url}: {len(sentences)} sentences")
            else:
                print(f"  │   ⚠ {url}: extracted no text")
        else:
            print(f"  │   ⚠ {url}: fetch failed")
        time.sleep(1)
    except Exception as e:
        print(f"  │   ⚠ {url}: {e}")

print(f"\n  ✓ Total LS candidates: {len(ls_sentences)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — SAS data: Vukuzenzele + gov.za PDFs + Trafilatura
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Collecting SAS data...")
print("█"*60)

sas_sentences = []

# Source 1: Vukuzenzele — pull ALL CSV files from simple_align_output
print("  ├─ Fetching Vukuzenzele (GitHub)...")
try:
    api_url = "https://api.github.com/repos/dsfsi/vukuzenzele-nlp/contents/data/simple_align_output"
    r = requests.get(api_url, timeout=30)
    items = r.json()
    csv_files = [f for f in items if isinstance(f, dict) and f["name"].endswith(".csv")]
    print(f"  │   Found {len(csv_files)} CSV files — downloading all...")

    for f in csv_files:
        try:
            r2 = requests.get(f["download_url"], timeout=30)
            df_raw = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip")
            # find Sesotho column
            for col in df_raw.columns:
                if any(x in col.lower() for x in ["sot", "sesotho", "st", "target"]):
                    lines = df_raw[col].dropna().astype(str).tolist()
                    sas_sentences.extend(lines)
                    break
            else:
                # aligned CSVs: Sesotho is usually the second column
                if len(df_raw.columns) >= 2:
                    lines = df_raw.iloc[:, 1].dropna().astype(str).tolist()
                    sas_sentences.extend(lines)
        except Exception as e2:
            print(f"  │   ⚠ {f['name']}: {e2}")

    print(f"  │   ✓ {len(sas_sentences)} sentences from Vukuzenzele")
except Exception as e:
    print(f"  │   ✗ Vukuzenzele failed: {e}")

# Source 2: gov.za PDFs with PyMuPDF (like your supervisor's example)
print("\n  ├─ Fetching gov.za PDFs with PyMuPDF...")
try:
    import fitz  # PyMuPDF

    pdf_urls = [
        "https://www.dwypd.gov.za/wp-content/uploads/2024/02/Sesotho.pdf",
        "https://www.nwpg.gov.za/wp-content/uploads/2024/02/SOPA-2024-SESOTHO-VERSION-.pdf",
        "https://www.sars.gov.za/wp-content/uploads/Docs/Translations/MediaReleases/2024/Media-release-SARS-media-statement-on-constitutional-court-judgement-updated-on-16-April-2024-Sesotho.pdf",
    ]

    for url in pdf_urls:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            doc = fitz.open(stream=r.content, filetype="pdf")
            pdf_sentences = []
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                sentences = split_sentences(text)
                pdf_sentences.extend(sentences)
            sas_sentences.extend(pdf_sentences)
            print(f"  │   ✓ {url.split('/')[-1]}: {len(pdf_sentences)} sentences")
        except Exception as e2:
            print(f"  │   ⚠ PDF failed: {e2}")

except ImportError:
    print("  │   ✗ PyMuPDF not installed. Run: pip install pymupdf")

# Source 3: Trafilatura on SAS sites
print("\n  ├─ Fetching SAS sites with Trafilatura...")
sas_urls = [
    "https://www.sabc.co.za/sabc/sesotho/",
    "https://www.sowetanlive.co.za/sesotho/",
]
for url in sas_urls:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False,
                                       include_tables=False)
            if text:
                sentences = split_sentences(text)
                sas_sentences.extend(sentences)
                print(f"  │   ✓ {url}: {len(sentences)} sentences")
            else:
                print(f"  │   ⚠ {url}: no text extracted")
        else:
            print(f"  │   ⚠ {url}: fetch failed")
        time.sleep(1)
    except Exception as e:
        print(f"  │   ⚠ {url}: {e}")

print(f"\n  ✓ Total SAS candidates: {len(sas_sentences)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — African Storybook (Selenium) — both variants
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: African Storybook scrape (Selenium)...")
print("  Note: Chrome browser will open — do not close it")
print("█"*60)

# These are known Sesotho book IDs on African Storybook
# Mix of SAS and LS — GlotLID will verify, source labelling handles variant
STORYBOOK_IDS = [
    ("21926", "LS"),
    ("21927", "LS"),
    ("21928", "LS"),
    ("22000", "SAS"),
    ("22001", "SAS"),
]

storybook_sentences = {
    "SAS": [],
    "LS": []
}

try:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # run without opening browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=options
    )
    wait = WebDriverWait(driver, 10)

    for book_id, variant in STORYBOOK_IDS:
        url = f"https://www.africanstorybook.org/newviewer/index.php?id={book_id}&bt=3&dual=false"
        print(f"\n  ├─ Book {book_id} ({variant}): {url}")
        driver.get(url)
        time.sleep(2)

        last_page_text = ""
        book_sentences = []

        while True:
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "p")))
                paragraphs = driver.find_elements(By.TAG_NAME, "p")
                page_texts = [p.text.strip() for p in paragraphs if p.text.strip()]
                combined_text = "\n".join(page_texts)

                if combined_text == last_page_text or not combined_text:
                    break

                for para in page_texts:
                    sentences = split_sentences(para)
                    book_sentences.extend(sentences)

                last_page_text = combined_text

                next_btn = wait.until(EC.element_to_be_clickable((By.ID, "next-page")))
                driver.execute_script("arguments[0].click();", next_btn)
                WebDriverWait(driver, 10).until(
                    lambda d: combined_text != "\n".join(
                        [p.text.strip() for p in d.find_elements(By.TAG_NAME, "p")
                         if p.text.strip()]
                    )
                )
            except TimeoutException:
                break
            except Exception as e:
                print(f"  │   ⚠ Page error: {e}")
                break

        storybook_sentences[variant].extend(book_sentences)
        print(f"  │   ✓ {len(book_sentences)} sentences from book {book_id}")

    driver.quit()
    print(f"\n  ✓ Storybook SAS sentences: {len(storybook_sentences['SAS'])}")
    print(f"  ✓ Storybook LS sentences:  {len(storybook_sentences['LS'])}")

    sas_sentences.extend(storybook_sentences["SAS"])
    ls_sentences.extend(storybook_sentences["LS"])

except Exception as e:
    print(f"  ✗ Selenium failed: {e}")
    print("  ⚠ Continuing without storybook data")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — GlotLID verification (threshold 0.7 for short text)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: GlotLID verification (threshold: 0.7)...")
print("█"*60)

print(f"\n  Verifying {len(ls_sentences)} LS candidates...")
ls_verified = verify_sesotho(
    [clean_text(s) for s in ls_sentences],
    confidence_threshold=0.7
)
print(f"  ✓ {len(ls_verified)} LS sentences passed")

print(f"\n  Verifying {len(sas_sentences)} SAS candidates...")
sas_verified = verify_sesotho(
    [clean_text(s) for s in sas_sentences],
    confidence_threshold=0.7
)
print(f"  ✓ {len(sas_verified)} SAS sentences passed")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 6 — Build and save labelled dataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 6: Building labelled dataset...")
print("█"*60)

all_data = []

for text, conf in ls_verified[:200]:
    all_data.append({
        "text": text,
        "label": "LS",
        "lid_confidence": conf,
        "source": "Mokhosi_SN_Zenodo_or_leihlolabasotho"
    })

for text, conf in sas_verified[:200]:
    all_data.append({
        "text": text,
        "label": "SAS",
        "lid_confidence": conf,
        "source": "Vukuzenzele_DSFSI_or_govza"
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