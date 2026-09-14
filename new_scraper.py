import time
import re
import unicodedata
import pandas as pd
import requests
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

test = "O monyane haholo"
labels, scores = model.predict(test, k=1)
print(f"  ✓ Test: '{test}' → {labels[0].replace('__label__','')} ({round(float(scores[0]),4)})")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text):
    raw = re.split(r"[.!?]", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 5]

def verify_sesotho(sentence, threshold=0.7):
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
BASE = "https://nalibali.org"
LANG = "st-ls"

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — Collect story URLs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Collecting story URLs...")
print("█"*60)

CATEGORY_URLS = [
    f"{BASE}/story_categories/traditional-tales-{LANG}/?lang={LANG}",
    f"{BASE}/story_categories/stories-with-animals-{LANG}/?lang={LANG}",
    f"{BASE}/story_categories/funny-stories-{LANG}/?lang={LANG}",
    f"{BASE}/story_categories/feel-good-stories-{LANG}/?lang={LANG}",
    f"{BASE}/story_categories/fantasy-stories-{LANG}/?lang={LANG}",
    f"{BASE}/story_categories/stories-with-life-lessons-{LANG}/?lang={LANG}",
    f"{BASE}/story_categories/stories-based-on-real-life-{LANG}/?lang={LANG}",
    f"{BASE}/story_categories/latest-stories-{LANG}/?lang={LANG}",
]

story_urls = set()
for cat_url in CATEGORY_URLS:
    try:
        r = requests.get(cat_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/stories/" in href and f"lang={LANG}" in href:
                if "/story_categories/" not in href and "/story-library/" not in href:
                    full = href if href.startswith("http") else BASE + href
                    story_urls.add(full)
        print(f"  ├─ {cat_url.split('/')[-2]}: {len(story_urls)} URLs so far")
        time.sleep(1)
    except Exception as e:
        print(f"  ⚠ {e}")

print(f"\n  ✓ Total unique story URLs: {len(story_urls)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — Scrape each story
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Scraping story text...")
print("  Using selector: elementor-widget-text-editor")
print("█"*60)

# paragraphs to skip — navigation, author lines, translator lines, questions
SKIP_PATTERNS = [
    "click here", "follow us", "author:", "translator:", "illustrator:",
    "this story is also available", "explore more", "donate", "quick links",
    "contact us", "privacy", "terms", "©", "eba mahlahahlaha",
    "o ikutlwa jwang", "na o", "o nahana", "moral", "web design",
]

def should_skip(text):
    t = text.lower()
    return any(p in t for p in SKIP_PATTERNS)

all_sentences = []
failed = 0

for i, url in enumerate(sorted(story_urls), 1):
    slug = url.split("/stories/")[1].split("/")[0]
    print(f"\n  ├─ [{i}/{len(story_urls)}] {slug}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        # use the confirmed selector
        content_divs = soup.find_all("div", class_="elementor-widget-text-editor")

        if not content_divs:
            print(f"  │   ⚠ No elementor-widget-text-editor found — trying fallback")
            # fallback: grab all paragraphs from hentry
            content_divs = soup.find_all("div", class_="hentry")

        # collect all paragraph text from content divs
        raw_paragraphs = []
        for div in content_divs:
            for p in div.find_all("p"):
                text = p.get_text().strip()
                if text and not should_skip(text):
                    raw_paragraphs.append(text)

        # join and split into sentences
        raw_text = " ".join(raw_paragraphs)
        cleaned = clean_text(raw_text)
        sentences = split_sentences(cleaned)

        verified_count = 0
        for sentence in sentences:
            is_sesotho, confidence = verify_sesotho(sentence)
            if is_sesotho:
                all_sentences.append({
                    "text": sentence,
                    "label": "LS",
                    "lid_confidence": confidence,
                    "source": "Nalibali_LS",
                    "source_url": url
                })
                verified_count += 1

        print(f"  │   ✓ {len(sentences)} candidates → {verified_count} verified Sesotho")

        # progress bar
        progress = int((i / len(story_urls)) * 30)
        bar = "█" * progress + "░" * (30 - progress)
        print(f"  │   [{bar}] {len(all_sentences)} total sentences")

        time.sleep(1.5)

    except Exception as e:
        print(f"  │   ✗ Failed: {e}")
        failed += 1

print(f"\n  ✓ Sentences collected: {len(all_sentences)}")
print(f"  ✗ Failed: {failed}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Merge with existing corpus
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Merging with existing sesotho_clean.csv...")
print("█"*60)

if not all_sentences:
    print("  ✗ No sentences collected — check output above")
else:
    new_df = pd.DataFrame(all_sentences)
    new_df = new_df.drop_duplicates(subset="text")
    new_df = new_df[new_df["text"].str.split().str.len() >= 5]
    print(f"  ✓ New sentences after dedup: {len(new_df)}")

    try:
        existing_df = pd.read_csv("sesotho_clean.csv")
        print(f"  ✓ Existing corpus: {len(existing_df)} sentences")
        print(f"    {existing_df['label'].value_counts().to_dict()}")
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="text")
    except FileNotFoundError:
        print("  ⚠ sesotho_clean.csv not found — saving new data only")
        combined = new_df

    combined.to_csv("sesotho_clean.csv", index=False, encoding="utf-8")

    print(f"\n  ✓ Final corpus: {len(combined)} sentences")
    print(f"  Label distribution:")
    print(combined["label"].value_counts().to_string(header=False))
    print(f"\n  Sample new LS sentences:")
    for t in new_df["text"].head(5).tolist():
        print(f"    '{t[:70]}'")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Done
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  DONE")
print("  Source: Nalibali.org — Sesotho stories (LS orthography)")
print("  Next: run orthographic_classifier.py to retrain on expanded data")
print("█"*60)