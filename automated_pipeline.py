import time
import re
import json
import pickle
import unicodedata
import pandas as pd
import fasttext
import trafilatura
from trafilatura.spider import focused_crawler
from trafilatura.sitemaps import sitemap_search
from trafilatura.feeds import find_feed_urls
from huggingface_hub import hf_hub_download


print("  CHECKPOINT 0: Loading models...")



model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
lid_model = fasttext.load_model(model_path)
print(" GlotLID loaded")


with open("ortho_classifier.pkl", "rb") as f:
    ortho_classifier = pickle.load(f)
print("  ✓ Orthographic classifier loaded")


SEED_DOMAINS = [
    # LS sources
    "https://www.leihlolabasotho.co.ls",
    "https://www.lmps.org.ls",
    "https://www.gov.ls",
    # SAS sources
    "https://www.vukuzenzele.gov.za",
    "https://www.sabc.co.za/sabc/sesotho/",
]


def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text):
    raw = re.split(r"[.!?\n]", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 3]

def verify_sesotho(sentence, threshold=0.7):
    clean = sentence.replace("\n", " ").strip()
    labels, scores = lid_model.predict(clean, k=1)
    lang = labels[0].replace("__label__", "")
    score = float(scores[0])
    return lang == "sot_Latn" and score >= threshold, round(score, 4)


print("  CHECKPOINT 1: Automated URL discovery...")


all_urls = []

for seed in SEED_DOMAINS:
    print(f"\n  ├─ Crawling: {seed}")
    discovered = []

    # Method 1: sitemap 
    try:
        sitemap_urls = sitemap_search(seed)
        if sitemap_urls:
            discovered.extend(sitemap_urls)
            print(f"  Sitemap: {len(sitemap_urls)} URLs")
    except Exception as e:
        print(f"  Sitemap failed: {e}")

    # Method 2: RSS/Atom feeds 
    try:
        feed_urls = find_feed_urls(seed)
        if feed_urls:
            discovered.extend(feed_urls)
            print(f"  Feeds: {len(feed_urls)} URLs")
    except Exception as e:
        print(f"  │Feeds failed: {e}")

    # Method 3: focused crawler 
    # max_seen_urls: how many pages to visit
    # max_known_urls: how many URLs to collect total
    try:
        todo, known = focused_crawler(
            seed,
            max_seen_urls=20,
            max_known_urls=50,
            lang="st"          
        )
        discovered.extend(todo)
        discovered.extend(known)
        print(f" Crawler: {len(todo)} todo + {len(known)} known URLs")
    except Exception as e:
        print(f"  Crawler failed: {e}")

    # deduplicate
    discovered = list(set(discovered))
    print(f"  Total unique URLs from {seed}: {len(discovered)}")
    all_urls.extend(discovered)
    time.sleep(2)  # polite delay between domains

all_urls = list(set(all_urls))
print(f"\n Total unique URLs discovered: {len(all_urls)}")


print("  CHECKPOINT 2: Fetching and extracting text...")


raw_sentences = []
failed = 0
empty = 0

for i, url in enumerate(all_urls, 1):
    print(f"\n  ├─ [{i}/{len(all_urls)}] {url[:70]}")
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            empty += 1
            print(f"  Empty response")
            continue

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            deduplicate=True
        )

        if not text:
            empty += 1
            print(f"  No text extracted")
            continue

        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)
        raw_sentences.extend([(s, url) for s in sentences])
        print(f"  {len(sentences)} sentences extracted")
        time.sleep(1)

    except Exception as e:
        failed += 1
        print(f"  Failed: {e}")

print(f"\n  Total raw sentences: {len(raw_sentences)}")
print(f"  Failed: {failed} | Empty: {empty}")



print("  CHECKPOINT 3: GlotLID language verification...")


verified_sentences = []
rejected = 0

for sentence, url in raw_sentences:
    is_sesotho, confidence = verify_sesotho(sentence)
    if is_sesotho:
        verified_sentences.append({
            "text": sentence,
            "source_url": url,
            "lid_confidence": confidence
        })
    else:
        rejected += 1

print(f"  Verified Sesotho: {len(verified_sentences)}")
print(f"  Rejected:         {rejected}")
print(f"  Retention rate:     {round(len(verified_sentences)/max(len(raw_sentences),1)*100, 1)}%")


print("  CHECKPOINT 4: Orthographic classification...")
print("  SAS vs LS — classified by spelling patterns, not source")


texts = [s["text"] for s in verified_sentences]

if not texts:
    print(" No verified sentences to classify. Check earlier stages.")
else:
    labels = ortho_classifier.predict(texts)
    probas = ortho_classifier.predict_proba(texts)
    max_probas = [round(max(p), 4) for p in probas]

    for i, sentence in enumerate(verified_sentences):
        sentence["ortho_label"]      = labels[i]
        sentence["ortho_confidence"] = max_probas[i]

    label_counts = pd.Series(labels).value_counts()
    print(f" Classification complete:")
    print(label_counts.to_string(header=False))
    print(f"\n  Sample classified sentences:")
    for s in verified_sentences[:5]:
        print(f"    [{s['ortho_label']} | conf:{s['ortho_confidence']}] "
              f"'{s['text'][:55]}'")


print("  CHECKPOINT 5: Quality filtering...")


if verified_sentences:
    df = pd.DataFrame(verified_sentences)
    before = len(df)

    # deduplication
    df = df.drop_duplicates(subset="text")
    print(f"  Removed {before - len(df)} duplicates")

    # length filter — at least 5 words
    df = df[df["text"].str.split().str.len() >= 5]
    print(f"   Removed short sentences (< 5 words)")

    # high confidence only
    df = df[df["lid_confidence"] >= 0.7]
    print(f"  Applied LID confidence threshold (>= 0.7)")

    print(f"\n  Final corpus size: {len(df)} sentences")
    if "ortho_label" in df.columns:
        print(f"  Label distribution:")
        print(df["ortho_label"].value_counts().to_string(header=False))



    print("  CHECKPOINT 6: Saving corpus as JSONL...")
   
    

    with open("sesotho_corpus.jsonl", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            entry = {
                "text":             row["text"],
                "ortho_label":      row.get("ortho_label", "unknown"),
                "ortho_confidence": row.get("ortho_confidence", 0.0),
                "lid_confidence":   row["lid_confidence"],
                "source_url":       row["source_url"],
                "token_count":      len(row["text"].split())
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Saved {len(df)} entries to sesotho_corpus.jsonl")
    print(f"\n  Sample JSONL entry:")
    sample = df.iloc[0]
    print(json.dumps({
        "text":           sample["text"][:60] + "...",
        "ortho_label":    sample.get("ortho_label", "unknown"),
        "lid_confidence": sample["lid_confidence"],
        "source_url":     sample["source_url"]
    }, indent=4, ensure_ascii=False))

else:
    print(" No data to save")
