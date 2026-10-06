import os
import re
import unicodedata
import pandas as pd
import fasttext
import fitz  # PyMuPDF
from huggingface_hub import hf_hub_download

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — point these at your local PDF folders
# ══════════════════════════════════════════════════════════════════════════════
HL_FOLDER  = r"C:\Users\ad\OneDrive\Desktop\Project Something Something\exam_papers"
FAL_FOLDER = r"C:\Users\ad\OneDrive\Desktop\Project Something Something\exam_pdfs_fal"
OUTPUT_CSV = "sesotho_clean.csv"   # merges with your existing corpus

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
    'does','said','each','made','most','after','both','through',
    'where','being','during','between','should','these','those',
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

def has_latex_artifact(text):
    return bool(re.search(r'\$[A-Za-z]\$', text))

def is_quality(text):
    if not text or len(text.split()) < 5:
        return False
    if has_latex_artifact(text):
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

# ══════════════════════════════════════════════════════════════════════════════
# EXAM PDF EXTRACTION
# Skips first 2 pages (cover, instructions) and last page (end matter)
# Skips pages that are mostly English (question instructions)
# ══════════════════════════════════════════════════════════════════════════════
def extract_from_pdf(pdf_path, label, source_name):
    results = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"  │   Pages: {total_pages}")

        # skip first 2 pages (cover + rubric) and last page
        for page_num in range(2, max(total_pages - 1, 3)):
            page_text = doc[page_num].get_text("text")

            # skip pages that look like English instructions
            # (exam papers mix Sesotho text with English question numbers)
            english_word_count = len([
                w for w in re.findall(r'\b[a-zA-Z]{4,}\b', page_text.lower())
                if w in ENGLISH_MARKERS
            ])
            total_words = len(page_text.split())
            if total_words > 0 and english_word_count / total_words > 0.15:
                print(f"  │   Page {page_num+1}: skipped (English-heavy)")
                continue

            cleaned = clean_text(page_text)
            sentences = split_sentences(cleaned)

            for sentence in sentences:
                if not is_quality(sentence):
                    continue
                ok, conf = verify_sesotho(sentence)
                if ok:
                    results.append({
                        "text": sentence,
                        "label": label,
                        "lid_confidence": conf,
                        "source": source_name,
                        "source_url": pdf_path
                    })

        print(f"  │   ✓ {len(results)} verified Sesotho sentences")
    except Exception as e:
        print(f"  │   ✗ Failed: {e}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — Process HL exam papers (SAS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Processing HL exam papers (SAS)...")
print("█"*60)

all_data = []

if os.path.exists(HL_FOLDER):
    pdf_files = [f for f in os.listdir(HL_FOLDER) if f.lower().endswith(".pdf")]
    print(f"  Found {len(pdf_files)} PDFs in HL folder")

    for i, pdf_file in enumerate(sorted(pdf_files), 1):
        pdf_path = os.path.join(HL_FOLDER, pdf_file)
        print(f"\n  ├─ [{i}/{len(pdf_files)}] {pdf_file}")
        results = extract_from_pdf(pdf_path, "SAS", "Matric_HL_Exam")
        all_data.extend(results)
else:
    print(f"  ⚠ Folder not found: {HL_FOLDER}")
    print(f"  Create folder and add PDF files, then rerun")

hl_count = sum(1 for d in all_data if d["source"] == "Matric_HL_Exam")
print(f"\n  ✓ Total from HL papers: {hl_count}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — Process FAL exam papers (SAS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Processing FAL exam papers (SAS)...")
print("█"*60)

if os.path.exists(FAL_FOLDER):
    pdf_files = [f for f in os.listdir(FAL_FOLDER) if f.lower().endswith(".pdf")]
    print(f"  Found {len(pdf_files)} PDFs in FAL folder")

    for i, pdf_file in enumerate(sorted(pdf_files), 1):
        pdf_path = os.path.join(FAL_FOLDER, pdf_file)
        print(f"\n  ├─ [{i}/{len(pdf_files)}] {pdf_file}")
        results = extract_from_pdf(pdf_path, "SAS", "Matric_FAL_Exam")
        all_data.extend(results)
else:
    print(f"  ⚠ Folder not found: {FAL_FOLDER}")
    print(f"  Create folder and add PDF files, then rerun")

fal_count = sum(1 for d in all_data if d["source"] == "Matric_FAL_Exam")
print(f"\n  ✓ Total from FAL papers: {fal_count}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Merge with existing corpus
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Merging with existing corpus...")
print("█"*60)

if not all_data:
    print("  ✗ No data extracted — check PDF folders above")
    exit()

new_df = pd.DataFrame(all_data)
new_df = new_df.drop_duplicates(subset="text")
new_df = new_df[new_df["text"].str.split().str.len() >= 5]
print(f"  ✓ New sentences after dedup: {len(new_df)}")
print(f"  New label distribution:")
print(new_df["label"].value_counts().to_string(header=False))

try:
    existing_df = pd.read_csv(OUTPUT_CSV, encoding="utf-8")
    print(f"\n  ✓ Existing corpus: {len(existing_df)} sentences")
    print(existing_df["label"].value_counts().to_string(header=False))
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="text")
except FileNotFoundError:
    print("  ⚠ No existing corpus found — creating fresh")
    combined = new_df

combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Report
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Final corpus report")
print("█"*60)
print(f"\n  Total sentences: {len(combined)}")
print(f"\n  Label distribution:")
print(combined["label"].value_counts().to_string(header=False))
print(f"\n  Sources breakdown:")
print(combined.groupby(["label", "source"]).size().to_string())
print(f"\n  Gap to target (2500 per variant):")
counts = combined["label"].value_counts()
for variant in ["SAS", "LS"]:
    count = counts.get(variant, 0)
    gap = max(0, 2500 - count)
    bar = "█" * min(int(count/2500*20), 20) + "░" * max(0, 20 - int(count/2500*20))
    print(f"  {variant}: [{bar}] {count}/2500  (need {gap} more)")
print(f"\n  ✓ Saved to {OUTPUT_CSV}")
print("█"*60)