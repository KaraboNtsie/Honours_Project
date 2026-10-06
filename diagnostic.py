import requests
import fasttext
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)

# fetch Mokhosi headlines
url = "https://zenodo.org/api/records/10531959/files/NewsSA.txt/content"
r = requests.get(url, timeout=60)
lines = [l.strip() for l in r.text.split("\n") if l.strip()][:20]

print("Testing first 20 Mokhosi headlines after normalisation:\n")
import re, unicodedata

def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

passed = 0
for line in lines:
    norm = normalize(line)
    labels, scores = model.predict(norm, k=1)
    lang = labels[0].replace("__label__", "")
    score = round(float(scores[0]), 4)
    status = "✓" if lang == "sot_Latn" and score >= 0.7 else "✗"
    if lang == "sot_Latn" and score >= 0.7:
        passed += 1
    print(f"  {status} [{lang}: {score}] '{norm[:60]}'")

print(f"\nPassed: {passed}/20 at threshold 0.7")