# Run this as a quick diagnostic script — save as diagnose.py
import pandas as pd
import requests
import io
import fasttext
from huggingface_hub import hf_hub_download

# load model
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)

# ── Test 1: what confidence do the LS headlines actually get? ─────────────────
print("=== TEST 1: GlotLID confidence on known LS sentences ===")
test_ls = [
    "MAEMONG A NEPAHETSENG HA HO MOTHO EA KA JANG NA",
    "MOKHA HA O TLISE PHETOHO HOHANG",
    "Lipolesa li fumane monna ea shotseng tlung ea hae",
    "Lekhotla le ahlolitse motho oa leshome le metso e meraro",
]
for s in test_ls:
    labels, scores = model.predict(s, k=3)
    print(f"\n  Text: '{s}'")
    for l, sc in zip(labels, scores):
        print(f"    {l.replace('__label__', '')}: {round(float(sc), 4)}")

# ── Test 2: check Vukuzenzele GitHub structure ────────────────────────────────
print("\n\n=== TEST 2: Vukuzenzele GitHub structure ===")
api_url = "https://api.github.com/repos/dsfsi/vukuzenzele-nlp/contents/"
r = requests.get(api_url, timeout=30)
items = r.json()
for item in items:
    print(f"  {item['type']}: {item['name']}")

# ── Test 3: check Zenodo file contents ───────────────────────────────────────
print("\n\n=== TEST 3: Zenodo SN dataset file contents ===")
zenodo_url = "https://zenodo.org/api/records/10531959"
r = requests.get(zenodo_url, timeout=30)
record = r.json()
for f in record.get("files", []):
    print(f"  File: {f['key']} | Size: {f['size']} bytes | URL: {f['links']['self']}")


df = pd.read_csv("sesotho_clean.csv")
print(f"Total sentences: {len(df)}")
print(df["label"].value_counts())
print("\nLS sample:")
print(df[df["label"]=="LS"]["text"].head(5).to_string(index=False))
print("\nSAS sample:")
print(df[df["label"]=="SAS"]["text"].head(5).to_string(index=False))