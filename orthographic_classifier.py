import re
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# Strips domain artifacts before classification so the model learns
# orthographic patterns (ea/oa vs ya/wa) rather than formatting signals
# (ALL CAPS headlines vs prose, periods vs no periods).
# ══════════════════════════════════════════════════════════════════════════════
# In normalize_for_classification, pad the text with spaces so boundary markers work
def normalize_for_classification(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " " + text + " "   # pad with spaces so " ea " matches at boundaries

# And update markers to be cleaner
LS_MARKERS  = [" ea ", " oa ", " li", " kh", " ch"]
SAS_MARKERS = [" ya ", " wa ", " kg", "tjh", " di"]

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load corpus
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading verified corpus...")
print("█"*60)

df = pd.read_csv("sesotho_clean.csv")
print(f"  {len(df)} sentences loaded")
print(f"  Label distribution:")
print(df["label"].value_counts().to_string(header=False))

# apply normalisation to the text column
# original text is preserved in df — we only normalise for classification
X_raw = df["text"].tolist()
y     = df["label"].tolist()

X = [normalize_for_classification(t) for t in X_raw]

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — Train/test split
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Train/test split (80/20, stratified)...")
print("█"*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print(f"  Train distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"  Test distribution:  {pd.Series(y_test).value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — Rule-based baseline (Makutoane 2022 markers)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Rule-based baseline...")
print("  Markers from Makutoane (2022) — applied after normalisation")
print("█"*60)

# Substitution rules from Makutoane (2022) Table 1
# Applied on normalised (lowercase, no punctuation) text
# Spaces around markers prevent partial matches inside longer words
#LS_MARKERS  = [" ea ", " oa ", " li", " ch", "kh"]
#SAS_MARKERS = [" ya ", " wa ", " kg", "tjh", " di"]

def rule_based_classify(text):
    # text is already normalised before calling this
    ls_score  = sum(text.count(m) for m in LS_MARKERS)
    sas_score = sum(text.count(m) for m in SAS_MARKERS)
    return "LS" if ls_score > sas_score else "SAS"

rule_preds = [rule_based_classify(x) for x in X_test]
rule_f1    = f1_score(y_test, rule_preds, average="macro")

print(f"\n  Rule-based classification report:")
print(classification_report(y_test, rule_preds, target_names=["LS", "SAS"]))
print(f"  Macro F1: {round(rule_f1, 4)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Train character n-gram classifier
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Training character n-gram classifier...")
print("  Features: char_wb n-grams (2-4), max 1000 features")
print("  strip_accents=None — preserves LS diacritics")
print("█"*60)

classifier = Pipeline([
    ("vectorizer", CountVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=1000,
        strip_accents=None,    # critical — LS diacritics are orthographic signal
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0
    ))
])

classifier.fit(X_train, y_train)
print(f"  ✓ Classifier trained on {len(X_train)} normalised sentences")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Evaluate on held-out test set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Evaluation on held-out test set...")
print("█"*60)

learned_preds = classifier.predict(X_test)
learned_f1    = f1_score(y_test, learned_preds, average="macro")

print(f"\n  Learned classifier report:")
print(classification_report(y_test, learned_preds, target_names=["LS", "SAS"]))

cm = confusion_matrix(y_test, learned_preds, labels=["LS", "SAS"])
print(f"  Confusion matrix:")
print(f"                Predicted LS   Predicted SAS")
print(f"  Actual LS         {cm[0][0]:<13}  {cm[0][1]}")
print(f"  Actual SAS        {cm[1][0]:<13}  {cm[1][1]}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 6 — Compare baseline vs learned
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 6: Baseline vs learned classifier...")
print("█"*60)

print(f"\n  {'Method':<40} {'Macro F1'}")
print(f"  {'-'*50}")
print(f"  {'Rule-based (Makutoane markers)':<40} {round(rule_f1, 4)}")
print(f"  {'Character n-gram + Logistic Regression':<40} {round(learned_f1, 4)}")
print(f"  {'Proposal target':<40} 0.85")
print()

if learned_f1 >= 0.85:
    print(f"  ✓ TARGET MET: F1 = {round(learned_f1, 4)}")
elif learned_f1 >= 0.75:
    print(f"  ⚠ F1 = {round(learned_f1, 4)} — below target, needs more data")
else:
    print(f"  ✗ F1 = {round(learned_f1, 4)} — investigate features")

if learned_f1 > rule_f1:
    print(f"  ✓ Learned beats rule-based by {round(learned_f1 - rule_f1, 4)}")
    print(f"    This confirms the model captures patterns beyond Makutoane's explicit rules")
else:
    print(f"  Rule-based is competitive — explicit markers may be sufficient")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 7 — Interpretability: what did the model learn?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 7: Top discriminating character n-grams...")
print("  These should match Makutoane (2022) rules if normalisation worked")
print("█"*60)

feature_names = classifier.named_steps["vectorizer"].get_feature_names_out()
coefficients  = classifier.named_steps["clf"].coef_[0]

top_n = 10
top_ls_idx  = coefficients.argsort()[:top_n]
top_sas_idx = coefficients.argsort()[-top_n:][::-1]

print(f"\n  Top features predicting LS:")
for i in top_ls_idx:
    print(f"    '{feature_names[i]}':  coef = {round(coefficients[i], 3)}")

print(f"\n  Top features predicting SAS:")
for i in top_sas_idx:
    print(f"    '{feature_names[i]}':  coef = {round(coefficients[i], 3)}")

print(f"\n  Expected LS features (Makutoane 2022): ea, oa, kh, ch, li")
print(f"  Expected SAS features (Makutoane 2022): ya, wa, kg, tjh")
print(f"  If top features match — classifier learned orthography, not formatting")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 8 — Demo on unseen sentences
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 8: Demo — classifying unseen sentences...")
print("  Sentences are normalised before classification (same as training)")
print("█"*60)

demo_sentences = [
    # known LS — ea, oa markers
    ("Lipolesa li fumane monna ea shotseng tlung ea hae",    "LS"),
    ("Lekhotla le ahlolitse motho oa leshome",               "LS"),
    # known SAS — ya, wa markers
    ("Motho ya yang sekolong wa ithuta taba tsa thuto",      "SAS"),
    ("Bana ba ya sekolong ka matsatsi a marobong",           "SAS"),
    # ambiguous — no strong markers either way
    ("Ke a leboha haholo thuso ena",                         "?"),
]

print(f"\n  {'Sentence':<52} {'Expected':<10} {'Predicted':<10} {'Conf'}")
print(f"  {'-'*80}")
correct = 0
total_known = 0
for sentence, expected in demo_sentences:
    norm = normalize_for_classification(sentence)
    pred    = classifier.predict([norm])[0]
    proba   = classifier.predict_proba([norm])[0]
    conf    = round(max(proba), 3)
    match   = "✓" if pred == expected else ("?" if expected == "?" else "✗")
    if expected != "?":
        total_known += 1
        if pred == expected:
            correct += 1
    print(f"  '{sentence[:50]}'")
    print(f"   Expected: {expected:<8} Predicted: {pred:<8} Conf: {conf}  {match}")

print(f"\n  Demo accuracy on known sentences: {correct}/{total_known}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 9 — Save model and results
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 9: Saving model and results...")
print("█"*60)

# save the full pipeline including normalisation info
# NOTE: normalisation is applied externally — wrap it for production use
with open("ortho_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
print("  ✓ Model saved: ortho_classifier.pkl")

results = {
    "corpus_size":        len(df),
    "train_size":         len(X_train),
    "test_size":          len(X_test),
    "label_distribution": df["label"].value_counts().to_dict(),
    "normalisation":      "lowercase + strip_punctuation (preserves diacritics)",
    "rule_based_f1":      round(rule_f1, 4),
    "learned_f1":         round(learned_f1, 4),
    "target_f1":          0.85,
    "target_met":         bool(learned_f1 >= 0.85),
    "beats_baseline":     bool(learned_f1 > rule_f1),
    "demo_accuracy":      f"{correct}/{total_known}"
}
with open("classifier_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  ✓ Results saved: classifier_results.json")

print("\n" + "█"*60)
print("  ORTHOGRAPHIC CLASSIFIER COMPLETE")
print("  Key change from previous version:")
print("  → normalize_for_classification() strips ALL CAPS and punctuation")
print("    so the model learns ea/oa/ya/wa patterns, not headline formatting")
print("█"*60)