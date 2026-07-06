import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 1: Loading verified corpus...")
print("█"*60)

df = pd.read_csv("sesotho_clean.csv")
print(f"  ✓ {len(df)} sentences loaded")
print(f"  Label distribution:")
print(df["label"].value_counts().to_string(header=False))

X = df["text"].tolist()
y = df["label"].tolist()

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2 — Train/test split
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 2: Train/test split (80/20, stratified)...")
print("█"*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✓ Train: {len(X_train)} | Test: {len(X_test)}")
print(f"  Train distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"  Test distribution:  {pd.Series(y_test).value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3 — Rule-based baseline (from your lit review Section 3.2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 3: Rule-based baseline...")
print("  Markers derived from Makutoane (2022) — your lit review Section 3.2")
print("█"*60)

LS_MARKERS  = [" ea ", " oa ", "li", " ch", "kh"]
SAS_MARKERS = [" ya ", " wa ", " kg", "tjh"]

def rule_based_classify(text):
    t = text.lower()
    ls_score  = sum(t.count(m) for m in LS_MARKERS)
    sas_score = sum(t.count(m) for m in SAS_MARKERS)
    return "LS" if ls_score > sas_score else "SAS"

rule_preds = [rule_based_classify(x) for x in X_test]
rule_f1 = f1_score(y_test, rule_preds, average="macro")

print(f"\n  Rule-based classification report:")
print(classification_report(y_test, rule_preds, target_names=["LS", "SAS"]))
print(f"  Macro F1: {round(rule_f1, 4)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 4 — Train character n-gram classifier
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 4: Training character n-gram classifier...")
print("  This is your novel pipeline component")
print("█"*60)

classifier = Pipeline([
    ("vectorizer", CountVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=1000,
        strip_accents=None,   # preserve diacritics — critical for LS detection
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0
    ))
])

classifier.fit(X_train, y_train)
print(f"  ✓ Classifier trained on {len(X_train)} sentences")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 5 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 5: Evaluation on held-out test set...")
print("█"*60)

learned_preds = classifier.predict(X_test)
learned_f1 = f1_score(y_test, learned_preds, average="macro")

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
print(f"  {'Your proposal target':<40} 0.85")
print()

if learned_f1 >= 0.85:
    print(f"  ✓ TARGET MET: F1 = {round(learned_f1, 4)}")
elif learned_f1 >= 0.75:
    print(f"  ⚠ F1 = {round(learned_f1, 4)} — below target, more data will close gap")
else:
    print(f"  ✗ F1 = {round(learned_f1, 4)} — needs investigation")

if learned_f1 > rule_f1:
    print(f"  ✓ Learned beats rule-based by {round(learned_f1 - rule_f1, 4)}")
else:
    print(f"  ⚠ Rule-based is competitive — interpretable baseline is strong")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 7 — What did the model actually learn?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 7: Top discriminating character n-grams...")
print("  Do these map back to Makutoane (2022)? They should.")
print("█"*60)

feature_names  = classifier.named_steps["vectorizer"].get_feature_names_out()
coefficients   = classifier.named_steps["clf"].coef_[0]

top_n = 10
top_ls_idx  = coefficients.argsort()[:top_n]
top_sas_idx = coefficients.argsort()[-top_n:][::-1]

print(f"\n  Top features predicting LS:")
for i in top_ls_idx:
    print(f"    '{feature_names[i]}':  coef = {round(coefficients[i], 3)}")

print(f"\n  Top features predicting SAS:")
for i in top_sas_idx:
    print(f"    '{feature_names[i]}':  coef = {round(coefficients[i], 3)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 8 — Demo: classify new sentences
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 8: Demo — classifying unseen sentences...")
print("█"*60)

demo_sentences = [
    # known LS patterns
    "Lipolesa li fumane monna ea shotseng tlung ea hae",
    "Lekhotla le ahlolitse motho oa leshome",
    # known SAS patterns
    "Motho ya yang sekolong wa ithuta taba tsa thuto",
    "Bana ba ya sekolong ka matsatsi a marobong",
    # ambiguous
    "Ke a leboha haholo thuso ena",
]

print(f"\n  {'Sentence':<55} {'Predicted'}")
print(f"  {'-'*65}")
for sentence in demo_sentences:
    pred = classifier.predict([sentence])[0]
    proba = classifier.predict_proba([sentence])[0]
    classes = classifier.classes_
    confidence = max(proba)
    print(f"  '{sentence[:52]}'")
    print(f"   → {pred} (confidence: {round(confidence, 3)})")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 9 — Save
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  CHECKPOINT 9: Saving model and results...")
print("█"*60)

with open("ortho_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
print("  ✓ Model saved: ortho_classifier.pkl")

results = {
    "corpus_size": len(df),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "label_distribution": df["label"].value_counts().to_dict(),
    "rule_based_f1": round(rule_f1, 4),
    "learned_f1": round(learned_f1, 4),
    "target_f1": 0.85,
    "target_met": bool(learned_f1 >= 0.85),
    "beats_baseline": bool(learned_f1 > rule_f1)
}
with open("classifier_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  ✓ Results saved: classifier_results.json")

print("\n" + "█"*60)
print("  ORTHOGRAPHIC CLASSIFIER COMPLETE")
print("  Key artifacts for your seminar:")
print("    sesotho_clean.csv         — verified labelled corpus")
print("    ortho_classifier.pkl      — trained classifier")
print("    classifier_results.json   — evaluation metrics")
print("█"*60)