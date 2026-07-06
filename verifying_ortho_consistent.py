import pandas as pd

df = pd.read_csv("sesotho_clean.csv")

# Makutoane (2022): ea/oa/li are LS markers, ya/wa are SAS markers
LS_MARKERS  = [" ea ", " oa ", " kg ", " tjh ", " tsh ", " d "]
SAS_MARKERS = [" ya ", " wa ", " kh ", " ch ", " tš ", " l "]

def count_markers(text, markers):
    text_lower = text.lower()
    return sum(text_lower.count(m) for m in markers)

df["ls_marker_count"]  = df["text"].apply(lambda x: count_markers(x, LS_MARKERS))
df["sas_marker_count"] = df["text"].apply(lambda x: count_markers(x, SAS_MARKERS))

print("=== Do source-based labels agree with spelling patterns? ===\n")

for label in ["LS", "SAS"]:
    subset = df[df["label"] == label]
    avg_ls  = subset["ls_marker_count"].mean()
    avg_sas = subset["sas_marker_count"].mean()
    print(f"{label} sentences ({len(subset)} total):")
    print(f"  avg LS markers per sentence:  {round(avg_ls, 3)}")
    print(f"  avg SAS markers per sentence: {round(avg_sas, 3)}")
    agreement = "✓ consistent" if (label == "LS" and avg_ls > avg_sas) or \
                                   (label == "SAS" and avg_sas > avg_ls) else "✗ inconsistent"
    print(f"  → {agreement} with source label\n")

print("=== Sample sentences with marker counts ===\n")
print("LS samples:")
for _, row in df[df["label"]=="LS"].head(5).iterrows():
    print(f"  '{row['text'][:60]}'")
    print(f"   LS markers: {row['ls_marker_count']} | SAS markers: {row['sas_marker_count']}")

print("\nSAS samples:")
for _, row in df[df["label"]=="SAS"].head(5).iterrows():
    print(f"  '{row['text'][:60]}'")
    print(f"   LS markers: {row['ls_marker_count']} | SAS markers: {row['sas_marker_count']}")