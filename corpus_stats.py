import pandas as pd, numpy as np, re, glob, random, json, collections, os, sys
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- EDIT THESE PATHS ----------
BASE      = "/home/karabo/Honours_Project/"
CURATED   = BASE + "sesotho_clean.csv"
NCHLT_TXT = BASE + "st/2.Corpora/CORP.NCHLT.st.CLEAN.2.0.txt"   # the NCHLT TEXT corpus, not the NER file
# Optional: leave as is if you don't have them; those parts are skipped
VUK_DIR   = BASE + "vukuzenzele-nlp/data/simple_align_output/"   # git clone https://github.com/dsfsi/vukuzenzele-nlp
NER_FILE  = BASE + "NCHLT Sesotho Named Entity Annotated Corpus/Dataset.NCHLT-II.st.NER.Full.txt"

# --------------------------------------

for p in [CURATED, NCHLT_TXT]:
    if not os.path.exists(p): sys.exit(f"Not found: {p}")

N=8626
# corpora
cur=pd.read_csv(CURATED,encoding="utf-8")
curated=cur.text.astype(str).tolist()
nchlt=[]
for line in open(NCHLT_TXT,encoding="utf-8"):
    s=line.strip()
    if not s or s.startswith("<") or s.startswith("_"): continue
    nchlt.append(s)
raw=[]
if os.path.isdir(VUK_DIR):
    for f in sorted(glob.glob(VUK_DIR+"*.csv")):
        d=pd.read_csv(f,on_bad_lines="skip")
        if len(d.columns)>=2: raw+=d.iloc[:,1].dropna().astype(str).tolist()
print(f"Loaded: curated={len(curated)}  nchlt={len(nchlt)}  raw={len(raw)}")
random.seed(42); nchlt_m=random.sample(nchlt,N)
C={"Curated":curated,"NCHLT (size-matched)":nchlt_m}
if len(raw)>=N:
    random.seed(42); C["Raw (unfiltered)"]=random.sample(raw,N)
else:
    print("Raw skipped (Vuk'uzenzele folder not found)")

norm=lambda s: re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",s.lower())).strip()
tok=lambda s: re.findall(r"[^\W\d_]+",s.lower())
EN=set("the and of to is in for that with are was this on be by as it from at an have has not".split())

# NER test vocab
random.seed(42)
S,t,l=[],[],[]
if os.path.exists(NER_FILE):
    for line in open(NER_FILE,encoding="utf-8"):
        line=line.rstrip()
        if not line:
            if t: S.append(t); t=[]
        else: t.append(line.split()[0])
    if t: S.append(t)
else:
    print("NER coverage skipped (NER file not found)")
random.shuffle(S); n=len(S); test=S[int(.8*n)+int(.1*n):]
test_tokens=[w for s in test for w in tok(" ".join(s))]

out={}
for name,sents in C.items():
    toks=[tok(s) for s in sents]; flat=[w for x in toks for w in x]
    nn=[norm(s) for s in sents]
    exact=1-len(set(sents))/len(sents)
    near=1-len(set(nn))/len(nn)
    # STTR over 1000-token windows
    w=[len(set(flat[i:i+1000]))/1000 for i in range(0,len(flat)-999,1000)]
    sttr=float(np.mean(w))
    vocab100k=len(set(flat[:100000]))
    alltext="".join(sents); chars=[c for c in alltext if not c.isspace()]
    nonalpha=sum(1 for c in chars if not c.isalpha())/len(chars)
    lens=[len(s.split()) for s in sents]
    short=np.mean([L<5 for L in lens]); long=np.mean([L>100 for L in lens])
    junk=np.mean([bool(re.search(r"�|Ã|â€|√",s)) for s in sents])
    eng=np.mean([ (sum(x in EN for x in tk)/max(len(tk),1))>=0.2 for tk in toks])
    V=set(flat)
    oov_type=float(np.mean([w not in V for w in set(test_tokens)])) if test_tokens else None
    oov_tok=float(np.mean([w not in V for w in test_tokens])) if test_tokens else None
    out[name]=dict(sentences=len(sents),tokens=len(flat),exact_dup=exact,near_dup=near,sttr=sttr,vocab_per_100k=vocab100k,
        nonalpha=nonalpha,short=short,long=long,junk=junk,english=eng,oov_type=oov_type,oov_token=oov_tok,mean_len=float(np.mean(lens)))

# purity: LID + orthography confidence (curated only)
out["Curated"]["lid_mean"]=float(cur.lid_confidence.mean())
out["Curated"]["lid_ge_09"]=float((cur.lid_confidence>=0.9).mean())
y=cur.label.where(cur.source!="Nalibali_LS","SAS").tolist()
X=[" "+norm(s)+" " for s in curated]
clf=Pipeline([("v",CountVectorizer(analyzer="char_wb",ngram_range=(2,4),max_features=1000)),("c",LogisticRegression(max_iter=1000,class_weight="balanced"))])
p=cross_val_predict(clf,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),method="predict_proba")
conf=p.max(1)
out["Curated"]["ortho_conf_ge_09"]=float((conf>=0.9).mean())
out["Curated"]["ortho_conf_mean"]=float(conf.mean())
out["_meta"]=dict(ner_test_types=len(set(test_tokens)),ner_test_tokens=len(test_tokens),nchlt_full=len(nchlt))
json.dump(out,open("corpus_stats.json","w"),indent=1)
for k,v in out.items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})