"""Step 5: pool nested-LOSO predictions, athlete-level bootstrap 95% CI (2000 resamples), selection counts."""
import json,glob,numpy as np,pandas as pd,collections
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
f=pd.read_csv("feat_raw.csv"); y=f.label.values; A=f.athlete_id.values; p=np.full(len(y),np.nan); sel=[]
for fn in glob.glob("nest_*.json"):
    d=json.load(open(fn)); p[d["idx"]]=d["p"]; sel.append((d["fold"],d["filter"],d["k"],d["feats"]))
assert not np.isnan(p).any()
ids=np.unique(A); idx={a:np.where(A==a)[0] for a in ids}; r=np.random.default_rng(0); F=[];U=[];C=[]
for _ in range(2000):
    s=np.concatenate([idx[a] for a in r.choice(ids,len(ids))]); F.append(f1_score(y[s],p[s]>=.5)); U.append(roc_auc_score(y[s],p[s])); C.append(accuracy_score(y[s],p[s]>=.5))
print("F1",round(f1_score(y,p>=.5),3),np.percentile(F,[2.5,97.5]).round(3)); print("Acc",round(accuracy_score(y,p>=.5),3),np.percentile(C,[2.5,97.5]).round(3)); print("AUC",round(roc_auc_score(y,p),3),np.percentile(U,[2.5,97.5]).round(3))
print(collections.Counter(s[1] for s in sel)); print(collections.Counter(s[2] for s in sel))
fc=collections.Counter(x for s in sel for x in s[3]); print(fc.most_common())
pa=[roc_auc_score(y[A==a],p[A==a]) for a in ids]; print("per-athlete median",np.median(pa).round(3),"min",np.min(pa).round(3))
np.save("p_nested.npy",p)
