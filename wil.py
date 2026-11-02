"""Step 3: paired Wilcoxon signed-rank tests of per-athlete ROC-AUC vs raw signal, Holm correction.
Requires p_<filter>_semg.npy from abl.py for all filters."""
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
f=pd.read_csv("feat_raw.csv"); y=f.label.values; A=f.athlete_id.values; ids=np.unique(A)
pa=lambda p: np.array([roc_auc_score(y[A==a],p[A==a]) for a in ids])
ks=["butter","kalman","wiener","wav4_sl","wav4_sD1","wav_D1D2","wav_D4","wav_D3D4"]
base=pa(np.load("p_raw_semg.npy")); res=[]
for k in ks:
    v=pa(np.load(f"p_{k}_semg.npy")); w=wilcoxon(v,base,zero_method="wilcox")
    res.append([k,np.median(base),np.median(v),np.median(v-base),w.statistic,w.pvalue])
r=pd.DataFrame(res,columns=["filter","med_raw","med_f","med_diff","W","p"]).sort_values("p")
m=len(r); ph=[]; run=0
for i,pv in enumerate(r.p.values): run=max(run,min(1,(m-i)*pv)); ph.append(run)
r["p_holm"]=ph
b=pa(np.load("p_butter_semg.npy")); d=pa(np.load("p_wav_D3D4_semg.npy")); w=wilcoxon(b,d)
print(r.round(4).to_string(index=False)); print("butter vs D3D4: med diff",np.median(b-d).round(4),"W",w.statistic,"p",round(w.pvalue,4))
r.to_csv("filter_wilcoxon.csv",index=False)
