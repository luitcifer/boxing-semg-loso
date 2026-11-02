"""Step 4: nested LOSO (primary result). Outer: 31 athletes; inner: 3-fold athlete-wise CV on 30 athletes.
Selected inside each outer fold: filter {raw, butter, wav_D3D4}, k in {5,10,15}, features ranked by Welch |t|.
Usage: python nested.py 1 2 3 ... 31   (writes nest_<fold>.json)"""
import numpy as np,pandas as pd,sys,json,warnings,time; warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
MYO=["RMS","MAV","var","WL","SSC","ZCR","MAD","WAMP","MDF","MNF","spectral_centroid","spectral_variance","bandwidth","slope","band_power_ratio"]
FILT=["raw","butter","wav_D3D4"]; KS=[5,10,15]
D={}
for k in FILT:
    f=pd.read_csv(f"feat_{k}.csv")
    for c in ["WL","ZCR","WAMP"]: f[c]=f[c]/f.n_samples   # length-invariant features
    D[k]=f
CAND=MYO+["pf_125_250","pf_250"]
y=D["raw"].label.values; A=D["raw"].athlete_id.values; ids=np.unique(A)
def mlp(): return MLPClassifier((64,16),solver="adam",learning_rate_init=1e-3,batch_size=32,max_iter=100,random_state=0)
def rank(X,yy): t=np.abs(ttest_ind(X[yy==1],X[yy==0],equal_var=False).statistic); return np.argsort(-t)
def fitpred(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr); return mlp().fit(sc.transform(Xtr),ytr).predict_proba(sc.transform(Xte))[:,1]
folds=[int(a) for a in sys.argv[1:]] or [int(a) for a in ids]
for a in folds:
    t=time.time(); tr_ids=ids[ids!=a]; rng=np.random.default_rng(int(a)); perm=rng.permutation(tr_ids); groups=np.array_split(perm,3)
    scores={}
    for k in FILT:
        X=D[k][CAND].values.astype(float)
        for kk in KS:
            aucs=[]
            for g in groups:
                itr=np.isin(A,tr_ids)&~np.isin(A,g); iva=np.isin(A,g)
                sel=rank(X[itr],y[itr])[:kk]; p=fitpred(X[itr][:,sel],y[itr],X[iva][:,sel]); aucs.append(roc_auc_score(y[iva],p))
            scores[(k,kk)]=np.mean(aucs)
    bk,bkk=max(scores,key=scores.get); X=D[bk][CAND].values.astype(float); tr=A!=a
    sel=rank(X[tr],y[tr])[:bkk]; p=fitpred(X[tr][:,sel],y[tr],X[~tr][:,sel])
    json.dump(dict(fold=int(a),filter=bk,k=bkk,feats=[CAND[i] for i in sel],inner={f"{x[0]}|{x[1]}":v for x,v in scores.items()},idx=np.where(~tr)[0].tolist(),p=p.tolist()),open(f"nest_{a}.json","w"))
    print(a,bk,bkk,round(roc_auc_score(y[~tr],p),3),round(time.time()-t),flush=True)
