"""Step 2: filter ablation, 31-fold LOSO, MLP 64-16 (Table 6).
Usage: python abl.py <filter> <semg|full>     e.g. python abl.py butter semg"""
import numpy as np,pandas as pd,sys,warnings,time; warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score
MYO=["RMS","MAV","var","WL","SSC","ZCR","MAD","WAMP","MDF","MNF","spectral_centroid","spectral_variance","bandwidth","slope","band_power_ratio"]
k,fs=sys.argv[1],sys.argv[2]; t=time.time()
f=pd.read_csv(f"feat_{k}.csv")
cols=MYO if fs=="semg" else MYO+["uzun","BMI","hrate"]
X=f[cols].values.astype(float); y=f.label.values; A=f.athlete_id.values; p=np.zeros(len(y))
for a in np.unique(A):
    tr=A!=a; sc=StandardScaler().fit(X[tr])          # scaler fitted on training athletes only
    m=MLPClassifier((64,16),solver="adam",learning_rate_init=1e-3,batch_size=32,max_iter=100,random_state=0).fit(sc.transform(X[tr]),y[tr])
    p[~tr]=m.predict_proba(sc.transform(X[~tr]))[:,1]
np.save(f"p_{k}_{fs}.npy",p)
print(k,fs,round(f1_score(y,p>=.5),3),round(roc_auc_score(y,p),3),round(time.time()-t),flush=True)
