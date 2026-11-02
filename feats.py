"""Step 1: filter each raw segment and compute features -> feat_<filter>.csv
Usage: python feats.py            (all filters)   or   python feats.py raw butter wav_D3D4"""
import numpy as np,pandas as pd,sys,time
sys.path.insert(0,'.'); from features_author import calculate_emg_features; from filters import FILTERS
s=pd.read_csv('segments_all.csv')
fr=pd.read_csv('features_raw.csv')[['file','athlete_id','label','muscle','segment']]
s=s.merge(fr,on='file',suffixes=('_s',''))
sigs=[np.array(v.split(),float) for v in s.signal]
cols=["MDF","MNF","spectral_centroid","spectral_variance","bandwidth","slope","band_power_ratio","RMS","MAV","var","WL","SSC","ZCR","MAD","WAMP","BMI","uzun","hrate"]
names=sys.argv[1:] or list(FILTERS)
for k in names:
    t=time.time(); rows=[]
    for sig,(i,r) in zip(sigs,s.iterrows()):
        y=FILTERS[k](sig); d=pd.DataFrame({'signal':y,'bmi':r.bmi,'uzun':r.uzun,'hrate':r.hrate})
        v=calculate_emg_features(d)
        X=np.fft.rfft(y); fq=np.fft.rfftfreq(len(y),1/1000); P=np.abs(X)**2; tot=P.sum()
        rows.append(v+[P[(fq>=125)&(fq<250)].sum()/tot, P[fq>=250].sum()/tot])
    o=pd.DataFrame(rows,columns=cols+["pf_125_250","pf_250"]); o=pd.concat([s[['file','athlete_id','label','muscle','segment']].reset_index(drop=True),o],axis=1)
    o["n_samples"]=o.uzun; o.to_csv(f"feat_{k}.csv",index=False); print(k,round(time.time()-t),flush=True)
