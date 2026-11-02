"""Denoising filters compared in the filter ablation (fs = 1000 Hz)."""
import numpy as np, pywt
from scipy.signal import butter, filtfilt, wiener
FS=1000
_b,_a=butter(4,[20/(FS/2),450/(FS/2)],btype='band')
def f_raw(x): return x
def f_butter(x): return filtfilt(_b,_a,x)          # 4th-order Butterworth 20-450 Hz, zero-phase
def f_wiener(x): return wiener(x)
def f_kalman(x,R=15.0,Q=0.05):                     # position-velocity Kalman filter
    F=np.array([[1,1/FS],[0,1]]); H=np.array([[1.0,0]]); Qm=Q*np.eye(2)
    s=np.array([x[0],0.0]); P=np.eye(2); out=np.empty_like(x)
    for i,z in enumerate(x):
        s=F@s; P=F@P@F.T+Qm
        K=P@H.T/(H@P@H.T+R); s=s+(K*(z-H@s)).ravel(); P=(np.eye(2)-K@H)@P; out[i]=s[0]
    return out
def _wav(x,levels,sigma_mode='D1',alpha=0.47):     # db4, J=4, level-adaptive soft threshold
    c=pywt.wavedec(x,'db4',level=4)                  # [A4,D4,D3,D2,D1]
    d1=c[-1]; sD1=np.median(np.abs(d1-np.median(d1)))/0.6745
    names={1:4,2:3,3:2,4:1}
    for i in range(1,5):
        L=names[i]
        if L not in levels: continue
        cl=c[i]; N=len(cl)
        s=sD1 if sigma_mode=='D1' else np.median(np.abs(cl-np.median(cl)))/0.6745
        if s<=0: continue
        Tu=s*np.sqrt(2*np.log(N)); E=np.mean(cl**2); W=np.mean(np.abs(cl)); mk=E/(E+s**2)
        rng=cl.max()-cl.min(); Tm=mk*Tu*W/rng if rng>0 else 0
        To=alpha*Tu+(1-alpha)*Tm
        c[i]=np.sign(cl)*np.maximum(np.abs(cl)-To,0)
    return pywt.waverec(c,'db4')[:len(x)]
FILTERS={"raw":f_raw,"butter":f_butter,"kalman":f_kalman,"wiener":f_wiener,
 "wav4_sl":lambda x:_wav(x,{1,2,3,4},'level'),"wav4_sD1":lambda x:_wav(x,{1,2,3,4}),
 "wav_D1D2":lambda x:_wav(x,{1,2}),"wav_D4":lambda x:_wav(x,{4}),"wav_D3D4":lambda x:_wav(x,{3,4})}
