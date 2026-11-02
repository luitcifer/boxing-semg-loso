"""18 sEMG + context features per segment (FFT-based), as used in the manuscript."""
import numpy as np, pandas as pd
def calculate_emg_features(data, fs=1000):
    data = data.fillna(0)
    signal = data['signal'].values.astype(float)
    bmi, uzun, hrate = data.loc[0,'bmi'], data.loc[0,'uzun'], data.loc[0,'hrate']
    N=len(signal); X=np.fft.rfft(signal); freqs=np.fft.rfftfreq(N,d=1.0/fs); PSD=np.abs(X)**2/N
    m0=PSD.sum(); MNF=(freqs*PSD).sum()/m0 if m0 else 0
    amp=np.abs(X); sc=(freqs*amp).sum()/amp.sum() if amp.sum() else 0
    MDF=freqs[np.where(np.cumsum(PSD)>=m0/2.0)[0][0]]
    sv=((freqs-MNF)**2*PSD).sum()/m0 if m0 else 0; sv=sv/(N-1); bw=np.sqrt(sv)
    v=freqs>0; slope,_=np.polyfit(freqs[v],np.log10(amp[v]+1e-8),1)
    lo=PSD[(freqs>=20)&(freqs<60)].sum(); hi=PSD[(freqs>=60)&(freqs<=500)].sum(); bpr=lo/hi if hi else np.nan
    RMS=np.sqrt(np.mean(signal**2)); MAV=np.mean(np.abs(signal)); var=np.var(signal); WL=np.abs(np.diff(signal)).sum()
    rng=np.ptp(signal); th=0.01*rng if rng>0 else 0.01
    d1=signal[1:-1]-signal[:-2]; d2=signal[2:]-signal[1:-1]
    ssc=np.sum((d1*d2<0)&((np.abs(d1)>th)|(np.abs(d2)>th)))/(N-1)
    ZCR=len(np.where(np.diff(np.sign(signal)))[0]); MAD=np.median(np.abs(signal-np.median(signal)))
    wamp=np.sum(np.abs(np.diff(signal))>th)
    return [MDF,MNF,sc,sv,bw,slope,bpr,RMS,MAV,var,WL,ssc,ZCR,MAD,wamp,bmi,uzun,hrate]
