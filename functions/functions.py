from scipy.signal import spectrogram, buttord, butter, sosfilt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from pywt import wavedec, dwt
from scipy import fft, signal
from statistics import mode
import scipy.stats as sst
from math import log2
import pandas as pd
import numpy as np


def sgram_part(
    sig,
    fs,
    win_width,
    frs,
    overlap=0,
    window='hann',
):
    f, t, Sxx = spectrogram(
        sig,
        fs=fs,
        nperseg=int(fs * win_width),
        noverlap=int(fs * win_width * overlap / 100),
        window=window,
    )

    df = f[1] - f[0]
    if frs[1] < 0:
        f = f[int(frs[0] / df):]
        Sxx = Sxx[int(frs[0] / df):, :]
    else:
        f = f[int(frs[0] / df) : (int(frs[1] / df) + 1)]
        Sxx = Sxx[int(frs[0] / df) : (int(frs[1] / df) + 1), :]

    PSD = []
    for i in range(len(t)):
        PSD.append(np.mean(Sxx[:, i]))

    return f, t, Sxx, PSD


def spear(dfX, dfY):
    dfYs = []
    labels = np.unique(np.array(dfY))
    for label in labels:
        dfYt = dfY.copy()
        dfYt[dfY != label] = 0
        dfYt[dfY == label] = 1
        dfYs.append(dfYt)
    names = np.array(dfX.columns)
    ret = []
    for i in names:
        if all(dfX.loc[:, i]) == dfX.loc[1, i]:
            print(f"{i} - constant")
            ret.append(0)
        else:
            r = [abs(sst.spearmanr(dfX.loc[:, i], dfY).statistic) for dfY in dfYs]
            ret.append(np.mean(r))
    return [x for _, x in sorted(zip(ret, names), reverse=True)], sorted(ret, reverse=True)


def r_peaks(ecg,fs,maxiter=100):
  sos = signal.butter(15, [1, 20], btype='bandpass', fs=fs, output="sos")
  ecg_prep = signal.sosfilt(sos, ecg)
  y = signal.filtfilt([50/fs]*int(fs/50), 1, ecg_prep)
  t = np.arange(len(y))/fs

  [_, cD] = dwt(y,'sym3')
  fs_c = fs*len(cD)/len(y)
  t_c = np.linspace(0,max(t),len(cD))

  cD2 = signal.medfilt(cD*cD,11)
  cD2 /= max(cD2)
  
  win_l = int(fs_c)
  for j in range(0,len(cD2)-win_l,int(win_l/2)):
    cD2[j:j+win_l] = cD2[j:j+win_l]/max(cD2[j:j+win_l])

  
  locs = signal.find_peaks(cD2,height=0.5,distance=int(0.3*fs_c))
  locs = np.array([int(x/fs_c*fs) for x in locs[0]])

  locs_tmp = locs.copy()
  locs_array = []
  locs_array.append(locs.copy())

  eq = False
  win_l = int(np.min(locs[1:-1]-locs[0:-2])/1.5)
  for j in range(maxiter):
    for i, loc in enumerate(locs[1:]):
      if locs[i]-locs[i-1] > np.mean(locs[1:]-locs[0:-1])*1.5:
        locs = np.append(locs,int((locs[i]+locs[i-1])/2))
        locs = np.sort(locs)
    
    for locs_in_arr in locs_array:
      if np.array_equal(locs_in_arr,locs):
        eq = True
        break
    if eq:
      break
    if np.array_equal(locs_tmp,locs):
      break
    else:
      locs_array.append(locs.copy())
      locs_tmp = locs

  locs_tmp = locs.copy()
  locs_array = []
  locs_array.append(locs.copy())

  eq = False
  win_l = int(np.min(locs[1:-1]-locs[0:-2])/1.25)
  for j in range(maxiter):
    for i, loc in enumerate(locs):
      win_start = max([0,loc-int(win_l/2)])
      win_stop = min([len(ecg),loc+int(win_l/2)])
      locs[i] = np.argmax(ecg[win_start:win_stop])+win_start
    
    for locs_in_arr in locs_array:
      if np.array_equal(locs_in_arr,locs):
        eq = True
        break
    if eq:
      break
    if np.array_equal(locs_tmp,locs):
      break
    else:
      locs_array.append(locs.copy())
      locs_tmp = locs

  locs = np.sort(np.unique(locs))
  time_locs = locs/fs
  RR = time_locs[1:] - time_locs[0:-1]
  RR = np.round(RR,3)

  return locs, time_locs, RR


def zeros_remover(df, labels=False):
    df2 = df.copy()
    c = 0
    if type(labels) is bool:
        labels = df.columns

    for i in labels:
        if any(df[i].isin([0])):
            print(f"Zero in {i}")
            c = c + 1
            median = df[i].median()
            df2[i] = df[i].replace(0, median)

    print(f"Replaced zero in {c} columns\n")
    return df2


def null_remover(df, labels=False):
    c = 0
    if type(labels) is bool:
        labels = df.columns

    for i in labels:
        if any(df[i].isnull()):
            if pd.api.types.is_numeric_dtype(df[i]):
                print(f"Empty in {i}")
                c = c + 1
                median = df[i].median()
                df[i] = df[i].fillna(median)
            else:
                c = c + 1
                mode = df[i].mode()[0]
                print(f"Empty in {i}")
                df[i] = df[i].fillna(mode)

    print(f"Replaced empty in {c} columns\n")
    return df


def nan_remover(df, labels=False):
    if type(labels) is bool:
        labels = df.columns[:-1]

    for i in labels:
        if pd.api.types.is_numeric_dtype(df[i]):
            continue
        
        try:
            numeric_column = pd.to_numeric(df[i], errors='coerce')
            if numeric_column.isnull().all():
                raise ValueError
            elif numeric_column.isnull().any():
                print(i)
                median_val = numeric_column.median()
                df[i] = numeric_column.fillna(median_val)

        except (TypeError, ValueError):
            unique_values = df[i].unique()
            if len(unique_values) > 2:
                le = LabelEncoder()
                df[i] = le.fit_transform(df[i].astype(str))
            else:
                mapping = {value: index for index, value in enumerate(unique_values)}
                df[i] = df[i].map(mapping)

    return df


def outliers_remover(df, max_iterations=5, labels=False):
    if type(labels) is bool:
        labels = df.columns

    for iteration in range(max_iterations):
        print(f"Iteration {iteration+1}")
        c = 0
        if iteration == 0:
            dfX = df[labels[:-1]].copy()

        for i in labels[:-1]:
            if len(dfX[i].unique()) < 10:
                print(f"{i} is categorical")
            else:
                Q1 = np.percentile(dfX[i], 25)
                Q3 = np.percentile(dfX[i], 75)
                IQR = Q3 - Q1

                min_val = Q1 - 1.5 * IQR
                max_val = Q3 + 1.5 * IQR
                outliers = (dfX[i] < min_val) | (dfX[i] > max_val)
                if outliers.any():
                    c += 1
                    dfX.loc[outliers, i] = None
            dfX[i].fillna(value=dfX[i].mode(), inplace=True)

        if c == 0:
            print("No outliers found\n")
            break

        print(f"Replaced outliers in {c} columns\n")
        df.update(dfX)

    return df


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[counts.argmax()]


def skewness(data):
    if len(data) < 3:
        return 0
    mean = data.mean()
    m3 = np.sum((data - mean) ** 3) / len(data)
    std3 = data.std() ** 3
    if std3 == 0:
        return 0
    return m3 / std3


def kurtosis(data):
    if len(data) < 4:
        return 0
    centered_data = data - data.mean()
    squared_deviations = centered_data**2
    variance_sq = squared_deviations.mean() ** 2
    if variance_sq == 0:
        return 0
    fourth_moment = np.mean(centered_data**4)
    return fourth_moment / variance_sq - 3


def time_params(sig, name, ent=False):
    diff = sig[1:-1] - sig[0:-2]
    ret = []
    names = []

    ret.append(sig.mean())
    names.append(f"{name} mean")
    ret.append(sig.std())
    names.append(f"{name} std")
    ret.append(diff.std())
    names.append(f"{name} diff std")
    ret.append(np.sqrt((sig**2).mean()))
    names.append(f"{name} RMS")
    ret.append(np.cov(sig))
    names.append(f"{name} covmat")
    ret.append(mode(sig))
    names.append(f"{name} mode")
    ret.append(kurtosis(sig))
    names.append(f"{name} kurt")
    ret.append(skewness(sig))
    names.append(f"{name} skew")
    if ent:
        new_vals, new_names = entropy_params(sig, name)
        ret = ret + new_vals
        names = names + new_names

    return ret, names


def eeg_freq_params(sig, fs, name):
    L = len(sig)
    sig_fft = abs(fft.fft(sig))[: int(L / 2)] * 2 / L
    freq = fft.fftfreq(len(sig), d=1 / fs)[: int(L / 2)]
    delta = sig_fft[np.where(np.logical_and(freq >= 1, freq < 4))] ** 2
    theta = sig_fft[np.where(np.logical_and(freq >= 4, freq < 8))] ** 2
    alpha = sig_fft[np.where(np.logical_and(freq >= 8, freq < 12))] ** 2
    beta1 = sig_fft[np.where(np.logical_and(freq >= 12, freq < 20))] ** 2
    beta2 = sig_fft[np.where(np.logical_and(freq >= 20, freq < 30))] ** 2
    gamma1 = sig_fft[np.where(np.logical_and(freq >= 30, freq < 60))] ** 2
    gamma2 = sig_fft[np.where(np.logical_and(freq >= 60, freq < 100))] ** 2
    all = sig_fft**2
    sum_all = sum(all)

    ret = []
    names = []

    ret.append(sum(delta) / sum_all / len(delta))
    names.append(f"{name} delta/all en")
    ret.append(sum(theta) / sum_all / len(theta))
    names.append(f"{name} theta/all en")
    ret.append(sum(alpha) / sum_all / len(alpha))
    names.append(f"{name} alpha/all en")
    ret.append(sum(beta1) / sum_all / len(beta1))
    names.append(f"{name} beta1/all en")
    ret.append(sum(beta2) / sum_all / len(beta2))
    names.append(f"{name} beta2/all en")
    ret.append(sum(gamma1) / sum_all / len(gamma1))
    names.append(f"{name} gamma1/all en")
    ret.append(sum(gamma2) / sum_all / len(gamma2))
    names.append(f"{name} gamma2/all en")
    ret.append(sum(alpha) / sum(beta1) / len(alpha) * len(beta1))
    names.append(f"{name} alpha/beta1 en")
    ret.append(sum(alpha) / sum(beta2) / len(alpha) * len(beta2))
    names.append(f"{name} alpha/beta2 en")
    ret.append(sum(alpha) / (sum(beta1) + sum(beta2)) / len(alpha) * (len(beta1) + len(beta2)))
    names.append(f"{name} alpha/(beta1+beta2) en")

    ret.append(delta.mean())
    names.append(f"{name} mean delta en")
    ret.append(theta.mean())
    names.append(f"{name} mean theta en")
    ret.append(alpha.mean())
    names.append(f"{name} mean alpha en")
    ret.append(beta1.mean())
    names.append(f"{name} mean beta1 en")
    ret.append(beta2.mean())
    names.append(f"{name} mean beta2 en")
    ret.append(gamma1.mean())
    names.append(f"{name} mean gamma1 en")
    ret.append(gamma2.mean())
    names.append(f"{name} mean gamma2 en")
    ret.append(all.mean())
    names.append(f"{name} mean all en")

    return ret, names


def hr_freq_params(sig, fs, name):
    L = len(sig)
    sig_fft = abs(fft.fft(sig))[: int(L / 2)] * 2 / L
    freq = fft.fftfreq(len(sig), d=1 / fs)[: int(L / 2)]

    HF = sig_fft[np.where(np.logical_and(freq >= 0.15, freq < 0.4))] ** 2
    LF = sig_fft[np.where(np.logical_and(freq >= 0.04, freq < 0.15))] ** 2
    VLF = sig_fft[np.where(np.logical_and(freq >= 0.015, freq < 0.04))] ** 2
    ULF = sig_fft[np.where(freq < 0.015)] ** 2
    VHF = sig_fft[np.where(freq >= 0.4)] ** 2
    all = sig_fft**2
    sum_all = sum(all)

    ret = []
    names = []

    ret.append(sum(HF) / sum_all / len(HF))
    names.append(f"{name} HF/all en")
    ret.append(sum(LF) / sum_all / len(LF))
    names.append(f"{name} LF/all en")
    ret.append(sum(VLF) / sum_all / len(VLF))
    names.append(f"{name} VLF/all en")
    ret.append(sum(ULF) / sum_all / len(ULF))
    names.append(f"{name} ULF/all en")
    ret.append(sum(VHF) / sum_all / len(VLF))
    names.append(f"{name} VHF/all en")
    ret.append(sum(VLF) / sum(ULF) / len(VLF) * len(ULF))
    names.append(f"{name} VLF/ULF en")
    ret.append(sum(VLF) / sum(VHF) / len(VLF) * len(VHF))
    names.append(f"{name} VLF/VHF en")
    ret.append(sum(VHF) / (sum(ULF) + sum(VLF)) / len(VHF) * (len(ULF) + len(VLF)))
    names.append(f"{name} VHF/(ULF+VLF) en")

    ret.append(HF.mean())
    names.append(f"{name} mean HF en")
    ret.append(LF.mean())
    names.append(f"{name} mean LF en")
    ret.append(VLF.mean())
    names.append(f"{name} mean VLF en")
    ret.append(ULF.mean())
    names.append(f"{name} mean ULF en")
    ret.append(VHF.mean())
    names.append(f"{name} mean VHF en")
    ret.append(all.mean())
    names.append(f"{name} mean all en")

    return ret, names


def ecg_freq_params(sig, fs, name):
    L = len(sig)
    sig_fft = abs(fft.fft(sig))[: int(L / 2)] * 2 / L
    freq = fft.fftfreq(len(sig), d=1 / fs)[: int(L / 2)]

    PLI = sig_fft[np.where(np.logical_and(freq >= 49, freq <= 51))] ** 2
    PT = sig_fft[np.where(np.logical_and(freq >= 1, freq <= 10))] ** 2
    MOT = sig_fft[np.where(np.logical_and(freq >= 3, freq <= 10))] ** 2
    QRS = sig_fft[np.where(np.logical_and(freq >= 2, freq <= 22))] ** 2
    MF = sig_fft[np.where(np.logical_and(freq >= 10, freq <= 49))] ** 2
    BW = sig_fft[np.where(freq <= 0.5)] ** 2
    HF = sig_fft[np.where(freq >= 50)] ** 2
    EMG = sig_fft[np.where(freq >= 10)] ** 2
    all = sig_fft**2
    sum_all = sum(all)

    ret = []
    names = []

    ret.append(sum(PLI) / sum_all / len(PLI))
    names.append(f"{name} PLI/all en")
    ret.append(sum(PT) / sum_all / len(PT))
    names.append(f"{name} P/T/all en")
    ret.append(sum(MOT) / sum_all / len(MOT))
    names.append(f"{name} Motion/all en")
    ret.append(sum(QRS) / sum_all / len(QRS))
    names.append(f"{name} QRS/all en")
    ret.append(sum(MF) / sum_all / len(MF))
    names.append(f"{name} MF/all en")
    ret.append(sum(BW) / sum_all / len(BW))
    names.append(f"{name} BW/all en")
    ret.append(sum(HF) / sum_all / len(HF))
    names.append(f"{name} HF/all en")
    ret.append(sum(EMG) / sum_all / len(EMG))
    names.append(f"{name} EMG/all en")

    ret.append(PLI.mean())
    names.append(f"{name} PLI mean en")
    ret.append(PT.mean())
    names.append(f"{name} P/T mean en")
    ret.append(MOT.mean())
    names.append(f"{name} Motion mean en")
    ret.append(QRS.mean())
    names.append(f"{name} QRS mean en")
    ret.append(MF.mean())
    names.append(f"{name} MF mean en")
    ret.append(BW.mean())
    names.append(f"{name} BW mean en")
    ret.append(HF.mean())
    names.append(f"{name} HF mean en")
    ret.append(EMG.mean())
    names.append(f"{name} EMG mean en")
    ret.append(all.mean())
    names.append(f"{name} all mean en")

    return ret, names


def wave_params(sig, name, ent, wlt, level):
    cA, *cDs = wavedec(sig, wlt, level=level)
    ret = []
    names = []

    en_sum = 0
    for i, cD in enumerate(cDs[::-1], start=1):
        cDen = sum(cD**2) / len(cD)
        ret.append(cDen)
        names.append(f"{name} cD{i} energy")
        en_sum = en_sum + cDen
    
    cAen = sum(cA**2) / len(cA)
    ret.append(cAen)
    names.append(f"{name} cA{level} energy")
    en_sum = en_sum + cAen

    for i, cD in enumerate(cDs[::-1], start=1):
        cDrelen = sum(cD**2) / len(cD) / en_sum
        ret.append(cDrelen)
        names.append(f"{name} cD{i} rel energy")

    cArelen = cAen / en_sum
    ret.append(cArelen)
    names.append(f"{name} cA{level} rel energy")

    for i, cD in enumerate(cDs[::-1], start=1):
        rt, nt = time_params(cD, name, ent)
        ret = ret + rt
        names = names + [f"cD{i} {x}" for x in nt]

    rt, nt = time_params(cA, name, ent)
    ret = ret + rt
    names = names + [f"cA{level} {x}" for x in nt]

    return ret, names


def create_filter(wp, ws, fs, btype="bandpass"):
    sgord, wn = buttord(wp, ws, 3, 60, fs=fs)
    sos = butter(sgord, wn, btype=btype, fs=fs, output="sos")
    return sos


def create_eeg_filters(fs, rhythms=range(7)):
    ws = [[1, 4], [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 100]]
    filters = []
    for i in rhythms:
        ww = ws[i]
        wwp = np.mean(ww) / 10
        filters.append(create_filter(ww, [ww[0] - wwp, ww[1] + wwp], fs))
    return filters


def create_hr_filters(fs, rhythms=range(4)):
    ws = [[0.15, 0.4], [0.04, 0.15], [0.015, 0.04], 0.015, 0.4]
    rts = ["HF", "LF", "VLF", "ULF", "VHF"]
    filters = []
    for i in rhythms:
        ww = ws[i]
        if rts[i] == "ULF":
            wwp = ww / 10
            filters.append(create_filter(ww, ww + wwp, fs, btype="lowpass"))
        elif rts[i] == "VHF":
            wwp = ww / 10
            filters.append(create_filter(ww, ww - wwp, fs, btype="highpass"))
        else:
            wwp = np.mean(ww) / 10
            filters.append(create_filter(ww, [ww[0] - wwp, ww[1] + wwp], fs))
    return filters


def create_ecg_filters(fs, rhythms=range(8)):
    ws = [[1, 10], [3, 10], [2, 22], [10, 49], [49, 51], 0.5, 10, 50]
    rts = ["PLI", "P/T", "Motion", "QRS", "MF", "BW", "HF", "EMG"]
    filters = []
    for i in rhythms:
        ww = ws[i]
        if rts[i] == "BW":
            wwp = ww / 10
            filters.append(create_filter(ww, ww + wwp, fs, btype="lowpass"))
        elif rts[i] in ["HF", "EMG"]:
            wwp = ww / 10
            filters.append(create_filter(ww, ww - wwp, fs, btype="highpass"))
        else:
            wwp = np.mean(ww) / 10
            filters.append(create_filter(ww, [ww[0] - wwp, ww[1] + wwp], fs))
    return filters


def eeg_rhythms(sig, name, filters, ent, rhythms=range(7)):
    rts = ["delta", "theta", "alpha", "beta1", "beta2", "gamma1", "gamma2"]
    ret = []
    names = []
    for i in rhythms:
        sos = filters[i]
        f = sosfilt(sos, sig)
        r, n = time_params(f, f"{name} {rts[i]}", ent)
        ret = ret + r
        names = names + n

    return ret, names


def hr_rhythms(sig, name, filters, ent, rhythms=range(4)):
    rts = ["HF", "LF", "VLF", "ULF", "VHF"]
    ret = []
    names = []
    for i in rhythms:
        sos = filters[i]
        f = sosfilt(sos, sig)
        r, n = time_params(f, f"{name} {rts[i]}", ent)
        ret = ret + r
        names = names + n

    return ret, names


def ecg_rhythms(sig, name, filters, ent, rhythms=range(4)):
    rts = ["PLI", "P/T", "Motion", "QRS", "MF", "BW", "HF", "EMG"]
    ret = []
    names = []
    for i in rhythms:
        sos = filters[i]
        f = sosfilt(sos, sig)
        r, n = time_params(f, f"{name} {rts[i]}", ent)
        ret = ret + r
        names = names + n

    return ret, names


def entropy(signal, order):
    if len(signal.shape) != 1:
        raise ValueError("Input signal must be a 1D array.")

    n = len(signal)
    embedded_matrix = np.empty((n - order + 1, order))
    for i in range(n - order + 1):
        embedded_matrix[i] = signal[i : i + order]

    _, counts = np.unique(embedded_matrix, axis=0, return_counts=True)
    symbol_probs = counts / np.sum(counts)

    entropy = 0.0
    for p in symbol_probs:
        if p > 0:
            entropy -= p * log2(p)

    return entropy


def entropy_params(sig, name):
    ret, names = [], []
    for i in range(3, 6):
        ret.append(entropy(sig, i))
        names.append(f"{name} {i}-th entr")

    return ret, names


def pRR50(sig):
    ret = 0
    for i in range(len(sig)):
        if abs(sig[i] - sig[i+1]) > 50e-3:
            ret = ret + 100
    return ret/(len(sig)-1)


def pRR20(sig):
    ret = 0
    for i in range(len(sig)):
        if abs(sig[i] - sig[i+1]) > 20e-3:
            ret = ret + 100
    return ret/(len(sig)-1)


def eeg_analyser(ds, fs, filters, wlt, level, ent=False):
    vals, names = [], []
    for name in ds:
        new_vals, new_names = eeg_freq_params(ds[name], fs, name)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = time_params(ds[name], name, ent)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = wave_params(ds[name], name, ent, wlt, level)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = eeg_rhythms(ds[name], name, filters, ent)
        vals = vals + new_vals
        names = names + new_names

    return vals, names


def hr_interp_analyser(ds, fs, filters, wlt, level, ent=False):
    vals, names = [], []
    for name in ds:
        new_vals, new_names = hr_freq_params(ds[name], fs, name)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = time_params(ds[name], name, ent)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = wave_params(ds[name], name, ent, wlt, level)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = hr_rhythms(ds[name], name, filters, ent)
        vals = vals + new_vals
        names = names + new_names

    return vals, names


def hr_analyser(ds):
    vals, names = [], []
    for name in ds:
       vals = vals + [
                      np.std(ds[name]),
                      np.std([ds[name][i+1] - ds[name][i] for i in range(len(ds[name])-1)]),
                      np.mean(ds[name]),
                      mode(ds[name]),
                      skewness(ds[name]),
                      kurtosis(ds[name]),
                      pRR50(ds[name]),
                      pRR20(ds[name]),
                      np.max(ds[name]) - np.min(ds[name])
                      ]
       name = name + [
                      f'{name} SDRR',
                      f'{name} RMSSD',
                      f'{name} mean',
                      f'{name} mode',
                      f'{name} skewness',
                      f'{name} kurtosis',
                      f'{name} pRR50',
                      f'{name} pRR20',
                      f'{name} Max-Min'
                      ]

    return vals, names


def ecg_analyser(ds, fs, filters, wlt, level, ent=False):
    vals, names = [], []
    for name in ds:
        new_vals, new_names = ecg_freq_params(ds[name], fs, name)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = time_params(ds[name], name, ent)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = wave_params(ds[name], name, ent, wlt, level)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = ecg_rhythms(ds[name], name, filters, ent)
        vals = vals + new_vals
        names = names + new_names

    return vals, names


def progress_tracker(logtxtbox, time_now, start_time, done, total):
    elap = time_now - start_time
    elap_m, elap_s = divmod(elap, 60)
    elap_h, elap_m = divmod(elap_m, 60)

    rem = elap * (total / done - 1)
    rem_m, rem_s = divmod(rem, 60)
    rem_h, rem_m = divmod(rem_m, 60)

    logtxtbox.text(
        f"Done: {done/total*100:.2f}%, Elapsed: {elap_h:02.0f}:{elap_m:02.0f}:{elap_s:02.0f}, Remaining: {rem_h:02.0f}:{rem_m:02.0f}:{rem_s:02.0f}"
    )
    return f"Done: {done/total*100:.2f}%, Elapsed: {elap_h:02.0f}:{elap_m:02.0f}:{elap_s:02.0f}, Remaining: {rem_h:02.0f}:{rem_m:02.0f}:{rem_s:02.0f}"
