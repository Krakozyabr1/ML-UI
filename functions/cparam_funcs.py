from scipy.signal import buttord, butter
from pywt import wavedec
from math import log2
import numpy as np


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
