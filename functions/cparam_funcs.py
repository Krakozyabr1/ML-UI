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
    
    centered = data - data.mean()
    m3 = np.mean(centered ** 3)  
    std3 = data.std() ** 3

    return m3 / std3 if std3 != 0 else 0 


def kurtosis(data):
    if len(data) < 4:
        return 0
    
    centered = data - data.mean()
    variance = np.mean(centered * centered) 

    if variance == 0:
        return 0
    
    m4 = np.mean(centered ** 4)

    return m4 / (variance * variance) - 3


def time_params(sig, name, ent=False, no_mean=False):
    diff = sig[1:] - sig[0:-1]
    ret = [sig.mean(),
           sig.std(),
           diff.std(),
           np.abs(diff).mean(),
        #    np.sqrt((sig * sig).mean()),
           np.median(sig),
           kurtosis(sig),
           skewness(sig)
           ]
    names = [f"{name} mean",
             f"{name} std",
             f"{name} diff std",
             f"{name} diff mean",
            #  f"{name} RMS",
             f"{name} median",
             f"{name} kurt",
             f"{name} skew"
             ]

    if ent:
        new_vals, new_names = entropy_params(sig, name)
        ret = ret + new_vals
        names = names + new_names

    return ret[1*int(no_mean):], names[1*int(no_mean):]


def wave_params(sig, name, ent, wlt, level_min, level):
    cA, *cDs_all = wavedec(sig, wlt, level=level)
    if level_min == 0:
        cDs = cDs_all
    else:
        cDs = cDs_all[:-level_min]

    coeffs = cDs[::-1] + [cA]
    coeff_names = [f"cD{i+1}" for i in range(len(cDs))] + [f"cA{level}"]

    energies = [np.mean(c**2) for c in coeffs]
    en_sum = np.sum(energies)
    relative_energies = [e / en_sum for e in energies]

    ret = energies
    names = [f"{name} {c_name} energy" for c_name in coeff_names]

    ret.extend(relative_energies)
    names.extend([f"{name} {c_name} rel energy" for c_name in coeff_names])

    for c_name, c in zip(coeff_names, coeffs):
        rt, nt = time_params(c, f"{name} {c_name}", ent) 
        ret.extend(rt)
        names.extend(nt)

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
