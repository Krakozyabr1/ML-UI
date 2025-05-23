from functions.cparam_funcs import *
from scipy.signal import sosfilt
from pyedflib import highlevel
from scipy import fft
import numpy as np
import time


def pRR50(sig):
    diffs = np.abs(np.diff(sig))
    return np.sum(diffs > 50e-3) / (len(sig) - 1) * 100


def pRR20(sig):
    diffs = np.abs(np.diff(sig))
    return np.sum(diffs > 20e-3) / (len(sig) - 1) * 100


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


def create_filters(fs, rhythms=range(4)):
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
       names = names + [
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


def generate_features_table(e, i, start_time, labels, filters, total_files, wlt, wlt_level, fs, selected_labels, logtxtbox):
    signals, _, _ = highlevel.read_edf(i)

    datarow = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i] and 'interp' not in label}
    datarow2 = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i] and 'interp' in label}

    val, param_name = [], []
    if len(datarow) > 0:
        val1, param_name1 = hr_analyser(datarow)
        val.extend(val1)
        param_name.extend(param_name1)

    if len(datarow2) > 0:
        val2, param_name2 = hr_interp_analyser(datarow2, fs, filters, wlt, wlt_level)
        val.extend(val2)
        param_name.extend(param_name2)

    _ = progress_tracker(logtxtbox, time.time(), start_time, e + 1, total_files)
    return {param_name[j]: val[j] for j in range(len(param_name))}
