from functions.cparam_funcs import *
from scipy.signal import sosfilt
# from pyedflib import highlevel
from functions.read_signals import read_signals
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
    sig_fft_mag = abs(fft.fft(sig) * 2 / L)[: L // 2]
    freq = fft.fftfreq(len(sig), d=1 / fs)[: L // 2]
    sig_fft_sq = sig_fft_mag * sig_fft_mag

    HF = sig_fft_sq[np.where(np.logical_and(freq >= 0.15, freq < 0.4))]
    LF = sig_fft_sq[np.where(np.logical_and(freq >= 0.04, freq < 0.15))]
    VLF = sig_fft_sq[np.where(np.logical_and(freq >= 0.015, freq < 0.04))]
    ULF = sig_fft_sq[np.where(freq < 0.015)]
    VHF = sig_fft_sq[np.where(freq >= 0.4)]
    sum_all = sum(sig_fft_sq)

    ret = [sum(HF) / sum_all,
           sum(LF) / sum_all,
           sum(VLF) / sum_all,
           sum(ULF) / sum_all,
           sum(VHF) / sum_all,
           sum(VLF) / sum(ULF),
           sum(VLF) / sum(VHF),
           sum(VHF) / (sum(ULF) + sum(VLF)),
           HF.mean(),
           LF.mean(),
           VLF.mean(),
           ULF.mean(),
           VHF.mean(),
           sig_fft_sq.mean()
    ]

    names = [f"{name} HF/all en",
             f"{name} LF/all en",
             f"{name} VLF/all en",
             f"{name} ULF/all en",
             f"{name} VHF/all en",
             f"{name} VLF/ULF en",
             f"{name} VLF/VHF en",
             f"{name} VHF/(ULF+VLF) en",
             f"{name} mean HF en",
             f"{name} mean LF en",
             f"{name} mean VLF en",
             f"{name} mean ULF en",
             f"{name} mean VHF en",
             f"{name} mean all en"
    ]

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


def hr_interp_analyser(ds, fs, filters, wlt, level_min, level, ent=False):
    vals, names = [], []
    for name in ds:
        new_vals, new_names = hr_freq_params(ds[name], fs, name)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = time_params(ds[name], name, ent)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = wave_params(ds[name], name, ent, wlt, level_min, level)
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


def generate_features_table(e, i, start_time, labels, filters, total_files, wlt, wlt_level_min, wlt_level, fs, selected_labels, logtxtbox):
    signals, *_ = read_signals(i)

    datarow = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i] and 'interp' not in label}
    datarow2 = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i] and 'interp' in label}

    val, param_name = [], []
    if len(datarow) > 0:
        val1, param_name1 = hr_analyser(datarow)
        val.extend(val1)
        param_name.extend(param_name1)

    if len(datarow2) > 0:
        val2, param_name2 = hr_interp_analyser(datarow2, fs, filters, wlt, wlt_level_min, wlt_level)
        val.extend(val2)
        param_name.extend(param_name2)

    _ = progress_tracker(logtxtbox, time.time(), start_time, e + 1, total_files)
    return {param_name[j]: val[j] for j in range(len(param_name))}
