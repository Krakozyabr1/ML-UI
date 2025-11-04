from functions.cparam_funcs import *
from scipy.signal import sosfilt
# from pyedflib import highlevel
from functions.read_signals import read_signals
from scipy import fft
import numpy as np
import time


def ecg_freq_params(sig, fs, name):
    L = len(sig)
    sig_fft_mag = abs(fft.fft(sig) * 2 / L)[: L // 2]
    freq = fft.fftfreq(len(sig), d=1 / fs)[: L // 2]
    sig_fft_sq = sig_fft_mag * sig_fft_mag

    PLI = sig_fft_sq[np.where(np.logical_and(freq >= 49, freq <= 51))]
    PT = sig_fft_sq[np.where(np.logical_and(freq >= 1, freq <= 10))]
    MOT = sig_fft_sq[np.where(np.logical_and(freq >= 3, freq <= 10))]
    QRS = sig_fft_sq[np.where(np.logical_and(freq >= 2, freq <= 22))]
    MF = sig_fft_sq[np.where(np.logical_and(freq >= 10, freq <= 49))]
    BW = sig_fft_sq[np.where(freq <= 0.5)]
    HF = sig_fft_sq[np.where(freq >= 50)]
    EMG = sig_fft_sq[np.where(freq >= 10)]
    sum_all = sum(sig_fft_sq)

    ret = [sum(PLI) / sum_all,
           sum(PT) / sum_all,
           sum(MOT) / sum_all,
           sum(QRS) / sum_all,
           sum(MF) / sum_all,
           sum(BW) / sum_all,
           sum(HF) / sum_all,
           sum(EMG) / sum_all,
           PLI.mean(),
           PT.mean(),
           MOT.mean(),
           QRS.mean(),
           MF.mean(),
           BW.mean(),
           HF.mean(),
           EMG.mean(),
           sig_fft_sq.mean(),
    ]

    names = [f"{name} PLI/all en",
             f"{name} P/T/all en",
             f"{name} Motion/all en",
             f"{name} QRS/all en",
             f"{name} MF/all en",
             f"{name} BW/all en",
             f"{name} HF/all en",
             f"{name} EMG/all en",
             f"{name} PLI mean en",
             f"{name} P/T mean en",
             f"{name} Motion mean en",
             f"{name} QRS mean en",
             f"{name} MF mean en",
             f"{name} BW mean en",
             f"{name} HF mean en",
             f"{name} EMG mean en",
             f"{name} all mean en",
    ]



    return ret, names


def create_filters(fs, rhythms=range(8)):
    ws = [[1, 10], [3, 10], [2, 22], [10, 49], [49, 51], 0.5, 10, 50]
    rts = ["PLI", "P/T", "Motion", "QRS", "MF", "BW", "HF", "EMG"]
    filters = []
    for i in rhythms:
        ww = ws[i]
        if rts[i] == "BW":
            wwp = ww / 10
            if ww + wwp < fs / 2:
                filters.append(create_filter(ww, ww + wwp, fs, btype="lowpass"))
        elif rts[i] in ["HF", "EMG"]:
            wwp = ww / 10
            if ww < fs / 2:
                filters.append(create_filter(ww, ww - wwp, fs, btype="highpass"))
        else:
            wwp = np.mean(ww) / 10
            if ww[1] + wwp < fs / 2:
                filters.append(create_filter(ww, [ww[0] - wwp, ww[1] + wwp], fs))
    return filters


def ecg_rhythms(sig, name, filters, ent, rhythms=None):
    if rhythms is None:
         rhythms = range(len(filters))
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


def ecg_analyser(ds, fs, filters, wlt, level_min, level, ent=False):
    vals, names = [], []
    for name in ds:
        new_vals, new_names = ecg_freq_params(ds[name], fs, name)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = time_params(ds[name], name, ent, no_mean=True)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = wave_params(ds[name], name, ent, wlt, level_min, level)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = ecg_rhythms(ds[name], name, filters, ent)
        vals = vals + new_vals
        names = names + new_names

    return vals, names


def generate_features_table(e, i, start_time, labels, filters, total_files, wlt, wlt_level_min, wlt_level, fs, selected_labels, logtxtbox, selected_ref):
                signals, *_ = read_signals(i, False)

                signals -= np.mean(signals, axis=1)[:, np.newaxis]
                
                datarow = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i]}

                val, param_name = ecg_analyser(datarow, fs, filters, wlt, wlt_level_min, wlt_level)
                _ = progress_tracker(logtxtbox, time.time(), start_time, e + 1, total_files)
                return {param_name[j]: val[j] for j in range(len(param_name))}
