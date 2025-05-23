from functions.cparam_funcs import *
from scipy.signal import sosfilt
from pyedflib import highlevel
from scipy import fft
import numpy as np
import time


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


def create_filters(fs, rhythms=range(8)):
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


def generate_features_table(e, i, start_time, labels, filters, total_files, wlt, wlt_level, fs, selected_labels, logtxtbox):
                signals, _, _ = highlevel.read_edf(i)

                datarow = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i]}

                val, param_name = ecg_analyser(datarow, fs, filters, wlt, wlt_level)
                _ = progress_tracker(logtxtbox, time.time(), start_time, e + 1, total_files)
                return {param_name[j]: val[j] for j in range(len(param_name))}
