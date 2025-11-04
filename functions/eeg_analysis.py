from functions.cparam_funcs import *
from scipy.signal import sosfilt
# from pyedflib import highlevel
from functions.read_signals import read_signals
from scipy import fft
import numpy as np
import time


def eeg_freq_params(sig, fs, name):
    L = len(sig)
    sig_fft_mag = abs(fft.fft(sig) * 2 / L)[: L // 2]
    freq = fft.fftfreq(len(sig), d=1 / fs)[: L // 2]
    sig_fft_sq = sig_fft_mag * sig_fft_mag

    delta = sig_fft_sq[np.where(np.logical_and(freq >= 1, freq < 4))]
    theta = sig_fft_sq[np.where(np.logical_and(freq >= 4, freq < 8))]
    alpha = sig_fft_sq[np.where(np.logical_and(freq >= 8, freq < 12))]
    beta1 = sig_fft_sq[np.where(np.logical_and(freq >= 12, freq < 20))]
    beta2 = sig_fft_sq[np.where(np.logical_and(freq >= 20, freq < 30))]
    gamma1 = sig_fft_sq[np.where(np.logical_and(freq >= 30, freq < 60))]
    gamma2 = sig_fft_sq[np.where(np.logical_and(freq >= 60, freq < 100))]
    sum_all = sum(sig_fft_sq)

    ret = [sum(delta) / sum_all / len(delta),
           sum(theta) / sum_all / len(theta),
           sum(alpha) / sum_all / len(alpha),
           sum(beta1) / sum_all / len(beta1),
           sum(beta2) / sum_all / len(beta2),
           sum(gamma1) / sum_all / len(gamma1),
           sum(gamma2) / sum_all / len(gamma2),
           sum(alpha) / sum(beta1) / len(alpha) * len(beta1),
           sum(alpha) / sum(beta2) / len(alpha) * len(beta2),
           sum(alpha) / (sum(beta1) + sum(beta2)) / len(alpha) * (len(beta1) + len(beta2)),
           delta.mean(),
           theta.mean(),
           alpha.mean(),
           beta1.mean(),
           beta2.mean(),
           gamma1.mean(),
           gamma2.mean(),
           sig_fft_sq.mean(),
    ]
    names = [f"{name} delta/all en",
             f"{name} theta/all en",
             f"{name} alpha/all en",
             f"{name} beta1/all en",
             f"{name} beta2/all en",
             f"{name} gamma1/all en",
             f"{name} gamma2/all en",
             f"{name} alpha/beta1 en",
             f"{name} alpha/beta2 en",
             f"{name} alpha/(beta1+beta2) en",
             f"{name} mean delta en",
             f"{name} mean theta en",
             f"{name} mean alpha en",
             f"{name} mean beta1 en",
             f"{name} mean beta2 en",
             f"{name} mean gamma1 en",
             f"{name} mean gamma2 en",
             f"{name} mean all en",
    ]

    return ret, names


def create_filters(fs, rhythms=range(7)):
    ws = [[1, 4], [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 100]]
    filters = []
    for i in rhythms:
        ww = ws[i]
        wwp = np.mean(ww) / 10
        if ww[1] + wwp < fs / 2:
            filters.append(create_filter(ww, [ww[0] - wwp, ww[1] + wwp], fs))
    return filters


def eeg_rhythms(sig, name, filters, ent, rhythms=None):
    if rhythms is None:
        rhythms = range(len(filters))
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


def eeg_analyser(ds, fs, filters, wlt, level_min, level, ent=False):
    vals, names = [], []
    for name in ds:
        new_vals, new_names = eeg_freq_params(ds[name], fs, name)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = time_params(ds[name], name, ent, no_mean=True)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = wave_params(ds[name], name, ent, wlt, level_min, level)
        vals = vals + new_vals
        names = names + new_names
        new_vals, new_names = eeg_rhythms(ds[name], name, filters, ent)
        vals = vals + new_vals
        names = names + new_names

    return vals, names


def generate_features_table(e, i, start_time, labels, filters, total_files, wlt, wlt_level_min, wlt_level, fs, selected_labels, logtxtbox, selected_ref):
    signals, *_ = read_signals(i, selected_ref)

    signals = signals - np.mean(signals, axis=1)[:, np.newaxis]
        
    datarow = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i]}

    val, param_name = eeg_analyser(datarow, fs, filters, wlt, wlt_level_min, wlt_level)
    _ = progress_tracker(logtxtbox, time.time(), start_time, e + 1, total_files)
    return {param_name[j]: val[j] for j in range(len(param_name))}