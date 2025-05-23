from functions.cparam_funcs import *
from scipy.signal import sosfilt
from pyedflib import highlevel
from scipy import fft
import numpy as np
import time


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


def create_filters(fs, rhythms=range(7)):
    ws = [[1, 4], [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 100]]
    filters = []
    for i in rhythms:
        ww = ws[i]
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


def generate_features_table(e, i, start_time, labels, filters, total_files, wlt, wlt_level, fs, selected_labels, logtxtbox):
    signals, _, _ = highlevel.read_edf(i)

    datarow = {label: signals[i] for i, label in enumerate(labels) if selected_labels[i]}

    val, param_name = eeg_analyser(datarow, fs, filters, wlt, wlt_level)
    _ = progress_tracker(logtxtbox, time.time(), start_time, e + 1, total_files)
    return {param_name[j]: val[j] for j in range(len(param_name))}