from pyedflib import highlevel
import numpy as np


def readedf(selected_file):
    signals, signal_headers, header = highlevel.read_edf(selected_file)
    sig_labels = [x["label"] for x in signal_headers]
    fs = signal_headers[0]["sample_frequency"]
    t = np.arange(len(signals[0])) / fs
    selected_labels = [False] * len(sig_labels)
    return signals, signal_headers, header, sig_labels, fs, t, selected_labels