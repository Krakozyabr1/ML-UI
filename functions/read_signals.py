# from pyedflib import highlevel
from mne.io import read_raw
import numpy as np


# def read_signals(selected_file, fs=-1):
#     signals, signal_headers, _ = highlevel.read_edf(selected_file)

#     sig_labels = [x["label"] for x in signal_headers]
#     dimensions = [signal_header["dimension"] for signal_header in signal_headers]

#     if fs < 0:
#         fs = signal_headers[0]["sample_frequency"]

#     selected_labels = [False] * len(sig_labels)
#     return signals, dimensions, sig_labels, fs, selected_labels

def read_signals(selected_file):
    raw = read_raw(selected_file)
        
    signals = raw.get_data()
    fs = raw.info['sfreq']
    sig_labels = raw.ch_names
    
    try:
        orig_units_dict = raw._orig_units
        dimensions = [orig_units_dict.get(ch_name, 'Unknown') for ch_name in sig_labels]
    except AttributeError:
        raise AttributeError("Failed to get original units from MNE object.")
    
    try:
        units_scaling_factors = np.array(raw._raw_extras[0]['units'])
    except (AttributeError, KeyError):
        raise AttributeError("Failed to get SI scaling factors from MNE object.")
    
    try:
        signal_types = raw._raw_extras[0]['ch_types']
    except (AttributeError, KeyError):
        raise AttributeError("Failed to get signal types from MNE object.")

    signals = signals / units_scaling_factors[:, np.newaxis] 

    selected_labels = [False] * len(sig_labels)
    return signals, signal_types, dimensions, sig_labels, fs, selected_labels