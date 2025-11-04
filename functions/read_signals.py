# from pyedflib import highlevel
from mne.io import read_raw
import numpy as np
from mne import pick_types

def median_absolute_deviation(data):
    median_val = np.median(data, axis=0)
    return np.median(np.abs(data - median_val), axis=0) * 1.4826

def huber_weights(residuals, scale, tuning_const=1.345):
    epsilon = 1e-6
    abs_residuals = np.abs(residuals)

    k = tuning_const * scale
    outlier_mask = abs_residuals > k[np.newaxis, :] 

    weights = np.ones_like(residuals)
    new_weights = k[np.newaxis, :] / (abs_residuals + epsilon)
    weights[outlier_mask] = new_weights[outlier_mask]
    
    return weights

def rCAR(signals, max_iter=20, convergence_tol=1e-5):
    reference_estimate = np.mean(signals, axis=0)
    
    for i in range(max_iter):
        residuals = signals - reference_estimate[np.newaxis, :]
        
        scale = median_absolute_deviation(residuals)
        weights = huber_weights(residuals, scale)
        
        weighted_signals = signals * weights
        sum_of_weights = np.sum(weights, axis=0)
        
        safe_sum_of_weights = np.where(sum_of_weights == 0, 1.0, sum_of_weights)
        
        new_reference_estimate = np.sum(weighted_signals, axis=0) / safe_sum_of_weights
        
        change = np.linalg.norm(new_reference_estimate - reference_estimate)
        reference_estimate = new_reference_estimate
        
        if change < convergence_tol:
            print(f"rCAR converged after {i+1} iterations.")
            break
    else:
        print(f"rCAR reached max iterations ({max_iter}) without full convergence.")

    return signals - reference_estimate[np.newaxis, :]

# def read_signals(selected_file, fs=-1):
#     signals, signal_headers, _ = highlevel.read_edf(selected_file)

#     sig_labels = [x["label"] for x in signal_headers]
#     dimensions = [signal_header["dimension"] for signal_header in signal_headers]

#     if fs < 0:
#         fs = signal_headers[0]["sample_frequency"]

#     selected_labels = [False] * len(sig_labels)
#     return signals, dimensions, sig_labels, fs, selected_labels

def read_signals(selected_file, selected_ref="Raw", selected_labels=None):
    raw = read_raw(selected_file, verbose=False, preload=True)
        
    fs = raw.info['sfreq']
    sig_labels = raw.ch_names
    
    try:
        orig_units_dict = raw._orig_units
        dimensions = [orig_units_dict.get(ch_name, 'Unknown') for ch_name in sig_labels]
    except AttributeError:
        raise AttributeError("Failed to get original units from MNE object.")
    
    try:
        signal_types = raw._raw_extras[0]['ch_types']
    except (AttributeError, KeyError):
        raise AttributeError("Failed to get signal types from MNE object.")

    try:
        units_scaling_factors = np.array(raw._raw_extras[0]['units'])
    except (AttributeError, KeyError):
        raise AttributeError("Failed to get SI scaling factors from MNE object.")
    
    signals = raw.get_data()

    eeg_indices = pick_types(raw.info, eeg=True, exclude=[])
    
    if len(eeg_indices) == 0 and selected_ref != 'Raw':
        print("Warning: No EEG channels found. Skipping referencing.")
    elif len(eeg_indices) == 1 and selected_ref != 'Raw':
        print("Warning: Only one EEG channel found. Skipping referencing.")

    ref_functions = {'CAR': np.mean,
                     'CMR': np.median}

    if selected_ref in ref_functions and len(eeg_indices) > 1:
        ref_func = ref_functions[selected_ref]
        eeg_signals = signals[eeg_indices, :]
        averaged_sig = ref_func(eeg_signals, axis=0)

        if not np.allclose(averaged_sig, 0):
            N = len(eeg_signals)
            factor = N / (N - 1)
            processed_eeg_signals = factor * (eeg_signals - averaged_sig)
            signals[eeg_indices, :] = processed_eeg_signals

    elif selected_ref == 'rCAR':
        signals[eeg_indices, :] = rCAR(signals[eeg_indices, :])

    signals = signals / units_scaling_factors[:, np.newaxis] 

    selected_labels = [False] * len(sig_labels)
    return signals, signal_types, dimensions, sig_labels, fs, selected_labels