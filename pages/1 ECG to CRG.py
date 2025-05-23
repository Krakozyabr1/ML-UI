from functions.read_edf import readedf
from functions.r_peaks import r_peaks
from scipy import signal, interpolate
from pyedflib import highlevel
import streamlit as st
import numpy as np
import os

st.set_page_config(layout="wide")
PAGE_NAME = "ECG to CRG"

def _reset_pre_selection_page_state():
    if st.session_state.get('last_active_page') != PAGE_NAME:
        st.session_state.df_path_confirmed = False
        st.session_state.pkl_path_confirmed = False
        st.session_state.current_df_path = ""
        st.session_state.current_pkl_path = ""
        st.session_state.current_analysis_type = "Classification"
        st.session_state.current_to_use_models = []
        st.session_state.calculation_triggered = False

_reset_pre_selection_page_state()
st.session_state['last_active_page'] = PAGE_NAME

@st.cache_resource
def call_readedf(selected_file):
    return readedf(selected_file)


left, right = st.columns(2)
with left:
    with st.form("file_selector_form", clear_on_submit=False):
        file_dir = st.text_input("Select .edf files directory:", value="").replace(
            '"', ""
        )

        if file_dir != "":
            if file_dir[-1] != "\\":
                file_dir += "\\"
            ls = [file_dir + "\\" + i for i in os.listdir(file_dir)]
        select_file_b = st.form_submit_button("Confirm", type="primary")


if file_dir != "":
    signals, signal_headers, header, sig_labels, fs, t, selected_labels = readedf(
        file_dir + "\\" + os.listdir(file_dir)[0]
    )


with right:
    with st.form("my_form", clear_on_submit=False, border=False):
        if file_dir != "":
            cols = st.columns(3)
            for i, label in enumerate(sig_labels):
                with cols[i % 3]:
                    selected_labels[i] = st.checkbox(
                        label,
                        value=("ECG" in label),
                    )
        toCRG = st.form_submit_button("Calculate", type="primary")


with left:
    if toCRG and file_dir != "":
        saveto = os.path.join(os.path.dirname(file_dir), "..", "CRG/crg_")
        selected_label = np.where(selected_labels)[0][0]

        for filename_short, file_name in zip(os.listdir(file_dir), ls):
            signals, _, _ = highlevel.read_edf(file_name)
            amp_label = signal_headers[selected_label]['dimension']
            fs = int(signal_headers[selected_label]['sample_rate'])
            ecg = signals[selected_label]

            ecg_fir_b = signal.firwin(numtaps=1000, cutoff=[0.5, 70], fs=fs, pass_zero='bandpass')
            ecg_prep = signal.filtfilt(ecg_fir_b, 1, ecg)

            _, time_locs, RR = r_peaks(ecg_prep,fs)
            cs = interpolate.CubicSpline(time_locs[1:],RR)
            new_t = np.arange(8*time_locs[1],8*time_locs[-1])/8
            RR_interp = cs(new_t)
            
            highlevel.write_edf(saveto+filename_short[:-3]+"edf", signals=[RR, RR_interp],
                                header=header,
                                signal_headers=[{'label': 'CRG', 'dimension': 's', 'sample_rate': 1.0, 'sample_frequency': 1.0, 'physical_max': max(abs(RR)),
                                                 'physical_min': -max(abs(RR)), 'digital_max': 32767, 'digital_min': -32767, 'prefilter': 'HP:0.000 Hz LP:0.0 Hz N:0.0', 'transducer': 'Unknown'},
                                                {'label': 'CRG interp', 'dimension': 's', 'sample_rate': 8.0, 'sample_frequency': 8.0, 'physical_max': max(abs(RR_interp)),
                                                 'physical_min': -max(abs(RR_interp)), 'digital_max': 32767, 'digital_min': -32767, 'prefilter': 'HP:0.000 Hz LP:0.0 Hz N:0.0', 'transducer': 'Unknown'}])
