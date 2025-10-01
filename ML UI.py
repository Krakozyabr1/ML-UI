from functions.sgram_part import sgram_part
from functions.read_signals import read_signals
import matplotlib.pyplot as plt
import streamlit as st
from scipy import fft
import numpy as np
import os

st.set_page_config(layout="wide")
PAGE_NAME = "Home"

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
def call_read_signals(selected_file):
    return read_signals(selected_file)


folders_to_create = [
    "Features/Learning",
    "Features/Selected features",
    "Features/ToClassify",
    "Models/Pre-selection",
    "Models/Trained",
    "Classified",
]

for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

variants = ["EEG", "ECG"]


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
            selected_file = st.selectbox("Select file", options=os.listdir(file_dir))
        select_file_b = st.form_submit_button("Confirm", type="primary")

if file_dir != "":
    signals, signal_types, dimensions, sig_labels, fs, selected_labels = call_read_signals(
        file_dir + "\\" + selected_file
    )

with right:
    if file_dir != "":
        variants_available = ["Manually", "All"]
        for var in variants:
            for sig_label in signal_types:
                if var in sig_label:
                    variants_available.append(var)
                    break
        selected_vars = st.selectbox("Select labels", options=variants_available)
    with st.form("my_form", clear_on_submit=False, border=False):
        if file_dir != "":
            cols = st.columns(3)
            for i, label in enumerate(sig_labels):
                with cols[i % 3]:
                    selected_labels[i] = st.checkbox(
                        label,
                        disabled=(selected_vars != "Manually"),
                        value=(selected_vars == "All" or selected_vars in signal_types[i]),
                    )
            t_left, t_right = st.columns(2)
            with t_left:
                t_start = float(st.text_input("t start, s:", value="0"))
            with t_right:
                t_end = float(st.text_input("t end, s:", value="-1"))

            pcols = st.columns(4)
            with pcols[0]:
                win_width = float(st.text_input("Window width, s:", value="0.1"))
            with pcols[1]:
                overlap = float(st.text_input("Overlap, %:", value="50"))
            with pcols[2]:
                f_start = int(st.text_input("Freq start, Hz:", value="0"))
            with pcols[3]:
                f_stop = int(st.text_input("Freq stop, Hz:", value="-1"))

            selected_plots = [False]*4

            selected_plots[0] = st.checkbox('Signal', value=True)

            logcol1, logcol2 = st.columns(2)

            with logcol1:
                selected_plots[1] = st.checkbox('Spectrum', value=False)
            with logcol2:
                log_spectrum = st.checkbox('Log', value=False, key='log_spectrum')

            with logcol1:
                selected_plots[2] = st.checkbox('Spectrogram', value=False)
            with logcol2:
                log_gram = st.checkbox('Log', value=False, key='log_gram')

            with logcol1:
                selected_plots[3] = st.checkbox('PSD', value=False)
            with logcol2:
                log_PSD = st.checkbox('Log', value=False, key='log_PSD')

        toPlot = st.form_submit_button("Plot selected", type="primary")

with left:
    if toPlot and file_dir != "":
        if any(selected_labels):
            eff_num = np.max([1, np.sum(selected_plots)])
            for i, label in enumerate(sig_labels):
                if selected_labels[i]:
                    t = np.arange(len(signals[i])) / fs
                    k = 0
                    if eff_num == 1:
                        fig = plt.figure(figsize=(10, 10))
                    else:
                        fig, axes = plt.subplots(nrows=eff_num, ncols=1)
                        fig.set_figheight(5*eff_num)
                        fig.set_figwidth(10)
                    for j in range(len(selected_plots)):
                        if selected_plots[j]:
                            k = k + 1
                            plt.subplot(eff_num, 1, k)
                            if k == 1: plt.title(sig_labels[i])
                            if j == 0:
                                t_start = int(np.ceil(t_start * fs))
                                if t_end > 0:
                                    t_end = int(t_end * fs)
                                else:
                                    t_end = -1
                                plt.plot(t[t_start:t_end], signals[i][t_start:t_end])
                                plt.xlim([min(t[t_start:t_end]), max(t[t_start:t_end])])
                                plt.ylabel(f'U, {dimensions[i]}')
                                plt.xlabel("t, s")
                            elif j == 1:
                                L = len(signals[i][t_start:t_end])
                                sig_fft = abs(fft.fft(signals[i][t_start:t_end] - np.mean(signals[i][t_start:t_end])))[: int(L / 2)] * 2 / L
                                freq = fft.fftfreq(len(signals[i][t_start:t_end]), d=1 / fs)[: int(L / 2)]
                                df = freq[1] - freq[0]
                                f1 = int(f_start / df)
                                if f_stop > 0:
                                    f2 = int(f_stop / df)
                                else:
                                    f2 = -1
                                if log_spectrum:
                                    plt.semilogy(freq[f1:f2],sig_fft[f1:f2])
                                else:
                                    plt.plot(freq[f1:f2],sig_fft[f1:f2])
                                plt.xlim([min(freq[f1:f2]), max(freq[f1:f2])])
                                plt.ylabel(f'A, {dimensions[i]}')
                                plt.xlabel("f, Hz")
                            elif j == 2:
                                ff, tt, Sxx, _ = sgram_part(signals[i][t_start:t_end], fs, win_width, [f_start, f_stop], overlap)
                                tt = [x+t[t_start] for x in tt]
                                plt.pcolormesh(tt, ff, Sxx, norm=[None, 'log'][log_gram], cmap="gnuplot")
                                plt.ylabel("f, Hz")
                                plt.xlabel("t, s")
                            elif j == 3:
                                ff, tt, _, PSD = sgram_part(signals[i][t_start:t_end], fs, win_width, [f_start, f_stop], overlap)
                                tt = [x+t[t_start] for x in tt]
                                if log_PSD:
                                    plt.semilogy(tt, PSD)
                                else:
                                    plt.plot(tt, PSD)
                                plt.xlim([min(tt), max(tt)])
                                plt.ylabel('PSD')
                                plt.xlabel("t, s")
                
                    st.pyplot(fig=fig)
