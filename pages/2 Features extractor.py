from functions.read_signals import read_signals
import streamlit as st
import pandas as pd
import warnings
import time
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(layout="wide")
PAGE_NAME = "Feature extractor"

def to_int(value):
    try:
        return int(value)
    except ValueError:
        st.warning(f"Cannot convert '{value}' to an integer: Invalid value.")
        return None
    except TypeError:
        st.warning(f"Cannot convert '{value}' to an integer: Invalid type.")
        return None
    except Exception as e:
        st.warning(f"Cannot convert '{value}' to an integer: {e}")
        return None

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


left, right = st.columns(2)
with left:
    signal_type = st.selectbox('Signals type', options=['EEG', 'ECG', 'CRG'])
    corr_wavelet = {'EEG': (2, 7),
                    'ECG': (3, 4),
                    'CRG': (8, 3)
                    }
    
    with st.form("file_selector_form", clear_on_submit=False):
        learning_check = st.checkbox('Learning', value=True)
        s_dir = st.text_input("Select signal files directory:", value="").replace('"', "")
        if s_dir != "":
            if learning_check:
                classes = os.listdir(s_dir)
                if s_dir[-2:] == "\\" or s_dir[-1] == "/":
                    dirs = [s_dir + i + "\\" for i in classes]
                else:
                    dirs = [s_dir + "\\" + i + "\\" for i in classes]
            else:
                if s_dir[-2:] == "\\": 
                    s_dir = s_dir[:-2]
                elif s_dir[-1] == "/":
                    s_dir = s_dir[:-1]

        filestoread_input = st.text_input("Files to read from each folder (ignore if not learning):", value="10")
        filestoread = to_int(filestoread_input)
        
        saveto_name = st.text_input("Select output .csv file name:", value="features").replace('.csv', "")
        if learning_check:
            saveto = os.path.join(os.path.dirname(__file__), "..", f"Features/Learning/{saveto_name}.csv")
        else:
            saveto = os.path.join(os.path.dirname(__file__), "..", f"Features/ToClassify/{saveto_name}.csv")

        wlt = st.selectbox("Select wavelet:", index=corr_wavelet[signal_type][0], options=[
                            'haar',
                            'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19',
                            'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38',
                            'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20',
                            'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17',
                            'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8',
                            'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8',
                            'dmey',
                            'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8',
                            'mexh',
                            'morl',
                            'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8',
                            'shan',
                            'fbsp',
                            'cmor',
                            ])
        st.text('Wavelet levels:')
        wlt_level_l, wlt_level_r = st.columns(2)
        with wlt_level_l:
            wlt_level_min_input = st.text_input("From:", value=f"1")
        with wlt_level_r:
            wlt_level_input = st.text_input("To:", value=f"{corr_wavelet[signal_type][1]}")

        wlt_level_min = max([to_int(wlt_level_min_input) - 1, 0])
        wlt_level = to_int(wlt_level_input)
        
        select_file_b = st.form_submit_button("Confirm", type="primary")
        

if s_dir != "":
    if learning_check:
        signals, signal_types, dimension, labels, fs, selected_labels = call_read_signals(
            dirs[0] + "\\" + os.listdir(dirs[0])[0]
        )
    else:
        signals, signal_types, dimension, labels, fs, selected_labels = call_read_signals(
            s_dir + "\\" + os.listdir(s_dir)[0]
        )

with right:
    if s_dir != "":
        with st.form("my_form", clear_on_submit=False, border=False):
            cols = st.columns(3)
            for i, label in enumerate(labels):
                with cols[i % 3]:
                    selected_labels[i] = st.checkbox(
                        label,
                        value=(signal_type in signal_types[i]),
                    )
            analyze_b = st.form_submit_button("Calculate", type="primary")

is_ready = True
for check in [s_dir, filestoread, wlt_level_min, wlt_level, filestoread]:
    if check is None:
        is_ready = False
        break

if 'analyze_b' in globals() and (is_ready and analyze_b):
    with right:
        logtxtbox = st.empty()
        if learning_check:
            M = []
            for class_ in classes:
                M.extend([class_] * filestoread)

        if learning_check:
            ls = []
            for dir_ in dirs:
                ls.extend([dir_ + i for i in os.listdir(dir_)][0:filestoread])
        else:
            ls = os.listdir(s_dir)
        
        total_files = len(ls)
        start_time = time.time()

        match signal_type:
            case 'EEG':
                import functions.eeg_analysis as analysis
            case 'ECG':
                import functions.ecg_analysis as analysis
            case 'CRG':
                import functions.crg_analysis as analysis

        filters = analysis.create_filters(fs)

        if learning_check:
            ds = [
                    analysis.generate_features_table(e, i, start_time, labels, filters, total_files,
                                                         wlt, wlt_level_min, wlt_level, fs, selected_labels, logtxtbox)
                    for e, i in enumerate(ls)
                ]
        else:
            ds = [
                analysis.generate_features_table(e, s_dir+'\\'+i, start_time, labels, filters, total_files, 
                                                     wlt, wlt_level_min, wlt_level, fs, selected_labels, logtxtbox)
                for e, i in enumerate(ls)
            ]

        df = pd.DataFrame(ds)
        if learning_check:
            df["Label"] = M
        else:
            df["Name"] = [x[:-4] for x in ls]
        df.to_csv(saveto, columns=df.columns, header=df.columns, index=False)

        elap = time.time() - start_time
        elap_h = elap // 3600
        elap_m = elap // 60 - 60 * elap_h
        elap_s = elap % 60
        st.text(f"Done! Time Elapsed: {elap_h:02.0f}:{elap_m:02.0f}:{elap_s:02.0f}")

    st.dataframe(df, column_config={x: st.column_config.NumberColumn(format="%.3e") for x in df.columns[:-1]})
