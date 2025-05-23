from sklearnex import patch_sklearn
import streamlit as st
import pandas as pd
import warnings
import pickle
import os
import functions.functions as fn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
patch_sklearn()

st.set_page_config(layout="wide")
PAGE_NAME = "Use"

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


@st.cache_resource(show_spinner=False)
def read_and_prepare(df_path):
    return fn.read_and_prepare(df_path)


@st.cache_resource(show_spinner=False)
def use_models(df_path,pkl_path):
    with open(pkl_path, 'rb') as f:
        if is_regression:
            scaler, loaded, yNames = pickle.load(f)
        else:
            scaler, loaded, yNames = pickle.load(f)
    
    df = read_and_prepare(df_path)
    labels = df.columns
    
    dfX = df[labels[:-1]]
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])
    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})
    
    ds = {'Name': list(df.iloc[:,-1])}

    status_text = ''
    logtxtbox = st.empty()
    for modelname in loaded:
        (est, selected_labels) = loaded[modelname]
        try:
            X = dfX.loc[:,selected_labels]
        except:
            X = dfX.iloc[:,selected_labels]
        status_text = status_text + f'{modelname+'...':<31}\t'
        logtxtbox.text(status_text)
        ds[modelname] = est.predict(X)
        status_text = status_text + f'Done!\n'
        logtxtbox.text(status_text)

    df_out = pd.DataFrame(ds)
    if not is_regression:
        di = {i: yName for i, yName in enumerate(yNames)}
        for modelname in loaded:
            df_out = df_out.replace({modelname: di})
    
    if len(df_out.columns[1:]) > 1:
        if is_regression:
            df_out['Mean'] = df_out.filter(df_out.columns[1:]).mean(axis=1)
        else:
            df_out['Majority'] = df_out.filter(df_out.columns[1:]).mode(axis=1)[0]

    return df_out


left, right = st.columns(2)
with left:
    with st.form("file_selector_form", clear_on_submit=False):
        dft_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/ToClassify")
        dft_path_ls = os.listdir(dft_path_folder)
        dft_path_option = st.selectbox("Select features file:", options=[""]+dft_path_ls)
        if dft_path_option != "":
            df_path = os.path.join(dft_path_folder, dft_path_option)

        pkl_path_folder = os.path.join(os.path.dirname(__file__), "..", "Models/Trained")
        pkl_path_ls = os.listdir(pkl_path_folder)
        pkl_path_option = st.selectbox("Select models file:", options=[""]+pkl_path_ls)
        if pkl_path_option != "":
            pkl_path = os.path.join(pkl_path_folder, pkl_path_option)

        saveto_name = st.text_input("Select output .csv file name:", value="classified").replace('.csv', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Classified/{saveto_name}.csv")
        is_regression = st.checkbox('Regression')
        select_file_b = st.form_submit_button("Confirm", type="primary")

if select_file_b:
    use_models.clear()

if (dft_path_option != "" and pkl_path_option != "") or (select_file_b and dft_path_option != "" and pkl_path_option != ""):
    with right:
        df_out = use_models(df_path,pkl_path)

    st.dataframe(df_out)
    df_out.to_csv(saveto, columns=df_out.columns, header=df_out.columns, index=False)
