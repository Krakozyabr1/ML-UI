from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearnex import patch_sklearn
from xgboost import XGBClassifier
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import warnings
import pickle
import os
from functions.functions import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
patch_sklearn()

st.set_page_config(layout="wide")

left, right = st.columns(2)
with left:
    with st.form("file_selector_form", clear_on_submit=False):
        dft_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/ToClassify")
        dft_path_ls = os.listdir(dft_path_folder)
        dft_path_option = st.selectbox("Select features file:", options=[""]+dft_path_ls)
        if dft_path_option != "":
            dft_path = os.path.join(dft_path_folder, dft_path_option)

        pkl_path_folder = os.path.join(os.path.dirname(__file__), "..", "Models/Trained")
        pkl_path_ls = os.listdir(pkl_path_folder)
        pkl_path_option = st.selectbox("Select models file:", options=[""]+pkl_path_ls)
        if pkl_path_option != "":
            pkl_path = os.path.join(pkl_path_folder, pkl_path_option)

        saveto_name = st.text_input("Select output .csv file name:", value="classified").replace('.csv', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Classified/{saveto_name}.csv")
        is_regression = st.checkbox('Regression')
        select_file_b = st.form_submit_button("Confirm", type="primary")

@st.cache_resource(show_spinner=False)
def main(dft_path,pkl_path):
    with open(pkl_path, 'rb') as f:
        if is_regression:
            scaler, loaded = pickle.load(f)
        else:
            scaler, loaded, yNames = pickle.load(f)
    
    dft = pd.read_csv(dft_path)
    dft = null_remover(dft)
    df = nan_remover(dft)
    # df = outliers_remover(dft)
    labels = dft.columns
    
    dfX = df[labels[:-1]]
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])
    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})
    
    ds = {'Name': list(df.iloc[:,-1])}

    status_text = ''
    logtxtbox = st.empty()
    for modelname in loaded:
        (est, selected_labels) = loaded[modelname]
        X = dfX.loc[:,selected_labels]
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

if select_file_b:
    main.clear()

if (dft_path_option != "" and pkl_path_option != "") or (select_file_b and dft_path_option != "" and pkl_path_option != ""):
    with right:
        df_out = main(dft_path,pkl_path)

    st.dataframe(df_out)
    df_out.to_csv(saveto, columns=df_out.columns, header=df_out.columns, index=False)
