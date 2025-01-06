# -*- coding: utf-8 -*-
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearnex import patch_sklearn
from xgboost import XGBClassifier
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import numpy as np
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
        dft_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Learning")
        dft_path_ls = os.listdir(dft_path_folder)
        dft_path_option = st.selectbox("Select features file:", options=[""]+dft_path_ls)
        if dft_path_option != "":
            dft_path = os.path.join(dft_path_folder, dft_path_option)

        pkl_path_folder = os.path.join(os.path.dirname(__file__), "..", "Models/Pre-selection")
        pkl_path_ls = os.listdir(pkl_path_folder)
        pkl_path_option = st.selectbox("Select models file:", options=[""]+pkl_path_ls)
        if pkl_path_option != "":
            pkl_path = os.path.join(pkl_path_folder, pkl_path_option)

        max_features = int(st.text_input("Maximum number of selected features:", value="-1"))
        pre_selection = int(st.text_input("Number of pre-selected features:", value="-1"))
        pre_selection_trees = int(st.text_input("Number of pre-selected features (tree-based):", value="-1"))
        saveto_name = st.text_input("Select output .pkl file name:", value="selected_features").replace('.pkl', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Features/Selected features/{saveto_name}.pkl")

        select_file_b = st.form_submit_button("Confirm", type="primary")

@st.cache_resource(show_spinner=False)
def main(dft_path,pkl_path):
    dft = pd.read_csv(dft_path)
    with open(pkl_path, 'rb') as f:
        scaler, le, models = pickle.load(f)
    logtxtbox = st.empty()
    status_text = f'{"Getting data...":<31}\t'
    logtxtbox.text(status_text)
    labels = dft.columns
    dft = null_remover(dft)
    dft = nan_remover(dft)
    df = outliers_remover(dft, [0.05, 0.95], labels[:-1])

    class_label = labels[-1]
    y = le.transform(np.array(df[class_label]))
    dfX = df[labels[:-1]]
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])
    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})

    X = dfX.copy()
    dfY = pd.DataFrame(data=y, columns=[class_label])
    status_text = status_text + "Done!\n"
    logtxtbox.text(status_text)

    ret = []

    for (model_name, est) in models:
        status_text = status_text + f"{model_name+'...':<31}\t"
        logtxtbox.text(status_text)
        df_X = X.copy()

        if model_name in ['DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier']:
            if pre_selection_trees != -1:
                est.fit(df_X, dfY.values.ravel())
                feature_importances = pd.Series(est.feature_importances_, index=df_X.columns)
                selected_features = feature_importances.nlargest(pre_selection_trees).index
                df_X = df_X[selected_features]
        elif model_name in ['SVC']:
            if pre_selection != -1:
                skb = SelectKBest(f_classif, k=pre_selection)
                skb.fit_transform(df_X, dfY.values.ravel())
                ft = skb.get_feature_names_out()
                df_X = df_X[ft]
        else:
            if pre_selection != -1:
                skb = SelectKBest(mutual_info_classif, k=pre_selection)
                skb.fit_transform(df_X, dfY.values.ravel())
                ft = skb.get_feature_names_out()
                df_X = df_X[ft]

        sfs = SFS(
            est,
            k_features=[(1, max_features), "best"][max_features == -1],
            forward=True,
            floating=False,
            verbose=2,
            scoring="accuracy",
            n_jobs=-1,
            cv=5,
        )

        sfs = sfs.fit(df_X, dfY.values.ravel())
        ft = sfs.k_feature_names_

        status_text = status_text + f"Done! {len(ft)} features (cv score: {round(100*(sfs.k_score_))}%)\n"
        logtxtbox.text(status_text)
        ret.append((model_name, ft))

    return ret, scaler, le

if select_file_b:
    main.clear()

if (dft_path_option != "" and pkl_path_option != "") or (select_file_b and dft_path_option != "" and pkl_path_option != ""):
    with right:
        models, scaler, le = main(dft_path, pkl_path)

    with st.form("my_form", clear_on_submit=False, border=False):
        Methods = [i for (i, _) in models]

        selected_models = [True]*len(Methods)
        for i in range(len(selected_models)):
            selected_models[i] = st.checkbox(f'{Methods[i]}',value=True)
        save_button = st.form_submit_button("Save selected", type="primary")

    if save_button:
        to_save = (scaler, le, [models[i] for i in range(len(Methods)) if selected_models[i]])
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)
            