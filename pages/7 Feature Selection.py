# -*- coding: utf-8 -*-
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
from sklearnex import patch_sklearn
from sklearn.base import clone
import streamlit as st
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import pickle
import os
from functions.functions import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
patch_sklearn()

def feature_selection_importance(X, y, model, cv=5, scoring='accuracy'):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute.")

    feature_indices = np.argsort(importance)[::-1]

    best_score = 0
    best_features = []

    def evaluate_features(selected_features):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            X_selected = X.iloc[:, selected_features]
            scores = cross_val_score(clone(model), X_selected, y, cv=cv, scoring=scoring)
        return np.mean(scores), selected_features

    results = Parallel(n_jobs=-1)(delayed(evaluate_features)(feature_indices[:i]) for i in tqdm(range(1, X.shape[1] + 1)))

    for mean_score, selected_features in results:
        if mean_score > best_score:
            best_score = mean_score
            best_features = selected_features

    return best_features.tolist(), best_score

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
        is_regression = st.checkbox('Regression')
        use_model_based = st.checkbox('Use model-specific scores/coefficients', value=True)
        selection_method = st.selectbox("Method:",options=['Forward', 'Backward'])

        select_file_b = st.form_submit_button("Confirm", type="primary")

@st.cache_resource(show_spinner=False)
def main(dft_path,pkl_path):
    dft = pd.read_csv(dft_path)
    with open(pkl_path, 'rb') as f:
        if is_regression:
            scaler, models = pickle.load(f)
        else:
            scaler, le, models = pickle.load(f)
    logtxtbox = st.empty()
    status_text = f'{"Getting data...":<31}\t'
    logtxtbox.text(status_text)
    labels = dft.columns
    dft = null_remover(dft)
    df = nan_remover(dft)
    # df = outliers_remover(dft)

    class_label = labels[-1]
    if is_regression:
        y = np.array(df[class_label])
    else:
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

        if model_name in ['DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','XGBRegressor','DecisionTreeRegressor']:
            if pre_selection_trees != -1:
                est.fit(df_X, dfY.values.ravel())
                feature_importances = pd.Series(est.feature_importances_, index=df_X.columns)
                selected_features = feature_importances.nlargest(pre_selection_trees).index
                df_X = df_X[selected_features]
            
            ft, best_score = feature_selection_importance(df_X,
                                         y,
                                         est,
                                         cv=5,
                                         scoring=["accuracy","r2"][is_regression])

        elif model_name in ['LogisticRegression', 'RidgeClassifier', 'Ridge', 'Lasso', 'ElasticNet']:
            if pre_selection != -1:
                skb = SelectKBest([f_classif, f_regression][is_regression], k=pre_selection)
                skb.fit_transform(df_X, dfY.values.ravel())
                ft = skb.get_feature_names_out()
                df_X = df_X[ft]
            
            ft, best_score = feature_selection_importance(df_X,
                                         y,
                                         est,
                                         cv=5,
                                         scoring=["accuracy","r2"][is_regression])
        else:
            if pre_selection != -1:
                skb = SelectKBest([mutual_info_classif, mutual_info_regression][is_regression], k=pre_selection)
                skb.fit_transform(df_X, dfY.values.ravel())
                ft = skb.get_feature_names_out()
                df_X = df_X[ft]

            sfs = SFS(
                est,
                k_features=[(1, max_features), "best"][max_features == -1],
                forward=selection_method == 'Forward',
                floating=False,
                verbose=1,
                scoring=["accuracy","r2"][is_regression],
                n_jobs=-1,
                cv=5,
            )

            sfs = sfs.fit(df_X, dfY.values.ravel())
            ft = sfs.k_feature_names_
            best_score = sfs.k_score_

        status_text = status_text + f"Done! {len(ft)} features (cv score: {round(100*(best_score))}%)\n"
        logtxtbox.text(status_text)
        ret.append((model_name, ft))

    if is_regression:
        return ret, scaler
    else:
        return ret, scaler, le

if select_file_b:
    main.clear()

if (dft_path_option != "" and pkl_path_option != "") or (select_file_b and dft_path_option != "" and pkl_path_option != ""):
    with right:
        if is_regression:
            models, scaler = main(dft_path, pkl_path)
        else:
            models, scaler, le = main(dft_path, pkl_path)

    with st.form("my_form", clear_on_submit=False, border=False):
        Methods = [i for (i, _) in models]

        selected_models = [True]*len(Methods)
        for i in range(len(selected_models)):
            selected_models[i] = st.checkbox(f'{Methods[i]}',value=True)
        save_button = st.form_submit_button("Save selected", type="primary")

    if save_button:
        if is_regression:
            to_save = (scaler, [models[i] for i in range(len(Methods)) if selected_models[i]])
        else:
            to_save = (scaler, le, [models[i] for i in range(len(Methods)) if selected_models[i]])
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)
            