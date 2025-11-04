# -*- coding: utf-8 -*-
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearnex import patch_sklearn
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import pickle
import os
import functions.functions as fn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
patch_sklearn()

st.set_page_config(layout="wide")
PAGE_NAME = "Selection"

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

@st.cache_resource(show_spinner=False)
def read_and_prepare(df_path):
    return fn.read_and_prepare(df_path)


@st.cache_resource(show_spinner=False)
def models_feature_selection(df_path, pkl_path, max_features, pre_selection, pre_selection_trees):
    df = read_and_prepare(df_path)
    with open(pkl_path, 'rb') as f:
        if is_regression:
            [scaler], models = pickle.load(f)
        else:
            [scaler, le], models = pickle.load(f)
    logtxtbox = st.empty()
    status_text = f'{"Getting data...":<31}\t'
    logtxtbox.text(status_text)
    labels = df.columns

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
        status_text = status_text + f"{model_name+'...':<31}\t "
        logtxtbox.text(status_text)
        df_X = X.copy()

        if model_name in ['DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','XGBRegressor','DecisionTreeRegressor']:
            if pre_selection_trees != -1:
                pre_selection_trees = min([pre_selection_trees, len(df_X.columns)])
                est.fit(df_X, dfY.values.ravel())
                feature_importances = pd.Series(est.feature_importances_, index=df_X.columns)
                selected_features = feature_importances.nlargest(pre_selection_trees).index
                df_X = df_X[selected_features]
                est.fit(df_X, dfY.values.ravel())
            
            max_features = min([max_features, len(df_X.columns)])
            ft, best_score = fn.feature_selection_importance(df_X,
                                         y,
                                         est,
                                         max_features,
                                         use_model_based,
                                         selection_method,
                                         is_regression,
                                         cv=5,
                                         scoring=["accuracy","r2"][is_regression])

        elif model_name in ['LogisticRegression', 'RidgeClassifier', 'Ridge', 'Lasso', 'ElasticNet']:
            if pre_selection != -1:
                pre_selection = min([pre_selection, len(df_X.columns)])
                skb = SelectKBest([f_classif, f_regression][is_regression], k=pre_selection)
                skb.fit_transform(df_X, dfY.values.ravel())
                ft = skb.get_feature_names_out()
                df_X = df_X[ft]
                est.fit(df_X, dfY.values.ravel())
            
            max_features = min([max_features, len(df_X.columns)])
            ft, best_score = fn.feature_selection_importance(df_X,
                                         y,
                                         est,
                                         max_features,
                                         use_model_based,
                                         selection_method,
                                         is_regression,
                                         cv=5,
                                         scoring=["accuracy","r2"][is_regression])
        else:
            if pre_selection != -1:
                pre_selection = min([pre_selection, len(df_X.columns)])
                skb = SelectKBest([mutual_info_classif, mutual_info_regression][is_regression], k=pre_selection)
                skb.fit_transform(df_X, dfY.values.ravel())
                ft = skb.get_feature_names_out()
                df_X = df_X[ft]
                est.fit(df_X, dfY.values.ravel())

            max_features = min([max_features, len(df_X.columns)])
            ft, best_score = fn.feature_selection_importance(df_X,
                                         y,
                                         est,
                                         max_features,
                                         use_model_based=False,
                                         selection_method=selection_method,
                                         is_regression=is_regression,
                                         cv=5,
                                         scoring=["accuracy","r2"][is_regression])

        status_text = status_text + f"Done! {len(ft)} features (cv score: {round(100*(best_score))}%)\n"
        logtxtbox.text(status_text)
        ret.append((model_name, ft))

    if is_regression:
        return ret, scaler
    else:
        return ret, scaler, le


left, right = st.columns(2)
with left:
    with st.form("file_selector_form", clear_on_submit=False):
        analysis_type = st.selectbox('Analysis type', options=['Classification', 'Regression'])
        is_regression = analysis_type == 'Regression'
        df_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Learning")
        df_path_ls = os.listdir(df_path_folder)
        df_path_option = st.selectbox("Select features file:", options=[""]+df_path_ls)
        if df_path_option != "":
            df_path = os.path.join(df_path_folder, df_path_option)

        pkl_path_folder = os.path.join(os.path.dirname(__file__), "..", "Models/Pre-selection")
        pkl_path_ls = os.listdir(pkl_path_folder)
        pkl_path_option = st.selectbox("Select models file:", options=[""]+pkl_path_ls)
        if pkl_path_option != "":
            pkl_path = os.path.join(pkl_path_folder, pkl_path_option)

        max_features_input = st.text_input("Maximum number of selected features:", value="-1")
        pre_selection_input = st.text_input("Number of pre-selected features:", value="-1")
        pre_selection_trees_input = st.text_input("Number of pre-selected features (tree-based):", value="-1")

        max_features = to_int(max_features_input)
        pre_selection = to_int(pre_selection_input)
        pre_selection_trees = to_int(pre_selection_trees_input)

        saveto_name = st.text_input("Select output .pkl file name:", value="selected_features").replace('.pkl', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Features/Selected features/{saveto_name}.pkl")
        use_model_based = st.checkbox('Use model-specific scores/coefficients', value=True)
        selection_method = st.selectbox("Method:",options=['Forward', 'Backward'])

        select_file_b = st.form_submit_button("Confirm", type="primary")

if select_file_b:
    models_feature_selection.clear()

if (df_path_option != "" and pkl_path_option != "") or (select_file_b and df_path_option != "" and pkl_path_option != ""):
    with right:
        if is_regression:
            models, scaler = models_feature_selection(df_path, pkl_path, max_features, pre_selection, pre_selection_trees)
        else:
            models, scaler, le = models_feature_selection(df_path, pkl_path, max_features, pre_selection, pre_selection_trees)

    with st.form("my_form", clear_on_submit=False, border=False):
        Methods = [i for (i, _) in models]

        selected_models = [True]*len(Methods)
        for i in range(len(selected_models)):
            selected_models[i] = st.checkbox(f'{Methods[i]}',value=True)
        save_button = st.form_submit_button("Save selected", type="primary")

    if save_button:
        if is_regression:
            to_save = ([scaler], [models[i] for i in range(len(Methods)) if selected_models[i]])
        else:
            to_save = ([scaler, le], [models[i] for i in range(len(Methods)) if selected_models[i]])
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)
            