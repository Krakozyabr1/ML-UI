from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearnex import patch_sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from skopt import BayesSearchCV
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
PAGE_NAME = "Train"

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
def call_plotter_reg(predictions, y_test, Methods, Accs, use_sig, use_round):
    columns2 = st.columns(2)
    fn.regression_plotter(columns2, predictions, y_test, Methods, Accs, use_sig, use_round)


@st.cache_resource(show_spinner=False)  
def call_plotter_class(Methods, Cs, yNames, Accs, cmap):
    columns2 = st.columns(2)
    fn.classification_plotter(columns2, Methods, Cs, yNames, Accs, cmap)


def prepare_data_2(df_path, to_encode=False):
    with open(pkl_path, 'rb') as f:
        if to_encode:
            [scaler, le], loaded = pickle.load(f)
        else:
            [scaler], loaded = pickle.load(f)

    df = read_and_prepare(df_path)
    labels = df.columns
    class_label = labels[-1]
    yNames = np.unique(np.array(df[class_label]))

    y = np.array(df[class_label])
    dfX = df[labels[:-1]]
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])

    if to_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})

    if to_encode:
        X_train, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2, stratify=y)
    else:
        X_train, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2)

    if to_encode:
        return X_train, X_test, y, y_test, loaded, yNames, scaler, le
    else:
        return X_train, X_test, y, y_test, loaded, yNames, scaler


@st.cache_resource(show_spinner=False)
def models_tuning_2(df_path, n_iter, analysis_type):
    if analysis_type == 'Classification':
        from functions.classification_config import CLASSIFICATION_PARAMS_SET, CLASSIFICATION_MODELS
        params_set = CLASSIFICATION_PARAMS_SET
        models = CLASSIFICATION_MODELS

        X_train, X_test, y, y_test, loaded, yNames, scaler, le = prepare_data_2(df_path, to_encode=True)

    elif analysis_type == 'Regression':
        from functions.regression_config import REGRESSION_PARAMS_SET, REGRESSION_MODELS
        params_set = REGRESSION_PARAMS_SET
        models = REGRESSION_MODELS
    
        X_train, X_test, y, y_test, loaded, yNames, scaler = prepare_data_2(df_path, to_encode=False)
    
    estimators = []

    status_text = ''
    logtxtbox = st.empty()
    predictions = []

    for modelname, selected_labels in loaded:
        try:
            X = X_train.loc[:,selected_labels]
        except:
            X = X_train.iloc[:,selected_labels]
        params = params_set[modelname]
        status_text = status_text + f'{modelname+'...':<31}\t '
        logtxtbox.text(status_text)
        if analysis_type == 'Classification':
                clf = BayesSearchCV(models[modelname], params, cv=5, n_points=2, n_iter=n_iter, n_jobs=-1)
        elif analysis_type == 'Regression':
            if modelname == 'SVR':
                clf = BayesSearchCV(models[modelname], params, cv=5, n_points=1, n_iter=n_iter, n_jobs=1, scoring='r2')
            else:
                clf = BayesSearchCV(models[modelname], params, cv=5, n_points=2, n_iter=n_iter, n_jobs=-1, scoring='r2')
        clf.fit(X, y)
        est = clf.best_estimator_
        estimators.append(est)
        try:
            X = X_test.loc[:,selected_labels]
        except:
            X = X_test.iloc[:,selected_labels]
        predictions.append(est.predict(X))
        status_text = status_text + f'Done! (cv score: {round(clf.best_score_*100)}%)\n'

    logtxtbox.text(status_text)

    return estimators, loaded, predictions, y_test, yNames, scaler


left, right = st.columns(2)
with left:
    analysis_type_options = ['Classification', 'Regression']
    analysis_type = st.selectbox(
        'Analysis type:',
        options=analysis_type_options,
        key="analysis_type_selector",
        on_change=lambda: st.session_state.update(df_path_confirmed=False, calculation_triggered=False)
    )
    with st.form("file_selector_form", clear_on_submit=False):
        df_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Learning")
        df_path_ls = os.listdir(df_path_folder)
        df_path_option = st.selectbox("Select features file:",
                                      options=[""]+df_path_ls,
                                      key="df_path_option_form_key")
        
        if df_path_option != "":
            df_path = os.path.join(df_path_folder, df_path_option)
        else:
            df_path = ""

        pkl_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Selected features")
        pkl_path_ls = os.listdir(pkl_path_folder)
        pkl_path_option = st.selectbox("Select models file:",
                                      options=[""]+pkl_path_ls,
                                      key="pkl_path_option_form_key")
        
        if pkl_path_option != "":
            pkl_path = os.path.join(pkl_path_folder, pkl_path_option)
        else:
            pkl_path = ""

        if analysis_type == 'Regression':
            left2, right2 = st.columns(2)
            with left2:
                use_sig = st.checkbox('Use sigmoid transform', value=False)
            with right2:
                use_round = st.checkbox('Round output', value=False)

        n_iter = int(st.text_input("Number of iterations for BayesSearchCV:", value="50"))

        saveto_name = st.text_input("Select output .pkl file name:", value="selected_models").replace('.pkl', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Models/Trained/{saveto_name}.pkl")
        select_file_b = st.form_submit_button("Confirm", type="primary")

        if select_file_b:
            if df_path_option == "":
                st.warning("Please select a features file before confirming.")
                st.session_state.df_path_confirmed = False
                st.session_state.calculation_triggered = False
            elif pkl_path_option == "":
                st.warning("Please select a models file before confirming.")
                st.session_state.pkl_path_confirmed = False
                st.session_state.calculation_triggered = False
            else:
                st.session_state.df_path_confirmed = True
                st.session_state.pkl_path_confirmed = True
                st.session_state.current_df_path = df_path
                st.session_state.current_pkl_path = pkl_path
                st.session_state.current_analysis_type = analysis_type
                st.session_state.calculation_triggered = True

                models_tuning_2.clear()

if st.session_state.calculation_triggered:
    if st.session_state.current_analysis_type == 'Classification':
        cmap = st.selectbox("Confusion Matrix color map:", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                                                                    'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                                                    'bone', 'pink',
                                                                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                                                                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
        
    with right:
        estimators, models, predictions, y_test, yNames, scaler = models_tuning_2(
                                                                    st.session_state.current_df_path,
                                                                    n_iter,
                                                                    st.session_state.current_analysis_type
                                                                    )

    with st.form("my_form", clear_on_submit=False, border=False):
        Methods = [i for i,_ in models]
        Accs = []
        Cs = []

        for y_pred in predictions:
            if st.session_state.current_analysis_type == 'Regression':
                if use_sig:
                    y_pred = fn.sigmoid(y_pred)
                if use_round:
                    y_pred = np.round(y_pred)
                Accs.append(r2_score(y_test,y_pred))
            elif st.session_state.current_analysis_type == 'Classification':
                C = confusion_matrix(y_test,y_pred)
                Cs.append(C)
                Accs.append(np.mean(C.diagonal()/C.sum(axis=1)))

        selected_models = [True]*len(models)
        for i in range(len(estimators)):
            selected_models[i] = st.checkbox(
                            f'({Accs[i]*100:.2f}%) {models[i][0]}',
                            value=True,
                        )
        save_button = st.form_submit_button("Save selected", type="primary")

    if st.session_state.current_analysis_type == 'Regression':
        call_plotter_reg(predictions, y_test, Methods, Accs, use_sig, use_round)
    elif st.session_state.current_analysis_type == 'Classification':
        call_plotter_class(Methods, Cs, yNames, Accs, cmap) 

    if save_button:
        to_save = (scaler, {Methods[i]: (estimators[i], models[i][1]) for i in range(len(Methods)) if selected_models[i]}, yNames)
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)
