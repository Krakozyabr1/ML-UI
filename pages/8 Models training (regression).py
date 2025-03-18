from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearnex import patch_sklearn
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
import os
from functions.functions import *

from skopt import BayesSearchCV

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

        left2, right2 = st.columns(2)
        with left2:
            use_sig = st.checkbox('Use sigmoid transform', value=False)
        with right2:
            use_round = st.checkbox('Round output', value=False)

        pkl_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Selected features")
        pkl_path_ls = os.listdir(pkl_path_folder)
        pkl_path_option = st.selectbox("Select models file:", options=[""]+pkl_path_ls)
        if pkl_path_option != "":
            pkl_path = os.path.join(pkl_path_folder, pkl_path_option)

        saveto_name = st.text_input("Select output .pkl file name:", value="selected_models").replace('.pkl', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Models/Trained/{saveto_name}.pkl")
        select_file_b = st.form_submit_button("Confirm", type="primary")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@st.cache_resource(show_spinner=False)
def main(dft_path):
    if 'estimators' not in globals():
        estimators = []

    with open(pkl_path, 'rb') as f:
        scaler, loaded = pickle.load(f)
        
    dft = pd.read_csv(dft_path)
    dft = null_remover(dft)
    dft = nan_remover(dft)
    df = outliers_remover(dft)
    labels = dft.columns
    class_label = labels[-1]
    y = np.array(df[class_label])
    dfX = df[labels[:-1]]
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])
    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})

    X_train, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2)

    models = {
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'XGBRegressor': XGBRegressor(),
            'GaussianProcessRegressor': GaussianProcessRegressor(),
            }

    params_set = {
        'Ridge': {
            'alpha': (1e-2, 1e+2, 'log-uniform'),
            'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        },
        'Lasso': {
            'alpha': (1e-2, 1e+2, 'log-uniform'),
            'selection': ['cyclic', 'random'],
        },
        'ElasticNet': {
            'alpha': (1e-2, 1e+2, 'log-uniform'),
            'selection': ['cyclic', 'random'],
            'l1_ratio': (0,1, 'uniform'),
        },
        'DecisionTreeRegressor': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': (1, 200, 'uniform'),
        },
        'SVR': [{
            'kernel': ['poly'],
            'C': (1e-3, 1e+3,'log-uniform'),
            'epsilon': (0.01, 0.3,'log-uniform'),
            'gamma': ['scale', 'auto', 0.1, 1],
            'degree': (2, 5),
        },
        {
            'C': (1e-3, 1e+3,'log-uniform'),
            'epsilon': (0.01, 0.3,'log-uniform'),
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['linear', 'rbf', 'sigmoid'],
        },
        ],
        'KNeighborsRegressor': {
            'n_neighbors': (1, 10),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],
            'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2','nan_euclidean'],
        },
        'XGBRegressor': {
            'learning_rate' : (0.1, 0.5, 'uniform'),
            'n_estimators' : (20, 100),
            'max_depth' : (2, 40),
        },
        'GaussianProcessRegressor': {
            # 'kernel': [RBF(), Matern(), RationalQuadratic()],
            'n_restarts_optimizer': (0, 5),
            'alpha': (1e-10, 1e-2, 'log-uniform'),
        },
    }

    status_text = ''
    logtxtbox = st.empty()
    for modelname, selected_labels in loaded:
        X = X_train.loc[:,selected_labels]
        params = params_set[modelname]
        status_text = status_text + f'{modelname+'...':<31}\t'
        logtxtbox.text(status_text)
        clf = BayesSearchCV(models[modelname], params, cv=5, n_points=2, n_iter=50, n_jobs=-1, scoring='r2')
        clf.fit(X, y)
        estimators.append(clf.best_estimator_)
        status_text = status_text + f'Done! (cv score: {round(clf.best_score_*100)}%)\n'

    logtxtbox.text(status_text)

    return estimators, X_test, y_test, loaded, scaler

if select_file_b:
    main.clear()

if (dft_path_option != "" and pkl_path_option != "") or (select_file_b and dft_path_option != "" and pkl_path_option != ""):
    with right:
        estimators, X, y_test, loaded, scaler = main(dft_path)

    with st.form("my_form", clear_on_submit=False, border=False):
        Methods = [i for i,_ in loaded]
        Accs = []
        
        for i, est in enumerate(estimators):
            X_test = X.loc[:,loaded[i][1]]
            y_pred = est.predict(X_test)
            if use_sig:
                y_pred = sigmoid(y_pred)
            if use_round:
                y_pred = np.round(y_pred)
            Accs.append(r2_score(y_test,y_pred))

        selected_models = [True]*8
        for i in range(len(estimators)):
            selected_models[i] = st.checkbox(
                            f'({round(Accs[i]*100)}%) {loaded[i][0]}',
                            value=True,
                        )
        save_button = st.form_submit_button("Save selected", type="primary")

    columns2 = st.columns(2)
    for i, (name, _) in enumerate(loaded):
        with columns2[i % 2]:
            fig, ax = plt.subplots(figsize=(7, 7))
            X_test = X.loc[:,loaded[i][1]]
            y_pred = estimators[i].predict(X_test)
            if use_sig:
                y_pred = sigmoid(y_pred)
            if use_round:
                y_pred = np.round(y_pred)
            ax.scatter(y_test, y_pred)
            sns.regplot(x=y_test, y=y_pred, scatter=False, line_kws={'linewidth': 3}, label='Regression Line')
            ax.plot([min(y_test),max(y_test)], [min(y_pred),max(y_pred)], 'r--', linewidth=3, label='Ideal Line')
            ax.legend()
            ax.set_xlabel('Real Values')
            ax.set_ylabel('Predicted Values')
            plt.title(f'R² Score: {Accs[i]*100:.2f}%')
            st.pyplot(fig=fig)

    fig, ax = plt.subplots(figsize=(10, 5))

    plt.bar(Methods,Accs)

    plt.xticks(rotation=90)
    ax.set_title("Methods comparison")
    ax.set_ylabel("R² Scores")
    ax.set_ylim(0,1)
    fig.tight_layout()
    st.pyplot(fig=fig)

    if save_button:
        to_save = (scaler, {Methods[i]: (estimators[i], loaded[i][1]) for i in range(len(Methods)) if selected_models[i]})
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)
