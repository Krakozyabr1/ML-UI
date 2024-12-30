from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearnex import patch_sklearn
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
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

        pkl_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Selected features")
        pkl_path_ls = os.listdir(pkl_path_folder)
        pkl_path_option = st.selectbox("Select models file:", options=[""]+pkl_path_ls)
        if pkl_path_option != "":
            pkl_path = os.path.join(pkl_path_folder, pkl_path_option)

        saveto_name = st.text_input("Select output .pkl file name:", value="selected_models").replace('.pkl', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Models/Trained/{saveto_name}.pkl")
        select_file_b = st.form_submit_button("Confirm", type="primary")

@st.cache_resource(show_spinner=False)
def main(dft_path):
    if 'estimators' not in globals():
        estimators = []

    with open(pkl_path, 'rb') as f:
        scaler, loaded = pickle.load(f)
        
    dft = pd.read_csv(dft_path)
    dft = null_remover(dft)
    df = outliers_remover(dft)
    labels = dft.columns
    class_label = labels[-1]
    yNames = np.unique(np.array(df[class_label]))

    le = LabelEncoder()
    y = le.fit_transform(np.array(df[class_label]))
    dfX = df[labels[:-1]]
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])
    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})

    X_train, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2, stratify=y)

    models = {'LogisticRegression': LogisticRegression(max_iter=1000),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GaussianNB': GaussianNB(),
            'GradientBoostingClassifier': XGBClassifier(),
            'RidgeClassifier': RidgeClassifier(),
            'SVC': SVC(),
            }

    params_set = {'LogisticRegression': {
        'penalty' : ['l1','l2'],
        'C' : [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver' : ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga', 'newton-cholesky'],
    },
    'KNeighborsClassifier' : {
        'n_neighbors' : [1, 2, 5, 10],
        'metric' : ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'nan_euclidean'],
    },
    'DecisionTreeClassifier': {
        'max_depth' : [1,5,10,20,30,40,50],
        'criterion' : ['gini', 'entropy', 'log_loss'],
    },
    'RandomForestClassifier': {
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'max_depth' : [10,20,30,40,50],
        'n_estimators' : [10,20,50,100,200,500],
    },
    'GaussianNB': {
        'var_smoothing' : [1e-7,1e-8,1e-9,1e-10,1e-11],
    },
    'GradientBoostingClassifier': {
        'learning_rate' : [0.1,0.2,0.5],
        'n_estimators' : [20,50,100,200,500],
        'max_depth' : [2,5,10,20,50],
    },
    'RidgeClassifier': {
        'alpha' : [0.1,1,10],
        'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    }
    }

    status_text = ''
    logtxtbox = st.empty()
    for modelname, selected_labels in loaded:
        X = X_train.loc[:,selected_labels]
        if modelname != 'SVC':
            params = params_set[modelname]
            status_text = status_text + f'{modelname+'...':<31}\t'
            logtxtbox.text(status_text)
            clf = GridSearchCV(models[modelname], params, cv=5, n_jobs=-1)
            clf.fit(X, y)
            estimators.append(clf.best_estimator_)
            status_text = status_text + f'Done! (cv score: {round(clf.best_score_*100)}%)\n'
        else:
            params1 = {
                'C' : [0.01, 0.1, 1.0, 10.0, 100.0],
                'kernel' : ['rbf', 'sigmoid'],
            }
            params2 = {
                'C' : [0.01, 0.1, 1.0, 10.0, 100.0],
                'degree' : [1,2,3,4,5,6,7],
                'kernel' : ['poly'],
            }

            status_text = status_text + f'{modelname+'...':<31}\t'
            clf1 = GridSearchCV(models[modelname], params1, cv=5, n_jobs=-1)
            clf1.fit(X, y)

            clf2 = GridSearchCV(models[modelname], params2, cv=5, n_jobs=-1)
            clf2.fit(X, y)

            estimators.append([clf1.best_estimator_, clf2.best_estimator_][int(clf1.best_score_ < clf2.best_score_)])
            status_text = status_text + f'Done! (cv score: {round(max([clf1.best_score_, clf2.best_score_])*100)}%)\n'
            logtxtbox.text(status_text)

    return estimators, X_test, y_test, yNames, loaded, scaler

if select_file_b:
    main.clear()

if (dft_path_option != "" and pkl_path_option != "") or (select_file_b and dft_path_option != "" and pkl_path_option != ""):
    cmap = st.selectbox("Confusion Matrix color map:", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                                                                'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                                                'bone', 'pink',
                                                                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                                                                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
    with right:
        estimators, X, y_test, yNames, loaded, scaler = main(dft_path)

    with st.form("my_form", clear_on_submit=False, border=False):
        Methods = [i for i,_ in loaded]
        Accs = []
        Cs = []
        for i, est in enumerate(estimators):
            X_test = X.loc[:,loaded[i][1]]
            y_pred = est.predict(X_test)
            C = confusion_matrix(y_test,y_pred)
            Cs.append(C)
            Accs.append(np.mean(C.diagonal()/C.sum(axis=1)))

        selected_models = [True]*8
        for i in range(len(estimators)):
            selected_models[i] = st.checkbox(
                            f'({round(Accs[i]*100)}%) {loaded[i][0]}',
                            value=True,
                        )
        save_button = st.form_submit_button("Save selected", type="primary")

    left2, right2 = st.columns(2)
    for i, ((name, _), C) in enumerate(zip(loaded, Cs)):
        with [left2, right2][i % 2]:
            fig, ax = plt.subplots(figsize=(7, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=C,display_labels=yNames)
            disp.plot(ax=ax,colorbar=False,cmap=cmap)
            Acc = C.diagonal()/C.sum(axis=1)
            accs_txt = f'{name}: '
            for yName in range(len(np.unique(yNames))):
                accs_txt = accs_txt + f'{round(Acc[yName]*100)}% '
            accs_txt = accs_txt + f'({round(np.mean(Acc)*100)}%)'
            plt.title(accs_txt)
            st.pyplot(fig=fig)

    fig, ax = plt.subplots(figsize=(10, 5))

    plt.bar(Methods,Accs)

    plt.xticks(rotation=90)
    ax.set_title("Methods comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0,1)
    fig.tight_layout()
    st.pyplot(fig=fig)

    if save_button:
        to_save = (scaler, {Methods[i]: (estimators[i], loaded[i][1]) for i in range(len(Methods)) if selected_models[i]}, yNames)
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)
