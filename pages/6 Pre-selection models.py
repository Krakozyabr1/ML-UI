from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
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

from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer

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

        pre_selected_num = int(
                st.text_input("Number of features to use for BayesSearchCV:", value="-1")
            )
        n_iter = int(st.text_input("Number of iterations for BayesSearchCV:", value="20"))
        
        saveto_name = st.text_input("Select output .pkl file name:", value="selected_models").replace('.pkl', "")
        saveto = os.path.join(os.path.dirname(__file__), "..", f"Models/Pre-selection/{saveto_name}.pkl")

        available_models = ['LogisticRegression','KNeighborsClassifier','GaussianNB',
                            'DecisionTreeClassifier','RandomForestClassifier',
                            'GradientBoostingClassifier','RidgeClassifier','SVC']
        
        to_use = [True]*8
        for i, to_use_label in enumerate(available_models):
            to_use[i] = st.checkbox(to_use_label, True)


        select_file_b = st.form_submit_button("Confirm", type="primary")

@st.cache_resource(show_spinner=False)
def main(dft_path, to_use, n_iter):
    if 'estimators' not in globals():
        estimators = []
    
    dft = pd.read_csv(dft_path)
    dft = null_remover(dft)
    df = nan_remover(dft)
    # df = outliers_remover(dft)
    labels = dft.columns
    class_label = labels[-1]
    yNames = np.array(df[class_label])

    le = LabelEncoder()
    y = le.fit_transform(np.array(df[class_label]))
    dfX = df[labels[:-1]]
    scaler = StandardScaler()
    scaler.fit(dfX)
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])
    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})
    dfY = pd.DataFrame(data=y, columns=[class_label])

    if pre_selected_num != -1:
        skb = SelectKBest(f_classif, k=pre_selected_num)
        skb.fit_transform(dfX, dfY.values.ravel())
        ft = skb.get_feature_names_out()
        dfX = dfX[ft]

    X, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2)
    models = [('LogisticRegression',LogisticRegression(max_iter=1000)),
            ('KNeighborsClassifier',KNeighborsClassifier()),
            ('GaussianNB',GaussianNB()),
            ('DecisionTreeClassifier',DecisionTreeClassifier()),
            ('RandomForestClassifier',RandomForestClassifier()),
            ('GradientBoostingClassifier',XGBClassifier()),
            ('RidgeClassifier',RidgeClassifier()),
            ('SVC',SVC()),
            ]

    params_set = [[{
        'penalty' : ['l1'],
        'C' : (1e-6, 1e+6, 'log-uniform'),
        'solver' : ['liblinear', 'saga'],
    },
    {
        'penalty' : ['l2'],
        'C' : (1e-6, 1e+6, 'log-uniform'),
        'solver' : ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga', 'newton-cholesky'],
    }
    ],
    {
        'n_neighbors' : (1, 10),
        'metric' : ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'nan_euclidean'],
    },
    {
        'var_smoothing' : (1e-7, 1e+11, 'log-uniform'),
    },
    {
        'max_depth' : (1, 40),
        'criterion' : ['gini', 'entropy', 'log_loss'],
    },
    {
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'max_depth' : (10, 40),
        'n_estimators' : (10, 200),
    },
    {
        'learning_rate' : (0.1, 0.5, 'uniform'),
        'n_estimators' : (10, 200),
        'max_depth' : (5, 40),
    },
    {
        'alpha' : (1e-2, 1e+2, 'log-uniform'),
        'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    },
    [{
        'C' : (1e-3, 1e+3, 'log-uniform'),
        'kernel' : ['rbf', 'sigmoid'],
    },
    {
        'C' : (1e-3, 1e+3, 'log-uniform'),
        'degree' : (1, 5),
        'kernel' : ['poly'],
    }
    ]
    ]

    status_text = ''
    logtxtbox = st.empty()
    for i, params in enumerate(params_set):
        if to_use[i]:
            status_text = status_text + f'{models[i][0]+'...':<31}\t'
            logtxtbox.text(status_text)
            clf = BayesSearchCV(models[i][1], params, cv=5, n_points=2, n_iter=n_iter, n_jobs=-1)
            clf.fit(X, y)
            estimators.append(clf.best_estimator_)
            status_text = status_text + f'Done! (cv score: {round(clf.best_score_*100)}%)\n'

    logtxtbox.text(status_text)
    return estimators, [model for i, model in enumerate(models) if to_use[i]], X_test, y_test, yNames, scaler, le

if select_file_b:
    main.clear()

if (dft_path_option != "" or (select_file_b and dft_path_option != "")) and sum(to_use) > 0:
    cmap = st.selectbox("Confusion Matrix color map:", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                                                                'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                                                'bone', 'pink',
                                                                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                                                                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
    with right:
        estimators, models, X_test, y_test, yNames, scaler, le = main(dft_path, to_use, n_iter)

    with st.form("my_form", clear_on_submit=False, border=False):
        Methods = [i for i,_ in models]
        Accs = []
        Cs = []
        for i, est in enumerate(estimators):
            y_pred = est.predict(X_test)
            C = confusion_matrix(y_test,y_pred)
            Cs.append(C)
            Accs.append(np.mean(C.diagonal()/C.sum(axis=1)))

        selected_models = [True]*8
        for i in range(len(estimators)):
            selected_models[i] = st.checkbox(
                            f'({Accs[i]*100:.2f}%) {models[i][0]}',
                            value=True,
                        )
        save_button = st.form_submit_button("Save selected", type="primary")

    columns2 = st.columns(2)
    for i, ((name, mod), C) in enumerate(zip(models, Cs)):
        with columns2[i % 2]:
            fig, ax = plt.subplots(figsize=(7, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=C,display_labels=np.unique(yNames))
            disp.plot(ax=ax,colorbar=False,cmap=cmap)
            Acc = C.diagonal()/C.sum(axis=1)
            accs_txt = f'{name}: '
            for yName in range(len(np.unique(yNames))):
                accs_txt = accs_txt + f'{round(Acc[yName]*100)}% '
            accs_txt = accs_txt + f'({np.mean(Acc)*100:.2f}%)'
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
        to_save = (scaler, le, [(Methods[i], estimators[i]) for i in range(len(Methods)) if selected_models[i]])
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)
