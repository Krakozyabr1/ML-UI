from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from sklearn.base import clone
import streamlit as st
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle


def feature_selection_importance(X, y, model, max_features, use_model_based, selection_method, is_regression, cv=5, scoring='accuracy'):
    if use_model_based:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute.")

        feature_indices = np.argsort(importance)[::-1]

        if max_features == -1:
            max_features = X.shape[1]

        best_score = 0
        best_features = []

        def evaluate_features(selected_features):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                X_selected = X.iloc[:, selected_features]
                scores = cross_val_score(clone(model), X_selected, y, cv=cv, scoring=scoring)
            return np.mean(scores), selected_features

        results = Parallel(n_jobs=-1)(delayed(evaluate_features)(feature_indices[:i]) for i in tqdm(range(1, max_features + 1)))

        for mean_score, selected_features in results:
            if mean_score > best_score:
                best_score = mean_score
                best_features = selected_features

        return best_features.tolist(), best_score
    else:
        sfs = SFS(
            model,
            k_features=[(1, max_features), "best"][max_features == -1],
            forward=selection_method == 'Forward',
            floating=False,
            verbose=1,
            scoring=["accuracy","r2"][is_regression],
            n_jobs=-1,
            cv=5,
        )

        sfs = sfs.fit(X, y)
        ft = sfs.k_feature_names_
        best_score = sfs.k_score_
        
        return ft, best_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def regression_plotter(columns2, predictions, y_test, Methods, Accs, use_sig, use_round):
        for i, name in enumerate(Methods):
            with columns2[i % 2]:
                fig, ax = plt.subplots(figsize=(7, 7))
                y_pred = predictions[i]
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
                plt.title(f'{name}\nR² Score: {Accs[i]*100:.2f}%')
                st.pyplot(fig=fig)

        fig, ax = plt.subplots(figsize=(10, 5))

        plt.bar(Methods,Accs)

        plt.xticks(rotation=90)
        ax.set_title("Methods comparison")
        ax.set_ylabel("R² Scores")
        ax.set_ylim(0,1)
        fig.tight_layout()
        st.pyplot(fig=fig)


def classification_plotter(columns2, Methods, Cs, yNames, Accs, cmap):
    
    for i, (name, C) in enumerate(zip(Methods, Cs)):
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

    plt.bar(Methods, Accs)

    plt.xticks(rotation=90)
    ax.set_title("Methods comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0,1)
    fig.tight_layout()
    st.pyplot(fig=fig)


def zeros_remover(df, labels=False):
    df2 = df.copy()
    c = 0
    if type(labels) is bool:
        labels = df.columns

    for i in labels:
        if any(df[i].isin([0])):
            print(f"Zero in {i}")
            c = c + 1
            median = df[i].median()
            df2[i] = df[i].replace(0, median)

    print(f"Replaced zero in {c} columns\n")
    return df2


def null_remover(df, labels=False):
    c = 0
    if type(labels) is bool:
        labels = df.columns

    for i in labels:
        if any(df[i].isnull()):
            if pd.api.types.is_numeric_dtype(df[i]):
                print(f"Empty in {i}")
                c = c + 1
                median = df[i].median()
                df[i] = df[i].fillna(median)
            else:
                c = c + 1
                mode = df[i].mode()[0]
                print(f"Empty in {i}")
                df[i] = df[i].fillna(mode)

    print(f"Replaced empty in {c} columns\n")
    return df


def nan_remover(df, labels=False):
    if type(labels) is bool:
        labels = df.columns[:-1]

    for i in labels:
        if pd.api.types.is_numeric_dtype(df[i]):
            continue
        
        try:
            numeric_column = pd.to_numeric(df[i], errors='coerce')
            if numeric_column.isnull().all():
                raise ValueError
            elif numeric_column.isnull().any():
                print(i)
                median_val = numeric_column.median()
                df[i] = numeric_column.fillna(median_val)

        except (TypeError, ValueError):
            unique_values = df[i].unique()
            if len(unique_values) > 2:
                le = LabelEncoder()
                df[i] = le.fit_transform(df[i].astype(str))
            else:
                mapping = {value: index for index, value in enumerate(unique_values)}
                df[i] = df[i].map(mapping)

    return df


def outliers_remover(df, max_iterations=5, labels=False):
    if type(labels) is bool:
        labels = df.columns

    for iteration in range(max_iterations):
        print(f"Iteration {iteration+1}")
        c = 0
        if iteration == 0:
            dfX = df[labels[:-1]].copy()

        for i in labels[:-1]:
            if len(dfX[i].unique()) < 10:
                print(f"{i} is categorical")
            else:
                Q1 = np.percentile(dfX[i], 25)
                Q3 = np.percentile(dfX[i], 75)
                IQR = Q3 - Q1

                min_val = Q1 - 1.5 * IQR
                max_val = Q3 + 1.5 * IQR
                outliers = (dfX[i] < min_val) | (dfX[i] > max_val)
                if outliers.any():
                    c += 1
                    dfX.loc[outliers, i] = None
            dfX[i].fillna(value=dfX[i].mode(), inplace=True)

        if c == 0:
            print("No outliers found\n")
            break

        print(f"Replaced outliers in {c} columns\n")
        df.update(dfX)

    return df


def read_and_prepare(df_path):
    df = pd.read_csv(df_path)
    df = null_remover(df)
    df = nan_remover(df)
    # df = outliers_remover(df)
    return df


def prepare_data_1(df_path, pre_selected_num, to_encode=False):
    estimators = []

    df = read_and_prepare(df_path)
    labels = df.columns
    class_label = labels[-1]
    if to_encode:
        yNames = np.unique(np.array(df[class_label]))
    else:
        yNames = None

    y = np.array(df[class_label])
    dfX = df[labels[:-1]]
    scaler = StandardScaler()
    scaler.fit(dfX)
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])

    if to_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})
    dfY = pd.DataFrame(data=y, columns=[class_label])

    if pre_selected_num != -1:
        skb = SelectKBest(f_classif, k=pre_selected_num)
        skb.fit_transform(dfX, dfY.values.ravel())
        ft = skb.get_feature_names_out()
        dfX = dfX[ft]

    if to_encode:
        X, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2, stratify=y)
    else:
        X, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2)

    if to_encode:
        return X, X_test, y, y_test, estimators, yNames, scaler, le
    else:
        return X, X_test, y, y_test, estimators, yNames, scaler
        

def models_tuning_1(df_path, to_use, n_iter, pre_selected_num, analysis_type):

    if analysis_type == 'Classification':
        from functions.classification_config import CLASSIFICATION_PARAMS_SET, CLASSIFICATION_MODELS
        params_set = CLASSIFICATION_PARAMS_SET
        models = CLASSIFICATION_MODELS

        X, X_test, y, y_test, estimators, yNames, scaler, le = prepare_data_1(df_path, pre_selected_num, to_encode=True)

    elif analysis_type == 'Regression':
        from functions.regression_config import REGRESSION_PARAMS_SET, REGRESSION_MODELS
        params_set = REGRESSION_PARAMS_SET
        models = REGRESSION_MODELS
    
        X, X_test, y, y_test, estimators, yNames, scaler = prepare_data_1(df_path, pre_selected_num, to_encode=False)

    status_text = ''
    logtxtbox = st.empty()
    for i, (name, model) in enumerate(models.items()):
        if to_use[i]:
            status_text = status_text + f'{name+'...':<31}\t '
            logtxtbox.text(status_text)
            if analysis_type == 'Classification':
                clf = BayesSearchCV(model, params_set[name], cv=5, n_points=2, n_iter=n_iter, n_jobs=-1)
            elif analysis_type == 'Regression':
                if name == 'SVR':
                    clf = BayesSearchCV(model, params_set[name], cv=5, n_points=1, n_iter=n_iter, n_jobs=1, scoring='r2')
                else:
                    clf = BayesSearchCV(model, params_set[name], cv=5, n_points=2, n_iter=n_iter, n_jobs=-1, scoring='r2')
            clf.fit(X, y)
            estimators.append(clf.best_estimator_)
            status_text = status_text + f'Done! (cv score: {round(clf.best_score_*100)}%)\n'

    logtxtbox.text(status_text)
    predictions = [estimator.predict(X_test) for estimator in estimators]

    if analysis_type == 'Classification':
        return estimators, [model for i, model in enumerate(models.keys()) if to_use[i]], predictions, y_test, yNames, scaler, le
    elif analysis_type == 'Regression':
        return estimators, [model for i, model in enumerate(models.keys()) if to_use[i]], predictions, y_test, yNames, scaler

   
def prepare_data_2(df_path, pkl_path, to_encode=False):
    with open(pkl_path, 'rb') as f:
        if to_encode:
            [scaler, le], loaded = pickle.load(f)
        else:
            [scaler], loaded = pickle.load(f)

    df = read_and_prepare(df_path)
    labels = df.columns
    class_label = labels[-1]
    if to_encode:
        yNames = np.unique(np.array(df[class_label]))
    else:
        yNames = None

    y = np.array(df[class_label])
    dfX = df[labels[:-1]]
    scaled = scaler.transform(dfX).T
    feature_names_out = scaler.get_feature_names_out(labels[:-1])

    if to_encode:
        y = le.fit_transform(y)

    dfX = pd.DataFrame({feature_names_out[i]: scaled[i] for i in range(len(feature_names_out))})

    if to_encode:
        X_train, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2, stratify=y)
    else:
        X_train, X_test, y, y_test = train_test_split(dfX, y, test_size=0.2)

    return X_train, X_test, y, y_test, loaded, yNames, scaler


def models_tuning_2(df_path, pkl_path, n_iter, analysis_type):
    if analysis_type == 'Classification':
        from functions.classification_config import CLASSIFICATION_PARAMS_SET, CLASSIFICATION_MODELS
        params_set = CLASSIFICATION_PARAMS_SET
        models = CLASSIFICATION_MODELS

        X_train, X_test, y, y_test, loaded, yNames, scaler = prepare_data_2(df_path, pkl_path, to_encode=True)

    elif analysis_type == 'Regression':
        from functions.regression_config import REGRESSION_PARAMS_SET, REGRESSION_MODELS
        params_set = REGRESSION_PARAMS_SET
        models = REGRESSION_MODELS
    
        X_train, X_test, y, y_test, loaded, yNames, scaler = prepare_data_2(df_path, pkl_path, to_encode=False)
    
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

def use_models(df_path, pkl_path, analysis_type):
    with open(pkl_path, 'rb') as f:
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
    if analysis_type == 'Classification':
        mapper = {i: yName for i, yName in enumerate(yNames)}
        for modelname in loaded:
            df_out = df_out.replace({modelname: mapper})
    
    if len(df_out.columns[1:]) > 1:
        if analysis_type == 'Regression':
            df_out['Mean'] = df_out.filter(df_out.columns[1:]).mean(axis=1)
        elif analysis_type == 'Classification':
            df_out['Majority'] = df_out.filter(df_out.columns[1:]).mode(axis=1)[0]

    return df_out