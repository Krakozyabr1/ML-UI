from sklearn.feature_selection import f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import scipy.stats as sst
import streamlit as st
import pandas as pd
import numpy as np


def spear(dfX, dfY):
    labels = np.unique(np.array(dfY))
    dfYs = [(dfY == label).astype(int) for label in labels]
    names = np.array(dfX.columns)
    ret = []
    for i in names:
        if dfX[i].nunique() == 1:
            print(f"{i} - constant")
            ret.append(0)
        else:
            r = [abs(sst.spearmanr(dfX.loc[:, i], dfY).statistic) for dfY in dfYs]
            ret.append(np.mean(r))

    sorted_ret, sorted_names = zip(*sorted(zip(ret, names), reverse=True))     
    return list(sorted_names), list(sorted_ret)


def evaluate_scores(dft,eval_method,to_plot=True):
    labels = dft.columns
    df_X = dft[labels[:-1]]
    dfX = df_X.copy()
    
    class_label = labels[-1]
    dfY = pd.DataFrame(data=np.array(dft[class_label]), columns=[class_label])

    if eval_method == "F score":
        f_statistic, _ = f_classif(dfX, dfY)
        np.nan_to_num(f_statistic, False, 0, 0, 0)
        names_sorted = [x for _, x in sorted(zip(f_statistic, labels[:-1]), reverse=True)]
        values_sorted = sorted(f_statistic, reverse=True)
        if to_plot:
            fig = plt.figure(figsize=(10,5))
            plt.bar(names_sorted[:min([10,len(labels[:-1])])],values_sorted[:min([10,len(labels[:-1])])])
            plt.ylabel('F score')
            plt.xticks(rotation = 90)
            st.pyplot(fig=fig)

    elif eval_method == 'Mutual info':
        mi = mutual_info_classif(dfX, dfY)
        names_sorted = [x for _, x in sorted(zip(mi, labels[:-1]), reverse=True)]
        values_sorted = sorted(mi, reverse=True)
        if to_plot:
            fig = plt.figure(figsize=(10,5))
            plt.bar(names_sorted[:min([10,len(labels[:-1])])],values_sorted[:min([10,len(labels[:-1])])])
            plt.ylabel('Mi')
            plt.xticks(rotation = 90)
            st.pyplot(fig=fig)

    else:
        names_sorted, values_sorted = spear(dfX, dfY)
        if to_plot:
            fig = plt.figure(figsize=(10,5))
            plt.bar(names_sorted[:min([10,len(labels[:-1])])],values_sorted[:min([10,len(labels[:-1])])])
            plt.ylabel('Mean Spearman coef')
            plt.xticks(rotation = 90)
            st.pyplot(fig=fig)

    return names_sorted
