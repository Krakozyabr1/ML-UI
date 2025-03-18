from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os
from functions.functions import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(layout="wide")

left, right = st.columns(2)
with left:
    with st.form("file_selector_form", clear_on_submit=False):
        dft_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Learning")
        dft_path_ls = os.listdir(dft_path_folder)
        dft_path_option = st.selectbox("Select features file:", options=[""]+dft_path_ls)
        if dft_path_option != "":
            dft_path = os.path.join(dft_path_folder, dft_path_option)

        eval_method = st.selectbox("Select features file:", options=['Mutual info', "F score", 'Mean Spearman coef'])
        select_file_b = st.form_submit_button("Confirm", type="primary")

@st.cache_resource(show_spinner=False)
def evals(dft,eval_method,to_plot=True):
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

@st.cache_resource(show_spinner=False)
def prepare_plot_selector(df):
    df = nan_remover(df)
    # df = outliers_remover(df)
    return df

if dft_path_option != "":
    with left:
        with st.form("plot_selector_form", clear_on_submit=False):
            df = pd.read_csv(dft_path)
            df = null_remover(df)
            # print(df)
            df = prepare_plot_selector(df)
            # print(df)
            labels = evals(df,eval_method,False)
            class_label =  df.columns[-1]
            hist_label = st.selectbox("Histogram:", options=labels, index=0)
            box_label = st.selectbox("Boxplot:", options=labels, index=0)
            l2, r2 = st.columns(2)
            with l2:
                pair_Xlabel = st.selectbox("Scatterplot X:", options=labels, index=0)        
            with r2:
                pair_Ylabel = st.selectbox("Scatterplot Y:", options=labels, index=1)  
            select_plot_b = st.form_submit_button("Plot", type="primary")


@st.cache_resource(show_spinner=False)
def main(df,hist_label,box_label,pair_Xlabel,pair_Ylabel):

    _, r3 = st.columns(2)
    with r3:
        fig1 = plt.figure(figsize=(10,5))
        sns.histplot(data=df,x=hist_label,hue=class_label)
        st.pyplot(fig=fig1)

        fig2 = plt.figure(figsize=(10,5))
        sns.boxplot(data=df,x=box_label,hue=class_label)
        st.pyplot(fig=fig2)

        fig3 = plt.figure(figsize=(10,5))
        sns.scatterplot(data=df,x=pair_Xlabel,y=pair_Ylabel,hue=class_label)
        st.pyplot(fig=fig3)


if select_file_b:
    evals.clear()

if dft_path_option != "" or (select_file_b and dft_path_option != ""):
    with right:
        _ = evals(df,eval_method)
    main(df,hist_label,box_label,pair_Xlabel,pair_Ylabel)
    