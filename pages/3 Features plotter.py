from functions.evaluate_scores import evaluate_scores
import functions.functions as fn
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(layout="wide")
PAGE_NAME = "Feature plotter"

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
def call_evaluate_scores(dft,eval_method,to_plot=True):
    return evaluate_scores(dft,eval_method,to_plot)


@st.cache_resource(show_spinner=False)
def plot_features(df,hist_label,box_label,pair_Xlabel,pair_Ylabel):

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


left, right = st.columns(2)
with left:
    with st.form("file_selector_form", clear_on_submit=False):
        df_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Learning")
        df_path_ls = os.listdir(df_path_folder)
        df_path_option = st.selectbox("Select features file:", options=[""]+df_path_ls)
        if df_path_option != "":
            df_path = os.path.join(df_path_folder, df_path_option)

        eval_method = st.selectbox("Select features file:", options=['Mutual info', "F score", 'Mean Spearman coef'])
        select_file_b = st.form_submit_button("Confirm", type="primary")

if df_path_option != "":
    with left:
        with st.form("plot_selector_form", clear_on_submit=False):
            df = read_and_prepare(df_path)
            labels = call_evaluate_scores(df,eval_method,False)
            class_label =  df.columns[-1]
            hist_label = st.selectbox("Histogram:", options=labels, index=0)
            box_label = st.selectbox("Boxplot:", options=labels, index=0)
            l2, r2 = st.columns(2)
            with l2:
                pair_Xlabel = st.selectbox("Scatterplot X:", options=labels, index=0)        
            with r2:
                pair_Ylabel = st.selectbox("Scatterplot Y:", options=labels, index=1)  
            select_plot_b = st.form_submit_button("Plot", type="primary")

if select_file_b:
    call_evaluate_scores.clear()

if df_path_option != "" or (select_file_b and df_path_option != ""):
    with right:
        _ = call_evaluate_scores(df,eval_method)
    plot_features(df,hist_label,box_label,pair_Xlabel,pair_Ylabel)
    