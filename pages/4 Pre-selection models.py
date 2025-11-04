from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearnex import patch_sklearn
import streamlit as st
import numpy as np
import warnings
import pickle
import os
import functions.functions as fn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
patch_sklearn()

st.set_page_config(layout="wide")
PAGE_NAME = "Pre-selection"

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
def call_plotter_reg(predictions, y_test, Methods, Accs, use_sig, use_round):
    columns2 = st.columns(2)
    fn.regression_plotter(columns2, predictions, y_test, Methods, Accs, use_sig, use_round)

@st.cache_resource(show_spinner=False)  
def call_plotter_class(Methods, Cs, yNames, Accs, cmap):
    columns2 = st.columns(2)
    fn.classification_plotter(columns2, Methods, Cs, yNames, Accs, cmap)

@st.cache_resource(show_spinner=False)
def models_tuning_1(df_path, to_use, n_iter, pre_selected_num, analysis_type):
    return fn.models_tuning_1(df_path, to_use, n_iter, pre_selected_num, analysis_type)

@st.cache_resource(show_spinner=False)
def models_tuning_1_reg(df_path, to_use, n_iter, pre_selected_num, analysis_type):
    return fn.models_tuning_1_reg(df_path, to_use, n_iter, pre_selected_num, analysis_type)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        analysis_type_options = ['Classification', 'Regression']
        analysis_type = st.selectbox(
            'Analysis type:',
            options=analysis_type_options,
            key="analysis_type_selector",
            on_change=lambda: st.session_state.update(df_path_confirmed=False, calculation_triggered=False)
        )
        with st.form("file_selector_form", clear_on_submit=False, border=False):
            df_path_folder = os.path.join(os.path.dirname(__file__), "..", "Features/Learning")
            df_path_ls = os.listdir(df_path_folder)
            df_path_option = st.selectbox("Select features file:",
                                        options=[""]+df_path_ls,
                                        key="df_path_option_form_key")
            
            if df_path_option != "":
                df_path = os.path.join(df_path_folder, df_path_option)
            else:
                df_path = ""

            if analysis_type == 'Regression':
                left2, right2 = st.columns(2)
                with left2:
                    use_sig = st.checkbox('Use sigmoid transform', value=False)
                with right2:
                    use_round = st.checkbox('Round output', value=False)

            pre_selected_num_input = st.text_input("Number of features to use for BayesSearchCV:", value="-1")
            n_iter_input = st.text_input("Number of iterations for BayesSearchCV:", value="20")

            pre_selected_num = to_int(pre_selected_num_input)
            n_iter = to_int(n_iter_input)

            saveto_name = st.text_input("Select output .pkl file name:", value="selected_models").replace('.pkl', "")
            saveto = os.path.join(os.path.dirname(__file__), "..", f"Models/Pre-selection/{saveto_name}.pkl")

            if analysis_type == 'Regression':
                from functions.regression_config import REGRESSION_MODELS
                models = REGRESSION_MODELS
                
            elif analysis_type == 'Classification':
                from functions.classification_config import CLASSIFICATION_MODELS
                models = CLASSIFICATION_MODELS
            
            available_models = models.keys()
            to_use = [True] * len(available_models)

            for i, to_use_label in enumerate(available_models):
                if to_use_label in ['LogisticRegression_L1','LogisticRegression_L2']:
                    to_use[i] = st.checkbox(to_use_label, value=False, key=f"model_checkbox_{i}_form")
                else:
                    to_use[i] = st.checkbox(to_use_label, value=True, key=f"model_checkbox_{i}_form")

            select_file_b = st.form_submit_button("Confirm", type="primary")

            if select_file_b:
                if df_path_option == "":
                    st.warning("Please select a features file before confirming.")
                    st.session_state.df_path_confirmed = False
                    st.session_state.calculation_triggered = False
                elif sum(to_use) == 0:
                    st.warning("Please select at least one model to use.")
                    st.session_state.df_path_confirmed = False
                    st.session_state.calculation_triggered = False
                else:
                    st.session_state.df_path_confirmed = True
                    st.session_state.current_df_path = df_path
                    st.session_state.current_analysis_type = analysis_type
                    st.session_state.current_to_use_models = to_use
                    st.session_state.calculation_triggered = True

                    models_tuning_1.clear()
    

if st.session_state.calculation_triggered:
    if st.session_state.current_analysis_type == 'Classification':
        cmap = st.selectbox("Confusion Matrix color map:", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                                                                'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                                                'bone', 'pink',
                                                                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                                                                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
    
    with right:
        estimators, Methods, predictions, y_test, yNames, *scaler = models_tuning_1(
                                                                    st.session_state.current_df_path,
                                                                    st.session_state.current_to_use_models,
                                                                    n_iter,
                                                                    pre_selected_num,
                                                                    st.session_state.current_analysis_type
                                                                    )

    with st.form("my_form", clear_on_submit=False, border=False):
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

        selected_models = [True]*sum(to_use)
        for i in range(len(estimators)):
            selected_models[i] = st.checkbox(
                            f'({Accs[i]*100:.2f}%) {Methods[i]}',
                            value=True,
                            key=f"model_selection_checkbox_{i}"
                        )
        save_button = st.form_submit_button("Save selected", type="primary")

    if st.session_state.current_analysis_type == 'Regression':
        call_plotter_reg(predictions, y_test, Methods, Accs, use_sig, use_round)
    elif st.session_state.current_analysis_type == 'Classification':
        call_plotter_class(Methods, Cs, yNames, Accs, cmap)    

    if save_button:
        to_save = (scaler, [(Methods[i], estimators[i]) for i in range(len(Methods)) if selected_models[i]])
        with open(saveto, 'wb') as f:
            pickle.dump(to_save, f)