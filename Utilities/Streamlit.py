import streamlit as st
import Models.Corn_USA_Yield as cy
import Utilities.Modeling as um
import Utilities.GLOBAL as GV

def initialize_Monitor_Corn_USA():
    if 'Corn_USA_Yield_download' not in st.session_state:
        st.session_state['Corn_USA_Yield_download'] = True
    
    if 'Corn_USA_Yield_update' not in st.session_state:
        st.session_state['Corn_USA_Yield_update'] = True

    if 'Corn_USA_Yield_raw_data' not in st.session_state:
        st.session_state['Corn_USA_Yield_raw_data'] = {}

    if 'Corn_USA_Yield_milestones' not in st.session_state:
        st.session_state['Corn_USA_Yield_milestones'] = {}

    if 'Corn_USA_Yield_intervals' not in st.session_state:
        st.session_state['Corn_USA_Yield_intervals'] = {}        

    if 'Corn_USA_Yield_train_df' not in st.session_state:
        st.session_state['Corn_USA_Yield_train_df'] = []   

    if 'Corn_USA_Yield_model' not in st.session_state:
        st.session_state['Corn_USA_Yield_model'] = []      

    if 'Corn_USA_Yield_pred_df' not in st.session_state:
        st.session_state['Corn_USA_Yield_pred_df'] = {}

    if 'Corn_USA_Yield_yields_pred' not in st.session_state:
        st.session_state['Corn_USA_Yield_yields_pred'] = {}


def initialize_Monitor_Soybean_USA():
    if 'Soybean_USA_Yield_download' not in st.session_state:
        st.session_state['Soybean_USA_Yield_download'] = True
    
    if 'Soybean_USA_Yield_update' not in st.session_state:
        st.session_state['Soybean_USA_Yield_update'] = True

    if 'Soybean_USA_Yield_raw_data' not in st.session_state:
        st.session_state['Soybean_USA_Yield_raw_data'] = {}

    if 'Soybean_USA_Yield_milestones' not in st.session_state:
        st.session_state['Soybean_USA_Yield_milestones'] = {}

    if 'Soybean_USA_Yield_intervals' not in st.session_state:
        st.session_state['Soybean_USA_Yield_intervals'] = {}        

    if 'Soybean_USA_Yield_train_df' not in st.session_state:
        st.session_state['Soybean_USA_Yield_train_df'] = []   

    if 'Soybean_USA_Yield_model' not in st.session_state:
        st.session_state['Soybean_USA_Yield_model'] = []      

    if 'Soybean_USA_Yield_pred_df' not in st.session_state:
        st.session_state['Soybean_USA_Yield_pred_df'] = {}

    if 'Soybean_USA_Yield_yields_pred' not in st.session_state:
        st.session_state['Soybean_USA_Yield_yields_pred'] = {}
