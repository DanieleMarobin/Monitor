import streamlit as st
import Models.Corn_USA_Yield as cy
import Utilities.Modeling as um
import Utilities.GLOBAL as GV

def initialize_Monitor_Corn_USA():
    if 'recalculate' not in st.session_state:
        st.session_state['recalculate'] = True

    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = {}

    if 'milestones' not in st.session_state:
        st.session_state['milestones'] = {}

    if 'intervals' not in st.session_state:
        st.session_state['intervals'] = {}        

    if 'train_df' not in st.session_state:
        st.session_state['train_df'] = []   

    if 'model' not in st.session_state:
        st.session_state['model'] = []      

    if 'pred_df' not in st.session_state:
        st.session_state['pred_df'] = {}

    if 'days_pred' not in st.session_state:
        st.session_state['days_pred'] = []

    if 'yields_pred' not in st.session_state:
        st.session_state['yields_pred'] = [] 
