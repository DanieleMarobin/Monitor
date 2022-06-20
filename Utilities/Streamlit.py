import streamlit as st
import Models.Corn_USA_Yield as cy
import Utilities.Modeling as um
import Utilities.GLOBAL as GV

def initialize_Monitor_Corn_USA():
    if 'download' not in st.session_state:
        st.session_state['download'] = True
    
    if 'update' not in st.session_state:
        st.session_state['update'] = True

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

    if 'yields_pred' not in st.session_state:
        st.session_state['yields_pred'] = [] 
