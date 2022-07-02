import streamlit as st
import Models.Corn_USA_Yield as cy
import Utilities.Modeling as um
import Utilities.GLOBAL as GV

def initialize_Monitor_USA_Yield(pf):
    # pf stands for "prefix"
    if pf not in st.session_state:
        st.session_state[pf] = {}
        st.session_state[pf]['download'] = True
        st.session_state[pf]['update'] = True

        st.session_state[pf]['raw_data'] = {}
        st.session_state[pf]['milestones'] = {}
        st.session_state[pf]['intervals'] = {}        

        st.session_state[pf]['train_df'] = []   
        st.session_state[pf]['model'] = []      

        st.session_state[pf]['pred_df'] = {}

        st.session_state[pf]['yields_pred'] = {}
