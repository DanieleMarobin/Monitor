import streamlit as st

def initialize():
    if 'recalculate' not in st.session_state:
        st.session_state['recalculate'] = True

    if 'daily_inputs' not in st.session_state:
        st.session_state['daily_inputs'] = {}

    if 'dates' not in st.session_state:
        st.session_state['dates'] = {}

    if 'days' not in st.session_state:
        st.session_state['days'] = []

    if 'yields' not in st.session_state:
        st.session_state['yields'] = []

    if 'final_df' not in st.session_state:
        st.session_state['final_df'] = []   

    if 'model' not in st.session_state:
        st.session_state['model'] = []        