import streamlit as st

def initialize():
    if 'count' not in st.session_state:
        st.session_state['count'] = 0
    if 'dates' not in st.session_state:
        st.session_state['dates'] = {}
