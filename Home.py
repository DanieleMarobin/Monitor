# https://share.streamlit.io/danielemarobin/monitor/main/Home.py

import streamlit as st



st.set_page_config(page_title="Home",layout="wide",initial_sidebar_state="expanded")


if 'prediction' not in st.session_state:
    st.session_state['prediction'] = 0
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'dates' not in st.session_state:
    st.session_state['dates'] = {}


st.sidebar.markdown("# Home")

st.markdown("# Home")
st.markdown("---")

