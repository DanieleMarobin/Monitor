# https://share.streamlit.io/danielemarobin/monitor/main/Home.py
# streamlit run Home.py

import streamlit as st
import Utilities.Streamlit as su

su.initialize_Monitor_Corn_USA()
su.initialize_Monitor_Soybean_USA()
st.set_page_config(page_title="Home",layout="wide",initial_sidebar_state="expanded")

st.markdown("# Home")
st.markdown("---")

st.sidebar.markdown("# Home")



