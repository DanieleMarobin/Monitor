# https://share.streamlit.io/danielemarobin/monitor/main/Home.py

import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc
import Models.USA_Corn_Yield as us_cy

st.set_page_config(page_title="US CORN Yield Calculation and Results",layout="wide",initial_sidebar_state="expanded")
st.markdown("# Model Calculation and Results")
st.markdown("---")
st.sidebar.markdown("# Model Calculation and Results")

us_cy.calculate_yield()