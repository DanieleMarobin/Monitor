# https://share.streamlit.io/danielemarobin/monitor/main/Home.py

import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc
import Models.USA_Corn_Yield as us_cy
import APIs.GDrive as gd

st.set_page_config(page_title="US CORN Yield Calculation and Results",layout="wide",initial_sidebar_state="expanded")

# st.markdown("Ok")
# files = gd.print_files()
# for item in files:
#     st.write(u'{0} ({1})'.format(item['name'], item['id']))

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = 0
if 'count' not in st.session_state:
    st.session_state['count'] = 0
	

col_model_text, col_calc_again = st.columns([3, 1])

with col_model_text:
    st.markdown("# Model Calculation and Results")

with col_calc_again:
    st.markdown("# ")
    calc_again = st.button('Calculate Again')

# A button to decrement the counter

if calc_again:
    st.session_state['count'] -= 1

# st.write('Count = ', st.session_state.count)



st.markdown("---")
st.sidebar.markdown("# Model Calculation and Results")

if (st.session_state['count']==0):
    st.session_state['prediction'] = us_cy.calculate_yield()
    st.session_state['count'] += 1

st.header('Prediction:')
st.subheader(st.session_state['prediction'])
    

