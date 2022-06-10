import streamlit as st
import pandas as pd
import Utilities.Weather as uw

st.write("Table")

# data=pd.read_csv('Data/dan.csv')
data=uw.get_w_sel_df()

st.dataframe(data)