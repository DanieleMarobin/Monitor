import streamlit as st
import pandas as pd

st.write("Table")

data=pd.read_csv('Data/dan.csv')

st.dataframe(data)