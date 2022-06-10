import streamlit as st
import pandas as pd

st.write("Table")

data=pd.read_csv('data/dan.csv')

st.dataframe(data)