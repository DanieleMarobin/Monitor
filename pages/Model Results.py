from datetime import datetime as dt
from turtle import update
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st

import Models.Corn_USA_Yield as cy
import APIs.QuickStats as qs

import Utilities.SnD as us
import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Charts as uc
import Utilities.Streamlit as su
import Utilities.GLOBAL as GV

su.initialize_Monitor_Corn_USA()
st.set_page_config(page_title="Model Results",layout="wide",initial_sidebar_state="expanded")

# Title, Settings, Recalculate Button etc
st.markdown("# Model Results")

progress_empty = st.empty()

metric_empty = st.empty()
chart_empty = st.empty()
line_empty = st.empty()
daily_input_empty= st.empty()
dataframe_empty = st.empty()

# Sidebar
st.sidebar.markdown("# Model Calculation Settings")
yield_analysis_start = st.sidebar.date_input("Yield Analysis Start", dt.today()+pd.DateOffset(-1))
prec_col, temp_col = st.sidebar.columns(2)

with prec_col:
    prec_units = st.radio("Precipitation Units",('mm','in'))
with temp_col:
    temp_units = st.radio("Temperature Units",('C','F'))

ext_mode = st.sidebar.radio("Projection",(GV.EXT_MEAN, GV.EXT_SHIFT_MEAN,GV.EXT_ANALOG,GV.EXT_LIMIT))
st.sidebar.markdown('---')
c1,c2,c3 = st.sidebar.columns(3)
with c2:
    update = st.button('Update')

if update:
    st.session_state['update'] = True

if st.session_state['download']:
    with st.spinner('Downloading Data from USDA as fast as I can...'):
        scope = cy.Define_Scope()

        raw_data = cy.Get_Data_fast(scope)

        st.session_state['raw_data'] = raw_data   

        st.session_state['download'] = False
else:
    raw_data = st.session_state['raw_data']



if st.session_state['update']:    
    with st.spinner('Building the Model...'):
        milestones =cy.Milestone_from_Progress(raw_data)
        intervals = cy.Intervals_from_Milestones(milestones)

        train_DF_instr = um.Build_DF_Instructions('weighted',GV.WD_HIST, prec_units=prec_units, temp_units=temp_units)        
        train_df = cy.Build_Train_DF(raw_data, milestones, intervals, train_DF_instr)

        model = um.Fit_Model(train_df,'Yield',GV.CUR_YEAR)

    with st.spinner('Evaulating Yield Evolution...'):
        pred_DF_instr=um.Build_DF_Instructions('weighted',GV.WD_H_GFS, prec_units=prec_units, temp_units=temp_units,ext_mode=ext_mode)
        pred_df = cy.Build_Pred_DF(raw_data,milestones,pred_DF_instr,GV.CUR_YEAR, yield_analysis_start)
        yields = model.predict(pred_df[model.params.index]).values        
 
        st.session_state['milestones'] = milestones
        st.session_state['intervals'] = intervals        
        st.session_state['train_df'] = train_df   
        st.session_state['model'] = model    
        st.session_state['pred_df'] = pred_df
        st.session_state['days_pred'] = pred_df.index.values
        st.session_state['yields_pred'] = yields

        st.session_state['update'] = False
else:
    milestones=st.session_state['milestones']
    intervals=st.session_state['intervals']        
    train_df=st.session_state['train_df']
    model=st.session_state['model']
    pred_df=st.session_state['pred_df']
    yields=st.session_state['yields_pred']   
    
# metric_empty.metric(label='Yield', value="{:.2f}".format(yields[-1]), delta= "{:.2f}".format(yields[-1]-yields[-2])+" bu/Ac")  
metric_empty.metric(label='Yield', value="{:.2f}".format(yields[-1]))
chart_empty.plotly_chart(uc.line_chart(x=pred_df.index.values,y=yields))    
line_empty.markdown('---')        
daily_input_empty.markdown('##### Prediction DataSet')
st.session_state['pred_df']['Yield']=yields
dataframe_empty.dataframe(st.session_state['pred_df'].drop(columns=['const']))    


# -------------------------------------------- Model Details --------------------------------------------
# coefficients
model_coeff=pd.DataFrame(columns=model.params.index)
model_coeff.loc[len(model_coeff)]=model.params.values
# model_coeff=model_coeff.drop(columns=['const'])
model_coeff.index=['Model Coefficients']

st.markdown('##### Coefficients')
st.dataframe(model_coeff)

# Training DataSet
st.markdown('---')
st.markdown('### Training DataSet')
st.dataframe(train_df.sort_index(ascending=False).loc[train_df['Trend']<GV.CUR_YEAR])

# summary
st.markdown("---")
st.subheader('Model Summary:')
st.write(model.summary())

# Correlation Matrix
st.markdown("---")
st.subheader('Correlation Matrix:')
st.plotly_chart(um.chart_corr_matrix(train_df.drop(columns=['Yield','const'])))
