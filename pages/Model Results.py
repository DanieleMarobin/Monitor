from datetime import datetime as dt
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
calc_again = st.sidebar.button('Re-Calculate')
if calc_again:
    st.session_state['recalculate'] = True


if st.session_state['recalculate']:
    with st.spinner('Getting Data from USDA as fast as I can...'):
        scope = cy.Define_Scope()

        raw_data = cy.Get_Data_fast(scope)
        milestones =cy.Milestone_from_Progress(raw_data)
        intervals = cy.Intervals_from_Milestones(milestones)

        train_DF_instr = um.Build_DF_Instructions('weighted',GV.WD_HIST, prec_units='in', temp_units='F')
        train_df = cy.Build_Train_DF(raw_data, milestones, intervals, train_DF_instr)
        model = um.Fit_Model(train_df,'Yield',GV.CUR_YEAR)

    with st.spinner('Evaulating Yield Evolution...'):
        pred_DF_instr=um.Build_DF_Instructions('weighted',GV.WD_H_GFS, prec_units='in', temp_units='F')
        pred_df = cy.Build_Pred_DF(raw_data,milestones,pred_DF_instr,GV.CUR_YEAR, dt(2022,6,10))
        yields = model.predict(pred_df[model.params.index]).values

        st.session_state['recalculate'] = False
        st.session_state['raw_data'] = raw_data   
        st.session_state['milestones'] = milestones
        st.session_state['intervals'] = intervals        
        st.session_state['train_df'] = train_df   
        st.session_state['model'] = model     
        st.session_state['pred_df'] = pred_df
        st.session_state['days_pred'] = pred_df.index.values
        st.session_state['yields_pred'] = yields       

else:
    print('Hello')


# declarations
corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
years=range(1985,2023)

days=[]
yields=[]

   
# Assign the saved values to the variables

df = st.session_state['train_df']    
days = st.session_state['days_pred']    
yields = st.session_state['yields_pred']    
stats_model = st.session_state['model']

# metric_empty.metric(label='Yield', value="{:.2f}".format(yields[-1]), delta= "{:.2f}".format(yields[-1]-yields[-2])+" bu/Ac")  
metric_empty.metric(label='Yield', value="{:.2f}".format(yields[-1]))
chart_empty.plotly_chart(uc.line_chart(x=days,y=yields))    
line_empty.markdown('---')        
daily_input_empty.markdown('##### Daily Inputs')    
dataframe_empty.dataframe(st.session_state['pred_df'])    




# -------------------------------------------- Model Details --------------------------------------------
# coefficients
model_coeff=pd.DataFrame(columns=stats_model.params.index)
model_coeff.loc[len(model_coeff)]=stats_model.params.values
# model_coeff=model_coeff.drop(columns=['const'])
model_coeff.index=['Model Coefficients']

st.markdown('##### Coefficients')
st.dataframe(model_coeff)
  

# final DataFrame
st.markdown('---')
st.markdown('### Final DataFrame')
st.dataframe(df.sort_index(ascending=False))

# summary
st.markdown("---")
st.subheader('Model Summary:')
st.write(stats_model.summary())

# Correlation Matrix
st.markdown("---")
st.subheader('Correlation Matrix:')
st.plotly_chart(um.chart_corr_matrix(df.drop(columns=['Yield'])))
