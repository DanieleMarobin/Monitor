from copy import deepcopy
from datetime import datetime as dt
import os
import pandas as pd
import streamlit as st

import Models.Corn_USA_Yield as cy

import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Charts as uc
import Utilities.Streamlit as su
import Utilities.GLOBAL as GV

su.initialize_Monitor_Corn_USA()
st.set_page_config(page_title="Model Results",layout="wide",initial_sidebar_state="expanded")

# Title, Declarations
st.markdown("## Model Results")
st.markdown("---")
s_WD = {GV.WD_H_GFS: 'GFS', GV.WD_H_ECMWF: 'ECMWF'} # Dictionary to translate into "Simple" words
sel_WD=[GV.WD_H_GFS, GV.WD_H_ECMWF]

# ------------------ Sidebar: All the setting (So the main body comes after as it reacts to this) ------------------
st.sidebar.markdown("# Model Calculation Settings")

yield_analysis_start = st.sidebar.date_input("Yield Analysis Start", dt.today()+pd.DateOffset(-1))
prec_col, temp_col = st.sidebar.columns(2)

with prec_col:
    st.markdown('### Precipitation')
    prec_units = st.radio("Units",('mm','in'))
    prec_ext_mode = st.radio("Projection ({0})".format(prec_units),(GV.EXT_MEAN, GV.EXT_ANALOG))
    prec_ext_analog=[]
    if prec_ext_mode==GV.EXT_ANALOG:
        prec_ext_analog = st.selectbox('Prec Analog Year', list(range(GV.CUR_YEAR-1,1984,-1)))
        prec_ext_mode=prec_ext_mode+'_'+str(prec_ext_analog)

with temp_col:
    st.markdown('### Temperature')
    temp_units = st.radio("Units",('C','F'))
    SDD_ext_mode = st.radio("Projection ({0})".format(temp_units),(GV.EXT_MEAN, GV.EXT_ANALOG))
    SDD_ext_analog=[]
    if SDD_ext_mode==GV.EXT_ANALOG:
        SDD_ext_analog = st.selectbox('SDD Analog Year', list(range(GV.CUR_YEAR-1,1984,-1)))
        SDD_ext_mode=SDD_ext_mode+'_'+str(SDD_ext_analog)

st.sidebar.markdown('---')
c1,update_col,c3 = st.sidebar.columns(3)
with update_col:
    update = st.button('Update')

scope = cy.Define_Scope()
if update:
    st.session_state['update'] = True

# Re-Downloading (cy.Get_Data_All_Parallel(scope))
if st.session_state['download']:
    with st.spinner('Downloading Data from USDA as fast as I can...'):
        raw_data = cy.Get_Data_All_Parallel(scope)
        st.session_state['download'] = False
# Just Retrieve
else:
    raw_data = st.session_state['raw_data']


# Re-Calculating
if st.session_state['update']:    
    os.system('cls')
    print('------------- Updating the Model -------------'); print('')

    # I need to re-build it to catch the Units Change
    with st.spinner('Building the Model...'):
        milestones =cy.Milestone_from_Progress(raw_data)
        intervals = cy.Intervals_from_Milestones(milestones)

        train_DF_instr = um.Build_DF_Instructions('weighted',GV.WD_HIST, prec_units=prec_units, temp_units=temp_units)        
        train_df = cy.Build_DF(raw_data, milestones, intervals, train_DF_instr)

        model = um.Fit_Model(train_df,'Yield',GV.CUR_YEAR)

    with st.spinner('Evaluating Yield Evolution...'):
        yields = {}
        pred_df = {}
        for WD in sel_WD:
            # Weather
            raw_data['w_df_all'] = uw.build_w_df_all(scope['geo_df'], scope['w_vars'], scope['geo_input_file'], scope['geo_output_column'])

            # Weighted Weather
            raw_data['w_w_df_all'] = uw.weighted_w_df_all(raw_data['w_df_all'], raw_data['weights'], output_column='USA')

            # Extention Modes
            ext_dict = {GV.WV_PREC:prec_ext_mode,  GV.WV_SDD_30:SDD_ext_mode}

            pred_DF_instr=um.Build_DF_Instructions('weighted',WD, prec_units=prec_units, temp_units=temp_units,ext_mode=ext_dict)

            pred_df[WD] = cy.Build_Pred_DF(raw_data, milestones, pred_DF_instr,GV.CUR_YEAR, yield_analysis_start)
            yields[WD] = model.predict(pred_df[WD][model.params.index]).values        
 
        st.session_state['raw_data'] = raw_data  

        st.session_state['milestones'] = milestones
        st.session_state['intervals'] = intervals        
        st.session_state['train_df'] = train_df   
        st.session_state['model'] = model    
        st.session_state['pred_df'] = pred_df
        st.session_state['yields_pred'] = yields

        st.session_state['update'] = False
# Just Retrieve
else:
    milestones=st.session_state['milestones']
    intervals=st.session_state['intervals']        
    train_df=st.session_state['train_df']
    model=st.session_state['model']
    pred_df=st.session_state['pred_df']
    yields=st.session_state['yields_pred']   


# ================================= Printing Results =================================
metric_cols = st.columns(len(sel_WD)+5)
for i,WD in enumerate(sel_WD):
    metric_cols[i].metric(label='Yield - '+s_WD[WD], value="{:.2f}".format(yields[WD][-1]))
# metric_empty.metric(label='Yield', value="{:.2f}".format(yields[-1]), delta= "{:.2f}".format(yields[-1]-yields[-2])+" bu/Ac")


days_dict={k:v.index.values for (k,v) in pred_df.items()}
st.plotly_chart(uc.line_chart(x_dict=days_dict, y_dict=yields))
st.markdown('---')
for WD in sel_WD:
    st.markdown('##### Prediction DataSet - ' + s_WD[WD])
    st.session_state['pred_df'][WD]['Yield']=yields[WD]
    st.dataframe(st.session_state['pred_df'][WD].drop(columns=['const']))


# -------------------------------------------- Model Details --------------------------------------------
# coefficients
st_model_coeff=pd.DataFrame(columns=model.params.index)
st_model_coeff.loc[len(st_model_coeff)]=model.params.values
st_model_coeff.index=['Model Coefficients']

st.markdown('##### Coefficients')
st.dataframe(st_model_coeff)
st.markdown('---')

# Training DataSet
st_train_df = deepcopy(train_df)
st.markdown('##### Training DataSet')
st.dataframe(st_train_df.sort_index(ascending=False).loc[st_train_df['Trend']<GV.CUR_YEAR])
st.markdown("---")

# -------------------------------------------- Milestones --------------------------------------------
dates_fmt = "%d %b %Y"
milestones_col, intervals_col = st.columns([1,2])
with milestones_col:
    st.markdown('##### Milestones')

col_80_planted, col_50_silked =  st.columns([1,1])

with intervals_col:
    st.markdown('##### Weather Windows')

col_plant_80, col_silk_50, i_1, i_2, i_3, i_4 = st.columns([1,1,1,1,1,1])

# 80% Planted
with col_plant_80:
    st.markdown('##### 80% Planted')  
    st.write('Self-explanatory')  
    styler = st.session_state['milestones']['80_pct_planted'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

# 50% Silking
with col_silk_50:
    st.markdown('##### 50% Silking')
    st.write('Self-explanatory')  
    styler = st.session_state['milestones']['50_pct_silked'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
    st.write(styler)    

# -------------------------------------------- Intervals --------------------------------------------
# Planting_Prec
with i_1:
    st.markdown('##### Planting Prec')
    st.write('80% planted -40 and +25 days')
    styler = st.session_state['intervals']['planting_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

# Jul_Aug_Prec
with i_2:
    st.markdown('##### Jul_Aug_Prec')    
    st.write('80% planted +26 and +105 days')
    styler = st.session_state['intervals']['jul_aug_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

# Regular_SDD
with i_3:
    # 50% Silking -15 and +15 days
    st.markdown('##### Pollination_SDD')
    st.write('50% Silking -15 and +15 days')
    styler = st.session_state['intervals']['pollination_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

# Pollination_SDD
with i_4:
    st.markdown('##### Regular_SDD')
    st.write('20 Jun - 15 Sep')
    styler = st.session_state['intervals']['regular_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)
    
st.markdown("---")

# -------------------------------------------- Summary --------------------------------------------
# Summary
st.subheader('Model Summary:')
st.write(model.summary())
st.markdown("---")

# Correlation Matrix
st.subheader('Correlation Matrix:')
st.plotly_chart(um.chart_corr_matrix(train_df.drop(columns=['Yield','const'])))
