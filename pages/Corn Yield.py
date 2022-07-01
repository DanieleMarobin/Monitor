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
import plotly.express as px

pf='Corn_USA_Yield_'
su.initialize_Monitor_Corn_USA()
su.initialize_Monitor_Soybean_USA()
st.set_page_config(page_title="Corn Yield",layout="wide",initial_sidebar_state="expanded")

# Title, Declarations
season_start = dt(GV.CUR_YEAR,4,10)
season_end = dt(GV.CUR_YEAR,9,20)

st.markdown("## Corn Yield")
st.markdown("---")

progress_str_empty = st.empty()
progress_empty = st.empty()

s_WD = {GV.WD_HIST: 'Hist', GV.WD_H_GFS: 'GFS', GV.WD_H_ECMWF: 'ECMWF'} # Dictionary to translate into "Simple" words
# sel_WD=[GV.WD_HIST, GV.WD_H_GFS, GV.WD_H_ECMWF]
sel_WD=[GV.WD_HIST, GV.WD_H_GFS]
simple_weights = True

# ------------------------------ Accessory functions ------------------------------
def add_intervals(chart, intervals):
    sel_intervals = [intervals['planting_interval'], intervals['jul_aug_interval'], intervals['regular_interval'], intervals['pollination_interval']]
    text = ['Planting', 'Growing Prec', 'Growing Temp', 'Pollination']
    position=['top left','top left','bottom left','bottom left']
    color=['blue','green','orange','red']

    uc.add_interval_on_chart(chart,sel_intervals,GV.CUR_YEAR,text,position,color)


# ------------------ Sidebar: All the setting (So the main body comes after as it reacts to this) ------------------
st.sidebar.markdown("# Model Settings")

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
    st.session_state[pf+'download'] = True


# ****************************** CORE CALCULATION ***********************************
scope = cy.Define_Scope()
if update:
    os.system('cls')
    st.session_state[pf+'update'] = True
    st.session_state[pf+'download'] = True

# Re-Downloading (sy.Get_Data_All_Parallel(scope))
if st.session_state[pf+'download']:
    progress_str_empty.write('Downloading Data from USDA...'); progress_empty.progress(0.0)

    raw_data = cy.Get_Data_All_Parallel(scope)
    
    if simple_weights: uw.add_Sdd_all(raw_data['w_w_df_all']) # This is the one that switches from simple to elaborate SDD

    st.session_state[pf+'download'] = False
# Just Retrieve
else:
    raw_data = st.session_state[pf+'raw_data']
    if len(raw_data)==0:
        raw_data = cy.Get_Data_All_Parallel(scope)

# Re-Calculating
if st.session_state[pf+'update']:    
    print('------------- Updating the Model -------------'); print('')

    # I need to re-build it to catch the Units Change
    progress_str_empty.write('Building the Model...'); progress_empty.progress(0.2)
    milestones =cy.Milestone_from_Progress(raw_data)
    
    intervals = cy.Intervals_from_Milestones(milestones)

    train_DF_instr = um.Build_DF_Instructions('weighted',GV.WD_HIST, prec_units=prec_units, temp_units=temp_units)        
    train_df = cy.Build_DF(raw_data, milestones, intervals, train_DF_instr)

    model = um.Fit_Model(train_df,'Yield',GV.CUR_YEAR)

    yields = {}
    pred_df = {}

    progress_str_empty.write('Trend Yield Evolution...'); progress_empty.progress(0.4)

    # Trend Yield
    trend_DF_instr=um.Build_DF_Instructions('weighted', prec_units=prec_units, temp_units=temp_units)

    pred_df['Trend'] = cy.Build_Progressive_Pred_DF(raw_data, milestones, trend_DF_instr,GV.CUR_YEAR, season_start, season_end,trend_yield_case=True)
    yields['Trend'] = model.predict(pred_df['Trend'][model.params.index]).values
    pred_df['Trend']['Yield']=yields['Trend']       
    prog=0.7
    for WD in sel_WD:
        progress_str_empty.write(s_WD[WD] + ' Yield Evolution...'); progress_empty.progress(prog); prog=prog+0.15

        # Weather
        raw_data['w_df_all'] = uw.build_w_df_all(scope['geo_df'], scope['w_vars'], scope['geo_input_file'], scope['geo_output_column'])

        # Weighted Weather
        raw_data['w_w_df_all'] = uw.weighted_w_df_all(raw_data['w_df_all'], raw_data['weights'], output_column='USA')
        if simple_weights: uw.add_Sdd_all(raw_data['w_w_df_all']) # This is the one that switches from simple to elaborate SDD

        # Extention Modes
        ext_dict = {GV.WV_PREC:prec_ext_mode,  GV.WV_SDD_30:SDD_ext_mode}

        pred_DF_instr=um.Build_DF_Instructions('weighted',WD, prec_units=prec_units, temp_units=temp_units,ext_mode=ext_dict)

        # pred_df[WD] = cy.Build_Pred_DF(raw_data, milestones, pred_DF_instr,GV.CUR_YEAR, yield_analysis_start)
        # pred_df[WD] = cy.Build_Pred_DF(raw_data, milestones, pred_DF_instr,GV.CUR_YEAR, yield_analysis_start, date_end=dt(2022,9,30))

        pred_df[WD] = cy.Build_Progressive_Pred_DF(raw_data, milestones, pred_DF_instr,GV.CUR_YEAR, season_start, season_end)

        yields[WD] = model.predict(pred_df[WD][model.params.index]).values        
        pred_df[WD]['Yield']=yields[WD]

        # Storing Session States
    st.session_state[pf+'raw_data'] = raw_data  

    milestones = cy.Extend_Milestones(milestones, dt.today())
    intervals = cy.Intervals_from_Milestones(milestones)

    st.session_state[pf+'milestones'] = milestones
    st.session_state[pf+'intervals'] = intervals
    st.session_state[pf+'train_df'] = train_df   
    st.session_state[pf+'model'] = model    
    st.session_state[pf+'pred_df'] = pred_df
    st.session_state[pf+'yields_pred'] = yields

    st.session_state[pf+'update'] = False
# Just Retrieve
else:
    milestones=st.session_state[pf+'milestones']
    intervals=st.session_state[pf+'intervals']        
    train_df=st.session_state[pf+'train_df']
    model=st.session_state[pf+'model']
    pred_df=st.session_state[pf+'pred_df']
    yields=st.session_state[pf+'yields_pred']   


# ================================= Printing Results =================================
progress_empty.progress(1.0); progress_empty.empty(); progress_str_empty.empty()
metric_cols = st.columns(len(sel_WD)+5)
for i,WD in enumerate(sel_WD):
    metric_cols[i].metric(label='Yield - '+s_WD[WD], value="{:.2f}".format(yields[WD][-1]))
# metric_empty.metric(label='Yield', value="{:.2f}".format(yields[-1]), delta= "{:.2f}".format(yields[-1]-yields[-2])+" bu/Ac")

# _______________________________________ CHART _______________________________________
if True:
    last_HIST_day = raw_data['w_df_all'][GV.WD_HIST].last_valid_index()
    last_GFS_day = raw_data['w_df_all'][GV.WD_GFS].last_valid_index()
    last_ECMWF_day = raw_data['w_df_all'][GV.WD_ECMWF].last_valid_index()
    last_day = pred_df[sel_WD[0]].last_valid_index()

    # Trend Yield
    df = pred_df['Trend']
    final_yield = yields['Trend'][-1]
    label_trend = "Trend: "+"{:.2f}".format(final_yield)
    yield_chart=uc.line_chart(x=pd.to_datetime(df.index.values), y=df['Yield'],name=label_trend,color='black', mode='lines',height=750)
    font=dict(size=20,color="black")
    yield_chart.add_annotation(x=last_day, y=final_yield,text=label_trend,showarrow=False,arrowhead=1,font=font,yshift=+20)

    # Historical Weather
    df = pred_df[GV.WD_HIST]
    df=df[df.index<=last_HIST_day]
    uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines+markers', name='Realized Weather', color='black', showlegend=True, legendrank=3)


    # Forecasts Historical
    df = pred_df[GV.WD_HIST]
    df=df[df.index>last_HIST_day]
    final_yield = yields[GV.WD_HIST][-1]
    label_hist = "Hist: "+"{:.2f}".format(final_yield)
    uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines+markers', name=label_hist, color='orange', marker_size=5, line_width=1.0, showlegend=True, legendrank=1)

    # Forecasts GFS
    df = pred_df[GV.WD_H_GFS]
    df=df[df.index>last_HIST_day]
    final_yield = yields[GV.WD_H_GFS][-1]
    label_gfs = "GFS: "+"{:.2f}".format(final_yield)
    uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines+markers', name=label_gfs, color='blue', marker_size=5, line_width=1.0, showlegend=True, legendrank=1)

    # Forecasts ECMWF
    if (GV.WD_H_ECMWF in sel_WD):
        df = pred_df[GV.WD_H_ECMWF]
        df=df[df.index>last_HIST_day]
        final_yield = yields[GV.WD_H_ECMWF][-1]
        label_ecmwf = "ECMWF: "+"{:.2f}".format(final_yield)
        uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines+markers', name=label_ecmwf, color='green', marker_size=5, line_width=1.0, showlegend=True, legendrank=2)

    # Projection Historical
    df = pred_df[GV.WD_HIST]
    df=df[df.index>last_HIST_day]
    uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines', name='Projections', color='red', line_width=2.0, showlegend=True, legendrank=4)
    final_yield = yields[GV.WD_HIST][-1]
    font=dict(size=20,color="red")
    yield_chart.add_annotation(x=last_day, y=final_yield,text=label_hist,showarrow=False,arrowhead=1,font=font,yshift=+15)

    # If GFS has higher yield, shift its label up and ecmwf down
    if ((GV.WD_H_ECMWF in sel_WD) and (yields[GV.WD_H_GFS][-1] > yields[GV.WD_H_ECMWF][-1])):
        gfs_y_shift=10
        ecmwf_y_shift=-20
    else:
        gfs_y_shift=-20
        ecmwf_y_shift=10

    # Projection GFS
    df = pred_df[GV.WD_H_GFS]
    df=df[df.index>last_GFS_day]
    uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines', name='Projections', color='red', line_width=2.0, showlegend=False, legendrank=4)
    final_yield = yields[GV.WD_H_GFS][-1]
    font=dict(size=20,color="blue")
    yield_chart.add_annotation(x=last_day, y=final_yield,text=label_gfs,showarrow=False,arrowhead=1,font=font,yshift=gfs_y_shift)

    # Projection ECMWF
    if (GV.WD_H_ECMWF in sel_WD):
        df = pred_df[GV.WD_H_ECMWF]
        df=df[df.index>last_ECMWF_day]
        uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines', name='Projection', color='red', line_width=2.0, showlegend=False, legendrank=4)
        final_yield = yields[GV.WD_H_ECMWF][-1]
        font=dict(size=20,color="green")
        yield_chart.add_annotation(x=last_day, y=final_yield,text=label_ecmwf,showarrow=False,arrowhead=1,font=font,yshift=ecmwf_y_shift)

    add_intervals(yield_chart, intervals)

    st.plotly_chart(yield_chart)

    st.markdown('---')

# -------------------------------------------- Model Details --------------------------------------------
# Yield impact decomposition
if False:
    st.dataframe(pred_df['Trend'].loc[season_end:season_end])
    input_arr = pred_df['Trend'].loc[season_end:season_end][model.params.index].values
    coeff_arr = model.params.values
    st.write('Input', input_arr)
    st.write('Coeff', coeff_arr)
    st.write('Yield Contribution', input_arr*coeff_arr)


# Coefficients
if True:
    st_model_coeff=pd.DataFrame(columns=model.params.index)
    st_model_coeff.loc[len(st_model_coeff)]=model.params.values
    st_model_coeff.index=['Model Coefficients']

    st.markdown('##### Coefficients')
    st.dataframe(st_model_coeff)
    st.markdown('---')

# Prediction DataSets
if True:
    st.markdown('##### Trend DataSet')
    st.dataframe(pred_df['Trend'].drop(columns=['const']))

    for WD in sel_WD:
        st.markdown('##### Prediction DataSet - ' + s_WD[WD])
        st.dataframe(st.session_state[pf+'pred_df'][WD].drop(columns=['const']))


# Training DataSet
if True:
    st_train_df = deepcopy(train_df)
    st.markdown('##### Training DataSet')
    st.dataframe(st_train_df.sort_index(ascending=True).loc[st_train_df['Trend']<GV.CUR_YEAR])
    st.markdown("---")

# -------------------------------------------- Milestones --------------------------------------------
if True:
    dates_fmt = "%d %b %Y"
    milestones_col, _ , intervals_col = st.columns([2,   0.5,   6])
    with milestones_col:
        st.markdown('##### Milestones')

    with intervals_col:
        st.markdown('##### Weather Windows')

    col_80_planted, col_50_silked ,_, i_1, i_2, i_3, i_4 = st.columns([1,1,   0.5,   1.5,1.5,1.5,1.5])

    # 80% Planted
    with col_80_planted:
        st.markdown('##### 80% Planted')  
        st.write('Self-explanatory')  
        styler = st.session_state[pf+'milestones']['80_pct_planted'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
        st.write(styler)

    # 50% Silking
    with col_50_silked:
        st.markdown('##### 50% Silking')
        st.write('Self-explanatory')  
        styler = st.session_state[pf+'milestones']['50_pct_silked'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
        st.write(styler)    

# -------------------------------------------- Intervals --------------------------------------------
if True:
    # Planting_Prec
    with i_1:
        st.markdown('##### Planting Prec')
        st.write('80% planted -40 and +25 days')
        styler = st.session_state[pf+'intervals']['planting_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
        st.write(styler)

    # Jul_Aug_Prec
    with i_2:
        st.markdown('##### Jul-Aug Prec')
        st.write('80% planted +26 and +105 days')
        styler = st.session_state[pf+'intervals']['jul_aug_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
        st.write(styler)

    # Pollination_SDD
    with i_3:
        # 50% Silking -15 and +15 days
        st.markdown('##### Pollination SDD')
        st.write('50% Silking -15 and +15 days')
        styler = st.session_state[pf+'intervals']['pollination_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
        st.write(styler)

    # Regular_SDD
    with i_4:
        st.markdown('##### Regular SDD')
        st.write('20 Jun - 15 Sep')
        styler = st.session_state[pf+'intervals']['regular_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
        st.write(styler)
        
    st.markdown("---")

# Stat Model Summary
if True:
    st.subheader('Model Summary:')
    st.write(model.summary())
    st.markdown("---")

# Analog Scenarios results
if False:
    st.subheader('Analog Scenarios Matrix:')

    heat_map_df =pd.read_csv('Analog_Scenarios.csv')
    heat_map_df = heat_map_df.pivot_table(index=['Precipitation'], columns=['Max Temperature'], values=['Yield'], aggfunc='mean')
    heat_map_df.columns = heat_map_df.columns.droplevel(level=0)
    fig = px.imshow(heat_map_df,color_continuous_scale='RdBu')
    fig.update_layout(width=1400,height=787)
    st.plotly_chart(fig)

    st.markdown("---")

# Correlation Matrix
if True:
    st.subheader('Correlation Matrix:')
    st.plotly_chart(um.chart_corr_matrix(train_df.drop(columns=['Yield','const'])))



