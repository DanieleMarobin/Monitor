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

def initialize_Monitor_USA_Yield(pf):
    # pf stands for "prefix"
    if pf not in st.session_state:
        st.session_state[pf] = {}
        st.session_state[pf]['full_analysis'] = False

        st.session_state[pf]['simple_weights'] = False
        st.session_state[pf]['raw_data'] = {}
        st.session_state[pf]['milestones'] = {}
        st.session_state[pf]['intervals'] = {}        

        st.session_state[pf]['train_df'] = []   
        st.session_state[pf]['model'] = []      

        st.session_state[pf]['pred_df'] = {}

        st.session_state[pf]['yields_pred'] = {}

def USA_Yield_Model_Template_old(id:dict):  
    # Preliminaries
    if True:
        os.system('cls')

        su.initialize_Monitor_USA_Yield(id['prefix'])
        st.set_page_config(page_title=id['title_str'],layout="wide",initial_sidebar_state="expanded")
        
        st.markdown("## "+id['title_str'])
        st.markdown("---")

        progress_str_empty = st.empty()
        progress_empty = st.empty()

        s_WD = {GV.WD_HIST: 'Hist', GV.WD_H_GFS: 'GFS', GV.WD_H_ECMWF: 'ECMWF'} # Dictionary to translate into "Simple" words

    # *************** Sidebar (Model User-Selected Settings) *******************
    if True:
        st.sidebar.markdown("# Model Settings")

        st.session_state[id['prefix']]['full_analysis']=st.sidebar.checkbox('Full Analysis', value=st.session_state[id['prefix']]['full_analysis'])
        st.session_state[id['prefix']]['simple_weights']=st.sidebar.checkbox('Simple Weights', value=st.session_state[id['prefix']]['simple_weights'])
        
        prec_col, temp_col = st.sidebar.columns(2)

        with prec_col:
            st.markdown('### Precipitation')
            prec_units = st.radio("Units",('mm','in'),1)
            prec_ext_mode = GV.EXT_MEAN

        with temp_col:
            st.markdown('### Temperature')
            temp_units = st.radio("Units",('C','F'),1)
            SDD_ext_mode = GV.EXT_MEAN

        ext_dict = {GV.WV_PREC:prec_ext_mode,  GV.WV_SDD_30:SDD_ext_mode}
        st.sidebar.markdown('---')
        c1,update_col,c3 = st.sidebar.columns(3)
        with update_col:
            update = st.button('Update')

    # **************************** Calculation *********************************
    # Scope
    if True:
        scope = id['func_Scope']()
       
    # Download Data
    if True:
        # Download Data
        progress_str_empty.write('Downloading Data from USDA...'); progress_empty.progress(0.0)

        raw_data = id['func_Raw_Data'](scope)
        
        if st.session_state[id['prefix']]['simple_weights']: uw.add_Sdd_all(raw_data['w_w_df_all']) # This is the one that switches from simple to elaborate SDD

        st.session_state[id['prefix']]['download'] = False

    # Calculation
    if True:
        # Re-Calculating
        print('------------- Updating the Model -------------'); print('')

        # I need to re-build it to catch the Units Change
        progress_str_empty.write('Building the Model...'); progress_empty.progress(0.2)
        milestones =id['func_Milestones'](raw_data)
        
        intervals = id['func_Intervals'](milestones)

        train_DF_instr = um.Build_DF_Instructions('weighted',GV.WD_HIST, prec_units=prec_units, temp_units=temp_units)        
        train_df = id['func_Build_DF'](raw_data, milestones, intervals, train_DF_instr)

        model = um.Fit_Model(train_df,'Yield',GV.CUR_YEAR)

        yields = {}
        pred_df = {}

        progress_str_empty.write('Trend Yield Evolution...'); progress_empty.progress(0.4)

        # for the full analysis, it is needed to start at the beginning of the season and finish at the end. But for the final yield I can just calculate the final point
        if st.session_state[id['prefix']]['full_analysis']:
            analysis_start = id['season_start']
            analysis_end = id['season_end']
        else:
            analysis_start = id['season_end']
            analysis_end = id['season_end']

        # Trend Yield
        trend_DF_instr=um.Build_DF_Instructions(WD_All='weighted', WD=GV.WD_HIST, prec_units=prec_units, temp_units=temp_units)
        pred_df['Trend'] = id['func_Progressive_Pred_DF'](raw_data, milestones, trend_DF_instr,GV.CUR_YEAR, analysis_start, analysis_end, trend_yield_case=True)
        yields['Trend'] = model.predict(pred_df['Trend'][model.params.index]).values
        pred_df['Trend']['Yield']=yields['Trend']

        st_prog=0.7
        for WD in id['sel_WD']:
            progress_str_empty.write(s_WD[WD] + ' Yield Evolution...'); progress_empty.progress(st_prog); st_prog=st_prog+0.15

            # Weather
            raw_data['w_df_all'] = uw.build_w_df_all(scope['geo_df'], scope['w_vars'], scope['geo_input_file'], scope['geo_output_column'])

            # Weighted Weather
            raw_data['w_w_df_all'] = uw.weighted_w_df_all(raw_data['w_df_all'], raw_data['weights'], output_column='USA')
            if st.session_state[id['prefix']]['simple_weights']: uw.add_Sdd_all(raw_data['w_w_df_all']) # This is the one that switches from simple to elaborate SDD

            # Instructions to build the prediction DataFrame
            pred_DF_instr=um.Build_DF_Instructions('weighted', WD=WD, prec_units=prec_units, temp_units=temp_units, ext_mode=ext_dict)

            pred_df[WD] = id['func_Progressive_Pred_DF'](raw_data, milestones, pred_DF_instr,GV.CUR_YEAR, analysis_start, analysis_end)
            yields[WD] = model.predict(pred_df[WD][model.params.index]).values        
            pred_df[WD]['Yield']=yields[WD]

        # Storing Session States
        st.session_state[id['prefix']]['raw_data'] = raw_data  

        milestones = id['func_Extend_Milestones'](milestones, dt.today())
        intervals = id['func_Intervals'](milestones)

        st.session_state[id['prefix']]['milestones'] = milestones
        st.session_state[id['prefix']]['intervals'] = intervals
        st.session_state[id['prefix']]['train_df'] = train_df   
        st.session_state[id['prefix']]['model'] = model    
        st.session_state[id['prefix']]['pred_df'] = pred_df
        st.session_state[id['prefix']]['yields_pred'] = yields

        st.session_state[id['prefix']]['update'] = False

    # ****************************** Results ***********************************
    # Metric
    if True:
        progress_empty.progress(1.0); progress_empty.empty(); progress_str_empty.empty()
        metric_cols = st.columns(len(id['sel_WD'])+5)
        for i,WD in enumerate(id['sel_WD']):
            metric_cols[i].metric(label='Yield - '+s_WD[WD], value="{:.2f}".format(yields[WD][-1]))

    # Chart
    if st.session_state[id['prefix']]['full_analysis']:
        last_HIST_day = raw_data['w_df_all'][GV.WD_HIST].last_valid_index()
        last_GFS_day = raw_data['w_df_all'][GV.WD_GFS].last_valid_index()
        last_ECMWF_day = raw_data['w_df_all'][GV.WD_ECMWF].last_valid_index()
        last_day = pred_df[id['sel_WD'][0]].last_valid_index()

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
        if (GV.WD_H_ECMWF in id['sel_WD']):
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
        if ((GV.WD_H_ECMWF in id['sel_WD']) and (yields[GV.WD_H_GFS][-1] > yields[GV.WD_H_ECMWF][-1])):
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
        if (GV.WD_H_ECMWF in id['sel_WD']):
            df = pred_df[GV.WD_H_ECMWF]
            df=df[df.index>last_ECMWF_day]
            uc.add_series(yield_chart, x=pd.to_datetime(df.index.values), y=df['Yield'], mode='lines', name='Projection', color='red', line_width=2.0, showlegend=False, legendrank=4)
            final_yield = yields[GV.WD_H_ECMWF][-1]
            font=dict(size=20,color="green")
            yield_chart.add_annotation(x=last_day, y=final_yield,text=label_ecmwf,showarrow=False,arrowhead=1,font=font,yshift=ecmwf_y_shift)

        id['func_add_chart_intervals'](yield_chart, intervals)

        st.plotly_chart(yield_chart)

        st.markdown('---')

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

        for WD in id['sel_WD']:
            st.markdown('##### Prediction DataSet - ' + s_WD[WD])
            st.dataframe(st.session_state[id['prefix']]['pred_df'][WD].drop(columns=['const']))

    # Training DataSet
    if True:
        st_train_df = deepcopy(train_df)
        st.markdown('##### Training DataSet')
        st.dataframe(st_train_df.sort_index(ascending=True).loc[st_train_df['Trend']<GV.CUR_YEAR])
        st.markdown("---")

    # Milestones & Intervals
    if True:
        id['func_st_milestones_and_intervals'](id)

    # Stat Model Summary
    if True:
        st.subheader('Model Summary:')
        st.write(model.summary())
        st.markdown("---")