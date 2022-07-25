import streamlit as st
from datetime import datetime as dt

import os
import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.SnD as us
import Utilities.GLOBAL as GV
import Utilities.Streamlit as su

def add_intervals(label,chart,intervals,cumulative):
    if 'Temp' in label:
        sel_intervals = [intervals['regular_interval'], intervals['pollination_interval']]
        text = ['SDD', 'Pollination']
        position=['bottom left','bottom left']
        color=['red','red']
        if not(cumulative):
            chart.add_hline(y=30,line_color='red')

    elif 'Sdd' in label:
        sel_intervals = [intervals['regular_interval'], intervals['pollination_interval']]
        text = ['SDD', 'Pollination']
        position=['top left','top left']
        color=['red','red']

    else:
        sel_intervals = [intervals['planting_interval'], intervals['jul_aug_interval']]
        text = ['Planting', 'Jul-Aug']
        position=['top left','top left']
        color=['blue','blue']

    uc.add_interval_on_chart(chart,sel_intervals,GV.CUR_YEAR,text,position,color)

# Initial Settings
if True:
    st.set_page_config(page_title='Weather Charts',layout='wide',initial_sidebar_state='expanded')
    sel_df = uw.get_w_sel_df()
    states_options=['USA', 'IA','IL','IN','OH','MO','MN','SD','NE']

with st.sidebar:
    st.markdown('# Weather Charts')
    crop=st.radio('Crop',('Corn','Soybean')); pf=crop+'_USA_Yield'
    simple_weights=st.sidebar.checkbox('Simple Weights', value=False)

    sel_states = st.multiselect( 'States',states_options,['USA'])    
    w_vars = st.multiselect( 'Weather Variables',[GV.WV_PREC,GV.WV_TEMP_MAX,GV.WV_TEMP_MIN,GV.WV_TEMP_AVG, GV.WV_SDD_30],[GV.WV_TEMP_MAX])
    slider_year_start = st.date_input('Seasonals Start', dt(2022, 1, 1))
    cumulative = st.checkbox('Cumulative')
    hovermode = st.selectbox('Hovermode',['x', 'y', 'closest', 'x unified', 'y unified'],index=0)
    # ext_mode = st.radio('Projection',[GV.EXT_MEAN, GV.EXT_ANALOG])
    ext_mode = GV.EXT_MEAN

    # the below 2 instructions need to stay here because they are a consequnce of the setting in the 'st.sidebar'
    ref_year_start = dt(GV.CUR_YEAR, slider_year_start.month, slider_year_start.day)
    ext_dict = {w_v : ext_mode for w_v in w_vars}

# Full USA
if True:
    os.system('cls')
    all_charts_usa={}
    if ('USA' in sel_states):
        download_states=['IA','IL','IN','OH','MO','MN','SD','NE']
        years=range(1985,2023)

        if ((pf in st.session_state) and ('raw_data' in st.session_state[pf]) and ('weights' in st.session_state[pf]['raw_data'])):
            weights = st.session_state[pf]['weights']
        else:
            if (crop=='Corn'):
                commodity='CORN'
            elif (crop=='Soybean'):
                commodity='SOYBEANS'
            
            weights= us.get_USA_prod_weights(commodity, 'STATE', years, download_states)        
        
        sel_df=sel_df[sel_df['state_alpha'].isin(download_states)]

        if len(sel_df)>0 and len(w_vars)>0:
            w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)
            
            # Calculate Weighted DF
            w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')
            if simple_weights: uw.add_Sdd_all(w_w_df_all, threshold=30)
            
            all_charts_usa = uc.Seas_Weather_Chart(w_w_df_all, ext_mode=ext_dict, cumulative = cumulative, ref_year_start= ref_year_start, hovermode=hovermode)

            for label, chart in all_charts_usa.all_figs.items():
                if (pf in st.session_state):
                    add_intervals(label,chart,st.session_state[pf]['intervals'],cumulative)

                st.markdown('#### '+label.replace('_',' '))
                st.plotly_chart(chart)
                st.markdown('#### ')

# Single States
if True:
    sel_df=sel_df[sel_df['state_alpha'].isin(sel_states)]
    all_charts_states={}
    if len(sel_df)>0 and len(w_vars)>0:
        w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)
        all_charts_states = uc.Seas_Weather_Chart(w_df_all, ext_mode=ext_dict, cumulative = cumulative, ref_year_start= ref_year_start, hovermode=hovermode)

        for label, chart in all_charts_states.all_figs.items():
            if (pf in st.session_state):
                add_intervals(label,chart,st.session_state[pf]['intervals'],cumulative)
                
            st.markdown('#### '+label.replace('_',' '))        
            st.plotly_chart(chart)
            st.markdown('#### ')