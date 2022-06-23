import streamlit as st
from datetime import datetime as dt

import os
import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.SnD as us
import Utilities.GLOBAL as GV
import Utilities.Streamlit as su

su.initialize_Monitor_Corn_USA()

st.set_page_config(page_title="Weather Charts",layout="wide",initial_sidebar_state="expanded")


su.initialize_Monitor_Corn_USA()
sel_df = uw.get_w_sel_df()
corn_states_options=['USA', 'IA','IL','IN','OH','MO','MN','SD','NE']

def add_intervals(label,chart,intervals):
    if 'Temp' in label:
        sel_intervals = [intervals['regular_interval'], intervals['pollination_interval']]
        text = ['SDD', 'Pollination']
        position=['bottom left','bottom left']
        color=['red','red']

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


with st.sidebar:
    st.markdown("# Weather Charts")
    sel_states = st.multiselect( 'States',corn_states_options,['USA'])
    w_vars = st.multiselect( 'Weather Variables',[GV.WV_PREC,GV.WV_TEMP_MAX,GV.WV_TEMP_MIN,GV.WV_TEMP_AVG, GV.WV_SDD_30],[GV.WV_TEMP_MAX])
    slider_year_start = st.date_input("Seasonals Start", dt(2022, 1, 1))
    cumulative = st.checkbox('Cumulative')
    ext_mode = st.radio("Projection",(GV.EXT_MEAN, GV.EXT_ANALOG))


ref_year_start = dt(GV.CUR_YEAR, slider_year_start.month, slider_year_start.day)
ext_dict = {w_v : ext_mode for w_v in w_vars}

# Full USA ---------------------------------------------------------------------------------------------------------
os.system('cls')
all_charts_usa={}
if ('USA' in sel_states):
    corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
    years=range(1985,2023)

    if 'weights' not in st.session_state:
        st.session_state['weights'] = us.get_USA_prod_weights('CORN', 'STATE', years, corn_states)
    weights = st.session_state['weights']

    sel_df=sel_df[sel_df['state_alpha'].isin(corn_states)]

    if len(sel_df)>0 and len(w_vars)>0:
        w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)

        # Calculate Weighted DF
        w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

        all_charts_usa = uc.Seas_Weather_Chart(w_w_df_all, ext_mode=ext_dict, cumulative = cumulative, ref_year_start= ref_year_start)

        for label, chart in all_charts_usa.all_figs.items():
            add_intervals(label,chart,st.session_state['intervals'])
            st.markdown("#### "+label.replace('_',' '))
            st.plotly_chart(chart)
            # st.markdown("---")
            st.markdown("#### ")


# Single States ---------------------------------------------------------------------------------------------------------
sel_df=sel_df[sel_df['state_alpha'].isin(sel_states)]
all_charts_states={}
if len(sel_df)>0 and len(w_vars)>0:
    w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)
    all_charts_states = uc.Seas_Weather_Chart(w_df_all, ext_mode=ext_dict, cumulative = cumulative, ref_year_start= ref_year_start)

    for label, chart in all_charts_states.all_figs.items():
        add_intervals(label,chart,st.session_state['intervals'])
        st.markdown("#### "+label.replace('_',' '))        
        st.plotly_chart(chart)
        # st.markdown("---")
        st.markdown("#### ")