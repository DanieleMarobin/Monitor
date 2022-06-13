import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.SnD as us

st.set_page_config(page_title="Weather Charts",layout="wide",initial_sidebar_state="expanded")
st.markdown("# Weather Charts")
st.sidebar.markdown("# Weather Charts")
st.markdown("---")

sel_df = uw.get_w_sel_df()
corn_states_options=['USA', 'IA','IL','IN','OH','MO','MN','SD','NE']

col_states, col_w_var = st.columns(2)

with col_states:
    sel_states = st.multiselect( 'States',corn_states_options,['IL'])

with col_w_var:
    w_vars = st.multiselect( 'Weather Variables',[uw.WV_PREC,uw.WV_TEMP_MAX,uw.WV_TEMP_MIN,uw.WV_TEMP_AVG],[uw.WV_TEMP_MAX])



# Full USA ---------------------------------------------------------------------------------------------------------
all_charts_usa={}
if ('USA' in sel_states):
    corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
    years=range(1985,2023)

    if 'weights' not in st.session_state:
        st.session_state['weights'] = us.get_USA_prod_weights('CORN', 'STATE', years, corn_states)
    weights = st.session_state['weights']

    sel_df=sel_df[sel_df['state_alpha'].isin(corn_states)]

    if len(sel_df)>0 and len(w_vars)>0:
        w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=uw.WS_UNIT_ALPHA, out_cols=uw.WS_UNIT_ALPHA)

        # Calculate Weighted DF
        w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

        all_charts_usa = uc.Seas_Weather_Chart(w_w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = False, ref_year_start= dt(uw.CUR_YEAR,1,1))

    for label, chart in all_charts_usa.all_figs.items():
        st.markdown("#### "+label.replace('_',' '))
        st.plotly_chart(chart)
        st.markdown("---")
        st.markdown("#### ")


# Single States ---------------------------------------------------------------------------------------------------------
sel_df=sel_df[sel_df['state_alpha'].isin(sel_states)]
all_charts_states={}
if len(sel_df)>0 and len(w_vars)>0:
    w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=uw.WS_UNIT_ALPHA, out_cols=uw.WS_UNIT_ALPHA)
    all_charts_states = uc.Seas_Weather_Chart(w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = False, ref_year_start= dt(uw.CUR_YEAR,1,1))

    for label, chart in all_charts_states.all_figs.items():
        st.markdown("#### "+label.replace('_',' '))
        st.plotly_chart(chart)
        st.markdown("---")
        st.markdown("#### ")    

# All Charts ---------------------------------------------------------------------------------------------------------



# for label, chart in all_charts.all_figs.items():
#     st.markdown("#### "+label.replace('_',' '))
#     st.plotly_chart(chart)
#     st.markdown("---")
#     st.markdown("#### ")