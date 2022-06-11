import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc
    
st.set_page_config(page_title="Weather Charts",layout="wide",initial_sidebar_state="expanded")
st.markdown("# Weather Charts")
st.sidebar.markdown("# Weather Charts")
st.markdown("---")

sel_df = uw.get_w_sel_df()
corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']

col_states, col_w_var = st.columns(2)

with col_states:
    sel_states = st.multiselect( 'States',corn_states,['IL'])

with col_w_var:
    w_vars = st.multiselect( 'Weather Variables',[uw.WV_PREC,uw.WV_TEMP_MAX,uw.WV_TEMP_MIN,uw.WV_TEMP_AVG],[uw.WV_TEMP_MAX])

sel_df=sel_df[
# (sel_df['state_alpha']=='IA')|
(sel_df['state_alpha']=='IL')
# (sel_df['state_alpha']=='IN')|
# (sel_df['state_alpha']=='OH')|
# (sel_df['state_alpha']=='MO')|
# (sel_df['state_alpha']=='MN')|
# (sel_df['state_alpha']=='SD')|
# (sel_df['state_alpha']=='NE')
]

if len(w_vars)>0:
    w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=uw.WS_UNIT_ALPHA, out_cols=uw.WS_UNIT_ALPHA)
    all_charts = uc.Seas_Weather_Chart(w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = False, ref_year_start= dt(uw.CUR_YEAR,1,1))

    for label, chart in all_charts.all_figs.items():
        st.markdown("#### "+label.replace('_',' '))
        st.plotly_chart(chart)
        st.markdown("---")
        st.markdown("#### ")