import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc
    
st.set_page_config(layout="wide")
st.markdown("# Weather Charts")
st.sidebar.markdown("# Weather Charts")

sel_df = uw.get_w_sel_df()

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

w_df_all = uw.build_w_df_all(sel_df,w_vars=[uw.WV_TEMP_MAX,uw.WV_PREC], in_files=uw.WS_UNIT_ALPHA, out_cols=uw.WS_UNIT_ALPHA)
all_charts = uc.Seas_Weather_Chart(w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = False, ref_year_start= dt(uw.CUR_YEAR,1,1))

for label, chart in all_charts.all_figs.items():
    st.plotly_chart(chart)