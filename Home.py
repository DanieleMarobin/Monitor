import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc

st.set_page_config(page_title="Model Results",layout="wide")
st.markdown("# Model Results")
st.sidebar.markdown("# Model Results")

sel_df = uw.get_w_sel_df()

sel_df=sel_df[
(sel_df['state_alpha']=='IA')|
(sel_df['state_alpha']=='IL')|
(sel_df['state_alpha']=='IN')|
(sel_df['state_alpha']=='OH')|
(sel_df['state_alpha']=='MO')|
(sel_df['state_alpha']=='MN')|
(sel_df['state_alpha']=='SD')|
(sel_df['state_alpha']=='NE')
]

st.dataframe(sel_df)