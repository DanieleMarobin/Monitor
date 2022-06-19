import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.SnD as us
import Utilities.GLOBAL as GV
import Utilities.Streamlit as su

su.initialize_Monitor_Corn_USA()

def find_on_x_axis(date, chart):
    id = 100*date.month+date.day
    for x in chart.data[0]['x']:
        if 100*x.month + x.day==id:
            return x

def add_w_dates(label, chart):  
    if len(st.session_state['dates'])>0:
        seas_year = 2022
        if 'Temp' in label:
            sel_dates = [st.session_state['dates']['regular'], st.session_state['dates']['pollination']]
            sel_text = ['SDD', 'Pollination']
            position='bottom left'
            color='red'

            if (not cumulative):
                chart.add_hline(y=30,line_color='red')

        elif ('Temp' in label) or ('Sdd' in label):
            sel_dates = [st.session_state['dates']['regular'], st.session_state['dates']['pollination']]
            sel_text = ['SDD', 'Pollination']
            position='top left'
            color='red'

        else:
            sel_dates = [st.session_state['dates']['planting'], st.session_state['dates']['jul_aug']]
            sel_text = ['Planting', 'Jul-Aug']
            position='top left'
            color='blue'

        for i,d in enumerate(sel_dates):
            s=find_on_x_axis(d['start'][seas_year],chart)
            e=find_on_x_axis(d['end'][seas_year],chart)
                    
            s_str=s.strftime("%Y-%m-%d")
            e_str=e.strftime("%Y-%m-%d")
            
            c= sel_text[i] +'   ('+s.strftime("%b%d")+' - '+e.strftime("%b%d")+')'

            chart.add_vrect(x0=s_str, x1=e_str,fillcolor=color, opacity=0.1,layer="below", line_width=0, annotation=dict(font_size=14,textangle=90,font_color=color), annotation_position=position, annotation_text=c)

st.set_page_config(page_title="Weather Charts",layout="wide",initial_sidebar_state="expanded")

# region initialization
su.initialize_Monitor_Corn_USA()
sel_df = uw.get_w_sel_df()
corn_states_options=['USA', 'IA','IL','IN','OH','MO','MN','SD','NE']
# endregion

# region controls
with st.sidebar:
    st.markdown("# Weather Charts")
    sel_states = st.multiselect( 'States',corn_states_options,['USA'])
    w_vars = st.multiselect( 'Weather Variables',[GV.WV_PREC,GV.WV_TEMP_MAX,GV.WV_TEMP_MIN,GV.WV_TEMP_AVG, GV.WV_SDD_30],[GV.WV_TEMP_MAX])
    slider_year_start = st.date_input("Seasonals Start", dt(2022, 1, 1))
    cumulative = st.checkbox('Cumulative')
    ext_mode = st.radio("Projection",(GV.EXT_MEAN, GV.EXT_SHIFT_MEAN,GV.EXT_ANALOG,GV.EXT_LIMIT))

ref_year_start = dt(GV.CUR_YEAR, slider_year_start.month, slider_year_start.day)
# endregion

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
        w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)

        # Calculate Weighted DF
        w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

        all_charts_usa = uc.Seas_Weather_Chart(w_w_df_all, ext_mode=[ext_mode], limit=[-1,1], cumulative = cumulative, ref_year_start= ref_year_start)

        for label, chart in all_charts_usa.all_figs.items():
            # add_w_dates(label,chart)
            st.markdown("#### "+label.replace('_',' '))
            st.plotly_chart(chart)
            # st.markdown("---")
            st.markdown("#### ")


# Single States ---------------------------------------------------------------------------------------------------------
sel_df=sel_df[sel_df['state_alpha'].isin(sel_states)]
all_charts_states={}
if len(sel_df)>0 and len(w_vars)>0:
    w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)
    all_charts_states = uc.Seas_Weather_Chart(w_df_all, ext_mode=[ext_mode], limit=[-1,1], cumulative = cumulative, ref_year_start= ref_year_start)

    for label, chart in all_charts_states.all_figs.items():
        # add_w_dates(label, chart)
        st.markdown("#### "+label.replace('_',' '))        
        st.plotly_chart(chart)
        # st.markdown("---")
        st.markdown("#### ")