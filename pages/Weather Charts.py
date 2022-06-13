import streamlit as st
from datetime import datetime as dt

import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.SnD as us

def find_on_x_axis(date, chart):
    id = 100*date.month+date.day
    for x in chart.data[0]['x']:
        if 100*x.month + x.day==id:
            return x


def add_w_dates(label, chart):
# st.session_state['dates']['jul_aug'] = jul_aug_dates
# st.session_state['dates']['planting'] = planting_dates
# st.session_state['dates']['pollination'] = pollination_dates
# st.session_state['dates']['regular'] = regular_dates
    
    if 'dates' in st.session_state:
        seas_year = 2022
        if 'Temp' in label:
            sel_dates = [st.session_state['dates']['regular'], st.session_state['dates']['pollination']]
            sel_text = ['SDD', 'Pollination']
            position='bottom left'
            color='red'
            if not cumulative:
                chart.add_hline(y=30,line_color='red')
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
st.markdown("# Weather Charts")
st.sidebar.markdown("# Weather Charts")
st.markdown("---")


sel_df = uw.get_w_sel_df()
corn_states_options=['USA', 'IA','IL','IN','OH','MO','MN','SD','NE']

col_states, col_w_var, year_start_col, cumulative_col = st.columns([2,2,1,1])

with col_states:
    sel_states = st.multiselect( 'States',corn_states_options,['USA'])

with col_w_var:
    w_vars = st.multiselect( 'Weather Variables',[uw.WV_PREC,uw.WV_TEMP_MAX,uw.WV_TEMP_MIN,uw.WV_TEMP_AVG],[uw.WV_TEMP_MAX])

with year_start_col:
    slider_year_start = st.date_input("Seasonals Start", dt(2022, 1, 1))

with cumulative_col:
    st.markdown('# ')
    cumulative = st.checkbox('Cumulative')

 

# slider_year_start = st.slider("Seasonal Start", value=dt(2020, 1, 1),min_value=dt(2020, 1, 1), max_value=dt(2020, 12, 31), format="DD MMM")
# st.write('Start Date:', slider_year_start)


ref_year_start = dt(uw.CUR_YEAR, slider_year_start.month, slider_year_start.day)


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

        all_charts_usa = uc.Seas_Weather_Chart(w_w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = cumulative, ref_year_start= ref_year_start)

        for label, chart in all_charts_usa.all_figs.items():
            add_w_dates(label,chart)
            st.markdown("#### "+label.replace('_',' '))
            st.plotly_chart(chart)
            st.markdown("---")
            st.markdown("#### ")


# Single States ---------------------------------------------------------------------------------------------------------
sel_df=sel_df[sel_df['state_alpha'].isin(sel_states)]
all_charts_states={}
if len(sel_df)>0 and len(w_vars)>0:
    w_df_all = uw.build_w_df_all(sel_df,w_vars=w_vars, in_files=uw.WS_UNIT_ALPHA, out_cols=uw.WS_UNIT_ALPHA)
    all_charts_states = uc.Seas_Weather_Chart(w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = False, ref_year_start= ref_year_start)

    for label, chart in all_charts_states.all_figs.items():
        add_w_dates(label, chart)
        st.markdown("#### "+label.replace('_',' '))        
        st.plotly_chart(chart)
        st.markdown("---")
        st.markdown("#### ")    


    




