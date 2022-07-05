from datetime import datetime as dt

import streamlit as st

import Models.Soybean_USA_Yield as sy
import Utilities.Streamlit as su
import Utilities.GLOBAL as GV

def st_milestones_and_intervals(id):
    # Milestones
    if True:
        dates_fmt = "%d %b %Y"
        milestones_col, _ , intervals_col = st.columns([1,   0.5,   6])
        with milestones_col:
            st.markdown('##### Milestones')

        with intervals_col:
            st.markdown('##### Weather Windows')

        m1,_, i1, i2, i3, i4 = st.columns([1,   0.5,   1.5,1.5,1.5,1.5])

        # 50% Silking
        with m1:
            st.markdown('##### 50% Bloomed')
            st.write('Self-explanatory')  
            styler = st.session_state[id['prefix']]['milestones']['50_pct_bloomed'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
            st.write(styler)    

    # Intervals
    if True:
        # Planting_Prec
        with i1:
            st.markdown('##### Planting Prec')
            st.write('10 May - 10 Jul')
            styler = st.session_state[id['prefix']]['intervals']['planting_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Jul_Aug_Prec
        with i2:
            st.markdown('##### Jul-Aug Prec')   
            st.write('11 Jul - 15 Sep')
            styler = st.session_state[id['prefix']]['intervals']['jul_aug_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Pollination_SDD
        with i3:
            # 50% Bloomed -10 and +10 days
            st.markdown('##### Pollination SDD')
            st.write('50% Bloomed -10 and +10 days')
            styler = st.session_state[id['prefix']]['intervals']['pollination_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Regular_SDD
        with i4:
            st.markdown('##### Regular SDD')
            st.write('25 Jun - 15 Sep')
            styler = st.session_state[id['prefix']]['intervals']['regular_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)
            
        st.markdown("---")


# Analysis preference
id={}
id['prefix']='Soybean_USA_Yield'
id['title_str'] = "Soybean Yield"

id['season_start'] = dt(GV.CUR_YEAR,4,10)
id['season_end'] = dt(GV.CUR_YEAR,9,20)    

id['sel_WD']=[GV.WD_HIST, GV.WD_H_GFS] # GV.WD_HIST, GV.WD_H_GFS, GV.WD_H_ECMWF
id['simple_weights'] = False

id['func_Scope'] = sy.Define_Scope
id['func_Raw_Data'] = sy.Get_Data_All_Parallel
id['func_Milestones'] = sy.Milestone_from_Progress
id['func_Extend_Milestones'] = sy.Extend_Milestones
id['func_Intervals'] = sy.Intervals_from_Milestones
id['func_Build_DF'] = sy.Build_DF
id['func_Progressive_Pred_DF'] = sy.Build_Progressive_Pred_DF

id['func_add_chart_intervals'] = sy.add_chart_intervals
id['func_st_milestones_and_intervals'] = st_milestones_and_intervals

# Create the Streamlit App
su.USA_Yield_Model_Template_old(id)