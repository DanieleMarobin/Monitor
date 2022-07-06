from datetime import datetime as dt

import streamlit as st

import Models.Corn_USA_Yield as cy
import Utilities.Streamlit as su
import Utilities.GLOBAL as GV

def st_milestones_and_intervals(id):
    # Milestones
    if True:
        dates_fmt = "%d %b %Y"
        milestones_col, _ , intervals_col = st.columns([2,   0.5,   6])
        
        with milestones_col:
            st.markdown('##### Milestones')

        with intervals_col:
            st.markdown('##### Weather Windows')

        m1, m2 ,_, i1, i2, i3, i4 = st.columns([1,1,   0.5,   1.5,1.5,1.5,1.5])

        # 80% Planted
        with m1:
            st.markdown('##### 80% Planted')  
            st.write('Self-explanatory')  
            styler = st.session_state[id['prefix']]['milestones']['80_pct_planted'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # 50% Silking
        with m2:
            st.markdown('##### 50% Silking')
            st.write('Self-explanatory')  
            styler = st.session_state[id['prefix']]['milestones']['50_pct_silked'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
            st.write(styler)    

    # Intervals
    if True:
        # Planting_Prec
        with i1:
            st.markdown('##### Planting Prec')
            st.write('80% planted -40 and +25 days')
            styler = st.session_state[id['prefix']]['intervals']['planting_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Jul_Aug_Prec
        with i2:
            st.markdown('##### Jul-Aug Prec')
            st.write('80% planted +26 and +105 days')
            styler = st.session_state[id['prefix']]['intervals']['jul_aug_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Pollination_SDD
        with i3:
            # 50% Silking -15 and +15 days
            st.markdown('##### Pollination SDD')
            st.write('50% Silking -15 and +15 days')
            styler = st.session_state[id['prefix']]['intervals']['pollination_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Regular_SDD
        with i4:
            st.markdown('##### Regular SDD')
            st.write('20 Jun - 15 Sep')
            styler = st.session_state[id['prefix']]['intervals']['regular_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)
            
        st.markdown("---")

# Analysis preference
id={}
id['prefix']='Corn_USA_Yield'
id['title_str'] = "Corn Yield"

id['season_start'] = dt(GV.CUR_YEAR,4,10)
id['season_end'] = dt(GV.CUR_YEAR,9,20)    

id['simple_weights'] = False

id['func_Scope'] = cy.Define_Scope
id['func_Raw_Data'] = cy.Get_Data_All_Parallel
id['func_Milestones'] = cy.Milestone_from_Progress
id['func_Extend_Milestones'] = cy.Extend_Milestones
id['func_Intervals'] = cy.Intervals_from_Milestones
id['func_Build_DF'] = cy.Build_DF
id['func_Progressive_Pred_DF'] = cy.Build_Progressive_Pred_DF

id['func_add_chart_intervals'] = cy.add_chart_intervals
id['func_st_milestones_and_intervals'] = st_milestones_and_intervals

# Create the Streamlit App
su.USA_Yield_Model_Template_old(id)