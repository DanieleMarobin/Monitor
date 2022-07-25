from datetime import datetime as dt

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

import Models.Soybean_USA_Yield_GA as sy

import Utilities.Streamlit as su

import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Utilities as uu
import Utilities.GLOBAL as GV


def st_milestones_and_intervals(milestones, intervals):
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
            styler = milestones['80_pct_planted'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # 50% Silking
        with m2:
            st.markdown('##### 50% Silking')
            st.write('Self-explanatory')  
            styler = milestones['50_pct_silked'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
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
            styler = intervals['jul_aug_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Pollination_SDD
        with i3:
            # 50% Silking -15 and +15 days
            st.markdown('##### Pollination SDD')
            st.write('50% Silking -15 and +15 days')
            styler = intervals['pollination_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)

        # Regular_SDD
        with i4:
            st.markdown('##### Regular SDD')
            st.write('20 Jun - 15 Sep')
            styler = intervals['regular_interval'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
            st.write(styler)
            
        st.markdown("---")

# Analysis preference
id={}
id['prefix']='Corn_USA_Yield'
id['title_str'] = "Corn Yield"

id['season_start'] = dt(GV.CUR_YEAR,4,10)
id['season_end'] = dt(GV.CUR_YEAR,9,20)    

id['simple_weights'] = False

id['func_Scope'] = sy.Define_Scope
id['func_Raw_Data'] = sy.Get_Data_All_Parallel
id['func_Milestones'] = sy.Milestone_from_Progress
id['func_Extend_Milestones'] = sy.Extend_Milestones
id['func_Intervals'] = sy.Intervals_from_Milestones
id['func_Build_DF'] = sy.Build_DF
id['func_Pred_DF'] = sy.Build_Pred_DF

id['func_add_chart_intervals'] = sy.add_chart_intervals
id['func_st_milestones_and_intervals'] = st_milestones_and_intervals

# Create the Streamlit App
su.USA_Yield_Model_Template_GA(id)



# From here -----------------------------------------------------
os.system('cls')
y_col  ='Yield'
file='GA_soy_7'; 
id=142
ref_year_start=dt(2022,1,1)

res = uu.deserialize(file,comment=False)
m = res['model'][id]

# Get the data
scope = sy.Define_Scope()
raw_data = sy.Get_Data_All_Parallel(scope)

# Elaborate the data
wws = um.var_windows_from_cols(m.params.index)
model_df = um.extract_yearly_ww_variables(w_df = raw_data['w_w_df_all']['hist'], var_windows= wws)
model_df = pd.concat([raw_data['yield'], model_df], sort=True, axis=1, join='inner')

list_WD=[GV.WD_H_GFS, GV.WD_H_GFS_EN, GV.WD_H_ECMWF, GV.WD_H_ECMWF_EN]

for WD in list_WD:
    print('--------------------')
    print(WD)
    w_df = raw_data['w_w_df_all'][WD]
    pred_df = uw.extend_with_seasonal_df(w_df, ref_year_start=ref_year_start)    

    model_df_ext = um.extract_yearly_ww_variables(w_df = pred_df, var_windows= wws)
    model_df_ext = pd.concat([raw_data['yield'], model_df_ext], sort=True, axis=1, join='outer')

    model_df_ext['year']=model_df_ext.index    
    model_df_ext=sm.add_constant(model_df_ext)
    
    model = um.Fit_Model(df=model_df_ext, y_col=y_col, exclude_from_year=GV.CUR_YEAR)

    # Compare Calculation vs Saved Results
    if False:
        print('Saved Results Comparison')
        df=model_df_ext.loc[model_df_ext.index < GV.CUR_YEAR]
        y_df = df[[y_col]]
        X_df=df[[c for c in m.params.index if c != 'const']]
        folds = um.folds_expanding(model_df=df, min_train_size=10)

        cv_score = um.stats_model_cross_validate(X_df, y_df, folds)

        comp_list =['cv_corr', 'cv_p', 'cv_r_sq', 'cv_MAE', 'cv_MAPE']

        for k in comp_list:
            saved = np.mean(res[k][id])
            calc = np.mean(cv_score[k])

            print('{0} Saved: {1}, Calculated: {2}, Difference: {3}'.format(k, saved, calc, saved-calc))

    yields = model.predict(model_df_ext[m.params.index].loc[GV.CUR_YEAR]).values    
    print('Yield Prediction:', yields)