from datetime import datetime as dt

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

import Models.Soybean_USA_Yield_GA as sy

import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Utilities as uu
import Utilities.GLOBAL as GV


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