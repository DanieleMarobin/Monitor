import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime as dt
from copy import deepcopy
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import plotly.express as px
import Utilities.GLOBAL as GV

def get_massive_df(w_df, date_start, date_end):
    w_df['date']=w_df.index
    w_df['year'] = pd.to_datetime(w_df['date']).dt.year
    w_df['time_id'] = 100*pd.to_datetime(w_df['date']).dt.month+pd.to_datetime(w_df['date']).dt.day 

    wws=[]

    start_list = pd.date_range(start = date_start, end = date_end, freq="1D")

    for s in tqdm(start_list):
        id_s = s.month * 100 + s.day
        end_list = pd.date_range(start=min(s + pd.DateOffset(days=0), date_end), end=date_end, freq="1D")

        for e in end_list:
            id_e = e.month * 100 + e.day
            ww = w_df[(w_df.time_id>=id_s) & (w_df.time_id<=id_e)]                                
            ww=ww.drop(columns=['time_id'])
            ww.columns=list(map(lambda x:'year'if x=='year'else x+'_'+s.strftime("%b%d")+'-'+e.strftime("%b%d"),list(ww.columns)))
            ww = ww.groupby('year').mean()
            ww.index=ww.index.astype(int)
            wws.append(ww)                                

    # Excluding everything: it exclude 2022  because some of the windows have not started yet
    fo = pd.concat(wws, sort=True, axis=1, join='inner')
    return fo


def seas_day(date, ref_year_start= dt(GV.CUR_YEAR,1,1)):
    """
    'seas_day' is the X-axis of the seasonal plot:
            - it makes sure to include 29 Feb
    """

    start_idx = 100 * ref_year_start.month + ref_year_start.day
    date_idx = 100 * date.month + date.day

    if (start_idx<300):
        if (date_idx>=start_idx):
            return dt(GV.LLY, date.month, date.day)
        else:
            return dt(GV.LLY+1, date.month, date.day)
    else:
        if (date_idx>=start_idx):
            return dt(GV.LLY-1, date.month, date.day)
        else:
            return dt(GV.LLY, date.month, date.day)



def generate_weather_windows_df(input_w_df, date_start, date_end):
    w_df=deepcopy(input_w_df)
    w_df['date']=w_df.index
    w_df['year'] = pd.to_datetime(w_df['date']).dt.year
    w_df['time_id'] = 100*pd.to_datetime(w_df['date']).dt.month+pd.to_datetime(w_df['date']).dt.day 

    wws=[]

    start_list = pd.date_range(start = date_start, end = date_end, freq="1D")

    for s in tqdm(start_list):
        id_s = s.month * 100 + s.day
        end_list = pd.date_range(start=min(s + pd.DateOffset(days=0), date_end), end=date_end, freq="1D")

        for e in end_list:
            id_e = e.month * 100 + e.day
            ww = w_df[(w_df.time_id>=id_s) & (w_df.time_id<=id_e)]                                
            ww=ww.drop(columns=['time_id'])
            ww.columns=list(map(lambda x:'year'if x=='year'else x+'_'+s.strftime("%b%d")+'-'+e.strftime("%b%d"),list(ww.columns)))
            ww = ww.groupby('year').mean()
            ww.index=ww.index.astype(int)
            wws.append(ww)                                

    # Excluding everything: it exclude 2022  because some of the windows have not started yet
    fo = pd.concat(wws, sort=True, axis=1, join='inner')
    return fo

def Build_DF_Instructions(WD_All='weighted', WD = GV.WD_HIST, prec_units = 'mm', temp_units='C', ext_mode = GV.EXT_DICT):
    fo={}

    if WD_All=='simple':
        fo['WD_All']='w_df_all'
    elif WD_All=='weighted':
        fo['WD_All']='w_w_df_all'

    fo['WD']=WD
        
    if prec_units=='mm':
        fo['prec_factor']=1.0
    elif prec_units=='in':
        fo['prec_factor']=1.0/25.4

    if temp_units=='C':
        fo['temp_factor']=1.0
    elif temp_units=='F':
        fo['temp_factor']=9.0/5.0

    fo['ext_mode']=ext_mode
    return fo

def Fit_Model(df, y_col, exclude_from_year=GV.CUR_YEAR):
    df=df.loc[df.index<exclude_from_year]

    y_df = df[[y_col]]
    X_df=df.drop(columns = y_col)

    return sm.OLS(y_df, X_df).fit()

def max_correlation(X_df, threshold=1.0):
    max_corr = np.abs(np.corrcoef(X_df,rowvar=False))
    max_corr = np.max(max_corr[max_corr<threshold])
    return max_corr

def chart_corr_matrix(X_df, threshold=1.0):
    corrMatrix = np.abs(X_df.corr())*100.0
    fig = px.imshow(corrMatrix,color_continuous_scale='RdBu_r')
    fig.update_traces(texttemplate="%{z:.1f}%")
    fig.update_layout(width=1400,height=787)
    return(fig)

def stats_model_cross_validate(X_df, y_df, folds):
    fo = {'cv_models':[], 'cv_corr':[], 'cv_p':[], 'cv_r_sq':[], 'cv_y_test':[],'cv_y_pred':[], 'cv_MAE':[], 'cv_MAPE':[]}
       
    X2_df = sm.add_constant(X_df)
    
    for split in folds:        
        train, test = split[0], split[1]

        max_corr=0
        if X_df.shape[1]>1:
            max_corr = np.abs(np.corrcoef(X_df.iloc[train],rowvar=False))
            max_corr=np.max(max_corr[max_corr<0.999]) 
        
        model = sm.OLS(y_df.iloc[train], X2_df.iloc[train]).fit()

        fo['cv_models']+=[model]
        fo['cv_corr']+=[max_corr]
        
        fo['cv_p']+=list(model.pvalues)
        fo['cv_r_sq']+=[model.rsquared]
                
        y_test = y_df.iloc[test]
        y_pred = model.predict(X2_df.iloc[test])
        
        fo['cv_y_test']+=[y_test]
        fo['cv_y_pred']+=[y_pred]
        
        fo['cv_MAE']+=[mean_absolute_error(y_test, y_pred)]
        fo['cv_MAPE']+=[mean_absolute_percentage_error(y_test, y_pred)]
        
    return fo

def folds_expanding(model_df, min_train_size=10):
    min_train_size= min(min_train_size,len(model_df)-3)
    folds_expanding = TimeSeriesSplit(n_splits=len(model_df)-min_train_size, max_train_size=0, test_size=1)
    folds = []
    folds = folds + list(folds_expanding.split(model_df))
    return folds

def print_folds(folds, X_df, years):
    '''
    Example (for the usual X_df with years as index): \n
    print_folds(folds, X_df, X_df.index.values)
    '''
    print('There are '+ str(len(folds)) +' folds:')
    if type(folds) == list:
        for f in folds:
            print(years[f[0]], "------>", years[f[1]])    
    else: 
        for train, test in folds.split(X_df): print(years[train], "------>", years[test])

def MAPE(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)