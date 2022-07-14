import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime as dt
from datetime import timedelta
from copy import deepcopy
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import plotly.express as px
import Utilities.GLOBAL as GV

def add_seas_year(w_df, ref_year=GV.CUR_YEAR, ref_year_start= dt(GV.CUR_YEAR,1,1), offset = 2):
    # yo = year offset
    # offset = 2 means:
    #       - first year is going to be first year - 2
    #       - last year is going to be ref_year + 2 = CUR_YEAR + 2

    os = w_df.index[0].year - ref_year -offset# offset start
    oe = w_df.index[-1].year - ref_year +offset  # offset end

    for yo in range(os, oe):
        value = ref_year+yo
        ss = ref_year_start+ pd.DateOffset(years=yo) # start slice
        es = ref_year_start+ pd.DateOffset(years=yo+1)+pd.DateOffset(days=-1) # end slice

        mask = ((w_df.index>=ss) & (w_df.index<=es))
        w_df.loc[mask,'year']=int(value)

    w_df['year'] = w_df['year'].astype('int')

    return w_df

def seas_day(date, ref_year_start= dt(GV.CUR_YEAR,1,1)):
    """
    'seas_day' is the X-axis of the seasonal plot:
            - it makes sure to include 29 Feb
            - it is very useful in creating weather windows
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

def generate_weather_windows_df(input_w_df, date_start, date_end, ref_year_start= dt(GV.CUR_YEAR,1,1), freq_start='1D', freq_end='1D'):
    wws=[]
    w_df=deepcopy(input_w_df)
    add_seas_year(w_df) # add the 'year' column
    w_df['seas_day'] = [seas_day(d,ref_year_start) for d in w_df.index]
    
    start_list = pd.date_range(start = date_start, end = date_end, freq=freq_start)

    for s in tqdm(start_list):
        id_s = seas_day(date=s, ref_year_start=ref_year_start)
        end_list = pd.date_range(start=min(s + pd.DateOffset(days=0), date_end), end=date_end, freq=freq_end)

        for e in end_list:
            id_e = seas_day(date=s, ref_year_start=ref_year_start)

            ww = w_df[(w_df['seas_day']>=id_s) & (w_df['seas_day']<=id_e)]
            ww=ww.drop(columns=['seas_day'])
            ww.columns=list(map(lambda x:'year'if x=='year'else x+'_'+s.strftime("%b%d")+'-'+e.strftime("%b%d"),list(ww.columns)))
            ww = ww.groupby('year').mean()
            ww.index=ww.index.astype(int)
            wws.append(ww)                                

    # Excluding everything: it exclude 2022  because some of the windows have not started yet
    fo = pd.concat(wws, sort=True, axis=1, join='inner')
    return fo


def windows(cols,year=2020):
    fo=[]
    
    for c in (x for x  in cols if '-' in x):
        split=re.split('_|-',c)
        
        if len(split)>1:
            start = dt.strptime(split[2]+str(year),'%b%d%Y')
            end = dt.strptime(split[3]+str(year),'%b%d%Y')
            fo.append((start,end))            
    return np.array(fo)

def windows_coverage(windows):
    fo = []
    for w in windows:
        fo.extend(np.arange(w[0], w[1] + timedelta(days = 1), dtype='datetime64[D]'))
    
    actual = set(fo)    
    if (len(actual)>0):
        full = np.arange(min(actual), max(actual) + np.timedelta64(1,'D'), dtype='datetime64[D]')
    else:        
        full=[]
        
    return full, actual



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
    """
    the 'threshold' is needed because when I want to analyze the 'max correlation'

    """

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

def print_folds(folds, years, X_df=None):
    '''
    Example (for the usual X_df with years as index):
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