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
import Utilities.Utilities as uu

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

def generate_weather_windows_df(input_w_df, date_start, date_end, ref_year = GV.CUR_YEAR, ref_year_start= dt(GV.CUR_YEAR,1,1), freq_start='1D', freq_end='1D'):
    wws=[]
    w_df=deepcopy(input_w_df)
    add_seas_year(w_df,ref_year, ref_year_start) # add the 'year' column
    w_df['seas_day'] = [seas_day(d,ref_year_start) for d in w_df.index]
    
    start_list = pd.date_range(start = date_start, end = date_end, freq=freq_start)

    for s in tqdm(start_list):
        id_s = seas_day(date=s, ref_year_start=ref_year_start)
        end_list = pd.date_range(start=min(s + pd.DateOffset(days=0), date_end), end=date_end, freq=freq_end)

        for e in end_list:
            id_e = seas_day(date=e, ref_year_start=ref_year_start)

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

def from_cols_to_var_windows(cols=[]):
    """
    Typical Use:
    ww = um.from_cols_to_var_windows(m.params.index)
    """
    # Make sure that this sub is related to the function "def windows(cols,year=2020):"
    var_windows=[]
    year = GV.LLY

    for c in (x for x  in cols if '-' in x):
        split=re.split('_|-',c)
        var = split[0]+'_'+split[1]
        
        if len(split)>1:
            start = dt.strptime(split[2]+str(year),'%b%d%Y')
            end = dt.strptime(split[3]+str(year),'%b%d%Y')
        
        var_windows.append({'variables':[var], 'windows':[{'start': start,'end':end}]})
    
    return var_windows

def extract_yearly_ww_variables(w_df, var_windows=[], join='inner', drop_na=True, drop_how='any'):
    w_df['date']=w_df.index
    wws=[]
    
    for v_w in var_windows:    
        # Get only needed variables
        w_cols=['date']
        w_cols.extend(v_w['variables'])
        w_df_sub = w_df[w_cols]

        w_df_sub['month'] = pd.to_datetime(w_df_sub['date']).dt.month
        w_df_sub['day'] = pd.to_datetime(w_df_sub['date']).dt.day
        w_df_sub['year'] = pd.to_datetime(w_df_sub['date']).dt.year
        w_df_sub['time_id'] = 100*w_df_sub['month']+w_df_sub['day']

        # Adding:
        #    1) 'time_id': to select the weather window
        #    2) 'year': to be able to group by year

        w_cols.extend(['time_id','year'])
        w_df_sub = w_df_sub[w_cols]
        
        for w in v_w['windows']:
            s = w['start']; id_s = s.month * 100 + s.day
            e = w['end']; id_e = e.month * 100 + e.day

            ww = w_df_sub[(w_df_sub.time_id>=id_s) & (w_df_sub.time_id<=id_e)]                                
            ww.drop(columns=['time_id'], inplace=True)
            ww.columns = list(map(lambda x:'year'if x=='year'else x+'_'+s.strftime("%b%d")+'-'+e.strftime("%b%d"),list(ww.columns)))
            ww = ww.groupby('year').mean()
            ww.index=ww.index.astype(int)
            wws.append(ww)                                  

    out_df = pd.concat(wws, sort=True, axis=1, join=join)        
    if drop_na: out_df.dropna(inplace=True, how=drop_how) # how : {'any', 'all'}
    return  out_df


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

def analyze_results(file_names=[]):
    print('--------------------------------------------------------')
    dm_best = {} # 1 key for every file to be analysed

    p_values_threshold = 0.05
    corr_threshold = 0.5

    rank_df = {'file':[],'idx':[], 'equation':[],'actual_cover':[],'holes_cover':[],'neg_cover':[], 'pos_cover':[],
            'r_sq':[],'corr':[],'MAE':[],'MAPE':[],
            'cv_r_sq':[],'cv_p':[],'cv_c':[],'cv_MAE':[],'cv_MAPE':[],'fitness':[],'cv_p_N':[],'cv_c_N':[]}

    for f in file_names:
        dm_best[f] = uu.deserialize(f)

        r=dm_best[f] # 'r' stands for Result

        for i,m in enumerate(r['model']):            
            wws = windows(m.params.index)
            cover = windows_coverage(wws)
            actual_cover =  len(cover[1])
            holes_cover =  len(cover[0])-len(cover[1])

            pos_prec_cover=actual_cover
            neg_prec_cover=0

            neg_prec = np.array([(m.params[x]<0 and 'Prec' in x) for x in m.params.index if '-' in x])

            if len(neg_prec)>0:
                pos_prec = ~neg_prec    
                neg_prec_cover = windows_coverage(wws[neg_prec])[1]
                pos_prec_cover = windows_coverage(wws[pos_prec])[1]

            cv_MAPE = np.array(r['cv_MAPE'][i])

            rank_df['file']+=[f]
            rank_df['idx']+=[i]

            coeff_2_digits = ['{:.2f}'.format(v) for v in m.params.values]
            rank_df['equation']+=[list(zip(coeff_2_digits,m.params.index))]

            rank_df['actual_cover']+=[actual_cover]
            rank_df['holes_cover']+=[holes_cover]
            rank_df['neg_cover']+=[neg_prec_cover]
            rank_df['pos_cover']+=[pos_prec_cover]

            rank_df['r_sq']+=[m.rsquared]
            rank_df['corr']+=[r['corr'][i]]
            rank_df['MAE']+=[r['MAE'][i]]
            rank_df['MAPE']+=[r['MAPE'][i]]

            rank_df['cv_r_sq']+=[np.mean(np.array(r['cv_r_sq'][i]))]
            rank_df['cv_p']+=[np.mean(np.array(r['cv_p'][i]))]
            rank_df['cv_c']+=[np.mean(np.array(r['cv_corr'][i]))]

            rank_df['cv_MAE']+=[np.mean(np.array(r['cv_MAE'][i]))]        
            rank_df['cv_MAPE']+=[np.mean(cv_MAPE)]
            rank_df['fitness']+=[r['fitness'][i]]

            rank_df['cv_p_N']+=[np.sum(np.array(r['cv_p'][i])>p_values_threshold)]
            rank_df['cv_c_N']+=[np.sum(np.array(r['cv_corr'][i])>corr_threshold)]                
    
    rank_df=pd.DataFrame(rank_df)
    return rank_df

def pick_model(results_file, model_id):
    result_file = uu.deserialize(results_file)
    model = result_file['model'][model_id]
    return model

def folds_expanding(model_df, min_train_size=10):    
    if 'const' in model_df.columns:
        col_n=len(model_df.columns)
    else:
        col_n=len(model_df.columns)+1
    
    min_train_size= max(min_train_size,col_n+1) # Obviously cannot run a model if I have 10 points and 10 columns: so I am adding 1 point to the columns size
    min_train_size= min(min_train_size,len(model_df)-3) # Adjusting for the number of datapoints

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