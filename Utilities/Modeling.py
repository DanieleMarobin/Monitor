import numpy as np

import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import plotly.express as px

import Utilities.GLOBAL as GV

def Build_DF_Instructions(WD_All='weighted', WD = GV.WD_HIST, prec_units = 'mm', temp_units='C', ext_mode = GV.EXT_MEAN):
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