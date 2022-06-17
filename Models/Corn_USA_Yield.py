import sys
sys.path.append(r'\\ac-geneva-24\E\grains trading\visual_studio_code\\')

from datetime import datetime as dt

import pandas as pd
import numpy as np
import statsmodels.api as sm

import APIs.QuickStats as qs

import Utilities.SnD as us
import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Charts as uc
import Utilities.Utilities as uu
import Utilities.GLOBAL as GV


def Define_Scope():
    fo={}

    # Geography    
    geo = uw.get_w_sel_df()
    fo['geo'] = geo[geo[GV.WS_COUNTRY_ALPHA] == 'USA']

    # Weather Variables
    fo['w_vars'] = [GV.WV_PREC, GV.WV_TEMP_MAX, GV.WV_SDD_30]

    # Time
    fo['years']=list(range(1985,GV.CUR_YEAR+1))
    
    return fo

def Get_Data(scope):
    fo={}

    # Select Weather Variables
    in_files = GV.WS_UNIT_ALPHA
    out_cols = GV.WS_UNIT_ALPHA
    w_df_all = uw.build_w_df_all(scope['geo'], scope['w_vars'], in_files, out_cols)

    # Build the Weights
    corn_states=scope['geo'][GV.WS_STATE_ALPHA]

    weights = us.get_USA_prod_weights('CORN', 'STATE', scope['years'], corn_states)

    # Weighted Weather DataFrame All (All = 'hist', 'GFS', Etc Etc)
    fo['w_w_df_all']  = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')


    fo['yield']=qs.get_yields(years=scope['years'],cols_subset=['year','Value'])
    fo['planting_progress']=qs.get_progress(progress_var='planting', years=scope['years'], cols_subset=['week_ending','Value'])    
    fo['silking_progress'] =qs.get_progress(progress_var='silking',  years=scope['years'], cols_subset=['week_ending','Value'])

    return fo

def Process_Data(scope,raw_data):
    """
    Process data like calculating weather intervals, or other useful info or stats
        - when they are needed for downstream calcs
        - when they are useful to be plotted or tabulated
    """

    fo={}

    # Planting Interval: 80% planted -40 and +25 days
    date_80_pct_planted=us.dates_from_progress(raw_data['planting_progress'], sel_percentage=80)    
    start=date_80_pct_planted['date']+pd.DateOffset(-40)
    end = date_80_pct_planted['date']+pd.DateOffset(+25)
    fo['planting_interval']=pd.DataFrame({'start':start,'end':end})

    # Jul Aug Interval: 80% planted +26 and +105 days
    start=date_80_pct_planted['date']+pd.DateOffset(+26)
    end = date_80_pct_planted['date']+pd.DateOffset(105)
    fo['jul_aug_interval']=pd.DataFrame({'start':start,'end':end})    

    # Pollination Interval: 50% planted -15 and +15 days
    date_50_pct_silked=us.dates_from_progress(raw_data['silking_progress'], sel_percentage=50)    
    start=date_50_pct_silked['date']+pd.DateOffset(-15)
    end = date_50_pct_silked['date']+pd.DateOffset(15)
    fo['pollination_interval']=pd.DataFrame({'start':start,'end':end})

    # Regular Interval: 20 Jun - 15 Sep
    start=[dt(y,6,20) for y in scope['years']]
    end=  [dt(y,9,25) for y in scope['years']]
    fo['regular_interval']=pd.DataFrame({'start':start,'end':end},index=scope['years'])
    return fo

def Build_Model_DF_Instructions(w_df, prec_units, temp_units):


def Build_Model_DF(scope, raw_data, processed_data):
    """
    The model DataFrame has 11 Columns:
            1) Yield (y)
            9) Variables
            1) Constant (added to be able to fit the model with 'statsmodels.api')

            1+9+1 = 11 Columns
    """
    w_df = GV.WD_HIST
    prec_factor = (1.0/25.4) 
    temp_factor = (9.0/5.0)

    # prec_factor = 1.0
    # temp_factor = 1.0

    # 1) Trend (first because I set the index and because it surely includes CUR_YEAR, while other variable might not have any value yet)
    df=pd.DataFrame(scope['years'], columns=['Trend'], index=scope['years'])
        
    # 2) Yield
    yields =  raw_data['yield']['Value'].values
    if not (GV.CUR_YEAR in yields): yields=np.append(yields, np.nan) # Because otherwise it cuts the GV.CUR_YEAR row
    df['Yield'] = yields    

    # 3) Percentage Planted as of 15th May
    df['Planted pct on May 15th']=us.progress_from_date(raw_data['planting_progress'], sel_date='2021-05-15')    

    # 4) Planting Precipitation - Based on 80% Planted Dates (What day was it when the crop was 80% planted)
    df['Planting Prec'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Prec']], processed_data['planting_interval'])*prec_factor

    # 5) Planting Prec Squared
    df['Planting Prec Squared'] = df['Planting Prec']**2

    # 6) Jul Aug Precipitation
    df['Jul Aug Prec'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Prec']], processed_data['jul_aug_interval'])*prec_factor

    # 7) Jul Aug Precipitation Squared
    df['Jul Aug Prec Squared'] = df['Jul Aug Prec']**2

    # 8) Precip Interaction = 'Planting Prec' * 'Jul Aug Prec'
    df['Prec Interaction'] = df['Planting Prec'] * df['Jul Aug Prec']

    # 9) Stress SDD - Based on 50% Silked Dates (What day was it when the crop was 50% silked)
    df['Pollination SDD'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Sdd30']], processed_data['pollination_interval'])*temp_factor

    # 10) Regular SDD: 20 Jun - 15 Sep
    df['Regular SDD'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Sdd30']], processed_data['regular_interval'])*temp_factor
    df['Regular SDD']=df['Regular SDD']-df['Pollination SDD']

    # 11) Constant
    df = sm.add_constant(df, has_constant='add')

    return df
    
def Fit_Model(df, y_col, exclude_from_year=GV.CUR_YEAR):
    df=df.loc[df.index<exclude_from_year]

    y_df = df[[y_col]]
    X_df=df.drop(columns = y_col)

    return sm.OLS(y_df, X_df).fit()

def Build_Prediction_DF(scope, raw_data, processed_data, df, date_start):   
        
    raw_data_pred = raw_data.copy()

    date_end = raw_data_pred['w_w_df_all'][GV.WD_H_GFS].index[-1]
    days_pred= list(pd.date_range(date_start, date_end))

    # Get the "dict_col_seas" from the 'hist' 
    w_df, dict_col_seas = uw.extend_with_seasonal_df(df_to_ext.loc[:day], return_dict_col_seas=True)
    for i, d in enumerate(days_pred):


    return 0



def main():
    scope = Define_Scope()
    raw_data = Get_Data(scope['geo'], scope['w_vars'], scope['years'])
    df = Build_Model_DF(raw_data)
    model = Fit_Model(df,'Yield',GV.CUR_YEAR)
    return model
    


if __name__=='__main__':    
    print('Corn_USA_Yield.py')