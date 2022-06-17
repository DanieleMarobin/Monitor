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
    """
    'geo_df':
        it is a dataframe (selection of rows of the weather selection file)
    'geo_input_file': 
        it needs to match the way files were named by the API
            GV.WS_STATE_NAME    ->  Mato Grosso_Prec.csv
            GV.WS_STATE_ALPHA   ->  MT_Prec.csv
            GV.WS_STATE_CODE    ->  51_Prec.csv

    'geo_output_column':
        this is how the columns will be renamed after reading the above files (important when matching weight matrices, etc)
            GV.WS_STATE_NAME    ->  Mato Grosso_Prec
            GV.WS_STATE_ALPHA   ->  MT_Prec
            GV.WS_STATE_CODE    ->  51_Prec
    """

    fo={}

    # Geography    
    geo = uw.get_w_sel_df()
    fo['geo_df'] = geo[geo[GV.WS_COUNTRY_ALPHA] == 'USA']
    fo['geo_input_file'] = GV.WS_UNIT_ALPHA 
    fo['geo_output_column'] = GV.WS_UNIT_ALPHA

    # Weather Variables
    fo['w_vars'] = [GV.WV_PREC, GV.WV_TEMP_MAX, GV.WV_SDD_30]

    # Time
    fo['years']=list(range(1985,GV.CUR_YEAR+1))
    
    return fo

def Get_Data(scope):
    fo={}
    fo['locations']=scope['geo_df'][GV.WS_STATE_ALPHA]

    # USDA    
    fo['yield']=qs.get_yields(years=scope['years'],cols_subset=['year','Value'])
    fo['weights'] = us.get_USA_prod_weights('CORN', 'STATE', scope['years'], fo['locations'])    
    fo['planting_progress']=qs.get_progress(progress_var='planting', years=scope['years'], cols_subset=['week_ending','Value'])    
    fo['silking_progress'] =qs.get_progress(progress_var='silking',  years=scope['years'], cols_subset=['week_ending','Value'])
    
    # Weather
    fo['w_df_all'] = uw.build_w_df_all(scope['geo_df'], scope['w_vars'], scope['geo_input_file'], scope['geo_output_column'])    
    fo['w_w_df_all']  = uw.weighted_w_df_all(fo['w_df_all'], fo['weights'], output_column='USA')

    return fo

def Process_Data(scope,raw_data):
    """
    Process data like calculating weather intervals, or other useful info or stats
        - when they are needed for downstream calcs
        - when they are useful to be plotted or tabulated
    """

    fo={}

    # Planting Interval: 80% planted -40 and +25 days
    fo['date_80_pct_planted']=us.dates_from_progress(raw_data['planting_progress'], sel_percentage=80)    
    start=fo['date_80_pct_planted']['date']+pd.DateOffset(-40)
    end = fo['date_80_pct_planted']['date']+pd.DateOffset(+25)
    fo['planting_interval']=pd.DataFrame({'start':start,'end':end})

    # Jul Aug Interval: 80% planted +26 and +105 days
    start=fo['date_80_pct_planted']['date']+pd.DateOffset(+26)
    end = fo['date_80_pct_planted']['date']+pd.DateOffset(105)
    fo['jul_aug_interval']=pd.DataFrame({'start':start,'end':end})    

    # Pollination Interval: 50% planted -15 and +15 days
    fo['date_50_pct_silked']=us.dates_from_progress(raw_data['silking_progress'], sel_percentage=50)    
    start=fo['date_50_pct_silked']['date']+pd.DateOffset(-15)
    end = fo['date_50_pct_silked']['date']+pd.DateOffset(15)
    fo['pollination_interval']=pd.DataFrame({'start':start,'end':end})

    # Regular Interval: 20 Jun - 15 Sep
    start=[dt(y,6,20) for y in scope['years']]
    end=  [dt(y,9,25) for y in scope['years']]
    fo['regular_interval']=pd.DataFrame({'start':start,'end':end},index=scope['years'])
    return fo

def Build_DF_Instructions(WD_All='weighted', WD = GV.WD_HIST, prec_units = 'mm', temp_units='C'):
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
    elif prec_units=='F':
        fo['temp_factor']=9.0/5.0

    return fo

def Build_Train_DF(scope, raw_data, processed_data, instructions):
    """
    The model DataFrame has 11 Columns:
            1) Yield (y)
            9) Variables
            1) Constant (added to be able to fit the model with 'statsmodels.api')

            1+9+1 = 11 Columns
    """
    w_all=instructions['WD_All']
    WD=instructions['WD']
    w_df = raw_data[w_all][WD]
    
    prec_factor = instructions['prec_factor']
    temp_factor = instructions['temp_factor']

    # 1) Trend (first because I set the index and because it surely includes CUR_YEAR, while other variable might not have any value yet)
    df=pd.DataFrame(scope['years'], columns=['Trend'], index=scope['years'])
        
    # 2) Yield
    yields =  raw_data['yield']['Value'].values
    if not (GV.CUR_YEAR in yields): yields=np.append(yields, np.nan) # Because otherwise it cuts the GV.CUR_YEAR row
    df['Yield'] = yields    

    # 3) Percentage Planted as of 15th May
    df['Planted pct on May 15th']=us.progress_from_date(raw_data['planting_progress'], sel_date='2021-05-15')    

    # 4) Planting Precipitation - Based on 80% Planted Dates (What day was it when the crop was 80% planted)
    df['Planting Prec'] = uw.extract_w_windows(w_df[['USA_Prec']], processed_data['planting_interval'])*prec_factor

    # 5) Planting Prec Squared
    df['Planting Prec Squared'] = df['Planting Prec']**2

    # 6) Jul Aug Precipitation
    df['Jul Aug Prec'] = uw.extract_w_windows(w_df[['USA_Prec']], processed_data['jul_aug_interval'])*prec_factor

    # 7) Jul Aug Precipitation Squared
    df['Jul Aug Prec Squared'] = df['Jul Aug Prec']**2

    # 8) Precip Interaction = 'Planting Prec' * 'Jul Aug Prec'
    df['Prec Interaction'] = df['Planting Prec'] * df['Jul Aug Prec']

    # 9) Stress SDD - Based on 50% Silked Dates (What day was it when the crop was 50% silked)
    df['Pollination SDD'] = uw.extract_w_windows(w_df[['USA_Sdd30']], processed_data['pollination_interval'])*temp_factor

    # 10) Regular SDD: 20 Jun - 15 Sep
    df['Regular SDD'] = uw.extract_w_windows(w_df[['USA_Sdd30']], processed_data['regular_interval'])*temp_factor
    df['Regular SDD']=df['Regular SDD']-df['Pollination SDD']

    # 11) Constant
    df = sm.add_constant(df, has_constant='add')

    return df
    
def Fit_Model(df, y_col, exclude_from_year=GV.CUR_YEAR):
    df=df.loc[df.index<exclude_from_year]

    y_df = df[[y_col]]
    X_df=df.drop(columns = y_col)

    return sm.OLS(y_df, X_df).fit()




def Build_Prediction_DF(scope, raw_data, processed_data, instructions):
    """
    for predictions I need to:
        1) extend the variables:
                1.1) at "raw_data" level: Weather
                1.2) at "processed_data" level:    

        2) cut the all the rows before CUR_YEAR so that the calculation is fast:
             because I will need to extend every day and recalculate
    """
    raw_data_pred = raw_data.copy()

    w_all=instructions['WD_All']
    WD=instructions['WD']

    w_df = raw_data[w_all][WD]

    # Try to uncomment this one
    # w_df_pred = raw_data_pred[w_all][WD]

    date_start=dt(2022,6,16)
    date_end = w_df.index[-1] # this one to check well what to do
            
    days_pred= list(pd.date_range(date_start, date_end))
    
    for i, day in enumerate(days_pred):
        if (i==0):
            raw_data_pred[w_all][WD], dict_col_seas = uw.extend_with_seasonal_df(w_df.loc[:day], return_dict_col_seas=True)
        else:
            raw_data_pred[w_all][WD] = uw.extend_with_seasonal_df(w_df.loc[:day], input_dict_col_seas = dict_col_seas)

        w_df_pred = Build_Train_DF(scope, raw_data_pred, processed_data, instructions) # Take only the GV.CUR_YEAR row and append
        return w_df_pred







def main():

    return 0
    


if __name__=='__main__':    
    print('Corn_USA_Yield.py')