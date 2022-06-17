from datetime import datetime as dt
from typing_extensions import dataclass_transform

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

# corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']

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

def Get_Data(geo, w_vars, years):
    fo={'Years':years}    
    fo.update(Get_USDA_Data(years))
    fo.update(Get_Weather_Data(geo,w_vars,years))

    return fo

def Get_USDA_Data(years):
    fo={}

    fo['Yield']=qs.get_yields(years=years,cols_subset=['year','Value'])
    fo['Planting']=qs.get_progress(progress_var='planting', years=years, cols_subset=['week_ending','Value'])    
    fo['Silking'] =qs.get_progress(progress_var='silking',  years=years, cols_subset=['week_ending','Value'])

    return fo

def Get_Weather_Data(geo, w_vars, years):
    fo={}

    # Select Weather Variables
    in_files = GV.WS_UNIT_ALPHA
    out_cols = GV.WS_UNIT_ALPHA
    w_df_all = uw.build_w_df_all(geo, w_vars, in_files, out_cols)

    # Build the Weights
    corn_states=geo[GV.WS_STATE_ALPHA]

    weights = us.get_USA_prod_weights('CORN', 'STATE', years, corn_states)

    # Weighted Weather DataFrame All (All = 'hist', 'GFS', Etc Etc)
    fo['w_w_df_all']  = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

    return fo

def Build_Model_DF(raw_data):
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
    df=pd.DataFrame(raw_data['Years'], columns=['Trend'], index=raw_data['Years'])
        
    # 2) Yield  
    yields =  raw_data['Yield']['Value'].values
    if not (GV.CUR_YEAR in yields): 
        yields=np.append(yields, np.nan) # Because otherwise it cuts the GV.CUR_YEAR row
    df['Yield'] = yields    

    # 3) Percentage Planted as of 15th May
    df['Planted pct on May 15th']=us.progress_from_date(raw_data['Planting'], sel_date='2021-05-15')    

    # 4) Planting Precipitation - Based on 80% Planted Dates (What day was it when the crop was 80% planted)
    date_80_pct_planted=us.dates_from_progress(raw_data['Planting'], sel_percentage=80)

    # Planting: 80% planted -40 and +25 days
    start=date_80_pct_planted['date']+pd.DateOffset(-40)
    end = date_80_pct_planted['date']+pd.DateOffset(+25)
    planting_interval=pd.DataFrame({'start':start,'end':end})
    df['Planting Prec'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Prec']], planting_interval)*prec_factor

    # 5) Planting Prec Squared
    df['Planting Prec Squared'] = df['Planting Prec']**2

    # 6) Jul Aug Precipitation: 80% planted +26 and +105 days
    start=date_80_pct_planted['date']+pd.DateOffset(+26)
    end = date_80_pct_planted['date']+pd.DateOffset(105)
    jul_aug_interval=pd.DataFrame({'start':start,'end':end})
    df['Jul Aug Prec'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Prec']], jul_aug_interval)*prec_factor

    # 7) Jul Aug Precipitation Squared
    df['Jul Aug Prec Squared'] = df['Jul Aug Prec']**2
    
    # 8) Precip Interaction = 'Planting Prec' * 'Jul Aug Prec'
    df['Prec Interaction'] = df['Planting Prec'] * df['Jul Aug Prec']

    # 9) Stress SDD - Based on 50% Silked Dates (What day was it when the crop was 50% silked)
    date_50_pct_silked=us.dates_from_progress(raw_data['Silking'], sel_percentage=50)

    # Pollination: 50% planted -15 and +15 days
    start=date_50_pct_silked['date']+pd.DateOffset(-15)
    end = date_50_pct_silked['date']+pd.DateOffset(15)
    pollination_interval=pd.DataFrame({'start':start,'end':end})
    df['Pollination SDD'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Sdd30']], pollination_interval)*temp_factor

    # 10) Regular SDD: 20 Jun - 15 Sep
    start=[dt(y,6,20) for y in df.index]
    end=[dt(y,9,25) for y in df.index]
    regular_dates=pd.DataFrame({'start':start,'end':end},index=df.index)
    df['Regular SDD'] = uw.extract_w_windows(raw_data['w_w_df_all'][w_df][['USA_Sdd30']], regular_dates)*temp_factor
    df['Regular SDD']=df['Regular SDD']-df['Pollination SDD']

    # 11) Regular SDD: 20 Jun - 15 Sep    
    df = sm.add_constant(df, has_constant='add')
    return df
    
def Fit_Model(df, y_col, cur_year):
    # Removing the current year
    df=df.loc[df.index<GV.CUR_YEAR]

    y_df = df[[y_col]]
    X_df=df.drop(columns = y_col)

    return sm.OLS(y_df, X_df).fit()

def Build_Prediction_DF(raw_data, df, date_start):   
    df_pred = pd.DataFrame(columns= df.columns)
    date_end = raw_data['w_w_df_all'][GV.WD_H_GFS].index[-1]
    days=pd.date_range(date_start, date_end)

    return df_pred

def main():
    scope = Define_Scope()
    raw_data = Get_Data(scope['geo'], scope['w_vars'], scope['years'])
    df = Build_Model_DF(raw_data)
    model = Fit_Model(df,'Yield',GV.CUR_YEAR)
    return model
    


if __name__=='__main__':
    print('Corn_USA_Yield.py')