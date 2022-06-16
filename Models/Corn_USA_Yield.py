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

# corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
days=[]
yields=[]


def Define_Scope():
    fo={}

    # Geography    
    geo = uw.get_w_sel_df()
    fo['geo'] = geo[geo[GV.WS_COUNTRY_ALPHA] == 'USA']

    # Weather Variables
    fo['w_vars'] = [GV.WV_PREC, GV.WV_TEMP_MAX, GV.WV_SDD_30]

    # Time
    fo['years']=range(1985,2023)

    return fo

def Get_Data(geo, w_vars, years):

    fo=Get_USDA_Data(years)
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
    weights = us.get_USA_prod_weights('CORN', 'STATE', years)

    # Weighted Weather DataFrame All (All = 'hist', 'GFS', Etc Etc)
    fo['w_w_df_all']  = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

    return fo

def Build_Model_DF(Raw_Data):
    """
    The model DataFrame has 11 Columns:
            1) Yield (y)
            9) Variables ()
            1) Constant (added to be able to fit the model with 'statsmodels.api')

            1+9+1 = 11 Columns
    """
    df={}
            

    return df

def Fit_Model():
    return 0

def Build_Prediction_DF():
    return 0


def main():
    Scope = Define_Scope()
    Raw_Data = Get_Data(Scope['geo'], Scope['w_vars'], Scope['years'])
    df = Build_Model_DF(Raw_Data)


def Corn_USA_Yield_Build_Model(data):


    # Yield    
    M_yield=data['Yield'].set_index('year')

    # 1) Percentage Planted as of 15th May
    M_plant_on_May15=us.progress_from_date(data['Planting'], sel_date='2021-05-15')



def Corn_USA_Yield(data):
    """
    The model DataFrame has 11 Columns:
            1) Yield (y)
            9) Variables ()
            1) Constant (added to be able to fit the model with 'statsmodels.api')

            1+9+1 = 11 Columns
    """



    # 80% Planted Dates (What day was it when the crop was 80% planted)
    date_80_pct_planted=us.dates_from_progress(planting_df, sel_percentage=80)

    # Planting dates are 80% planted -40 and +25 days
    start=date_80_pct_planted['date']+pd.DateOffset(-40)
    end = date_80_pct_planted['date']+pd.DateOffset(+25)
    planting_dates=pd.DataFrame({'start':start,'end':end})

    # DATES: jul_aug are 80% planted +26 and +105 days
    start=date_80_pct_planted['date']+pd.DateOffset(+26)
    end = date_80_pct_planted['date']+pd.DateOffset(105)
    jul_aug_dates=pd.DataFrame({'start':start,'end':end})
    
    # 50% Silked Dates (What day was it when the crop was 50% silked)
    date_50_pct_silked=us.dates_from_progress(silking_df, sel_percentage=50)

    # Pollination dates are 50% planted -15 and +15 days
    start=date_50_pct_silked['date']+pd.DateOffset(-15)
    end = date_50_pct_silked['date']+pd.DateOffset(15)
    pollination_dates=pd.DataFrame({'start':start,'end':end})

    # Regular SDD dates are 20 Jun to 15 Sep
    start=[dt(y,6,20) for y in years]
    end=[dt(y,9,25) for y in years]
    regular_dates=pd.DataFrame({'start':start,'end':end},index=years)

    

    # ------------------------------------- CHOOSE which FORECAST to EXTEND -------------------------------------
    # select which dataframe to extend
    df_to_ext =  w_w_df_all[GV.WD_H_GFS] # Extending the GFS
    # df_to_ext =  w_w_df_all[GV.WD_H_ECMWF] # Extending the ECMWF

    w_df = df_to_ext.copy()

    # -------------------------------- 9 Variables --------------------------------
    # Trend                                                                             # 1
    # M_plant_on_May15                                                                  # 2
    M_jul_aug_prec = uw.extract_w_windows(w_df[['USA_Prec']],jul_aug_dates)             # 3
    # M_jul_aug_prec SQ                                                                 # 4
    M_planting_prec = uw.extract_w_windows(w_df[['USA_Prec']],planting_dates)           # 5
    # M_planting_prec                                                                   # 6
    M_pollination_sdd = uw.extract_w_windows(w_df[['USA_Sdd30']], pollination_dates)    # 7
    M_regular_sdd = uw.extract_w_windows(w_df[['USA_Sdd30']], regular_dates)            # 8
    # Precip_Interaction                                                                # 9

    # Combining the 2 SDD columns
    M_sdd = pd.concat([M_pollination_sdd, M_regular_sdd],axis=1)
    M_sdd.columns=['Pollination_SDD','Regular_SDD']
    M_sdd['Regular_SDD']=M_sdd['Regular_SDD']-M_sdd['Pollination_SDD']

    cols_names = ['Yield','Plant_Progr_May15','Jul_Aug_Prec','Pollination_SDD','Regular_SDD', 'Planting_Prec']

    M_df=[M_yield, M_plant_on_May15, M_jul_aug_prec/25.4, M_sdd*9/5, M_planting_prec/25.4]

    M_df=pd.concat(M_df,axis=1)
    M_df.columns=cols_names

    M_df['Trend']=M_df.index

    M_df['Jul_Aug_Prec_Sq']=M_df['Jul_Aug_Prec']**2 # Sq
    M_df['Planting_Prec_Sq']=M_df['Planting_Prec']**2 # Sq
    M_df['Precip_Interaction']=M_df['Planting_Prec']*M_df['Jul_Aug_Prec']

    y_col='Yield'
    df=M_df.dropna()

    y_df = df[[y_col]]
    X_df=df.drop(columns = y_col)
    X2_df = sm.add_constant(X_df)
    
    stats_model = sm.OLS(y_df, X2_df).fit()



def predict_something_something():
    # silk_50_pct_CUR_YEAR=pd.Series([dt(GV.CUR_YEAR,d.month,d.day) for d in silk_50_pct['date']]) # Adding current estimate for silking dates
    # silk_50_pct.loc[GV.CUR_YEAR]= np.mean(silk_50_pct_CUR_YEAR)
    return 0



if __name__=='__main__':
    print('Corn_USA_Yield.py')