import pandas as pd
import numpy as np
import statsmodels.api as sm

import APIs.QuickStats as qs

import Utilities.SnD as us
import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Charts as uc
import Utilities.Utilities as uu
import Utilities.GLOBAL as gv

from datetime import datetime as dt


# declarations
corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
years=range(1985,2023)

days=[]
yields=[]

def Corn_USA_Yield():
    # Yield    
    df_yield=qs.get_QS_yields(years=years,cols_subset=['year','Value'])
    M_yield=df_yield.set_index('year')

    # Progress
    planting_df=qs.get_QS_progress(progress_var='planting', years=years, cols_subset=['week_ending','Value'])
    silking_df=qs.get_QS_progress(progress_var='silking',years=years,cols_subset=['week_ending','Value'])

    # Progress as of 15th May (Yifu calls it "Planting Date" in his file)
    M_plant_on_May15=us.progress_from_date(planting_df, sel_date='2021-05-15')

    # Select the Weather Stations
    df_w_sel = uw.get_w_sel_df()
    df_w_sel = df_w_sel[df_w_sel[gv.WS_COUNTRY_ALPHA] == 'USA']

    # Build the Weather DF
    sel_w_vars = [gv.WV_PREC, gv.WV_TEMP_MAX, gv.WV_SDD_30]
    in_files = gv.WS_UNIT_ALPHA
    out_cols = gv.WS_UNIT_ALPHA
    w_df_all = uw.build_w_df_all(df_w_sel, sel_w_vars, in_files, out_cols)

    # Build the Weights
    weights = us.get_USA_prod_weights('CORN', 'STATE', years, corn_states)

    # Weighted DataFrame
    w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

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
    df_to_ext =  w_w_df_all[gv.WD_H_GFS] # Extending the GFS
    # df_to_ext =  w_w_df_all[gv.WD_H_ECMWF] # Extending the ECMWF

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
    # silk_50_pct_CUR_YEAR=pd.Series([dt(gv.CUR_YEAR,d.month,d.day) for d in silk_50_pct['date']]) # Adding current estimate for silking dates
    # silk_50_pct.loc[gv.CUR_YEAR]= np.mean(silk_50_pct_CUR_YEAR)
    return 0