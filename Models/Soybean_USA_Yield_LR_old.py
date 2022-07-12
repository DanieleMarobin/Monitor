import sys;
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

from datetime import datetime as dt
from copy import deepcopy
import concurrent.futures

import pandas as pd; pd.options.mode.chained_assignment = None
import numpy as np
import statsmodels.api as sm

import APIs.QuickStats as qs

import Utilities.SnD as us
import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Charts as uc
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

def Get_Data_Single(scope: dict, var: str = 'yield', fo = {}):
    
    if (var=='yield'):
        return qs.get_yields('SOYBEANS', years=scope['years'],cols_subset=['year','Value'])

    elif (var=='weights'):
        return us.get_USA_prod_weights('SOYBEANS', 'STATE', scope['years'], fo['locations'])

    elif (var=='planting_progress'):
        return qs.get_progress('SOYBEANS',progress_var='planting', years=scope['years'], cols_subset=['week_ending','Value'])

    elif (var=='blooming_progress'):
        return qs.get_progress('SOYBEANS',progress_var='blooming',  years=scope['years'], cols_subset=['week_ending','Value'])

    elif (var=='w_df_all'):
        return uw.build_w_df_all(scope['geo_df'], scope['w_vars'], scope['geo_input_file'], scope['geo_output_column'])

    elif (var=='w_w_df_all'):
        # For this one to work, it is obvious that both:
        #       - "fo['w_df_all']" and 
        #       - "fo['weights']"
        #  need to be passed in the input "fo = {}"
        return uw.weighted_w_df_all(fo['w_df_all'], fo['weights'], output_column='USA')

    return fo

def Get_Data_All_Parallel(scope):
    # https://towardsdatascience.com/multi-tasking-in-python-speed-up-your-program-10x-by-executing-things-simultaneously-4b4fc7ee71e

    fo={}

    # Time
    fo['years']=scope['years']

    # Space
    fo['locations']=scope['geo_df'][GV.WS_STATE_ALPHA]

    download_list=['yield','weights','planting_progress','blooming_progress','w_df_all']
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results={}
        for variable in download_list:
            results[variable] = executor.submit(Get_Data_Single, scope, variable, fo)
    
    for var, res in results.items():
        fo[var]=res.result()
    
    # Weighted Weather: it is here because it needs to wait for the 2 main in ingredients (1) fo['w_df_all'], (2) fo['weights'] to be calculated first
    variable = 'w_w_df_all'
    fo[variable]  = Get_Data_Single(scope, variable, fo)

    return fo



def Milestone_from_Progress(raw_data):
    """
    Process data like calculating weather intervals, or other useful info or stats
        - when they are needed for downstream calcs
        - when they are useful to be plotted or tabulated
    """

    fo={}

    # 50% silked
    fo['50_pct_bloomed']=us.dates_from_progress(raw_data['blooming_progress'], sel_percentage=50)

    # I define 'fix_milestone' to be able to fill it in the 'Intervals_from_Milestones' function
    fo['fix_milestone']=pd.DataFrame(columns=['date'], index=raw_data['years'])

    # To check for planting pct
    fo['10th_June_pct_planted']=us.progress_from_date(raw_data['planting_progress'], progress_date=dt(GV.CUR_YEAR,6,10))
    return fo

def Extend_Milestones(milestones, simulation_day, year_to_ext = GV.CUR_YEAR):
    fo={}
    m_copy=deepcopy(milestones)

    # Fix milestone
    fo['fix_milestone']=m_copy['fix_milestone']

    # 50% bloomed
    fo['50_pct_bloomed']=us.extend_date_progress(m_copy['50_pct_bloomed'],day=simulation_day, year= year_to_ext)

    # To check for planting pct
    fo['10th_June_pct_planted']=us.extend_progress(m_copy['10th_June_pct_planted'],progress_date=dt(GV.CUR_YEAR,6,10), day=simulation_day)
    return fo



def Intervals_from_Milestones(milestones):
    fo={}

    # Planting Interval: 10 May - 10 Jul
    start=[dt(y,5,10) for y in milestones['fix_milestone'].index]
    end=  [dt(y,7,10) for y in milestones['fix_milestone'].index]
    fo['planting_interval']=pd.DataFrame({'start':start,'end':end}, index=milestones['fix_milestone'].index)

    # Jul Aug Interval: 80% planted +26 and +105 days
    start=[dt(y,7,11) for y in milestones['fix_milestone'].index]
    end=  [dt(y,9,15) for y in milestones['fix_milestone'].index]
    fo['jul_aug_interval']=pd.DataFrame({'start':start,'end':end}, index=milestones['fix_milestone'].index)

    # Pollination Interval: 50% bloomed -10 and +10 days
    start=milestones['50_pct_bloomed']['date']+pd.DateOffset(-10)
    end = milestones['50_pct_bloomed']['date']+pd.DateOffset(10)
    fo['pollination_interval']=pd.DataFrame({'start':start,'end':end})

    # Regular Interval: 25 Jun - 15 Sep
    start=[dt(y,6,25) for y in milestones['fix_milestone'].index]
    end=  [dt(y,9,15) for y in milestones['fix_milestone'].index]
    fo['regular_interval']=pd.DataFrame({'start':start,'end':end}, index=milestones['fix_milestone'].index)

    return fo

def Build_DF(raw_data, milestones, intervals, instructions):
    """
    The model DataFrame has 11 Columns:
            1) Yield (y)
            8) Variables
            1) Constant (added to be able to fit the model with 'statsmodels.api')

            1+8+1 = 10 Columns
    """

    w_all=instructions['WD_All'] # 'simple'->'w_df_all', 'weighted'->'w_w_df_all'
    WD=instructions['WD']
    w_df = raw_data[w_all][WD]
    
    prec_factor = instructions['prec_factor']
    temp_factor = instructions['temp_factor']

    # 1) Trend (first because I set the index and because it surely includes CUR_YEAR, while other variable might not have any value yet)
    df=pd.DataFrame(raw_data['years'], columns=['Trend'], index=raw_data['years'])
        
    # 2) Yield
    yields =  raw_data['yield']['Value'].values
    if not (GV.CUR_YEAR in yields): yields=np.append(yields, np.nan) # Because otherwise it cuts the GV.CUR_YEAR row
    df['Yield'] = yields    

    # 3) Percentage Planted as of 10th June
    df['Planted pct on Jun 10th']=milestones['10th_June_pct_planted']

    # 4) Planting Precipitation: 10 May - 10 Jul
    df['Planting Prec'] = uw.extract_w_windows(w_df[['USA_Prec']], intervals['planting_interval'])*prec_factor

    # 5) Planting Prec Squared
    df['Planting Prec Squared'] = df['Planting Prec']**2

    # 6) Jul Aug Precipitation
    df['Jul Aug Prec'] = uw.extract_w_windows(w_df[['USA_Prec']], intervals['jul_aug_interval'])*prec_factor

    # 7) Jul Aug Precipitation Squared
    df['Jul Aug Prec Squared'] = df['Jul Aug Prec']**2

    # 8) Stress SDD - Based on 50% Silked Dates (What day was it when the crop was 50% bloomed)
    df['Pollination SDD'] = uw.extract_w_windows(w_df[['USA_Sdd30']], intervals['pollination_interval'])*temp_factor

    # 9) Regular SDD: 25 Jun - 15 Sep
    df['Regular SDD'] = uw.extract_w_windows(w_df[['USA_Sdd30']], intervals['regular_interval'])*temp_factor
    df['Regular SDD']=df['Regular SDD']-df['Pollination SDD']

    # 10) Constant
    df = sm.add_constant(df, has_constant='add')

    return df
    
def Build_Pred_DF(raw_data, milestones, instructions, year_to_ext = GV.CUR_YEAR,  date_start=dt.today(), date_end=None, trend_yield_case= False):
    """
    for predictions I need to:
        1) extend the variables:
                1.1) Weather
                1.2) All the Milestones
                1.3) Recalculate the Intervals (as a consequence of the Milestones shifting)

        2) cut the all the rows before CUR_YEAR so that the calculation is fast:
             because I will need to extend every day and recalculate
    """
    
    dfs = []
    w_all=instructions['WD_All']
    WD=instructions['WD']
    ext_dict = instructions['ext_mode']
    ref_year_start=dt(2022,1,1)


    # print('---> Prediction Dataset {0}, {1}, Mode: {2}'.format(w_all,WD,ext_dict)); print('')

    raw_data_pred = deepcopy(raw_data)
    w_df = raw_data[w_all][WD]
    
    if (date_end==None): date_end = w_df.index[-1] # this one to check well what to do
    days_pred= list(pd.date_range(date_start, date_end))


    for i, day in enumerate(days_pred):
        if trend_yield_case:
            keep_duplicates='last'
            extend_milestones_day=days_pred[0]
        else:
            keep_duplicates='first'
            extend_milestones_day=days_pred[i]


        # Extending the Weather        
        if (i==0):
            # Picks the analog on the first day (ex: Jun 1st), and then just uses it till the end            
            # raw_data_pred[w_all][WD], dict_col_seas = uw.extend_with_seasonal_df(w_df, return_dict_col_seas=True, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates= keep_duplicates) # good
            raw_data_pred[w_all][WD], dict_col_seas = uw.extend_with_seasonal_df(w_df[w_df.index<=day], return_dict_col_seas=True, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates= keep_duplicates)
            # raw_data_pred[w_all][WD], dict_col_seas = uw.extend_with_seasonal_df(w_df.loc[:day], return_dict_col_seas=True, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates= keep_duplicates) # good
        else:
            # raw_data_pred[w_all][WD] = uw.extend_with_seasonal_df(w_df, input_dict_col_seas = dict_col_seas, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates=keep_duplicates) # good
            raw_data_pred[w_all][WD] = uw.extend_with_seasonal_df(w_df[w_df.index<=day], input_dict_col_seas = dict_col_seas, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates=keep_duplicates)
            # raw_data_pred[w_all][WD] = uw.extend_with_seasonal_df(w_df.loc[:day], input_dict_col_seas = dict_col_seas, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates=keep_duplicates) # good
        
        # Extending the Milestones
        milestones_pred = Extend_Milestones(milestones, extend_milestones_day)

        # Calculate the intervals
        intervals_pred = Intervals_from_Milestones(milestones_pred)

        # Keep only the selected year to speed up the calculations
        for i in intervals_pred: 
            intervals_pred[i] = intervals_pred[i].loc[year_to_ext:year_to_ext]
            end=min(day,intervals_pred[i].loc[year_to_ext]['end'])
            intervals_pred[i].loc[year_to_ext,'end']=end


        # Build the 'Simulation' DF
        w_df_pred = Build_DF(raw_data_pred, milestones_pred, intervals_pred, instructions) # Take only the GV.CUR_YEAR row and append

        # Append row to the final matrix (to pass all at once for the daily predictions)
        dfs.append(w_df_pred.loc[year_to_ext:year_to_ext])
    
    fo = pd.concat(dfs)

    fo.index= days_pred.copy()

    return fo


def add_chart_intervals(chart, intervals):
    sel_intervals = [intervals['planting_interval'], intervals['jul_aug_interval'], intervals['regular_interval'], intervals['pollination_interval']]
    text = ['Planting', 'Growing Prec', 'Growing Temp', 'Pollination']
    position=['top left','top left','bottom left','bottom left']
    color=['blue','green','orange','red']

    uc.add_interval_on_chart(chart,sel_intervals,GV.CUR_YEAR,text,position,color)


def Scenario_Calc(prec_units,temp_units,sce_dict,raw_data,milestones,sce_date,model):
    pred_DF_instr=um.Build_DF_Instructions('weighted',GV.WD_H_GFS, prec_units=prec_units, temp_units=temp_units, ext_mode = sce_dict)
    pred_df = Build_Pred_DF(raw_data,milestones,pred_DF_instr,GV.CUR_YEAR,date_start=sce_date, date_end=sce_date)
    yields = model.predict(pred_df[model.params.index]).values
    return yields[0]

def Analog_scenarios():
    # Declarations
    sce_start = 1985
    sce_end = 2022    
    prec_units='in'
    temp_units='F'

    years = range(sce_start,sce_end)
    scope = Define_Scope()
    raw_data = Get_Data_All_Parallel(scope)

    milestones = Milestone_from_Progress(raw_data)
    intervals = Intervals_from_Milestones(milestones)

    train_DF_instr = um.Build_DF_Instructions('weighted',GV.WD_HIST, prec_units=prec_units, temp_units=temp_units, ext_mode = GV.EXT_DICT)
    train_df = Build_DF(raw_data, milestones, intervals, train_DF_instr)
    model = um.Fit_Model(train_df,'Yield',GV.CUR_YEAR)

    # Scenarios Initialization
    sce_date = raw_data['w_w_df_all'][GV.WD_GFS].last_valid_index()
    scenarios_EXT_dict = {}
    results = {}

    for p in years:
        for t in years:
            scenarios_EXT_dict[str(p)+'_'+str(t)]={GV.WV_PREC:GV.EXT_ANALOG+'_'+str(p),GV.WV_SDD_30:GV.EXT_ANALOG+'_'+str(t)}

    print('Scenarios to compute:', len(scenarios_EXT_dict))

    # Scenarios Calculation    
    # i=0
    # for sce_name, sce_dict in scenarios_EXT_dict.items():
    #     i+=1
    #     if i%100==0: print(i)
    #     results[sce_name]=Scenario_Calc(prec_units,temp_units,sce_dict,raw_data,milestones,sce_date,model)

    fo={}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results={}
        for sce_name, sce_dict in scenarios_EXT_dict.items():
            results[sce_name] = executor.submit(Scenario_Calc, prec_units, temp_units, sce_dict, raw_data, milestones, sce_date,model)
    
    for var, res in results.items():
        fo[var]=res.result()

    print('All Done')
    return fo



def main():
    scope = Define_Scope()

    raw_data = Get_Data_All_Parallel(scope)
    milestones =Milestone_from_Progress(raw_data)
    intervals = Intervals_from_Milestones(milestones)

    train_DF_instr = um.Build_DF_Instructions('weighted',GV.WD_HIST, prec_units='in', temp_units='F',ext_mode = GV.EXT_DICT)
    train_df = Build_DF(raw_data, milestones, intervals, train_DF_instr)
    model = um.Fit_Model(train_df,'Yield',GV.CUR_YEAR)
    print(model.summary())

    season_end = dt(2022,10,1)
    pred_DF_instr=um.Build_DF_Instructions('weighted',GV.WD_H_GFS, prec_units='in', temp_units='F',ext_mode = GV.EXT_DICT)
    pred_df = Build_Pred_DF(raw_data, milestones, pred_DF_instr, GV.CUR_YEAR, date_start=season_end, date_end=season_end)
    print('_____________________ pred_df _____________________')
    print(pred_df)

    yields = model.predict(pred_df[model.params.index]).values    
    print('Final Yield (end of season):', yields[-1])
    print('All Done')

if __name__=='__main__':
    main()