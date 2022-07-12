"""
This file relies on the library 'pygad' for the Genetic Algorithms calculations
Unfortunately there are certain functions that do not accept external inputs
so the only way to pass variables to them is to have some global variables
"""

import sys;
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

import os
from datetime import datetime as dt
from copy import deepcopy
import concurrent.futures

import pandas as pd; pd.options.mode.chained_assignment = None
import numpy as np

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import pygad

import APIs.QuickStats as qs

import Utilities.SnD as us
import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Charts as uc
import Utilities.GLOBAL as GV
import Utilities.Utilities as uu


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
            # Picks the extension column and then just uses it till the end            
            raw_data_pred[w_all][WD], dict_col_seas = uw.extend_with_seasonal_df(w_df[w_df.index<=day], return_dict_col_seas=True, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates= keep_duplicates)
        else:
            raw_data_pred[w_all][WD] = uw.extend_with_seasonal_df(w_df[w_df.index<=day], input_dict_col_seas = dict_col_seas, var_mode_dict=ext_dict, ref_year_start=ref_year_start,keep_duplicates=keep_duplicates)
        
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



def on_generation(ga_instance):    
    best_fitness = ga_instance.best_solution()[1]
    gen = ga_instance.generations_completed

    if best_fitness > dm_best['best_fitness']:
        ga_instance.mutation_probability=0.5
        print()
        elapsed = dt.now() - start_times['all']
        c = dm_best['corr'][-1]
        r = dm_best['model'][-1].rsquared
        mae = dm_best['MAE'][-1]
        mape = dm_best['MAPE'][-1]
        
        print('=========================================================================================>','%.9f'% r,' - ', '%.9f'%best_fitness)
        print('Time:',dt.now(),'- Elapsed:', elapsed,'Solutions:',len(dm_best['model']),'Gen: ', gen)
        print( 'Fit', '%.5f'%best_fitness,'- Corr:','%.3f'%c,'- MAE:','%.3f'%mae,'- MAPE:','%.4f'%mape,'R-Sqr:','%.6f'% r)
        
        dm_best['best_fitness'] = best_fitness
        
        print();                
        print('  '.join(dm_best['model'][-1].params.index))        
        print(['%.8f' % x for x in dm_best['model'][-1].params])
        print(['%.8f' % x for x in dm_best['model'][-1].pvalues])
        print()
        
        if len(dm_best['cv_p_values'])>0:
            n=len(dm_best['cv_p_values'][-1])
            n_p = np.sum(np.array(dm_best['cv_p_values'][-1]) > p_values_threshold)
            nc=len(dm_best['cv_corr'][-1])
            n_c = np.sum(np.array(dm_best['cv_corr'][-1]) > corr_threshold)
            
            p = np.mean(dm_best['cv_p_values'][-1])
            r = np.mean(dm_best['cv_r_squared'][-1])
            mae = np.mean(dm_best['cv_MAE'][-1])
            mape = np.mean(dm_best['cv_MAPE'][-1])
            corr = np.mean(dm_best['cv_corr'][-1])
            
            print('CV P-s:','%.5f'%p,'(',n_p,'/',n,') CV R-sq:','%.5f' %r,\
                  'CV MAE:','%.4f' %mae,'CV MAPE:','%.4f' %mape,'CV corr:','%.4f' %corr,'(',n_c,'/',nc,')')
            print('_____________________________________________________________________________________________________________')
            
        if gen > 500:uu.serialize(dm_best,save_file,False)
                        
    if (gen % 1000 == 0): 
        elapsed = dt.now() - start_times['generation']
        print(dt.now(),'Completed Generation: ', gen,'in:',elapsed,'-',sel_state)
        start_times['generation'] = dt.now()

def fitness_func_cross_validation(solution, solution_idx):
    fitness=0
    cols=np.append(X_cols_fixed, model_cols[solution[solution>-1]])
    X_df = model_df[cols]    
        
    # Max correlation condition
    max_corr=0
    if X_df.shape[1]>1:
        max_corr = np.abs(np.corrcoef(X_df,rowvar=False))
        max_corr=np.max(max_corr[max_corr<1])     
        if max_corr > corr_threshold: return fitness
        
    # Overall model creation
    X2_df = sm.add_constant(X_df)    
    stats_model = sm.OLS(y_df, X2_df).fit()

    # P-Values condition
    non_sign = np.sum(stats_model.pvalues.values > p_values_threshold)
    if non_sign > 0: return fitness
    
    # Min Coverage
    wws = um.windows(stats_model.params.index)
    cover = um.windows_coverage(wws)
    actual_cover =  len(cover[1])
    holes_cover =  len(cover[0])-actual_cover

    if (actual_cover < min_coverage) or (holes_cover>0): return fitness
    
    # Negative and Positive Precipitation Masks        
    neg_prec = np.array([(stats_model.params[x]<0 and 'Prec' in x) for x in stats_model.params.index if '-' in x])
    if len(neg_prec)>0:
        pos_prec = ~neg_prec    
        neg_prec_cover =  len(um.windows_coverage(wws[neg_prec])[1])
        pos_prec_cover =  len(um.windows_coverage(wws[pos_prec])[1])
        if (pos_prec_cover>0) and (neg_prec_cover / pos_prec_cover) > 0.5: return fitness                    
        
    # Cross-Validation calculation
    cv_score = um.stats_model_cross_validate(X_df, y_df, folds)
    
    p_values_mean=np.mean(cv_score['p_values'])
    r_squared_mean=np.mean(cv_score['r_squared'])
    MAPE_mean=np.mean(cv_score['MAPE'])
    corr_mean=np.mean(cv_score['corr'])    
    
    fitness = r_squared_mean - p_values_mean - MAPE_mean

    if np.isnan(fitness): return 0            

    if fitness > dm_best['best_fitness']:            
        dm_best['model']+=[stats_model]
        dm_best['MAE']+=[mean_absolute_error(y_df, stats_model.predict(X2_df))]
        dm_best['MAPE']+=[mean_absolute_percentage_error(y_df, stats_model.predict(X2_df))]
        dm_best['fitness']+=[fitness]
        dm_best['corr']+=[max_corr]
                
        dm_best['cv_p_values']+=[cv_score['p_values']]
        dm_best['cv_r_squared']+=[cv_score['r_squared']]
        dm_best['cv_MAE']+=[cv_score['MAE']]
        dm_best['cv_MAPE']+=[cv_score['MAPE']]
        dm_best['cv_corr']+=[cv_score['corr']]        
    return fitness


def GA_model_search(args):
    # Main selections ----------------------------------------------------------------------------
    period_start= dt(2021,2,1)
    period_end =  dt(2021,6,30)
    
    global p_values_threshold; p_values_threshold = 0.05
    global corr_threshold; corr_threshold = 0.4
    global min_coverage; min_coverage = 60 # in days
    
    y_col  ='Yield'
    global X_cols_fixed; X_cols_fixed = ['year'] # ['year']  

    # Genetic Algorithm
    GA_n_variables = 4
    fitness_func = fitness_func_cross_validation
          
    num_generations = 10000000000

    solutions_per_population = 10 # Number of solutions (i.e. chromosomes) within the population
    num_parents_mating = 4

    parent_selection_type='rank'    
    mutation_type='random'
    mutation_probability=1.0

    stop_criteria=["reach_1000000", "saturate_20000"]    
    # ------------------------------------------------------------------------------------------
    
    sel_letters=args[0]
    
    # These below need to be global because inside the fitness function of the library    
    global y_df
    global model_cols
    global model_df
    global dm_best
    global folds
    global start_times
    global sel_state                      
    
    yield_trend_df=1 # WIP # get_yield_df(sel_letters)
    w_df=1 # WIP # weighted_w_df(w_df=all_w_df['hist'],weights=weights,w_vars=w_vars, output_column=sel_state)                
        
    massive_df=um.generate_weather_windows_df(w_df,period_start,period_end)

    model_df = pd.concat([yield_trend_df,massive_df], sort=True, axis=1, join='inner')
    model_df=model_df.dropna()

    print(dt.now(), 'Calculated Model Dataframe')
    print(model_df.shape)

    # Train Test Splits 
    min_train_size= min(10,len(model_df)-3); 
    folds_expanding = TimeSeriesSplit(n_splits=len(model_df)-min_train_size, max_train_size=0, test_size=1)
    folds = []; folds = folds + list(folds_expanding.split(model_df))

    # -------------------------------------------------------------------------------------------------
    y_df = model_df[[y_col]]

    model_cols=list(model_df.columns)
    model_cols.remove(y_col) # Remove "Y"
    for x in X_cols_fixed: model_cols.remove(x) # Remove " Fixed Columns "
    model_cols=np.array(model_cols)

    sel_n_variables = GA_n_variables - len(X_cols_fixed)

    cols_n = len(model_cols)
    gene_space=[]
    for i in range(sel_n_variables): gene_space.append(range(-1, cols_n))

    print(dt.now(), 'Set up the Genetic Algorithm')

    while True:    
        if (os.path.exists(save_file)):
            dm_best=uu.deserialize(save_file)
            dm_best['best_fitness']=0
        else:
            dm_best={'best_fitness':0,'model':[],'MAE':[],'MAPE':[],'fitness':[],'corr':[],'cv_p_values':[],
                     'cv_r_squared':[],'cv_MAE':[],'cv_MAPE':[],'cv_corr':[]}

        start_times={'all':dt.now(),'generation':dt.now()}

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_genes=sel_n_variables,
                               sol_per_pop=solutions_per_population,

                               num_parents_mating=num_parents_mating,
                               parent_selection_type=parent_selection_type,                       

                               mutation_type=mutation_type,
                               mutation_probability=mutation_probability,                       

                               fitness_func=fitness_func,

                               gene_type=int,
                               gene_space=gene_space,                       
                               allow_duplicate_genes=False,

                               on_generation=on_generation,
                               stop_criteria=stop_criteria)


        print('******************************** Start a new Run ********************************')
        dm_best['best_fitness']=0
        ga_instance.run()    


# Global Variables to be used inside the 'pypgad' functions
save_file= 'daniele'

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