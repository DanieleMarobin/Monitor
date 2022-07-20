"""
This file relies on the library 'pygad' for the Genetic Algorithms calculations
Unfortunately there are certain functions that do not accept external inputs
so the only way to pass variables to them is to have some global variables
"""

import sys

import re
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

import warnings; warnings.filterwarnings("ignore")


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
        df = qs.get_yields('SOYBEANS', years=scope['years'],cols_subset=['year','Value'])
        df = df.rename(columns={'Value':'Yield'})
        df=df.set_index('year',drop=False)
        return df

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



def on_generation(ga_instance):    
    best_fitness = ga_instance.best_solution()[1]
    gen = ga_instance.generations_completed

    if best_fitness > dm_best['best_fitness']:
        m=dm_best['model'][-1]
        ga_instance.mutation_probability=0.5
        GA_pref['corr_threshold']=max(dm_best['corr'][-1], final_corr_threshold)
        GA_pref['p_values_threshold']=max(np.max(m.pvalues), final_p_values_threshold)

        print()
        elapsed = dt.now() - start_times['all']
        c = dm_best['corr'][-1]
        r = m.rsquared
        mae = dm_best['MAE'][-1]
        mape = dm_best['MAPE'][-1]
        
        print('=========================================================================================>','%.9f'% r,' - ', '%.9f'%best_fitness)
        print('Time:',dt.now(),'corr_threshold',GA_pref['corr_threshold'],'p_values_threshold',GA_pref['p_values_threshold'])

        print('Time:',dt.now(),'- Elapsed:', elapsed,'Solutions:',len(dm_best['model']),'Gen: ', gen)
        print( 'Fit', '%.5f'%best_fitness,'- Corr:','%.3f'%c,'- MAE:','%.3f'%mae,'- MAPE:','%.4f'%mape,'R-Sqr:','%.6f'% r)
        
        dm_best['best_fitness'] = best_fitness
        
        print();     

        coeff_2_digits = ['{:.2f}'.format(v) for v in m.params.values]
        equation=[list(zip(coeff_2_digits,m.params.index))]

        print('Equation',equation)        

        print(['%.8f' % x for x in m.pvalues])
        print()
        
        if len(dm_best['cv_p'])>0:
            n=len(dm_best['cv_p'][-1])
            n_p = np.sum(np.array(dm_best['cv_p'][-1]) > GA_pref['p_values_threshold'])
            nc=len(dm_best['cv_corr'][-1])
            n_c = np.sum(np.array(dm_best['cv_corr'][-1]) > GA_pref['corr_threshold'])
            
            p = np.mean(dm_best['cv_p'][-1])
            r = np.mean(dm_best['cv_r_sq'][-1])
            mae = np.mean(dm_best['cv_MAE'][-1])
            mape = np.mean(dm_best['cv_MAPE'][-1])
            corr = np.mean(dm_best['cv_corr'][-1])
            
            print('CV P-s:','%.5f'%p,'(',n_p,'/',n,') CV R-sq:','%.5f' %r,'CV MAE:','%.4f' %mae,'CV MAPE:','%.4f' %mape,'CV corr:','%.4f' %corr,'(',n_c,'/',nc,')')
            print('_____________________________________________________________________________________________________________')
            
        if ((GA_pref['p_values_threshold']<=final_p_values_threshold) and (GA_pref['corr_threshold']<=final_corr_threshold)):
            uu.serialize(dm_best,save_file,False)
                        
    if (gen % 1000 == 0): 
        elapsed = dt.now() - start_times['generation']
        print(dt.now(),'Completed Generation: ', gen,'in:', elapsed)
        start_times['generation'] = dt.now()

def fitness_func_cross_validation(solution, solution_idx):
    fitness=0
    cols=np.append(X_cols_fixed, model_cols[solution[solution>-1]])
    X_df = model_df[cols]    
        
    # Max correlation condition
    max_corr=0
    if X_df.shape[1]>1:
        max_corr = um.max_correlation(X_df, threshold=0.99)
        max_corr=np.max(max_corr[max_corr<1])     
        if max_corr > GA_pref['corr_threshold']: return fitness
        
    # Overall model creation
    X2_df = sm.add_constant(X_df)    
    stats_model = sm.OLS(y_df, X2_df).fit()

    # Overall P-Values condition
    non_sign = np.sum(stats_model.pvalues.values > GA_pref['p_values_threshold'])
    if non_sign > 0: return fitness
    
    # Min Coverage and No holes conditions
    wws = um.var_windows_from_cols(stats_model.params.index)
    cover = um.var_windows_coverage(wws)
    actual_cover =  len(cover[1])
    holes_cover =  len(cover[0])-actual_cover

    if (actual_cover < min_coverage) or (holes_cover>0): return fitness
    
    # Positive Precipitation condition
    pos_prec_mask = np.array([(stats_model.params[x]>0 and 'Prec' in x) for x in stats_model.params.index if '-' in x]) # [True, False, True, False] signaling which of the variables are positive Precipitations
    pos_prec_cover = len(um.var_windows_coverage(wws[pos_prec_mask])[1])
    if (pos_prec_cover == 0): return fitness
    
    # Negative Precipitation cover smaller than Positive Precipitation condition
    neg_prec_mask = np.array([(stats_model.params[x]<0 and 'Prec' in x) for x in stats_model.params.index if '-' in x])
    if len(neg_prec_mask)>0:
        neg_prec_cover = len(um.var_windows_coverage(wws[neg_prec_mask])[1])
        
        if (neg_prec_cover / pos_prec_cover) > 0.5: return fitness
        
    # Cross-Validation calculation
    cv_score = um.stats_model_cross_validate(X_df, y_df, folds)
    
    cv_p_mean=np.mean(cv_score['cv_p'])
    cv_r_sq_mean=np.mean(cv_score['cv_r_sq'])
    cv_MAPE_mean=np.mean(cv_score['cv_MAPE'])
    
    fitness = cv_r_sq_mean - cv_p_mean - cv_MAPE_mean

    if np.isnan(fitness): return 0            

    if fitness > dm_best['best_fitness']:            
        dm_best['model']+=[stats_model]
        dm_best['MAE']+=[mean_absolute_error(y_df, stats_model.predict(X2_df))]
        dm_best['MAPE']+=[mean_absolute_percentage_error(y_df, stats_model.predict(X2_df))]
        dm_best['fitness']+=[fitness]
        dm_best['corr']+=[max_corr]
                
        dm_best['cv_p']+=[cv_score['cv_p']]
        dm_best['cv_r_sq']+=[cv_score['cv_r_sq']]
        dm_best['cv_MAE']+=[cv_score['cv_MAE']]
        dm_best['cv_MAPE']+=[cv_score['cv_MAPE']]
        dm_best['cv_corr']+=[cv_score['cv_corr']]        
    return fitness

def GA_model_search(raw_data):
 
    # These below need to be global because inside the fitness function of the library    
    global y_df
    global model_cols
    global model_df
    global dm_best
    global folds
        
    model_df = pd.concat([raw_data['yield'],raw_data['multi_ww_df']], sort=True, axis=1, join='inner')
    model_df=model_df.dropna() # Needed because (maybe) the current year has some windows that have not started yet

    # Train Test Splits 
    min_train_size= min(10,len(model_df)-3)
    folds_expanding = TimeSeriesSplit(n_splits=len(model_df)-min_train_size, max_train_size=0, test_size=1)
    folds = []
    folds = folds + list(folds_expanding.split(model_df))

    y_df = model_df[[y_col]]
    model_cols=list(model_df.columns)
    model_cols.remove(y_col) # Remove "Y"
    for x in X_cols_fixed: model_cols.remove(x) # Remove " Fixed Columns "
    model_cols=np.array(model_cols)

    sel_n_variables = GA_n_variables - len(X_cols_fixed)

    gene_space=[]
    for i in range(sel_n_variables): 
        gene_space.append(range(-1, len(model_cols)))

    while True:    
        if (os.path.exists(save_file)):
            dm_best=uu.deserialize(save_file)
            dm_best['best_fitness']=0
        else:
            dm_best={'best_fitness':0,'model':[],'MAE':[],'MAPE':[],'fitness':[],'corr':[],
            'cv_p':[],'cv_r_sq':[],'cv_MAE':[],'cv_MAPE':[],'cv_corr':[]}

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
        GA_pref['p_values_threshold'] = 0.2
        GA_pref['corr_threshold'] = 0.9
        ga_instance.run()

# Global Variables to be used inside the 'pypgad' functions
if True:
    save_file= 'GA_soy'
    start_times={'all':dt.now(),'generation':dt.now()}

    # Preliminaries
    if True:
        multi_ww_dt_s=dt(2022,5,1)
        multi_ww_dt_e=dt(2022,9,1)

        multi_ww_freq_start='1D'
        multi_ww_freq_end='1D'

        multi_ww_ref_year_s=dt(2022,1,1)

    # Genetic Algorithm
    if True:
        y_col  ='Yield'
        X_cols_fixed = ['year']

        GA_pref={}

        GA_pref['p_values_threshold'] = 0.2 # 0.05
        GA_pref['corr_threshold'] = 0.9 # 0.4

        final_p_values_threshold=0.05
        final_corr_threshold=0.6

        min_coverage = 0.0 # 60 # in days
        
        GA_n_variables = 6
        fitness_func = fitness_func_cross_validation
                
        num_generations = 10000000000

        solutions_per_population = 10 # Number of solutions (i.e. chromosomes) within the population
        num_parents_mating = 4

        parent_selection_type='rank'    
        mutation_type='random'
        mutation_probability=1.0

        stop_criteria=["reach_1000000", "saturate_20000"]    

def main():
    scope = Define_Scope()
    raw_data = Get_Data_All_Parallel(scope)    
    raw_data['multi_ww_df']=um.generate_weather_windows_df(raw_data['w_w_df_all']['hist'], date_start=multi_ww_dt_s, date_end=multi_ww_dt_e, ref_year_start=multi_ww_ref_year_s, freq_start=multi_ww_freq_start, freq_end=multi_ww_freq_end)

    GA_model_search(raw_data)    

if __name__=='__main__':
    if False:
        main()
    else:
        rank_df=um.analyze_results(['GA_soy'])    
        uu.show_excel(rank_df)