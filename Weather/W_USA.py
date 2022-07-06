import pandas as pd
import concurrent.futures

import APIs.NWS as nw
import APIs.Bloomberg as ba

import Utilities.Utilities as uu
import Utilities.Weather as uw
import Utilities.GLOBAL as GV

from datetime import datetime as dt

def parallel_save(file, df):
    df.columns=['value']
    df.to_csv(file)

def save_w_files(NWS_grid_results):
    file_df_dict={}
    for state, value in NWS_grid_results.items():
        for county, df in value.items():
            for c in df.columns:                
                save_file = GV.W_DIR + str(county)+'_'+c+'_hist.csv'
                file_df_dict[save_file]=df[[c]]
                # df[[c]].to_csv(save_file)

    uu.log('Files to save: '+str(len(file_df_dict)))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for file, df in file_df_dict.items():
            executor.submit(parallel_save, file, df)

    uu.log('All Saved')

def update_NWS_hist_weather(states = ['IL','IA'], start_date='1950-01-01', end_date='2023-12-31'):    
    # area_reduce='county_mean'
    area_reduce='state_mean'

    w_df_states=nw.get_states_weather(states,start_date,end_date,area_reduce=area_reduce)
    
    save_w_files(w_df_states)
    uu.log('Updated the States Averages')

def udpate_USA_Bloomberg(run, states = ['IL','IA'],  model = 'GFS', model_type = 'DETERMINISTIC'):
    """
    model = 'GFS', 'ECMWF'
    model_type = 'DETERMINISTIC', 'ENSEMBLE_MEAN'
    """
       
    blp = ba.BLPInterface('//blp/exrsvc')
    run_str = run.strftime("%Y-%m-%dT%H:%M:%S")

    suffix=''
    if model_type=='ENSEMBLE_MEAN': suffix='En'

    for s in states:        
        location = 'US_'+ s
        
        overrides = {'location': location, 'fields':'TEMPERATURE|PRECIPITATION', 'model':model,'publication_date':run_str,'location_time':True,'type':model_type}

        df_GFS = blp.bsrch('comdty:weather', overrides)
        df_GFS['Location Time'] =  pd.to_datetime(df_GFS['Location Time'])
        df_GFS['date']=df_GFS['Location Time'].dt.date
        df_GFS['Precipitation (mm)'].iloc[0]=0

        # Prec
        file_name = GV.W_DIR+ s+'_Prec_'+ model.lower()+ suffix +'.csv'
        df = df_GFS.groupby('date')[['Precipitation (mm)']].sum()
        df=df.rename(columns={'Precipitation (mm)': 'value'})
        df.to_csv(file_name)
        print('Saved', file_name)

        # TempMax
        file_name = GV.W_DIR+ s+'_TempMax_'+ model.lower()+ suffix + '.csv'
        df = df_GFS.groupby('date')[['Temperature (°C)']].max()
        df=df.rename(columns={'Temperature (°C)': 'value'})
        df.to_csv(file_name)
        print('Saved', file_name)