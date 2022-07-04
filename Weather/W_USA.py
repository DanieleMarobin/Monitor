import concurrent.futures

import APIs.NWS as nw
import Utilities.Utilities as uu
import Utilities.Weather as uw
import Utilities.GLOBAL as GV


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

def update_USA_weather(states = ['IL','IA'], start_date='1950-01-01', end_date='2023-12-31'):    
    # area_reduce='county_mean'
    area_reduce='state_mean'

    w_df_states=nw.get_states_weather(states,start_date,end_date,area_reduce=area_reduce)
    
    save_w_files(w_df_states)
    uu.log('Updated the States Averages')
    
    

