import sys;
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

import os
from datetime import datetime as dt

import Weather.W_USA as wu

import APIs.Geosys as ge
import APIs.Bloomberg as ba

import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.Utilities as uu
import Utilities.GLOBAL as GV

def update_weather(download_hist=False, download_geosys=False, download_bloomberg=False):
    """
    model = 'GFS', 'ECMWF'
    model_type = 'DETERMINISTIC', 'ENSEMBLE_MEAN'
    """

    run_gfs=        dt(2022,7,7,6,0,0)
    run_gfs_en=     dt(2022,7,7,6,0,0)

    run_ecmwf=      dt(2022,7,7,6,0,0)
    run_ecmwf_en=   dt(2022,7,7,0,0,0)

    states=['IA','IL','IN','OH','MO','MN','SD','NE']

    if download_hist:
        uu.log('USA NWS Historical Weather')
        wu.update_NWS_hist_weather(states = states, start_date='1985-01-01', end_date='2023-12-31')

    if download_geosys:
        uu.log('Geosys Weather')
        ge.update_Geosys_Weather()

    if download_bloomberg:
        # runs_df=ba.latest_weather_run_df()

        model='GFS'
        uu.log('USA Bloomberg GFS Operational ----------------------------------------')
        wu.udpate_USA_Bloomberg(run_gfs, states, model=model, model_type='DETERMINISTIC')
        uu.log('USA Bloomberg GFS Ensemble ----------------------------------------')
        wu.udpate_USA_Bloomberg(run_gfs_en, states, model=model, model_type='ENSEMBLE_MEAN')        
    
        model='ECMWF'
        uu.log('USA Bloomberg ECMWF Operational ----------------------------------------')
        wu.udpate_USA_Bloomberg(run_ecmwf, states, model=model, model_type='DETERMINISTIC')
        uu.log('USA Bloomberg ECMWF Ensemble ----------------------------------------')
        wu.udpate_USA_Bloomberg(run_ecmwf_en, states, model=model, model_type='ENSEMBLE_MEAN')        

    uu.log('-----------------------------------------------------------')
    print('Done With the Weather Download')

def hello_world_seas_chart():
    states=['IA','IL']
    # states=['IA','IL','IN','OH','MO','MN','SD','NE']
    # Get the Weather Selection File
    sel_df = uw.get_w_sel_df()

    # Select the State/Unit to chart ('IL' and 'IA' for this example)
    # ['IA','IL','IN','OH','MO','MN','SD','NE']
    sel_df=sel_df[sel_df['state_alpha'].isin(states)]

    # Build all the Weather Datasets
    w_df_all = uw.build_w_df_all(sel_df,w_vars=[GV.WV_TEMP_MAX,GV.WV_PREC], in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)
    # w_df_all = uw.build_w_df_all(sel_df,w_vars=[GV.WV_PREC], in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)
    # w_df_all = uw.build_w_df_all(sel_df,w_vars=[GV.WV_TEMP_MAX], in_files=GV.WS_UNIT_ALPHA, out_cols=GV.WS_UNIT_ALPHA)

    # Chart the Weather Dataframe
    all_charts = uc.Seas_Weather_Chart(w_df_all)
    

    for label, chart in all_charts.all_figs.items():
        chart.show('browser')

if __name__=='__main__':
    # hello_world_seas_chart()
    os.system('cls')
    update_weather(download_hist=False, download_bloomberg=True)