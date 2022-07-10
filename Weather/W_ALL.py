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

    states=['IA','IL','IN','OH','MO','MN','SD','NE']

    if download_hist:
        uu.log('USA NWS Historical Weather')
        wu.update_NWS_hist_weather(states = states, start_date='1985-01-01', end_date='2023-12-31')

    if download_geosys:
        uu.log('Geosys Weather')
        ge.update_Geosys_Weather()

    if download_bloomberg:
        runs_df=ba.latest_weather_run_df(finished=True)
        os.system('cls')
        
        blp = ba.BLPInterface('//blp/exrsvc')

        for i, row in runs_df.iterrows():
            run_str = row['Run'].strftime("%Y-%m-%dT%H:%M:%S")
            print(row['model'],row['model_type'], run_str,'----------------------')
            wu.udpate_USA_Bloomberg(row['Run'], states, model=row['model'], model_type=row['model_type'], blp=blp)

        runs_df.to_csv(GV.W_LAST_UPDATE_FILE)
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
    os.system('cls')
    update_weather(download_hist=True, download_bloomberg=True)