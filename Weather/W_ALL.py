import sys;
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

import os
from datetime import datetime as dt

import Weather.W_USA as wu
import threading

import APIs.Geosys as ge
import APIs.Bloomberg as ba

import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.Utilities as uu
import Utilities.GLOBAL as GV
import warnings; warnings.filterwarnings("ignore")

def update_weather(download_hist=False, download_geosys=False, download_bloomberg=True, loop_interval=60):
    """
    'loop_interval' in seconds
    model = 'GFS', 'ECMWF'
    model_type = 'DETERMINISTIC', 'ENSEMBLE_MEAN'
    """
    if loop_interval>0:
        threading.Timer(loop_interval, update_weather,[download_hist, download_geosys, download_bloomberg,loop_interval]).start()

    states=['IA','IL','IN','OH','MO','MN','SD','NE']

    if download_hist:
        uu.log('USA NWS Historical Weather')
        wu.update_NWS_hist_weather(states = states, start_date='1985-01-01', end_date='2023-12-31')

    if download_geosys:
        uu.log('Geosys Weather')
        ge.update_Geosys_Weather()

    if download_bloomberg:
        runs_df=ba.latest_weather_run_df(finished=True)

        # to compare with previous Runs
        if False:
            runs_df.loc['ECMWF Ensemble','Run']=dt(2022,7,13,12,0,0)
                
        blp = ba.BLPInterface('//blp/exrsvc')

        for i, row in runs_df.iterrows():
            run_str = row['Run'].strftime("%Y-%m-%dT%H:%M:%S")
            print(row['model'],row['model_type'], run_str,'----------------------')
            wu.udpate_USA_Bloomberg(row['Run'], states, model=row['model'], model_type=row['model_type'], blp=blp)
            print()

        runs_df.to_csv(GV.W_LAST_UPDATE_FILE)

    uu.log('Done With the Weather Download -----------------------------------------------------------')
      

if __name__=='__main__':
    os.system('cls')
    update_weather(download_hist=True, download_bloomberg=True, loop_interval=600)