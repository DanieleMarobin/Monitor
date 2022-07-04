import sys
from tkinter.tix import Tree
sys.path.append(r'\\ac-geneva-24\E\grains trading\visual_studio_code\\')

from datetime import datetime as dt

import Weather.W_USA as wu
import APIs.Geosys as ge

import Utilities.Weather as uw
import Utilities.Charts as uc
import Utilities.Utilities as uu
import Utilities.SnD as us
import Utilities.GLOBAL as GV


def update_weather(download_hist=True, download_geosys=True, overwrite=False):
    if download_hist:
        uu.log('-------------- USA Historical Weather --------------')
        corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
        wu.update_USA_weather(states = corn_states, start_date='1985-01-01', end_date='2023-12-31')

    if download_geosys:
        uu.log('-------------- Geosys Weather --------------')
        ge.update_Geosys_Weather(overwrite=overwrite)

    uu.log('-------------- Copying all the files to the Monitor Folder --------------')
    source_dir = GV.W_DIR
    destination_dir = r"\\ac-geneva-24\E\grains trading\Streamlit\Monitor\Weather\\"
    uu.copy_folder(source_dir,destination_dir,verbose=True)

    uu.log('----------------------------')
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
    
    # uc.Seas_Weather_Chart(w_df_all, ext_mode=[uw.EXT_MEAN], limit=[-1,1], cumulative = False)
    # uc.Seas_Weather_Chart(w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = True, ref_year_start= dt(uw.CUR_YEAR-1,9,1))
    # uc.Seas_Weather_Chart(w_df_all, ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = False, ref_year_start= dt(uw.CUR_YEAR,1,1))
    # return
    
    # -------------------------------------------------------------- with Weights --------------------------------------------------------------
    # # Build the Weights
    # years=[2015,2017,2020,2021]
    # weights = us.get_USA_prod_weights('CORN', 'STATE', years, states)

    # # Chart the Weighted Dataframe
    # w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

    # uc.Seas_Weather_Chart(w_w_df_all)
    # uc.Seas_Weather_Chart(w_w_df_all,ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = False, ref_year_start= dt(uw.CUR_YEAR-0,1,1))
    # uc.Seas_Weather_Chart(w_w_df_all,ext_mode=[uw.EXT_ANALOG], limit=[-1,1], cumulative = True, ref_year_start= dt(uw.CUR_YEAR-0,1,1))

    for label, chart in all_charts.all_figs.items():
        chart.show('browser')

if __name__=='__main__':
    # hello_world_seas_chart()
    update_weather(download_hist=True,download_geosys=True, overwrite=True)