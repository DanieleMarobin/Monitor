# region imports
import os
import numpy as np
import pandas as pd
import calendar
from datetime import datetime as dt
import streamlit as st
#endregion

# region STATIC VAR
W_DIR = 'Weather/'
W_SEL_FILE = W_DIR + "weather_selection.csv"
CUR_YEAR = dt.today().year

# Weather Data types
WD_HIST='hist'
WD_GFS='gfs'
WD_ECMWF='ecmwf'
WD_H_GFS='hist_gfs'
WD_H_ECMWF='hist_ecmwf'

# WV = Weather variable
WV_PREC='Prec'

WV_TEMP_MAX='TempMax'
WV_TEMP_MIN='TempMin'
WV_TEMP_AVG='TempAvg'
WV_TEMP_SURF='TempSurf'
WV_SDD_30='Sdd30'

WV_SOIL='Soil'
WV_HUMI='Humi'
WV_VVI='VVI'

# Extention Modes
EXT_LIMIT='Limit'
EXT_MEAN='Mean'
EXT_ANALOG='Analog'
EXT_SHIFT_MEAN='Shifted_Mean'

EXT_DICT = {
    WV_PREC : {'mode': EXT_MEAN, 'limit': [0,0]},

    WV_TEMP_MAX: {'mode': EXT_ANALOG, 'limit': [-1,1]},
    WV_TEMP_MIN: {'mode': EXT_ANALOG, 'limit': [-1,1]},
    WV_TEMP_AVG: {'mode': EXT_LIMIT, 'limit': [-1,1]},
    WV_TEMP_SURF: {'mode': EXT_LIMIT, 'limit': [-1,1]},

    WV_SOIL: {'mode': EXT_LIMIT, 'limit': [-1,1]},
    WV_HUMI: {'mode': EXT_LIMIT, 'limit': [-1,1]},
    WV_VVI: {'mode': EXT_LIMIT, 'limit': [0,0]},
}

# Projection
PROJ='_Proj'
ANALOG='_Analog'

# w_sel file columns
WS_AMUIDS='amuIds'
WS_COUNTRY_NAME='country_name'
WS_COUNTRY_ALPHA='country_alpha'
WS_COUNTRY_CODE='country_code'
WS_UNIT_NAME='unit_name'
WS_UNIT_ALPHA='unit_alpha'
WS_UNIT_CODE='unit_code'
WS_STATE_NAME='state_name'
WS_STATE_ALPHA='state_alpha'
WS_STATE_CODE='state_code'
#endregion

# region accessories
def from_cols_to_w_vars(cols):
    fo = [c.split('_')[1] for c in cols]
    fo = list(set(fo))
    return fo

def last_leap_year():    
    start=dt.today().year
    while(True):
        if calendar.isleap(start): return start
        start-=1

LLY = last_leap_year()    
#endregion

# region w_sel
def get_w_sel_df():    
    return pd.read_csv(W_SEL_FILE,dtype=str)

def open_w_sel_file():
    program = r'"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"'
    os.system("start " + program + " "+W_SEL_FILE)

def update_w_sel_file(amuIds_results):
    """
    used like this normally:
        amuIds_results=ge.get_BRA_states_amuIds([],bearer_token)
        df=uw.update_w_sel_file(amuIds_results)
    """
    
    dict = {}
    cols=list(amuIds_results[0].keys()); cols.remove('coor'), cols.remove('func')

    for c in cols:
        dict[c]=[]
        for i in amuIds_results:
            dict[c].append(str(i[c]))
    
    updated_df=pd.DataFrame(dict)

    df_w_sel = get_w_sel_df()
    df_w_sel = df_w_sel.append(updated_df)
    df_w_sel = df_w_sel.drop_duplicates()
    
    df_w_sel.to_csv(W_SEL_FILE, index=False)
    
    return df_w_sel    
#endregion

# region build w_df_all
def build_w_df_all(df_w_sel, w_vars=[WV_PREC, WV_TEMP_MAX], in_files=WS_AMUIDS, out_cols=WS_UNIT_NAME):
    """
    in_files: MUST match the way in which files were written (as different APIS have different conventions)

    """    
    w_vars_copy = w_vars.copy()

    if WV_SDD_30 in w_vars_copy:
        w_vars_copy.append(WV_TEMP_MAX)                
    
    w_vars_copy=list(set(w_vars_copy))

    fo = {WD_HIST: [], WD_GFS: [], WD_ECMWF: []}

    # Looping 'WD_HIST', 'WD_GFS', 'WD_ECMWF'
    for key, value in fo.items():
        w_dfs = []
        dict_col_file = {}

        # creating the dictionary 'IL_Prec' from file 'E:/Weather/etc etc
        for index, row in df_w_sel.iterrows():
            for v in w_vars_copy:
                file = row[in_files]+'_'+v+'_'+key+'.csv'
                col = row[out_cols]+'_'+v
                dict_col_file[col] = file

        # reading the files
        for col, file in dict_col_file.items():
            if (os.path.exists(W_DIR+file)):
                w_dfs.append(pd.read_csv(W_DIR+file, parse_dates=['time'], index_col='time', names=['time', col], header=0))

        # concatenating the files
        if len(w_dfs) > 0:
            w_df = pd.concat(w_dfs, axis=1, sort=True)
            w_df = w_df.dropna(how='all')
            fo[key] = w_df

        # Adding 'derivatives' columns
        if WV_SDD_30 in w_vars_copy:            
            add_Sdd(fo[key], source_WV=WV_TEMP_MAX, threshold=30)

    # Create the DF = Hist + Forecasts
    if (len(fo[WD_GFS])):
        fo[WD_H_GFS] = pd.concat([fo[WD_HIST], fo[WD_GFS]], axis=0, sort=True)
    if (len(fo[WD_ECMWF])):
        fo[WD_H_ECMWF] = pd.concat([fo[WD_HIST], fo[WD_ECMWF]], axis=0, sort=True)

    # Remove the temporary columns (used to add derivatives)
    cols_to_remove = list(set(w_vars_copy) - set(w_vars))

    for key, value in fo.items():
        fo[key] = remove_w_col(fo[key], cols_to_remove=cols_to_remove)

    return fo

def weighted_w_df(w_df, weights, w_vars=[], output_column='Weighted'):
    # w_vars = [] needs to be a list

    fo_list = []
    if len(w_vars)==0: 
        w_vars=from_cols_to_w_vars(w_df.columns)
    
    w_df_years = w_df.index.year.unique()
    weights_years=weights.index.unique()
    
    # Add missing years
    missing_weights = list(set(w_df_years) - set(weights_years))
    weight_mean=weights.mean()
    for m in missing_weights:
        weights.loc[m]=weight_mean

    # Remove useless years
    weights=weights.loc[w_df_years]
    weights=weights.sort_index()

    for v in w_vars:
        fo = w_df.copy()
        fo = fo.reset_index(drop=True).set_index(w_df.index.year)
        
        var_weights = weights.copy()
        var_weights.columns = [c+'_'+v for c in weights.columns]

        w_w_df = fo * var_weights

        w_w_df=w_w_df.set_index(w_df.index)

        w_w_df = w_w_df.dropna(how='all', axis=1)
        w_w_df = w_w_df.dropna(how='all', axis=0)
        
        # w_w_df=w_w_df.set_index(w_df.index)

        fo = w_w_df.sum(axis=1)
        fo = pd.DataFrame(fo)

        fo = fo.rename(columns={0: output_column+'_'+v})
        fo_list.append(fo)

    return pd.concat(fo_list, axis=1)

def weighted_w_df_all(all_w_df, weights, w_vars=[], output_column='Weighted'):
    fo={}
    for key,value in all_w_df.items():
        if len(value)>0:
            fo[key]=weighted_w_df(value,weights,w_vars,output_column)                        
    return fo
# endregion

# region Derivatives Columns
def remove_w_col(w_df, cols_to_remove):
    cols=[]
    for c_rem in cols_to_remove:
        cols += [c for c in w_df.columns if c_rem in c]
    
    fo = w_df.drop(columns=cols)
    return fo

def add_Sdd(w_df, source_WV=WV_TEMP_MAX, threshold=30):
    for col in w_df.columns:
        geo, w_var= col.split('_')
        if w_var == source_WV:
            new_w_var = geo+'_Sdd'+str(threshold)
            w_df[new_w_var]=w_df[col]
            mask=w_df[new_w_var]>threshold
            w_df[new_w_var][mask]=w_df[new_w_var][mask]-threshold
            w_df[new_w_var][~mask]=0  
    return w_df
    
# endregion

# region Weather Windows
def extract_w_windows(w_df, windows_df: pd.DataFrame):
    """
    the 'windows_df' needs to have 'start' and 'end' columns
    """
    fo=pd.DataFrame(columns=w_df.columns)

    for i in windows_df.index:
        sd=windows_df.loc[i]['start']
        ed=windows_df.loc[i]['end']
        # fo.append(w_df[(w_df.index>=sd) & (w_df.index<=ed)].sum())
        # fo.loc[i]=w_df[(w_df.index>=sd) & (w_df.index<=ed)].sum()
        fo.loc[i]= np.sum(w_df[(w_df.index>=sd) & (w_df.index<=ed)])

    return fo
#endregion

# region seasonals

def add_seas_year(w_df, ref_year=CUR_YEAR, ref_year_start= dt(CUR_YEAR,1,1), offset = 2):
    # yo = year offset
    # offset = 2 means:
    #       - first year is going to be first year - 2
    #       - last year is going to be ref_year + 2 = CUR_YEAR + 2

    os = w_df.index[0].year - ref_year -offset# offset start
    oe = w_df.index[-1].year - ref_year +offset  # offset end

    for yo in range(os, oe):
        value = ref_year+yo
        ss = ref_year_start+ pd.DateOffset(years=yo) # start slice
        es = ref_year_start+ pd.DateOffset(years=yo+1)+pd.DateOffset(days=-1) # end slice

        w_df.loc[ss:es,'year']=int(value)

    w_df['year'] = w_df['year'].astype('int')

    return w_df

def seas_day(date, ref_year_start= dt(CUR_YEAR,1,1)):
    # This function returns both the year and seas_day
    # seas_day is basically the X-axis of the seasonal plot
    start_idx = 100 * ref_year_start.month + ref_year_start.day
    date_idx = 100 * date.month + date.day

    if (start_idx<300):
        if (date_idx>=start_idx):
            return dt(LLY, date.month, date.day)
        else:
            return dt(LLY+1, date.month, date.day)
    else:
        if (date_idx>=start_idx):
            return dt(LLY-1, date.month, date.day)
        else:
            return dt(LLY, date.month, date.day)


def seasonalize(w_df, col=None, mode = 'Mean', limit=[-1,1], ref_year=CUR_YEAR, ref_year_start = dt(CUR_YEAR,1,1)):
    # This function MUST do only 1 column at a time

    # 'ref_year' = reference year
    # 'ref_seas_start' = reference seasonal start
    
    if col==None: col = w_df.columns[0]
    w_df=w_df[[col]]
    
    add_seas_year(w_df,ref_year,ref_year_start)
    w_df['seas_day'] = [seas_day(d,ref_year_start) for d in w_df.index]

    pivot = w_df.pivot_table(index=['seas_day'], columns=['year'], values=[col], aggfunc='mean')
    pivot.columns = pivot.columns.droplevel(level=0)

    # Drop columns that don't start from the beginning of the crop year (ref_year_start)
    cols_to_drop = [c for c in pivot.columns if np.flatnonzero(~np.isnan(pivot[c]))[0] > 0]
    pivot=pivot.drop(columns=cols_to_drop)

    # the below interpolation is to fill 29 Feb every year
    pivot.interpolate(inplace=True, limit_area='inside')
            
    cur_year_v = pivot[ref_year].values
    lvi = np.flatnonzero(~np.isnan(cur_year_v))[-1] # Last Valid Index of the current year

    cols_no_cur_year = list(pivot.columns); cols_no_cur_year.remove(ref_year)
    max_no_cur_year_v = pivot[cols_no_cur_year].max(axis=1).values
    min_no_cur_year_v = pivot[cols_no_cur_year].min(axis=1).values
    avg_no_cur_year_v = pivot[cols_no_cur_year].mean(axis=1).values            

    # analogue identification
    analog_col=None
    analog_pivot = pivot.cumsum()
    df_sub=analog_pivot.drop(columns=[ref_year]).subtract(analog_pivot[ref_year],axis=0).abs()

    if (len(df_sub.columns)>0):        
        dt_s=analog_pivot.index[0]
        dt_e=analog_pivot.index[lvi]

        abs_error= df_sub.loc[dt_s:dt_e].sum()        
        analog_col=abs_error.index[np.argmin(abs_error)]
        # if (analog_col!=None): print(analog_col)

    pivot['Max']=max_no_cur_year_v
    pivot['Min']=min_no_cur_year_v
    pivot['Mean']=avg_no_cur_year_v
    
    delta = avg_no_cur_year_v[lvi] - cur_year_v[lvi] # Difference between current year value and average in the lvi

    # initialize the projection as the current year values (all 366 values still including the NaN at the end)
    proj = np.array(cur_year_v)

    # the below condition kicks in only if we actually need a projection:
    # basically saying for 'hist' len(cur_year_v) == 366 so just calculate if 'lvi' is less
    # for GFS and ECMWF the 2 are the same, so it will not do anything
    
    if len(cur_year_v)>lvi+1:
        # The below only work on the 'projection part' lvi+1:
        shifted_mean = avg_no_cur_year_v[lvi+1:] - delta # This is in just the average translated to match the last day        
                
        if mode==EXT_LIMIT:           
            limit_curve = shifted_mean

            # Minimum
            no_cur_year_fwd_min = min_no_cur_year_v[lvi+1:]
            min_diff = limit_curve - no_cur_year_fwd_min
            min_diff[np.where(min_diff < limit[0])] = limit[0]
            limit_curve = no_cur_year_fwd_min + min_diff

            # Maximum
            no_cur_year_fwd_max = max_no_cur_year_v[lvi+1:]
            max_diff = limit_curve - no_cur_year_fwd_max
            max_diff[np.where(max_diff > limit[1])] = limit[1]
            limit_curve = no_cur_year_fwd_max + max_diff        
        

        # Attaching the "projection" part to the "proj" column
        # 'Limit' takes the 'Shifted_Mean' and then apply the limits (as it can be seen above)
        if mode==EXT_LIMIT:         proj[lvi+1:] = limit_curve # With limits control (for variables like Soil moisture, Temperature)
        elif mode==EXT_MEAN:        proj[lvi+1:] = avg_no_cur_year_v[lvi+1:]  # Avg weather (for variables like Precipitation)
        elif mode==EXT_SHIFT_MEAN:  proj[lvi+1:] = shifted_mean
        elif mode==EXT_ANALOG:      proj[lvi+1:] = pivot[analog_col][lvi+1:]
            
    pivot[str(ref_year)+PROJ] = proj
    
    if analog_col!=None: pivot[str(analog_col)+ANALOG] = pivot[analog_col]

    return pivot

def cumulate_seas(df, excluded_cols = [], ref_year=CUR_YEAR):
    df=df.drop(columns=excluded_cols)

    cols_no_cur_year = list(df.columns); 
    cols_no_cur_year.remove(ref_year)

    df = df.cumsum()
    df['Max']=df[cols_no_cur_year].max(axis=1)
    df['Min']=df[cols_no_cur_year].min(axis=1)
    df['Mean']=df[cols_no_cur_year].mean(axis=1)
    return df
#endregion

# region extending

def extend_with_seasonal_df(w_df, cols_to_extend=[], seas_cols_to_use=[], modes=[], limits=[],ref_year=CUR_YEAR, ref_year_start= dt(CUR_YEAR,1,1)):
    w_df_ext_s=[]
    if len(cols_to_extend)==0:
        cols_to_extend = w_df.columns
    
    for idx, col in enumerate(cols_to_extend):
        w_var=col.split('_')[1]
        # choosing the column to extract from the "Seasonalize" function
        if len(seas_cols_to_use)==0:
            seas_col_to_use = str(CUR_YEAR)+PROJ
        else:
            i = min(idx,len(seas_cols_to_use)-1)
            seas_cols_to_use[i]

        # Picking the 'mode'
        if len(modes)==0:
            if w_var in EXT_DICT:
                mode=EXT_DICT[w_var]['mode']
            else:
                mode=EXT_MEAN
        else:
            i = min(idx,len(modes)-1)
            mode=modes[i]

        # Picking the 'limit'
        if w_var in EXT_DICT:
            limit=EXT_DICT[w_var]['limit']
        else:
            limit=[-1,1]

        # if len(limits)==0: 
        #     limit=EXT_DICT[w_var]['limit']
        # else:
        #     i = min(idx,len(limits)-1)
        #     limit = limits[i]
                
        # Calculate the seasonal
        seas = seasonalize(w_df, col, mode=mode, limit=limit, ref_year=ref_year, ref_year_start=ref_year_start)
        
        ext_year = pd.to_datetime(w_df.last_valid_index()).year

        if not calendar.isleap(ext_year):
            seas=seas.drop(str(LLY)+'-02-29') # Remove 29 Feb if not leap year

        seas['time'] = [dt(year=ext_year, month=x.month, day=x.day) for x in seas.index]
        seas=seas.set_index('time') 

        seas=seas.rename(columns={seas_col_to_use:col})
        
        w_df_ext = pd.concat([w_df[[col]], seas[[col]]])

        w_df_ext['time']=w_df_ext.index
        w_df_ext=w_df_ext.drop_duplicates(ignore_index=True, subset=['time'], keep='first')
        w_df_ext=w_df_ext.set_index('time')        
        
        w_df_ext_s.append(w_df_ext.copy())

    fo= pd.concat(w_df_ext_s,axis=1)
    fo=fo.sort_index(ascending=True)
    return fo
#endregion