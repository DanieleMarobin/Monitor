import numpy as np
import pandas as pd
from datetime import datetime as dt

import APIs.QuickStats as qs
import Utilities.GLOBAL as GV


def get_USA_prod_weights(commodity='CORN', aggregate_level='STATE', years=[], subset=[]):
    """
    rows:       years \n
    columns:    region \n    
    """
    fo=qs.get_production(commodity=commodity,aggregate_level=aggregate_level, years=years)
    fo = pd.pivot_table(fo,values='Value',index='state_alpha',columns='year')

    if (len(subset))>0: fo=fo.loc[subset]

    fo=fo/fo.sum()

    return fo.T

def dates_from_progress(df, sel_percentage=50.0, time_col='week_ending', value_col='Value'):
    """
    Question answered:
    "What day the crop was 50% planted for each year?"
    """

    fo_dict={'year':[],'date':[]}
    df[time_col]=pd.to_datetime(df[time_col])

    # Remove current year information, if the we have no enough data to interpolate the current year
    if True:
        mask=(df[time_col]>dt(GV.CUR_YEAR,1,1))
        cur_year_df=df[mask]

        if (len(cur_year_df)>0):
            if (cur_year_df[value_col].max() < sel_percentage):
                mask=(df[time_col]<dt(GV.CUR_YEAR,1,1))
                df=df[mask]

    df=df.set_index(time_col, drop=False)
    df=df.asfreq('1D')

    df[value_col]=df[value_col].interpolate(limit_area='inside')

    # To avoid interpolation from previous year end of planting (100%) to next year beginning of planting (0%)
    mask=df[value_col]>df[value_col].shift(fill_value=0)
    df=df[mask]

    df['diff']=abs(df[value_col]-sel_percentage)

    min_diff = df.groupby(df.index.year).min()
    
    for y in min_diff.index:
        sel_df=df.loc[(df['diff']==min_diff.loc[y]['diff']) & (df.index.year==y)]

        fo_dict['year'].append(y)
        fo_dict['date'].append(sel_df.index[0])

    fo=pd.DataFrame(fo_dict)
    fo=fo.set_index('year')
    return fo

def extend_date_progress(date_progress_df: pd.DataFrame, year=GV.CUR_YEAR, day=dt.today(), col='date'):
    """
    Same as the weather extention wwith seasonals, but with dates of crop progress

    Args:
        date_progress_df (pd.DataFrame): index = year, columns = 'date' (for every year: when was the crop 80% planted? or 50% silked etc)

        year (int): the year that I need to have a value for

        day (datetime): the simulation day. It simulates not knowing anything before this day (included). Useful to avoid the "49" late planting


    Explanation:
        if we have data already all is good:
            -> 'if year in fo.index: return fo'
        
        Otherwise we have to pick the later between:
            - the average of previous years
            - simulation day
        
        case 1) there is no value yet for 80% planted in on 'June 15th':
            - the average is going to be 15th May
            - but being on June 15th and not having a value yet, it means that the value cannot be May 15th (otherwise we would have had a value)
            -> so return 'June 15th' that is Max('June 15th', 'May 15th')
        
        case 2) there is no value yet for 80% planted in on Feb 17th:
            - the average is going to be 15th May
            -> so return  'May 15th' that is Max('Feb 17th', 'May 15th')    
    """

    fo = date_progress_df
    if year in fo.index: return fo
    
    fo_excl_YEAR=fo.loc[fo.index<year]
    fo_excl_YEAR=pd.Series([dt(year,d.month,d.day) for d in fo_excl_YEAR[col]])

    avg_day = np.mean(fo_excl_YEAR)
    avg_day = dt(avg_day.year,avg_day.month,avg_day.day)

    if ((avg_day > day) or (avg_day > dt.today())):
        fo.loc[year] = avg_day
    else:
        fo.loc[year] = day
    
    return fo


def progress_from_date(df: pd.DataFrame, progress_date, time_col='week_ending', value_col='Value'):
    """
    Args:
        df (pd.DataFrame): _description_
        sel_date (_type_): _description_
        time_col (str, optional): _description_
        value_col (str, optional): _description_

    Returns:
        df (pd.DataFrame): index = year, columns = 'Value' (for every year: % progress on 'sel_date')
    """
    fo_dict={'year':[],value_col:[]}

    df[time_col]=pd.to_datetime(df[time_col])
    df=df.set_index(time_col)
    df=df.asfreq('1D')
    df[value_col]=df[value_col].interpolate(limit_area='inside')


    dates = [dt(y,progress_date.month,progress_date.day) for y in df.index.year.unique()]
    df = df.loc[dates]
    
    fo_dict['year']=df.index.year
    fo_dict['Value']=df[value_col]
    fo=pd.DataFrame(fo_dict)
    fo=fo.set_index('year')

    return fo
def extend_progress(progress_df: pd.DataFrame, progress_date, year=GV.CUR_YEAR, day=dt.today()):
    """_summary_

    Args:
        progress_df (pd.DataFrame): index = year, columns = 'Value' (for every year: % progress on 'progress_date')
        progress_date (datetime): '15th May' would indicate that the 'Value' is % progess on the '15th May'
        year (int): year to extend (create the row 2022)
        day (datetime): the simulation day. It simulates not knowing anything before this day (included). Useful to avoid the "49" late planting
        col (str):

    Returns:
        Same as the input but extended by 1 row
        progress_df (pd.DataFrame): index = year, columns = 'Value' (for every year: % progress on 'progress_date')
    """
    # Ex: May 15th % planted

    # if we are before May 15th -> take the average of the previous years (overwriting the previous value)
    # if there is no value -> take the average of the previous years    

    fo = progress_df
    if ((day<progress_date) or not(year in fo.index)):
        fo_excl_YEAR=fo.loc[fo.index<year]
        fo.loc[year] = fo_excl_YEAR.mean()     

    return fo