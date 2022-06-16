import pandas as pd
from datetime import datetime as dt

import APIs.QuickStats as qs


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
    Question answered: \n
    "What day the crop was 50% planted for each year?" \n
    """
    fo_dict={'year':[],'date':[]}

    df[time_col]=pd.to_datetime(df[time_col])
    df=df.set_index(time_col)
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

def progress_from_date(df, sel_date='2021-05-15', time_col='week_ending', value_col='Value'):
    """
    Question answered: \n
    "What progress the crop was on May 15th?" \n
    The output is a dict { year : progress}
    """    
    fo_dict={'year':[],value_col:[]}

    df[time_col]=pd.to_datetime(df[time_col])
    df=df.set_index(time_col)
    df=df.asfreq('1D')
    df[value_col]=df[value_col].interpolate(limit_area='inside')


    sel_date= pd.to_datetime(sel_date)
    dates = [dt(y,sel_date.month,sel_date.day) for y in df.index.year.unique()]
    df = df.loc[dates]
    
    fo_dict['year']=df.index.year
    fo_dict['Value']=df[value_col]
    fo=pd.DataFrame(fo_dict)
    fo=fo.set_index('year')

    return fo