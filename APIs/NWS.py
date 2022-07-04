"""
Documentation: \n
http://www.rcc-acis.org/docs_webservices.html \n

Abbreviation: \n
https://simple.wikipedia.org/wiki/U.S._postal_abbreviations \n
"""

import requests
import pandas as pd
import numpy as np
import json

import concurrent.futures
import Utilities.Utilities as uu
import Utilities.GLOBAL as GV


W_VAR_DICT={'pcpn':GV.WV_PREC,'mint':GV.WV_TEMP_MIN,'maxt':GV.WV_TEMP_MAX}


def get_NWS_multi_stn(state='', county='', start_date='', end_date='', variables=['pcpn', 'mint', 'maxt'], return_meta=False):
    """
    start_date='2020-01-01 \n
    end_date='2020-01-01 \n
    Variables \n
    'elems': maxt, mint, avgt, pcpn \n
    Metadata \n
    'name', 'state', 'sids', 'sid_dates', 'll', 'elev', 'uid', 'county', 'climdiv' \n
    """

    metadata = ['name', 'state', 'uid', 'county']
    params = {'sdate': start_date, 'edate': end_date,
              'elems': ','.join(variables)}

    if county != '':
        params['county'] = county
    elif state != '':
        params['state'] = state

    if len(metadata) > 0:
        params['meta'] = ','.join(metadata)

    response = requests.get(
        'https://data.rcc-acis.org/MultiStnData', params=params)

    js = response.json()

    fo_meta = []
    county_dfs = {}

    dates = np.arange(np.datetime64(start_date), np.datetime64(
        end_date) + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    cols = variables.copy()
    cols.append('date')
    has_prec = 'pcpn' in cols
    # cols.append('avgt')

    for station in js['data']:
        meta = station['meta']
        fo_meta.append(meta)

        if 'county' in meta:
            meta_county = meta['county']
        else:
            print(station['meta'])
            meta_county = meta['state'] + str(meta['uid'])

        station_data = station['data']
        station_df = pd.DataFrame(station_data)

        for col in station_df.columns:
            station_df[col] = pd.to_numeric(station_df[col], errors='coerce')

        station_df['date'] = dates
        station_df.columns = cols

        # Interpolating the Temperatures (interpolation doesn't work if there are nan -> mask below)
        station_df[['mint', 'maxt']] = station_df[['mint', 'maxt']
                                                  ].interpolate(limit=60, limit_area='inside')

        # Filling "inner" the Precipitations
        if has_prec:
            for c in ['pcpn']:
                y = station_df[c].values
                mask = ~np.isnan(y)
                if np.sum(mask) > 0:
                    n0 = np.nonzero(mask)
                    f_n0 = n0[0][0]
                    l_n0 = n0[0][-1]
                    y[~mask] = 0
                    y[0:f_n0] = np.nan
                    y[l_n0+1:] = np.nan

                    station_df[c] = y

        if (meta_county in county_dfs):
            county_dfs[meta_county].append(station_df)
        else:
            county_dfs[meta_county] = [station_df]

    fo = {}
    for key, c_dfs in county_dfs.items():
        df = pd.concat(c_dfs)

        df.columns = cols
        df = df.melt(id_vars='date')
        # dropna:bool, default True Do not include columns whose entries are all NaN
        df = pd.pivot_table(df, values='value', index=['date'], columns=[
                            'variable'], aggfunc=np.mean, dropna=True)
        df.index.name = None
        df.columns.name = None

        if (('mint' in df.columns) and ('maxt' in df.columns)):
            df['avgt'] = (df['mint'].values+df['maxt'].values)/2
        fo[key] = df

    if return_meta:
        return fo, fo_meta
    else:
        return fo

def get_NWS_grid(state, start_date='', end_date='', variables=['pcpn', 'mint', 'maxt'], area_reduce='county_mean'):

    units_dict = {'pcpn': 'mm', 'mint': 'degreeC', 'maxt': 'degreeC'}

    params = {'sdate': start_date, 'edate': end_date, 'grid': '1'}
    params['state'] = state
    elems = [{'name': v, 'area_reduce': area_reduce, 'units': units_dict[v]} for v in variables]
    params['elems'] = elems
    params = {'params': json.dumps(params)}

    uu.log('Downloading '+state+'...')

    response = requests.get('https://data.rcc-acis.org/GridData', params=params)
    js = response.json()

    uu.log('Downloaded '+state)

    dfs = []

    for day in js['data']:
        df = pd.json_normalize(day).transpose()
        df[0] = np.datetime64(day[0])
        dfs.append(df)

    df = pd.concat(dfs)

    cols = ['time']
    cols.extend(variables)
    df.columns = cols
    df[GV.WV_TEMP_AVG] = (df['mint'].values+df['maxt'].values)/2

    fo = {}
    for out_county in df.index.unique():
        temp_df = df.loc[out_county]
        temp_df = temp_df.set_index('time')
        temp_df=temp_df.rename(columns=W_VAR_DICT)
        fo[out_county] = temp_df

    return fo


def get_NWS_general_state(state=''):
    """
    Documentation: \n
    http://www.rcc-acis.org/docs_webservices.html \n

    available metas: id, name, bbox, geojson, state
    """
    # response = requests.get('https://data.rcc-acis.org/General/county?id=17113&meta=id,name,state,geojson')
    response = requests.get('https://data.rcc-acis.org/General/state?state='+state+'&meta=id,name,geojson,state')
    js = response.json()['meta'][0]
    return js

def get_NWS_general_county(county_id=''):
    """
    Documentation: \n
    http://www.rcc-acis.org/docs_webservices.html \n

    available metas: id, name, bbox, geojson, state
    """
    response = requests.get('https://data.rcc-acis.org/General/county?id='+county_id+'&meta=id,name,state,geojson')
    # response = requests.get('https://data.rcc-acis.org/General/state?state='+state+'&meta=id,name,geojson,state')
    js = response.json()['meta'][0]
    return js


def get_NWS_county_geo(county=[]):
    fo = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for c in county:
            fo[c] = executor.submit(get_NWS_general_county, c)

    for key, value in fo.items():
        fo[key] = value.result()

    return fo

def get_NWS_states_geo(states=[]):
    fo = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for s in states:
            fo[s] = executor.submit(get_NWS_general_state, s)

    for key, value in fo.items():
        fo[key] = value.result()

    return fo

def get_states_weather(states=['IA','IL'], start_date='2021-01-01', end_date='2022-05-25', variables=['pcpn', 'mint', 'maxt'], area_reduce='state_mean'):
    """
    area_reduce='state_mean', 'county_mean' \n
    corn_states=['IA','IL','IN','OH','MO','MN','SD','NE'] \n
    """

    fo = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for s in states:
            fo[s] = executor.submit(get_NWS_grid, s, start_date, end_date, variables, area_reduce)

    for key, value in fo.items():
        fo[key] = value.result()

    return fo

