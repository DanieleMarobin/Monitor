#region imports
import os
import pandas as pd
import geopandas as gpd
import requests
import concurrent.futures
from datetime import datetime as dt

import APIs.NWS as nw

import Utilities.Weather as uw
import Utilities.Utilities as uu
import Utilities.GLOBAL as GV
from Utilities.GIS import pole_of_inaccessibility
#endregion

#region Weather Variables Functions
func_prec={
    'name':GV.WV_PREC,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/daily-precipitation',
    'indicator':[2,4,5]}

func_temp_avg={
    'name':GV.WV_TEMP_AVG,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/average-temperature',
    'indicator':[2,4,5]}

func_temp_min={
    'name':GV.WV_TEMP_MIN,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/min-temperature',
    'indicator':[2,4,5]}

func_temp_max={
    'name':GV.WV_TEMP_MAX,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/max-temperature',
    'indicator':[2,4,5]}

func_temp_surf={
    'name':GV.WV_TEMP_SURF,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/surface-temperature',
    'indicator':[2]}

func_soil={
    'name':GV.WV_SOIL,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/soil-moisture',
    'indicator':[2,4,5]}

func_vvi={
    'name':GV.WV_VVI,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/vegetation-vigor-index',
    'indicator':[1]}

func_humi={
    'name':GV.WV_HUMI,
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/relative-humidity',
    'indicator':[2]}
#endregion

#region Lasso functions
func_lasso_municipios={
    'name':'lasso_municipios',
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/admin-unit/lasso-coverage/SouthAmerica-MUNICIPIOS',
    'idBlock':230}

func_lasso_USA_county={
    'name':'lasso_municipios',
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/admin-unit/lasso-coverage/County',
    'idBlock':141}

func_lasso_world={
    'name':'lasso_world',
    'func':'https://api.geosys-na.net/Agriquest/Geosys.Agriquest.CropMonitoring.WebApi/v0/api/admin-unit/lasso-coverage/World',
    'idBlock':129}    
#endregion


def get_bearer_token():
    headers_token = {'Content-Type': 'application/x-www-form-urlencoded'}

    data_token = 'grant_type=password&'
    data_token+='username=AQ_AVERE_mln&'
    data_token+='password=AQ_AVERE_mln&'
    data_token+='client_id=agriquest_web&'
    data_token+='client_secret=agriquest_web.secret&'
    data_token+='scope=openid offline_access geo6:ndvigraph'

    response_token = requests.post('https://identity.geosys-na.com/v2.1/connect/token', headers=headers_token, data=data_token)

    js_token = response_token.json()

    bearer_token = 'Bearer '+js_token['access_token']
    return bearer_token

def get_amuIds(bearer_token, download_input, offset=0.0000000000001):
    """
    'download_input' is a dictionary that must contain the keys['coor','func'] \n
    """

    coor_amuIds, func_amuIds = download_input['coor'], download_input['func']
    response_text=''
    
    headers_amuIds = {'Accept': '*/*','Authorization': bearer_token, 'Connection': 'keep-alive'}
    
    while len(response_text)==0:
        json_data_amuIds = {
        'coordinates': [
            [coor_amuIds[0]-offset/2,coor_amuIds[1]-offset/2],
            [coor_amuIds[0]+offset/2,coor_amuIds[1]-offset/2],
            [coor_amuIds[0]+offset/2,coor_amuIds[1]+offset/2],
            [coor_amuIds[0]-offset/2,coor_amuIds[1]+offset/2],
            [coor_amuIds[0]-offset/2,coor_amuIds[1]-offset/2],
        ],
        'zoom': 17,}

        response_amuIds = requests.post(func_amuIds['func'], headers=headers_amuIds, json=json_data_amuIds)
        response_text=response_amuIds.text
        # print('response_amuIds',response_text,'len(response_amuIds)',len(response_text))
        # print('response_json',response_amuIds.json())
        offset=offset*1.2
    
    download_input[GV.WS_AMUIDS]=response_amuIds.json()
    download_input['idBlock']=func_amuIds['idBlock']
    return download_input



def get_USA_county_amuIds(county=[], bearer_token=None):

    country_name = 'United States'
    country_alpha = 'USA'
    country_code = '840'

    if bearer_token==None:
        print('Generating the bearer_token')
        bearer_token=get_bearer_token()
        
    # Get the geography from NWS
    geography = nw.get_NWS_county_geo(county)
    print('got the geography')
    # Creating the Download list
    download_list=[]
    for s in geography:        
        g=geography[s]
        pole = pole_of_inaccessibility(g['geojson'],0.01)
        coor = [pole.x, pole.y]
        download_list+=[{
            'coor':coor, 'func':func_lasso_USA_county,
            GV.WS_COUNTRY_NAME: country_name, GV.WS_COUNTRY_ALPHA: country_alpha, GV.WS_COUNTRY_CODE: country_code,
            GV.WS_UNIT_NAME:g['name'],GV.WS_UNIT_ALPHA:g['id'],GV.WS_UNIT_CODE:'wip',
            GV.WS_STATE_NAME:g['name'],GV.WS_STATE_ALPHA:g['id'],GV.WS_STATE_CODE:'wip'}]

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results={}
        for download_input in download_list:
            results[download_input['unit_name']] = executor.submit(get_amuIds, bearer_token, download_input)

    fo = extract_results(results)
    return fo


def get_USA_states_amuIds(states=[], bearer_token=None):

    country_name = 'United States'
    country_alpha = 'USA'
    country_code = '840'

    if bearer_token==None:
        print('Generating the bearer_token')
        bearer_token=get_bearer_token()
        
    # Get the geography from NWS
    geography = nw.get_NWS_states_geo(states)
    
    # Creating the Download list
    download_list=[]
    for s in geography:        
        g=geography[s]
        pole = pole_of_inaccessibility(g['geojson'],0.01)
        coor = [pole.x, pole.y]
        download_list+=[{
            'coor':coor, 'func':func_lasso_world,
            GV.WS_COUNTRY_NAME: country_name, GV.WS_COUNTRY_ALPHA: country_alpha, GV.WS_COUNTRY_CODE: country_code,
            GV.WS_UNIT_NAME:g['name'],GV.WS_UNIT_ALPHA:g['id'],GV.WS_UNIT_CODE:'wip',
            GV.WS_STATE_NAME:g['name'],GV.WS_STATE_ALPHA:g['id'],GV.WS_STATE_CODE:'wip'}]

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results={}
        for download_input in download_list:
            results[download_input['unit_name']] = executor.submit(get_amuIds, bearer_token, download_input)

    fo = extract_results(results)
    return fo

def get_BRA_states_amuIds(states=[], bearer_token=None):
    # to get the states boundaries: https://portaldemapas.ibge.gov.br/portal.php#mapa218980
    # List of ISO 3166 country codes: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes

    country_name = 'Brazil'
    country_alpha = 'BRA'
    country_code = '076'

    g_dir = r"\\ac-geneva-24\E\grains trading\visual_studio_code\GeoData\BRA\\"
    
    if bearer_token==None:
        print('Generating the bearer_token')
        bearer_token=get_bearer_token()

    # Get the geography from downloeaded shapefiles
    states_df = gpd.read_file(g_dir+'lim_unidade_federacao_a.shp')

    # Creating the Download list
    download_list=[]

    for index,row in states_df.iterrows():
        pole = pole_of_inaccessibility(row['geometry'],0.01)
        coor = [pole.x, pole.y]
        download_list+=[{
            'coor':coor, 'func':func_lasso_world,
            GV.WS_COUNTRY_NAME: country_name, GV.WS_COUNTRY_ALPHA: country_alpha, GV.WS_COUNTRY_CODE: country_code,
            GV.WS_UNIT_NAME:row['nome'],GV.WS_UNIT_ALPHA:row['sigla'],GV.WS_UNIT_CODE:row['geocodigo'],
            GV.WS_STATE_NAME:row['nome'],GV.WS_STATE_ALPHA:row['sigla'],GV.WS_STATE_CODE:row['geocodigo']}]

    
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results={}
        for download_input in download_list:
            results[download_input['unit_code']] = executor.submit(get_amuIds, bearer_token, download_input)

    fo = extract_results(results)
    return fo    



def extract_results(results):
    fo=[]
    for code, res in results.items():
        fo.append(res.result())
        if (len(fo[-1][GV.WS_AMUIDS]) !=1 ):
            print('***************************** Issues *****************************')
        else:
            fo[-1][GV.WS_AMUIDS]= str(fo[-1][GV.WS_AMUIDS][0])
    return fo

def variables_downloads(row):
    variables = []

    if row[GV.WV_PREC]=='y': variables.append(func_prec)

    if row[GV.WV_TEMP_AVG]=='y': variables.append(func_temp_avg)
    if row[GV.WV_TEMP_MIN]=='y': variables.append(func_temp_min)
    if row[GV.WV_TEMP_MAX]=='y': variables.append(func_temp_max)
    if row[GV.WV_TEMP_SURF]=='y': variables.append(func_temp_surf)
    
    if row['Soil']=='y': variables.append(func_soil)
    if row[GV.WV_HUMI]=='y': variables.append(func_humi)
    if row[GV.WV_VVI]=='y': variables.append(func_vvi)
    
    return variables

def extract_final_df(dfs):
    final_df = pd.concat(dfs,axis=0,ignore_index=True)
    final_df=final_df.drop_duplicates(ignore_index=True, subset=['time'])
    final_df=final_df[['value','time']]
    final_df=final_df.set_index('time')
    final_df=final_df.sort_index(ascending=True)
    return final_df

def update_df(new_df, old_df):
    final_df = pd.concat([new_df,old_df],axis=0,ignore_index=False)
    final_df['time']=final_df.index
    final_df=final_df.drop_duplicates(ignore_index=True, subset=['time'],keep='last')

    final_df=final_df[['value','time']]
    final_df=final_df.set_index('time')
    final_df=final_df.sort_index(ascending=True)
    return final_df


def download_weather(bearer_token, download_input):
    w_id, w_block, var = download_input[GV.WS_AMUIDS], download_input['idBlock'], download_input['var']        
    forecast_only = download_input['forecast_only']
    file_name = download_input['file_name']

    headers_parallel = {'Accept': 'application/json, text/plain, */*',
                        'Authorization': bearer_token,
                        'Connection': 'keep-alive',
                        'Content-Type': 'application/json;charset=UTF-8'}
            
    json_data_var = {'amuIds': [w_id],
                     'idBlock': w_block,
                     'idPixelType': 1,
                     'indicatorTypeIds': var['indicator'],
                     'startDate': '',
                     'endDate': ''}    

    intervals = []        
    dfs,gfss,ecmwfs=[],[],[]
    df_saved={}

    save_file = GV.W_DIR + file_name+'_'+var['name']+'_hist.csv'
    save_gfs_file = GV.W_DIR + file_name+'_'+var['name']+'_gfs.csv'
    save_ecmwf_file = GV.W_DIR + file_name+'_'+var['name']+'_ecmwf.csv'
    

    if forecast_only:
        start_date = dt(dt.now().year - 1,1,1).strftime("%Y-%m-%d")
        intervals.append({'start': start_date,'end': '2035-12-31'})

    elif os.path.exists(save_file):
        df_saved = pd.read_csv(save_file,index_col='time')
        start_date = pd.to_datetime(df_saved.last_valid_index())
        start_date = dt(start_date.year,1,1).strftime("%Y-%m-%d")

        intervals.append({'start': start_date,'end': '2035-12-31'})

    else:
        intervals.append({'start': '1985-01-01','end': '2010-12-31'})
        intervals.append({'start': '2010-01-01','end': '2035-12-31'})   


    for date in intervals:
        json_data_var['startDate']=date['start']
        json_data_var['endDate']=date['end']

        response_var = requests.post(var['func'], headers=headers_parallel, json=json_data_var)
        if (len(response_var.text)==0):                   
            print('******** No Data for: ', var['name'], ' From: ', date['start'],' to: ', date['end'],'********')
            break

        js_var = response_var.json()
        
        df_fore,df_gfs,df_ecmwf = pd.DataFrame(js_var['forecastMeasures']),{},{}

        
        if (not forecast_only):
            df = pd.DataFrame(js_var['observedMeasures'])

            if (len(df)>0):dfs.append(df)
            final_df = extract_final_df(dfs)

            if (len(df_saved)>0): final_df = update_df(df_saved, final_df) # Mofify if there was a pre-existing file
            
        
        if len(df_fore)>0:
            df_gfs=df_fore[df_fore.indicatorTypeId==5]
            df_ecmwf=df_fore[df_fore.indicatorTypeId==4]                                                                          
        
        if (len(df_gfs)>0):gfss.append(df_gfs)
        if (len(df_ecmwf)>0):ecmwfs.append(df_ecmwf)

    if (not forecast_only): final_df.to_csv(save_file)
    if len(gfss)>0:final_gfs = extract_final_df(gfss); final_gfs.to_csv(save_gfs_file)
    if len(ecmwfs)>0:final_ecmwf = extract_final_df(ecmwfs);final_ecmwf.to_csv(save_ecmwf_file)
    
    uu.log('Saved file: '+ save_file)
        

def update_Geosys_Weather(overwrite=False):
    w_sel_df = uw.get_w_sel_df()

    bearer_token=get_bearer_token()

    # Prepare the download list
    download_list=['']

    while len(download_list)>0:
        already_selected=[]
        download_list=[]

        for index, row in w_sel_df.iterrows():            
            variables = variables_downloads(row)
            forecast_only = False

            for var in variables:
                if row['country_alpha']=='USA':
                    file_name = str(row['state_alpha'])
                    forecast_only=True

                elif row['country_alpha']=='BRA':
                    file_name = str(row[GV.WS_AMUIDS])
                    forecast_only=False
                

                if file_name in already_selected: continue

                save_file = GV.W_DIR + file_name+'_'+var['name']+'_hist.csv'
                save_gfs_file = GV.W_DIR + file_name+'_'+var['name']+'_gfs.csv'
                save_ecmwf_file = GV.W_DIR + file_name+'_'+var['name']+'_ecmwf.csv'
                
                if forecast_only:
                    save_file = save_gfs_file

                # the below is to avoid checking forecast files for variables that don't have forecasts
                if (var['name']==GV.WV_HUMI) or(var['name']==GV.WV_VVI) or(var['name']==GV.WV_TEMP_SURF): 
                    save_gfs_file = save_ecmwf_file = save_file

                if (overwrite) or (not uu.updated_today(save_file)) or (not uu.updated_today(save_gfs_file)) or (not uu.updated_today(save_ecmwf_file)):
                    download_list+=[{GV.WS_AMUIDS:row[GV.WS_AMUIDS], 'idBlock':row['idBlock'], 'var':var,'file_name':file_name, 'forecast_only':forecast_only}]

                    already_selected+=[file_name]

        # Downloading
        overwrite=False
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            for index, download in enumerate(download_list):
                executor.submit(download_weather, bearer_token, download)

