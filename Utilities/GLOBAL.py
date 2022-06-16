from datetime import datetime as dt
from calendar import isleap

def last_leap_year():    
    start=dt.today().year
    while(True):
        if isleap(start): return start
        start-=1

# region STATIC VAR
W_DIR = 'Weather/'
W_SEL_FILE = W_DIR + "weather_selection.csv"
CUR_YEAR = dt.today().year
LLY = last_leap_year()

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