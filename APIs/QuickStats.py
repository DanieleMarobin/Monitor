# Quick Stats - Download Interface: https://quickstats.nass.usda.gov
# Quick Stats - API documentation: https://quickstats.nass.usda.gov/api
# Quick Stats - html encoding: https://www.w3schools.com/tags/ref_urlencode.asp
# 50'000 records is the limit for a single call


import pandas as pd

class QS_input():
    def __init__(self):
        self.source_desc=[]
        self.commodity_desc=[]
        self.short_desc=[]
        self.years=[]
        self.reference_period_desc=[]
        self.domain_desc=[]
        self.agg_level_desc=[]

def QS_url(input:QS_input):
    url = 'http://quickstats.nass.usda.gov/api/api_GET/?key=96002C63-2D1E-39B2-BF2B-38AA97CC7B18&'

    for i in input.source_desc:
        url=url + 'source_desc=' + i +'&'
    for i in input.commodity_desc:
        url=url + 'commodity_desc=' + i +'&'
    for i in input.short_desc: 
        url=url + 'short_desc=' + i +'&'
    for i in input.years: 
        url=url + 'year=' + str(i) +'&'
    for i in input.reference_period_desc: 
        url=url + 'reference_period_desc=' + i +'&'
    for i in input.domain_desc:
        url=url + 'domain_desc=' + i +'&'
    for i in input.agg_level_desc:
        url=url + 'agg_level_desc=' + i +'&'

    url=url+'format=CSV'
    url = url.replace(" ", "%20")
    return url

def get_data(input: QS_input):
    url = QS_url(input)        
    fo = pd.read_csv(url,low_memory=False)  
    return fo

def get_yields(commodity='CORN', aggregate_level='NATIONAL', years=[],cols_subset=[]):
    """
    df_yield = qs.get_QS_yields(commodity='SOYBEANS',aggregate_level='NATIONAL', columns_output=['year','Value'])\n
    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.years.extend(years)
    dl.commodity_desc.append(commodity)

    if commodity=='CORN':
        dl.short_desc.append(commodity+', GRAIN - YIELD, MEASURED IN BU / ACRE')
    elif commodity=='SOYBEANS':
        dl.short_desc.append(commodity+' - YIELD, MEASURED IN BU / ACRE')

    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)

    return fo

def get_progress(commodity='CORN', progress_var='PLANTING', aggregate_level='NATIONAL', years=[], cols_subset=[]):
    """
    df_planted=qs.get_QS_planting_progress(commodity='SOYBEANS', aggregate_level='NATIONAL', years=[2017],columns_output=['year','week_ending','Value'])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.years.extend(years)
    dl.commodity_desc.append(commodity)

    if progress_var.lower()=='planting':
        dl.short_desc.append(commodity+' - PROGRESS, MEASURED IN PCT PLANTED')
    elif progress_var.lower()=='silking':
        dl.short_desc.append(commodity+' - PROGRESS, MEASURED IN PCT SILKING')

    dl.agg_level_desc.append(aggregate_level)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='week_ending',ascending=True)

    return fo

def get_production(commodity='CORN', aggregate_level='NATIONAL', years=[], cols_subset=[]):
    """
    df_prod=qs.get_QS_production('soybeans', aggregate_level='COUNTY', years=[2017])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.source_desc.append('SURVEY')
    dl.years.extend(years)
    dl.commodity_desc.append(commodity)

    if commodity=='CORN':
        dl.short_desc.append(commodity+', GRAIN - PRODUCTION, MEASURED IN BU')
    elif commodity=='SOYBEANS':
        dl.short_desc.append(commodity+' - PRODUCTION, MEASURED IN BU')

    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)
    fo['Value'] = fo['Value'].str.replace(',','').astype(float)

    return fo