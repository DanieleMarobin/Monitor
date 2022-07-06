""" Usage
Reference File: # E:\grains trading\jupyter\support\Bloomberg Weather

blp = BLPInterface('//BLP/refdata')

Excel BDP Function
df = blp.bdp('c h2 comdty', 'px_last')
df = blp.bdp('c h2 comdty', ['px_last','volume'])
df = blp.bdp(['c h2 comdty','w h2 comdty'], 'px_last')
df = blp.bdp(['c h2 comdty','w h2 comdty'], ['px_last','volume'])


Excel BDS Function
df = blp.bds('c h2 comdty', 'FUT_CHAIN_LAST_TRADE_DATES')
overrides = {'INCLUDE_EXPIRED_CONTRACTS': 'Y'}
df = blp.bds('c h2 comdty', 'FUT_CHAIN_LAST_TRADE_DATES', overrides)


Excel BDH Function
overrides = {'startDate':'20201231', 'endDate':'20221231', 'periodicitySelection': 'DAILY'}

df = blp.bdh('c h2 comdty', 'PX_LAST', overrides)
df = blp.bdh(['c h2 comdty','w h2 comdty'], 'PX_LAST', overrides)
df = blp.bdh('c h2 comdty', ['px_last','volume'], overrides)
df = blp.bdh(['c h2 comdty','w h2 comdty'], ['px_last','volume'], overrides)


Excel BSRCH Function
blp = BLPInterface('//blp/exrsvc')
overrides = {'location': 'KNYC','fields':'TEMPERATURE','model':'gfs'}
df = blp.bsrch('comdty:weather', overrides)
"""

import blpapi
import pandas as pd
import numpy as np

from datetime import datetime as dt
from pandas import Series
from pandas import DataFrame
import Utilities.GLOBAL as GV


class RequestError(Exception):
    """A RequestError is raised when there is a problem with a Bloomberg API response."""
    def __init__ (self, value, description):
        self.value = value
        self.description = description
        
    def __str__ (self):
        return self.description + '\n\n' + str(self.value)

class BLPInterface:
    """ A wrapper for the Bloomberg API that returns DataFrames.  This class
        manages a //BLP/refdata service and therefore does not handle event
        subscriptions.
    
        All calls are blocking and responses are parsed and returned as 
        DataFrames where appropriate. 
    
        A RequestError is raised when an invalid security is queried.  Invalid
        fields will fail silently and may result in an empty DataFrame.
    """ 
    def __init__ (self, service='//BLP/refdata', host='localhost', port=8194, open=True):
        self.active = False
        self.host = host
        self.port = port
        self.service = service
        if open:
            self.open()
        
    def open (self):
        if not self.active:
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost(self.host)
            sessionOptions.setServerPort(self.port)
            self.session = blpapi.Session(sessionOptions)
            self.session.start()
            self.session.openService(self.service)
            self.refDataService = self.session.getService(self.service)
            self.active = True    
    def close (self):
        if self.active:
            self.session.stop()
            self.active = False

    def bdh(self, securities, fields, overrides={}):
        """ Equivalent to the Excel BDH Function.
        
            If securities are provided as a list, the returned DataFrame will
            have a MultiIndex.
        """

        response = self.sendRequest('HistoricalData', securities = securities, fields = fields, overrides = overrides)
        
        data = []
        keys = []
        
        for msg in response:
            securityData = msg.getElement('securityData')
            fieldData = securityData.getElement('fieldData')
            fieldDataList = [fieldData.getValueAsElement(i) for i in range(fieldData.numValues())]
            
            df = DataFrame()
            
            for fld in fieldDataList:
                for v in [fld.getElement(i) for i in range(fld.numElements()) if fld.getElement(i).name() != 'date']:
                    df.loc[fld.getElementAsDatetime('date'), str(v.name())] = v.getValue()

            df.index = pd.to_datetime(df.index)
            df.replace('#N/A History', np.nan, inplace=True)
            
            keys.append(securityData.getElementAsString('security'))
            data.append(df)
        
        if len(data) == 0:
            return DataFrame()
        if type(securities) == str:
            data = pd.concat(data, axis=1)
            #data.columns.name = 'Field'
        else:
            #data = pd.concat(data, keys=keys, axis=1, names=['Security','Field'])
            data = pd.concat(data, axis=1)
            
        data.index.name = 'Date'
        return data        
    def bdp (self, securities, fields, overrides={}):
        """ Equivalent to the Excel BDP Function.
        
            If either securities or fields are provided as lists, a DataFrame
            will be returned.
        """
        response = self.sendRequest('ReferenceData', securities = securities, fields = fields, overrides = overrides)
        
        data = DataFrame()
        
        for msg in response:
            securityData = msg.getElement('securityData')
            securityDataList = [securityData.getValueAsElement(i) for i in range(securityData.numValues())]
            
            for sec in securityDataList:
                fieldData = sec.getElement('fieldData')
                fieldDataList = [fieldData.getElement(i) for i in range(fieldData.numElements())]
                
                for fld in fieldDataList:
                    data.loc[sec.getElementAsString('security'), str(fld.name())] = fld.getValue()

        
        if data.empty:
            return data
        else: 
            #data.index.name = 'Security'
            #data.columns.name = 'Field'
            return data.iloc[0,0] if ((type(securities) == str) and (type(fields) == str)) else data        
    def bds (self, securities, fields, overrides={}):
        """ Equivalent to the Excel BDS Function.
        
            If securities are provided as a list, the returned DataFrame will
            have a MultiIndex.
            
            You may pass a list of fields to a bulkRequest.  An appropriate
            Index will be generated, however such a DataFrame is unlikely to
            be useful unless the bulk data fields contain overlapping columns.
        """
        response = self.sendRequest('ReferenceData', securities = securities, fields = fields, overrides = overrides)

        data = []
        keys = []
        
        for msg in response:
            securityData = msg.getElement('securityData')
            securityDataList = [securityData.getValueAsElement(i) for i in range(securityData.numValues())]
            
            for sec in securityDataList:
                fieldData = sec.getElement('fieldData')
                fieldDataList = [fieldData.getElement(i) for i in range(fieldData.numElements())]
                
                df = DataFrame()
                
                for fld in fieldDataList:
                    for v in [fld.getValueAsElement(i) for i in range(fld.numValues())]:
                        s = Series()
                        for d in [v.getElement(i) for i in range(v.numElements())]:
                            s[str(d.name())] = d.getValue()
                        df = df.append(s, ignore_index=True)

                if not df.empty:
                    keys.append(sec.getElementAsString('security'))
                    data.append(df.set_index(df.columns[0]))
                    
        if len(data) == 0:
            return DataFrame()
        if type(securities) == str:
            data = pd.concat(data, axis=1)
            #data.columns.name = 'Field'
        else:
            data = pd.concat(data, keys=keys, axis=0, names=['Security',data[0].index.name])
            
        return data
    def bsrch(self, domain, overrides={}):
        response = self.sendRequest('ExcelGetGrid', domain = domain, overrides =  overrides)

        results = []
        for message in response:
            try:
                element = message.asElement()
                titles = element.getElement('ColumnTitles')
                titles = [titles.getValueAsString(i) for i in range(titles.numValues())]
            except Exception as e:
                raise e
            if titles:
                records = element.getElement('DataRecords')
                rows = []
                for i in range(records.numValues()):
                    record = records.getValueAsElement(i)
                    field_list = record.getElement('DataFields')
                    row = []
                    for j in range(field_list.numValues()):
                        field = field_list.getValueAsElement(j)
                        row.append(field.getElement(0).getValue())
                    rows.append(row)
                df = pd.DataFrame(rows, columns=titles)
                #df = df.set_index('Reported Time')
                results.append(df)
            else:
                print("Invalid parameter passed: {}".format(message))
        try:
            return results[0]
        except:
            return results
        
    def sendRequest (self, requestType, securities=[], fields=[], domain='', overrides={}):
        """ Prepares and sends a request then blocks until it can return 
            the complete response.
            
            Depending on the complexity of your request, incomplete and/or
            unrelated messages may be returned as part of the response.
        """
        request = self.refDataService.createRequest(requestType + 'Request')
        
        if type(securities) == str:
            securities = [securities]
        if type(fields) == str:
            fields = [fields]
        
        for s in securities:
            request.getElement("securities").appendValue(s)
        for f in fields:
            request.getElement("fields").appendValue(f)
         
        if requestType == 'HistoricalData':
            for k, v in overrides.items():
                if type(v) == dt:
                    v = v.strftime('%Y%m%d')
                request.set(k, v)

        elif requestType =='ReferenceData':
            for k, v in overrides.items():
                element = request.getElement("overrides").appendElement()
                element.setElement("fieldId", k)
                element.setElement("value", v)        

        elif requestType =='ExcelGetGrid':
            request.set('Domain', domain)

            for k, v in overrides.items():
                element = request.getElement('Overrides').appendElement()
                element.setElement('name', k)
                element.setElement('value', v)

        self.session.sendRequest(request)

        response = []
        while True:
            event = self.session.nextEvent(100)
            for msg in event:
                if msg.hasElement('responseError'):
                    raise RequestError(msg.getElement('responseError'), 'Response Error')
                if msg.hasElement('securityData'):
                    if msg.getElement('securityData').hasElement('fieldExceptions') and (msg.getElement('securityData').getElement('fieldExceptions').numValues() > 0):
                        raise RequestError(msg.getElement('securityData').getElement('fieldExceptions'), 'Field Error')
                    if msg.getElement('securityData').hasElement('securityError'):
                        raise RequestError(msg.getElement('securityData').getElement('securityError'), 'Security Error')
                
                if (msg.messageType() == requestType + 'Response') or (msg.messageType() == 'GridResponse'):
                    response.append(msg)
                
            if event.eventType() == blpapi.Event.RESPONSE:
                break
                
        return response

    def __enter__ (self):
        self.open()
        return self        
    def __exit__ (self, exc_type, exc_val, exc_tb):
        self.close()
    def __del__ (self):
        self.close()

def latest_weather_run(model = 'GFS', model_type = 'DETERMINISTIC', region='US_IL', blp=None, finished=True):
    if (blp==None):
        print('blp==None')
        blp = BLPInterface('//blp/exrsvc')

    sd=dt.today()+pd.DateOffset(days=1) # Start Date
    run = dt(sd.year,sd.month,sd.day,0,0,0)

    df={}

    key='_'.join([model,model_type,str(run.hour)])
    finished_run_len = GV.BB_RUNS_DICT[key]

    while True:
        try:
            run_str = run.strftime("%Y-%m-%dT%H:%M:%S")
            overrides = {'location': region, 'fields':'TEMPERATURE|PRECIPITATION', 'model':model,'publication_date':run_str,'location_time':True, 'type':model_type}
            df = blp.bsrch('comdty:weather', overrides)
            rows = len(df)
            key='_'.join([model,model_type,str(run.hour)])
            finished_run_len = GV.BB_RUNS_DICT[key]

            if finished and rows==finished_run_len:
                return run, rows
            elif not finished and rows>0:
                return run, rows

        finally:
            run=run+pd.DateOffset(hours=-6)
    
