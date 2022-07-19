"""
Project:
    https://console.cloud.google.com/welcome?project=monitor-353019

On the side bar navigation go to
    1) "APIs and Services"
    2) "Enabled APIs and services" and enable Google Drive API
    3) go to Credentials and create OAuth 2.0 Client IDs
    4) download into the the working folder (E:\grains trading\Streamlit\Monitor) 
       client secret .json file (output of "OAuth client created") and rename it 'credentials.json'
    5) run the 'get_credentials' function and authenticate with google
    6) in the working folder (E:\grains trading\Streamlit\Monitor) a 'token.json' file has been created

Sources:
  Official Guide
    https://developers.google.com/drive/api/guides/about-sdk

    https://discuss.streamlit.io/t/google-drive-csv-file-link-to-pandas-dataframe/8057
    https://developers.google.com/drive/api/v2/reference/files/get

  Scope error:
    https://stackoverflow.com/questions/52135293/google-drive-api-the-user-has-not-granted-the-app-error

Files properties:
    https://developers.google.com/drive/api/v3/reference/files
"""

import sys;
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

import os
import os.path
from io import BytesIO
import pandas as pd
import pandas._libs.lib as lib
import concurrent.futures

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import  Utilities.GLOBAL as GV

def get_credentials() -> Credentials:
  # If modifying these scopes, delete the file token.json.
  # SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
  SCOPES = ['https://www.googleapis.com/auth/drive']

  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists('token.json'):
      creds = Credentials.from_authorized_user_file('token.json', SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
          creds.refresh(Request())
      else:
          flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
          creds = flow.run_local_server(port=0)
      # Save the credentials for the next run
      with open('token.json', 'w') as token:
          token.write(creds.to_json())
  
  return creds

def print_name_id(creds: Credentials, pageSize: int=10) -> None:
  try:
      service = build('drive', 'v3', credentials=creds)

      # Call the Drive v3 API
      results = service.files().list(pageSize=pageSize, fields="nextPageToken, files(id, name)").execute()
      items = results.get('files', [])

      if not items:
          print('No files found.')
          return
          
      print('Files:')
      for item in items:
          print(u'{0} ({1})'.format(item['name'], item['id']))

  except HttpError as error:
    # TODO(developer) - Handle errors from drive API.
    print(f'An error occurred: {error}')

def execute_query(service, query = "name = 'last_update.csv'",fields='files(id, name, mimeType, parents)'):
    fo = []
    page_token = None
    while True:
        response = service.files().list(q=query,spaces='drive',fields='nextPageToken,'+fields,pageToken=page_token).execute()
        fo.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)

        if page_token is None:
            break
    return fo

def download_file_from_id(service, file_id):
    request = service.files().get_media(fileId=file_id)
    file = BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()

    file.seek(0)    
    return file


def download_file_from_path(creds: Credentials, file_path):
    service = build('drive', 'v3', credentials=creds,cache_discovery=False)

    split = file_path.split('/')
    folders = split[0:-1]
    file_name = split[-1]

    fields='files(id, name, mimeType, parents)'

    files_query=query=f"name = '{file_name}'"
    files = execute_query(service=service, query=files_query, fields=fields)

    files_dict={}
    for f in files:
        files_dict[f['id']]={'name':f['name'],'id':f['id'],'parents':f['parents']}

    folders_query = "trashed = false and mimeType = 'application/vnd.google-apps.folder'"
    folders = execute_query(service=service, query=folders_query, fields=fields)

    folders_dict={}
    for f in folders:
        if 'parents' in f:
            folders_dict[f['id']]={'name':f['name'],'id':f['id'],'parents':f['parents']}

    dict_paths_id={}

    for f in files_dict:
        fo=[files_dict[f]['name']]
        dict_paths_id['/'.join(get_parent(id=files_dict[f]['parents'][0],folders_dict=folders_dict,fo=fo))]=f
            
    return download_file_from_id(service=service, file_id= dict_paths_id[file_path])


def get_parent(id,folders_dict,fo):
    if (id in folders_dict) and ('parents' in folders_dict[id]):
        fo.insert(0,folders_dict[id]['name'])
        get_parent(folders_dict[id]['parents'][0],folders_dict,fo)    
    return fo

def read_csv_parallel(donwload_dict,creds=None,max_workers=500):
    fo={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results={}
        for i,v in enumerate(donwload_dict['file_path']):
            file_path=donwload_dict['file_path'][i]
            dtype=donwload_dict['dtype'][i] if 'dtype' in donwload_dict else None
            parse_dates=donwload_dict['parse_dates'][i] if 'parse_dates' in donwload_dict else False
            index_col=donwload_dict['index_col'][i] if 'index_col' in donwload_dict else None
            names=donwload_dict['names'][i] if 'names' in donwload_dict else lib.no_default
            header=donwload_dict['header'][i] if 'header' in donwload_dict else 'infer'
            dayfirst=donwload_dict['dayfirst'][i] if 'dayfirst' in donwload_dict else False

            results[file_path] = executor.submit(read_csv, file_path, creds, dtype, parse_dates, index_col, names, header, dayfirst)
    
    for file_path, res in results.items():
        fo[file_path]=res.result()

    return fo

def read_csv(file_path, creds=None, dtype=None, parse_dates=False, index_col=None, names=lib.no_default, header='infer', dayfirst=False):
    if not os.path.exists(file_path):
        # print('from GDrive:',file_path)
        if creds==None:
            creds = get_credentials()
        file_path=download_file_from_path(creds,file_path)
    # else:
    #     print('from Local',file_path)
        
    return pd.read_csv(file_path,dtype=dtype,parse_dates=parse_dates,index_col=index_col,names=names,header=header,dayfirst=dayfirst)

if __name__ == "__main__":
    get_credentials()