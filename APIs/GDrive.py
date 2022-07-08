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
"""

import sys;
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

import os
import os.path
from io import BytesIO
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from apiclient import http

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

def search_file(creds: Credentials, query: str = "name = 'last_update.csv'"):
    """
    Search file in drive location

    Search for 'Query string examples' in the below webpage:
        https://developers.google.com/drive/api/guides/search-files

          -> query: str = "mimeType='text/csv'"
          -> query: str = "name = 'daniele.csv'"
    """
    # creds, _ = google.auth.default() # doesn't work

    try:
        # create gmail api client
        service = build('drive', 'v3', credentials=creds)
        files = []
        page_token = None
        while True:
            # pylint: disable=maybe-no-member
            response = service.files().list(q=query,spaces='drive',fields='nextPageToken, ''files(id, name)',pageToken=page_token).execute()
            for file in response.get('files', []):
                # Process change
                print(F'Found file: {file.get("name")}, {file.get("id")}')
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

    except HttpError as error:
        print(F'An error occurred: {error}')
        files = None

    return files

def download_file(creds: Credentials, folder_name = None, file_name = None, file_id = None):
    try:
        # create gmail api client
        service = build('drive', 'v3', credentials=creds)

        # if we don't have a 'file_id' it is necessary to find it
        if (file_id==None):
          # Folder and Filename
          if folder_name!=None:
            query=f"name = '{folder_name}'"
            folder_id=search_file(creds=creds, query=query)[0]['id']
          else:
            folder_id=None

          if (folder_id!=None) and (file_name!=None):
            query=f"'{folder_id}' in parents and name = '{file_name}'"
            print('query ->',query)
            file_id=search_file(creds=creds, query=query)[0]['id']
            print('file_id ->',file_id)


          # Filename only
          elif (folder_name == None) and (file_name!=None):
            query=f"name = '{file_name}'"            
            file_id=search_file(creds=creds, query=query)[0]['id']

        request = service.files().get_media(fileId=file_id)
        file = BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None

    file.seek(0)
    
    return file

if __name__=='__main__':  
  os.system('cls')
  creds = get_credentials()

  # print('*************************************')  
  # test_f=download_file(creds=creds, file_id='1BtJc0osxWlsW7zDttOi7sKPuwbei5U54')
  # df=pd.read_csv(test_f)
  # print(df)

  # print('*************************************')
  # test_f=download_file(creds=creds, file_name='last_update.csv')
  # df=pd.read_csv(test_f)
  # print(df)
  
  print('*************************************')
  test_f=download_file(creds=creds, folder_name='Test 2', file_name='daniele.csv')
  df=pd.read_csv(test_f)
  print(df)

  print('Done')
  