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

def download_file_old(creds: Credentials, folder_name = None, file_name = None, file_id = None):
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

def download_file(creds: Credentials, file_path = None, file_id = None):
    try:
        # create gmail api client
        service = build('drive', 'v3', credentials=creds)

        # if we don't have a 'file_id' it is necessary to find it
        if (file_id==None):
            split = file_path.split('/')
            folders = split[0:-1]
            file_name = split[-1]
            
            print('Folders:', folders)
            print('File Name:', file_name)

            path_folders_id=[search_file(creds=creds, query=f"name = '{folder}'")[0]['id']  for folder in split[0:-1]]
            queries = [f"'{folder_id}' in parents" for folder_id in path_folders_id]
            print('folders_queries', queries)

            queries= [queries[-1]] # This is because GDrive has only 1 parent
            queries= [] # This is because GDrive has only 1 parent

            queries.append(f"name = '{file_name}'")
            query=' and '.join(queries)

            print('Final query:',query)
            found_files=search_file(creds=creds, query=query)
            print('found_files',found_files)
            file_id=found_files[0]['id']

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



# From here
# https://stackoverflow.com/questions/41741520/how-do-i-search-sub-folders-and-sub-sub-folders-in-google-drive


FOLDER_TO_SEARCH = '123456789'  # ID of folder to search
DRIVE_ID = '654321'  # ID of shared drive in which it lives
MAX_PARENTS = 500  # Limit set safely below Google max of 599 parents per query.




def get_all_folders_in_drive(creds):
    """
    Return a dictionary of all the folder IDs in a drive mapped to their parent folder IDs (or to the
    drive itself if a top-level folder). That is, flatten the entire folder structure.
    """
    drive_api_ref = build('drive', 'v3', credentials=creds)
    folders_in_drive_dict = {}
    page_token = None
    max_allowed_page_size = 1000
    just_folders = "trashed = false and mimeType = 'application/vnd.google-apps.folder'"
    while True:
        results = drive_api_ref.files().list(
            pageSize=max_allowed_page_size,
            fields="nextPageToken, files(id, name, mimeType, parents)",
            includeItemsFromAllDrives=True, supportsAllDrives=True,
            # corpora='drive',
            # driveId=DRIVE_ID,
            pageToken=page_token,
            q=just_folders).execute()
        folders = results.get('files', [])
        page_token = results.get('nextPageToken', None)
        for folder in folders:
            # folders_in_drive_dict[folder['id']] = folder['parents'][0]
            print('folder',folder)
            print('folderid',folder['id'])
            if 'parents'in folder:
                print('parents',folder['parents'])

        if page_token is None:
            break
    return folders_in_drive_dict


def get_subfolders_of_folder(folder_to_search, all_folders):
    """
    Yield subfolders of the folder-to-search, and then subsubfolders etc. Must be called by an iterator.
    :param all_folders: The dictionary returned by :meth:`get_all_folders_in-drive`.
    """
    temp_list = [k for k, v in all_folders.items() if v == folder_to_search]  # Get all subfolders
    for sub_folder in temp_list:  # For each subfolder...
        yield sub_folder  # Return it
        yield from get_subfolders_of_folder(sub_folder, all_folders)  # Get subsubfolders etc


def get_relevant_files(self, relevant_folders):
    """
    Get files under the folder-to-search and all its subfolders.
    """
    relevant_files = {}
    chunked_relevant_folders_list = [relevant_folders[i:i + MAX_PARENTS] for i in
                                     range(0, len(relevant_folders), MAX_PARENTS)]
    for folder_list in chunked_relevant_folders_list:
        query_term = ' in parents or '.join('"{0}"'.format(f) for f in folder_list) + ' in parents'
        relevant_files.update(get_all_files_in_folders(query_term))
    return relevant_files


def get_all_files_in_folders(self, parent_folders, creds):
    """
    Return a dictionary of file IDs mapped to file names for the specified parent folders.
    """
    drive_api_ref = build('drive', 'v3', credentials=creds)
    files_under_folder_dict = {}
    page_token = None
    max_allowed_page_size = 1000
    just_files = f"mimeType != 'application/vnd.google-apps.folder' and trashed = false and ({parent_folders})"
    while True:
        results = drive_api_ref.files().list(
            pageSize=max_allowed_page_size,
            fields="nextPageToken, files(id, name, mimeType, parents)",
            includeItemsFromAllDrives=True, supportsAllDrives=True,
            corpora='drive',
            driveId=DRIVE_ID,
            pageToken=page_token,
            q=just_files).execute()
        files = results.get('files', [])
        page_token = results.get('nextPageToken', None)
        for file in files:
            files_under_folder_dict[file['id']] = file['name']
        if page_token is None:
            break
    return files_under_folder_dict







if __name__=='__daniele__':
    # os.system('cls')
    print('*************************************')
    creds = get_credentials()

    file_path=GV.W_LAST_UPDATE_FILE
    file_path='Data/Test1/moline/Weather/last_update.csv'
    
    test_f=download_file(creds=creds, file_path=file_path)
    # test_f=download_file(creds=creds, file_path='last_update.csv')

    # test_f=download_file_old(creds=creds, folder_name = 'Weather', file_name = 'last_update.csv')

    df=pd.read_csv(test_f)
    print(df)

    print('Done')
  

if __name__ == "__main__":
    print('*************************************************')
    creds = get_credentials()
    all_folders_dict = get_all_folders_in_drive(creds)  # Flatten folder structure
    
    print(all_folders_dict)
    print('Done')
    # relevant_folders_list = [FOLDER_TO_SEARCH]  # Start with the folder-to-archive
    # for folder in get_subfolders_of_folder(FOLDER_TO_SEARCH, all_folders_dict):
    #     relevant_folders_list.append(folder)  # Recursively search for subfolders
    # relevant_files_dict = get_relevant_files(relevant_folders_list)  # Get the files