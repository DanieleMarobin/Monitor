"""
Project:
https://console.cloud.google.com/welcome?project=monitor-353019

On the side bar navigation go to
  1) "APIs and Services"
  2) "Enabled APIs and services" and enable Google Drive API
  3) go to Credentials and create OAuth 2.0 Client IDs

"""


# https://stackoverflow.com/questions/52135293/google-drive-api-the-user-has-not-granted-the-app-error
# https://discuss.streamlit.io/t/google-drive-csv-file-link-to-pandas-dataframe/8057
# https://developers.google.com/drive/api/v2/reference/files/get


from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

try:
    service = build('drive', 'v3', credentials=creds)

    # Call the Drive v3 API
    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
        
    print('Files:')
    for item in items:
        print(u'{0} ({1})'.format(item['name'], item['id']))
except HttpError as error:
    print(f'An error occurred: {error}')










from apiclient import http


def print_file_metadata(service, file_id):
  """Print a file's metadata.

  Args:
    service: Drive API service instance.
    file_id: ID of the file to print metadata for.
  """

  return(service.files().get(fileId=file_id).execute())


def print_file_content(service, file_id):
  """Print a file's content.

  Args:
    service: Drive API service instance.
    file_id: ID of the file.

  Returns:
    File's content if successful, None otherwise.
  """
  # print (service.files().get_media(fileId=file_id).execute())
  return service.files().get_media(fileId=file_id).execute()


def download_file(service, file_id, local_fd):
  """Download a Drive file's content to the local filesystem.

  Args:
    service: Drive API Service instance.
    file_id: ID of the Drive file that will downloaded.
    local_fd: io.Base or file object, the stream that the Drive file's
        contents will be written to.
  """
  request = service.files().get_media(fileId=file_id)
  media_request = http.MediaIoBaseDownload(local_fd, request)

  while True:
    download_progress, done = media_request.next_chunk()

    if download_progress:
      print ('Download Progress: %d%%' % int(download_progress.progress() * 100))
    if done:
      print ('Download Complete')
      return