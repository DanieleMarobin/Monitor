# from __future__ import print_function

# import os.path

# from google.oauth2.credentials import Credentials
# from googleapiclient.discovery import build

# # If modifying these scopes, delete the file token.json.
# SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

# def get_gdrive_files_metadata():
#     """Shows basic usage of the Drive v3 API.
#     Prints the names and ids of the first 10 files the user has access to.
#     """
#     creds = None
#     # The file token.json stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists('token.json'):
#         creds = Credentials.from_authorized_user_file('token.json', SCOPES)
#     # If there are no (valid) credentials available, let the user log in.

#     service = build('drive', 'v3', credentials=creds)

#     # Call the Drive v3 API
#     results = service.files().list(pageSize=100, fields="nextPageToken, files(id, name)").execute()
#     items = results.get('files', [])

#     if not items:
#         print('No files found.')
#         return

#     return items
        