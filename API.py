# Install this first pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

import os
import io
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

def main():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    # ID of the folder you want to download files from.
    folder_id = 'YOUR_FOLDER_ID_HERE'

    # Query to get files from the folder
    query = f"'{folder_id}' in parents"

    # Call the Drive v3 API to get the files
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    # Create a directory for the downloaded files
    local_folder_name = 'Downloaded_Files'
    if not os.path.exists(local_folder_name):
        os.makedirs(local_folder_name)

    if not items:
        print('No files found.')
    else:
        print('Downloading files...')
        for item in items:
            file_id = item['id']
            file_name = item['name']
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f'Download {int(status.progress() * 100)}%.')

            with open(os.path.join(local_folder_name, file_name), 'wb') as f:
                fh.seek(0)
                f.write(fh.read())

            print(f'File "{file_name}" downloaded to the "{local_folder_name}" folder.')

if __name__ == '__main__':
    main()
