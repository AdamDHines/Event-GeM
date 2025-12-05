import os
import requests
import torch

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)


def download_file_from_google_drive(destination, file_id="1kH9GAzNHvEBXIcM69TuCgk9VQ9psQYhe"):
    """
    Download a file from Google Drive, handling the virus-scan confirmation
    page for large files.
    """
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    _save_response_content(response, destination)
