import requests
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def download_file(url, output_folder):
    """
    Downloads a file from the given URL and saves it in the specified folder with the original filename.

    Parameters:
    url (str): The URL of the file to download.
    output_folder (str): The folder to save the downloaded file in.
    """
    # Extract the filename from the URL
    filename = os.path.basename(url)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the full path for the output file
    filepath = os.path.join(output_folder, filename)

    # Setup retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    # Send a GET request to the URL
    try:
        response = http.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        # Write the content to a file
        with open(filepath, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file. Error: {e}")
