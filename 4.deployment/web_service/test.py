import requests

# Define the year and month for which you want to get the average predicted duration
data = {
    "year": 2023,
    "month": 5,
}

url = "http://127.0.0.1:9696/predict"

response = requests.post(url, json=data)

try:
    response_data = response.json()
    print(response_data)
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON response:")
    print(response.text)
