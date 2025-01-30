import requests

API_URL = "http://127.0.0.1:5001/predict"

data = {"text": "This is an example of a fake news article."}
response = requests.post(API_URL, json=data)

print("Response:", response.json())
