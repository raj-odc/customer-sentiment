# Test the API
import requests
import json

# Test data matching TRD example
test_data = {
    "text": "The new UI is beautiful, but the app keeps crashing on startup and the login is slow."
}

response = requests.post('http://localhost:5000/analyze', json=test_data)
print(json.dumps(response.json(), indent=2))