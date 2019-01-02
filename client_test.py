import requests
import base64
import json

addr = 'http://localhost:5000'
test_url = addr + '/api/classify'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

with open("data/Equations/Clean/eq1_hr.jpg", 'rb') as f:
    img_encode = base64.b64encode(f.read())

data = {
    "img_encode": img_encode.decode(),
    "format": "file"
}

# send http request with image and receive response
response = requests.post(test_url, json=json.dumps(data), headers=headers)

print(response.text)