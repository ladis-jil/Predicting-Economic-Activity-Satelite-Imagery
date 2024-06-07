import numpy as np
import requests

name = 'Addis Ababa'
api_url = 'https://api.api-ninjas.com/v1/city?name={}'.format(name)
response = requests.get(api_url, headers={'X-Api-Key': ''})
if response.status_code == requests.codes.ok:
    print(response.text)
else:
    print("Error:", response.status_code, response.text)
