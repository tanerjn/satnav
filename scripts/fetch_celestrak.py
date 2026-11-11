#!/usr/bin/env python3
import json
import requests

url = "https://celestrak.org/NORAD/elements/gp.php"
params = {"CATNR": 25544, "FORMAT": "json"}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()
print(json.dumps(data, indent=2))

