#!/usr/bin/env python3
import requests

# Your Space-Track credentials
username = ""
password = ""

# Base URL and JSON endpoints for NORAD 25544 (ISS)
base_url = "https://www.space-track.org"
satcat_endpoint = (
    "/basicspacedata/query/class/satcat/NORAD_CAT_ID/25544/format/json"
)
tle_endpoint = (
    "/basicspacedata/query/class/tle/NORAD_CAT_ID/25544/orderby/EPOCH"
    " desc/limit/1/format/json"
)

# Fetch SATCAT data (catalog info)
response = requests.get(
    base_url + satcat_endpoint, auth=(username, password)
)
if response.status_code == 200:
  satcat_json = response.json()

# Fetch latest TLE orbit data in JSON format
response_tle = requests.get(base_url + tle_endpoint, auth=(username, password))
if response_tle.status_code == 200:
  tle_json = response_tle.json()

