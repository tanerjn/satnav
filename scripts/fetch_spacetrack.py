import os
import requests
from dotenv import load_dotenv

# Load environment variables from a local .env file if present
load_dotenv()

USERNAME = os.getenv("SPACETRACK_USERNAME")
PASSWORD = os.getenv("SPACETRACK_PASSWORD")
NORAD_ID = 45532  # Example: Starlink satellite

if not USERNAME or not PASSWORD:
    raise ValueError(
        "❌ Missing credentials! Please set SPACETRACK_USERNAME and SPACETRACK_PASSWORD "
        "as environment variables or inside a .env file."
    )

def fetch_space_track_data(norad_id):
    session = requests.Session()

    # 1. Authenticate via Space-Track login endpoint
    login_url = "https://www.space-track.org/ajaxauth/login"
    payload = {"identity": USERNAME, "password": PASSWORD}

    print("Authenticating with Space-Track...")
    response = session.post(login_url, data=payload)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to authenticate. Status code: {response.status_code}")

    # 2. Query the GP endpoint for the specified NORAD ID (returns OMM JSON with TLE lines)
    api_url = f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/limit/1/format/json"

    print(f"Fetching orbital data for NORAD ID {norad_id}...")
    data_response = session.get(api_url)

    if data_response.status_code == 200:
        satellite_data = data_response.json()
        if satellite_data:
            record = satellite_data[0]
            print("\n--- Successfully Fetched Satellite Record ---")
            print(f"Satellite Name: {record.get('OBJECT_NAME')}")
            print(f"NORAD ID:       {record.get('NORAD_CAT_ID')}")
            print(f"Epoch:          {record.get('EPOCH')}")
            print(f"TLE Line 1:     {record.get('TLE_LINE1')}")
            print(f"TLE Line 2:     {record.get('TLE_LINE2')}")
            return record
        else:
            print(f"⚠️ No data found for NORAD ID {norad_id}.")
            return None
    else:
        print(f"❌ Error fetching data: {data_response.status_code} - {data_response.text}")
        return None

if __name__ == "__main__":
    fetch_space_track_data(NORAD_ID)
