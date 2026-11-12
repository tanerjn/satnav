import json

# Example snippet to check the first and last timestamps in the JSON file
with open("STARLINK-45532_2026-01-01_2026-02-01.json", "r") as f:
    data = json.load(f)

# Assuming it's a list of states or dictionary with time keys
print("First record time:", data[0] if isinstance(data, list) else list(data.keys())[0])
print("Last record time:", data[-1] if isinstance(data, list) else list(data.keys())[-1])
