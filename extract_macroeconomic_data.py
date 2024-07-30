import requests

def fetch_macro_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

# macro data
macro_url = "https://data.nber.org/data/cycles/"

# Fetch and handle macroeconomic data
macro_data = fetch_macro_data(macro_url)
if macro_data:
    print("Successfully fetched macroeconomic data")
else:
    print("Failed to fetch macroeconomic data")
