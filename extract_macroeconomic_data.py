import requests

def fetch_macro_data(url: str) -> dict:
    """
    Fetch macroeconomic data from a given URL.

    Args:
        url (str): The URL from which to fetch the macroeconomic data.

    Returns:
        dict: A dictionary containing the macroeconomic data if successful.
        None: If the data could not be fetched or an error occurred.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching macro data: {http_err}")
    except Exception as err:
        print(f"An error occurred while fetching macro data: {err}")
    return None

if __name__ == '__main__':
    # URL for macroeconomic data
    macro_url = "https://data.nber.org/data/cycles/"

    # Fetch and handle macroeconomic data
    macro_data = fetch_macro_data(macro_url)
    if macro_data:
        print("Successfully fetched macroeconomic data.")
        # Process the macro_data as needed
    else:
        print("Failed to fetch macroeconomic data.")
