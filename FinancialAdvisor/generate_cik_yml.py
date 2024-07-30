import requests
from bs4 import BeautifulSoup
import yaml
import pandas as pd

def fetch_cik(company_name):
    base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    headers = {
        "User-Agent": "Mozilla/5.0 (your.email@example.com)"
    }
    variations = [
        company_name,
        company_name.replace(" Inc.", ""),
        company_name.replace(" Inc.", ""),
        company_name.replace(" Corp.", ""),
        company_name.replace(" Corporation", ""),
        company_name.replace(" Corp", "")
    ]
    
    for name_variation in variations:
        params = {
            "company": name_variation,
            "owner": "exclude",
            "action": "getcompany",
            "output": "atom"
        }
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "xml")
            company_info = soup.find("company-info")
            if company_info:
                cik_tag = company_info.find("cik")
                if cik_tag:
                    return cik_tag.text.strip()
        else:
            print(f"Failed to fetch data for {name_variation}. Status code: {response.status_code}")
    print(f"CIK tag not found for {company_name}. Response content: {response.content}")
    return None

def create_cik_yaml(companies, output_file):
    company_list = []
    for company in companies:
        cik = fetch_cik(company)
        if cik:
            company_list.append({
                "name": company,
                "cik": str(cik)  # Ensure CIK is a string
            })
        else:
            print(f"CIK not found for {company}")
    
    with open(output_file, 'w') as file:
        yaml.dump({"companies": company_list}, file)

def read_companies_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df['name'].tolist()

csv_file = 'in/companies.csv'
output_file = 'out/companies_with_cik.yml'

companies = read_companies_from_csv(csv_file)
create_cik_yaml(companies, output_file)

with open(output_file, 'r') as file:
    print(file.read())



# app-1  | companies:
# app-1  | - cik: 0000320193
# app-1  |   name: Apple Inc.
# app-1  | - cik: 0000789019
# app-1  |   name: Microsoft Corp
# app-1  | - cik: 0001728203
# app-1  |   name: Tesla Inc.

