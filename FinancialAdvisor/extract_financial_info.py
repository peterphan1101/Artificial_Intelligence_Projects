import requests
import pandas as pd
import yaml
import time
from requests.exceptions import ConnectionError, Timeout, RequestException
import os
import argparse

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['companies']

def fetch_financial_data(cik: str, max_retries=3, backoff_factor=0.3) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {
        "User-Agent": "Your Name <your.email@example.com>",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except (ConnectionError, Timeout) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
        except RequestException as e:
            print(f"Request failed: {e}")
            return None
    print(f"Failed to fetch data for CIK {cik} after {max_retries} attempts")
    return None

def extract_financial_info(filings):
    if filings:
        financial_data = {
            'cik': filings['cik'],
            'company_name': filings['entityName'],
            'financials': filings['facts']
        }
        return financial_data
    return None

def parse_financial_data(financials):
    data = []
    for report_type, reports in financials.items():
        for report in reports.values():
            for instance in report['units'].values():
                for record in instance:
                    data.append({
                        'report_type': report_type,
                        'date': record.get('end', 'N/A'),
                        'value': record.get('val', 'N/A')
                    })
    return data

def main(yml_path, output_csv_path):
    companies = read_yaml(yml_path)
    all_data = []

    for company in companies:
        cik = company['cik']
        filings = fetch_financial_data(cik)
        if filings:
            financial_info = extract_financial_info(filings)
            if financial_info:
                parsed_data = parse_financial_data(financial_info['financials'])
                for data in parsed_data:
                    data.update({
                        'cik': financial_info['cik'],
                        'company_name': financial_info['company_name']
                    })
                    all_data.append(data)

    df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Financial data saved to {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest financial data and save to CSV')
    parser.add_argument('--yml_path', required=True, help='Path to the YAML file with CIK codes')
    parser.add_argument('--output_csv_path', required=True, help='Path to save the output CSV file')

    args = parser.parse_args()

    main(args.yml_path, args.output_csv_path)