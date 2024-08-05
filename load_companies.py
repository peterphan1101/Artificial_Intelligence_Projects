import requests
import pandas as pd
import yaml
import time
from requests.exceptions import ConnectionError, Timeout, RequestException
import os
import argparse
from edgar_data_load import get_edgar_filings, upload_to_mongo
from load_to_vectdb import load_to_vectdb

def read_sample_companies(file_path):
    with open(file_path, 'r') as file:
        sample_companies = yaml.safe_load(file)
    return sample_companies['companies']

def load_companies_to_mongo(companies, mongo_db):
    for _company in companies:
        _company_cik = _company['cik']
        _filings = get_edgar_filings(_company_cik)
        # print(f"Inside load_companies_to_mongo. Loading _filings : {_filings}")
        upload_to_mongo(_filings, mongo_db)

def load_companies_to_vectdb(companies, mongo_db, vector_db):
    print(f"Inside load_companies_to_vectdb. \n companies : {companies}, \n mongo_db : {mongo_db} \n vector_db : {vector_db}\n")
    for _company in companies:
        _company_cik = _company['cik']
        load_to_vectdb(_company_cik, mongo_db, vector_db)

