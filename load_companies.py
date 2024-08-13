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
    """
    read_sample_companies method takes the path of sample companies yaml file and return a dictionary with company cik and names
    
    Parameters:
    file_path (str): file path to sample companies yaml file.
    
    Returns:
    sample_companies (list): list of companies with their cik and name.
    """     
    with open(file_path, 'r') as file:
        sample_companies = yaml.safe_load(file)
    return sample_companies['companies']

def load_companies_to_mongo(companies, mongo_db):
    """
    load_companies_to_mongo method gets the filings data from SEC Edgar Api and loads them to MongoDB.
    
    Parameters:
    companies (list): list of companies with cik and name.
    mongo_db (object): MongoDB client with connection to MongoDB
    
    Returns:
    None
    """        
    for _company in companies:
        _company_cik = _company['cik']
        _filings = get_edgar_filings(_company_cik)
        # print(f"Inside load_companies_to_mongo. Loading _filings : {_filings}")
        upload_to_mongo(_filings, mongo_db)

def load_companies_to_vectdb(companies, mongo_db, vector_db):
    """
    load_companies_to_vectdb method gets the data from MongoDB and loads them vector db using the generated embeddings.
    
    Parameters:
    companies (list): list of companies with cik and name.
    mongo_db (object): MongoDB client with connection to MongoDB
    vector_db (object): FAISS vector DB connection object
    
    Returns:
    None
    """ 
    # print(f"Inside load_companies_to_vectdb. \n companies : {companies}, \n mongo_db : {mongo_db} \n vector_db : {vector_db}\n")
    for _company in companies:
        _company_cik = _company['cik']
        load_to_vectdb(_company_cik, mongo_db, vector_db)

