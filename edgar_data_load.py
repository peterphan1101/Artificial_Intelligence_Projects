import os
from edgar import Company, set_identity
from edgar.financials import Financials
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import argparse
from datetime import datetime, date

# Load OPENAI_API_KEY, MONGO_URI and other secrets/sensitive information from .env file
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

# Set identity of client for SEC Edgar Api
set_identity("Financial Advisor using LLM sdasari5@jhu.edu")

def get_edgar_filings(ticker_or_cik :str):
    """
    get_edgar_filings function gets the last 3 years of 10-K and 10-Q filings of a given company and returns those filings for further processing.
    
    Parameters:
    ticker_or_cik (str): Company ticker symbol or CIK.
    
    Returns:
    _filings (list): list of company 10-K and 10-Q filings.
    """     
    _num_filings = 12  # last 3 years
    _filings = Company(ticker_or_cik).get_filings(form=["10-K", "10-Q"]).latest(_num_filings)
    return _filings

def upload_to_mongo(filings, mongo_db):
    """
    upload_to_mongo function uploads the Edgar Api returned filings data to Mongo DB.
    
    Parameters:
    filings (list): list of company 10-K and 10-Q filings.
    mongo_db (object): Mongo DB connection object.
    
    Returns:
    None
    """    
    mongo_collection = mongo_db['FinDocs_text']
    fin_docs = []
    for _filing in filings:
        _doc = {
            "company": _filing.company, 
            "cik":_filing.cik,
            "form": _filing.form,
            "filing_date": _filing.filing_date.isoformat(), 
            "accession_no": _filing.accession_no,
            "financial_doc": _filing.text()
            }
        fin_docs.append(_doc)
    mongo_collection.insert_many(fin_docs)

def main(company_ticker):
    """
    main method loads data from SEC Edgar Api to MongoDB. This is used for testing and offline loading of a specific company data to MongoDB.
    
    Parameters:
    company_ticker (str): Company ticker symbol (or CIK).
    
    Returns:
    None
    """    
    print(company_ticker)
    _filings = get_edgar_filings(company_ticker)
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client['llm_mongo_db']     # 'llm_mongo_db' Mongo DB name
    upload_to_mongo(_filings, mongo_db)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch data from SEC Edgar API and load the documents to MongoDB')
    parser.add_argument('--company_ticker', required=True, help='Ticker of the company for which the data need to be fetched from SEC Edgar DB and loaded into MongoDB. E.g MSFT, APPL etc.,')
    args = parser.parse_args()
    main(args.company_ticker)