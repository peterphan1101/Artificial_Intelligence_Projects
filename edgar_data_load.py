import os
from edgar import Company, set_identity
from edgar.financials import Financials
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import argparse
from datetime import datetime, date

load_dotenv()
set_identity("Financial Advisor using LLM sdasari5@jhu.edu")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

def get_edgar_filings(ticker_or_cik :str):
    _num_filings = 4  # last 1 year
    _filings = Company(ticker_or_cik).get_filings(form=["10-K", "10-Q"]).latest(_num_filings)
    return _filings

def upload_to_mongo(filings, mongo_db):
    # mongo_client = MongoClient(MONGO_URI)
    # mongo_db = mongo_client['llm_mongo_db']
    # mongo_collection = mongo_db['FinDocs']
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
        # print(f"Inside upload_to_mongo. _doc : {_doc}")
        fin_docs.append(_doc)
    mongo_collection.insert_many(fin_docs)
    # print(f"Data uploaded successfully! for filings : {filings}")

def main(company_ticker):
    print(company_ticker)
    _filings = get_edgar_filings(company_ticker)
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client['llm_mongo_db']
    upload_to_mongo(_filings, mongo_db)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch data from SEC Edgar API and load the documents to MongoDB')
    parser.add_argument('--company_ticker', required=True, help='Ticker of the company for which the data need to be fetched from SEC Edgar DB and loaded into MongoDB. E.g MSFT, APPL etc.,')
    args = parser.parse_args()
    main(args.company_ticker)