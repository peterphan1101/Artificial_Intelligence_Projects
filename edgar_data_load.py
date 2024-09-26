import argparse
import os

from dotenv import load_dotenv
from edgar import Company, set_identity
from pymongo import MongoClient

# Load OPENAI_API_KEY, MONGO_URI and other secrets/sensitive information from `.env` file
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

# Set identity for SEC EDGAR API
# Including your email helps the SEC identify legitimate users and comply with their guidelines
set_identity(f"Financial Advisor using LLM {EMAIL_ADDRESS}") 

def get_edgar_filings(ticker_or_cik: str):
    """
    Retrieve the last 3 years of 10-K and 10-Q filings for a given company from the SEC EDGAR database.

    Args:
        ticker_or_cik (str): The company's ticker symbol or Central Index Key (CIK).

    Returns:
        list: A list of filing objects for the company.
    """
    num_filings = 12  # Approximate number for the last 3 years (quarterly filings)
    company = Company(ticker_or_cik)
    filings = company.get_filings(form=["10-K", "10-Q"]).latest(num_filings)
    return filings


def upload_to_mongo(filings, mongo_db):
    """
    Upload the retrieved filings to a MongoDB collection.

    Args:
        filings (list): A list of filing objects retrieved from the SEC EDGAR API.
        mongo_db (pymongo.database.Database): The MongoDB database object.

    Returns:
        None
    """
    mongo_collection = mongo_db['FinDocs_text']
    fin_docs = []
    for filing in filings:
        doc = {
            "company": filing.company,
            "cik": filing.cik,
            "form": filing.form,
            "filing_date": filing.filing_date.isoformat(),
            "accession_no": filing.accession_no,
            "financial_doc": filing.text(),
        }
        fin_docs.append(doc)

    if fin_docs:
        result = mongo_collection.insert_many(fin_docs)
        print(f"Inserted {len(result.inserted_ids)} documents into MongoDB collection 'FinDocs_text'.")
    else:
        print("No documents to insert.")


def main(company_ticker: str):
    """
    Main function to retrieve company filings from the SEC EDGAR API and upload them to MongoDB.

    Args:
        company_ticker (str): The company's ticker symbol or CIK.

    Returns:
        None
    """
    print(f"Fetching filings for {company_ticker}...")
    filings = get_edgar_filings(company_ticker)
    if not filings:
        print("No filings found for the given company.")
        return

    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client['llm_mongo_db']  # MongoDB database name
        upload_to_mongo(filings, mongo_db)
    except Exception as e:
        print(f"An error occurred while connecting to MongoDB: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch data from SEC EDGAR API and load the documents into MongoDB.'
    )
    parser.add_argument(
        '--company_ticker',
        required=True,
        help='Ticker symbol or CIK of the company (e.g., MSFT, AAPL).'
    )
    args = parser.parse_args()
    main(args.company_ticker)
