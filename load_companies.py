import os
import yaml
import logging
from edgar_data_load import get_edgar_filings, upload_to_mongo
from load_to_vectdb import load_to_vectdb
from requests.exceptions import ConnectionError, Timeout, RequestException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_sample_companies(file_path):
    """
    Reads sample companies from a YAML file.

    Parameters:
    file_path (str): Path to the sample companies YAML file.

    Returns:
    list: List of companies with their CIK and name.
    """
    try:
        with open(file_path, 'r') as file:
            sample_companies = yaml.safe_load(file)
        return sample_companies['companies']
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return []
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return []

def load_companies_to_mongo(companies, mongo_db):
    """
    Fetches SEC filings for the given companies and uploads them to MongoDB.

    Parameters:
    companies (list): List of companies with CIK and name.
    mongo_db (object): MongoDB client connection object.

    Returns:
    None
    """
    for company in companies:
        company_cik = company['cik']
        company_name = company['name']
        try:
            filings = get_edgar_filings(company_cik)
            if filings:
                upload_to_mongo(filings, mongo_db)
                logging.info(f"Uploaded filings for {company_name} (CIK: {company_cik}) to MongoDB.")
            else:
                logging.warning(f"No filings found for {company_name} (CIK: {company_cik}).")
        except (ConnectionError, Timeout, RequestException) as e:
            logging.error(f"Failed to fetch filings for {company_name} (CIK: {company_cik}): {e}")
        except Exception as e:
            logging.error(f"Unexpected error while processing {company_name} (CIK: {company_cik}): {e}")

def load_companies_to_vectdb(companies, mongo_db, vector_db):
    """
    Loads company data from MongoDB into FAISS vector database using embeddings.

    Parameters:
    companies (list): List of companies with CIK and name.
    mongo_db (object): MongoDB client connection object.
    vector_db (object): FAISS vector DB connection object.

    Returns:
    None
    """
    for company in companies:
        company_cik = company['cik']
        company_name = company['name']
        try:
            load_to_vectdb(company_cik, mongo_db, vector_db)
            logging.info(f"Company {company_name} (CIK: {company_cik}) loaded into FAISS vector DB.")
        except Exception as e:
            logging.error(f"Error loading {company_name} (CIK: {company_cik}) into vector DB: {e}")
