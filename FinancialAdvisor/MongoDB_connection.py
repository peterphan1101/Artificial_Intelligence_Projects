import os
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MONGO_URI=mongodb+srv://peterphan1101:a3kjIzvUl9oo1TUo@cluster0.twctxve.mongodb.net/


MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

#  MongoDB connection
client = MongoClient(MONGO_URI)
db = client['LC_10k_10q_12schedulea_data']
collection = db['FinancialReports']

# Load data from CSV
data = pd.read_csv('output_files/financial_info.csv')
records = data.to_dict('records')

# Insert data into MongoDB
collection.insert_many(records)
print("Data uploaded successfully!")
