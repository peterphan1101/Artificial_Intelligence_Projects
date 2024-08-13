# Financial Advisor using LLM
### A Retrieval Augmented Generation Application

Financial Advisor application is a Large Language Model (LLM) based personal financial advising application that leverages the cutting-edge capabilities of LLMs such as Natural Langugage Understanding (NLU), Natural Langugage Processing (NLP) and Generation (GenAI). It digests and understands copious amount of freely available and proprietary financial data to advise on financial status of publicly traded companies and enables users make informed decision on personal investing in stocks and other financial instruments. These Retrieval Augmented Generation capabilities combined with Search tool enable users to not only get historical data of a given company, but also current financial information such as opening closing stock pricess and sentiment from search results.

As part of this limited prototype and demonstration of Large Language Models technical capabilities of NLU, NLP, GenAI, Agentic and RAG capabilities for financial advising, we used Form 10-Q (Quarterly) and Form 10-K (Annual) filings of sample public traded companies such as Apple (APPL), Microsoft (MSFT) and Tesla (TSLA). All publicly traded companies are mandated by United States  Securities and Exchange Comission (SEC) to file Form 10-Q and Form-10K. These forms include wealth of information on the health of the publicly traded companies to allow investors to make informed decisions. A sample Apple Form 10-K with file name `APPL_0000320193.pdf` is included in this repository to provide details on various financial information such as revenue, profit, business units, risks and executive compensation etc., This information is used by LLMs to provide answers and advise on relevant questions asked by users of this application. 

While this prototype is limited to Form 10-Q and Form-K data of sample companies, it demonstrates the ability of LLMs ability to understand large amounts of any financial data and current events (in the form search tool) to provide data-driven advices. 

**Responsible AI Disclaimer:** As with any Artificial Intelligence (AI) generated responses, the AI advices can be incorrect. User of the application should verify the generated responses to make informed investment decisions. Also, note that all investments carry financial risks and can result in loss of money. This is a prototype to demonstrate LLM technical capabilities, use at your own risk.


## Table of Contents

- [Langchain Extract Financial Info](#langchain-extract-financial-info)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
  - [MongoDB Setup](#mongodb-setup)
  - [Debugging MongoDB Connection](#debugging-mongodb-connection)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Environment Variables](#environment-variables)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

The Langchain Extract Financial Info project helps users retrieve and analyze financial documents from the SEC Edgar database. It combines modern machine learning techniques and traditional data retrieval methods to provide insightful information on various companies.

## Features

- Retrieve financial documents (10-K, 10-Q) from the SEC Edgar database.
- Store and manage financial data using MongoDB.
- Utilize FAISS for efficient vector search.
- Interactive Streamlit interface for querying financial data.

## Installation

To get started with the Financial Agent project, follow these steps:

1. Clone the repository:
    
    ```
    git clone [<https://github.com/your-username/financial-agent.git>](https://github.com/peterphan1101/Artificial_Intelligence_Projects)

    ```
    
2. Install the required Python packages:
    
    ```
    pip install -r requirements.txt

    ```
    

## MongoDB Setup

To set up MongoDB for this project, follow these instructions:

1. **Install Docker Desktop** (if not already installed):
    - Download and install Docker Desktop from the official Docker website.
2. **Pull MongoDB Docker Image**:
    
    ```
    docker pull mongo
    ```
    
3. **Run MongoDB Container**:
    
    ```
    docker run --restart=unless-stopped --name llm_mongo -p 27017:27017 -v /workspace/mongo:/workspace/mongo -e MONGO_INITDB_ROOT_USERNAME=mongoadmin -e MONGO_INITDB_ROOT_PASSWORD=YourSecretPasswd mongo

    ```
    
4. **Access MongoDB Shell**:
    
    ```
    docker exec -it llm_mongo mongosh -u mongoadmin -p YourSecretPasswd --authenticationDatabase admin llm_mongo_db
    
    ```
    
5. **Start MongoDB Container**:
    
    ```
    docker start -i llm_mongo

    ```
    

## Debugging MongoDB Connection

If you encounter issues connecting to MongoDB, follow these steps to debug:

1. **Check MongoDB Service**:
Ensure that the MongoDB service is running. You can check this using Docker commands:
    
    ```
    docker ps -a
    docker start llm_mongo

    ```
    
2. **Verify Connection String**:
Make sure the `MONGO_URI` in your `.env` file is correct. It should follow this format:
    
    ```
    MONGO_URI=mongodb://mongoadmin:YourSecretPasswd@localhost:27017/llm_mongo_db

    ```
    
3. **Access MongoDB Shell**:
Try accessing the MongoDB shell to verify credentials:
    
    ```
    docker exec -it llm_mongo mongosh -u mongoadmin -p YourSecretPasswd --authenticationDatabase admin llm_mongo_db

    ```
    
4. **Check Network Issues**:
Ensure there are no network issues preventing access to MongoDB. If you are running the application in a different environment, make sure the MongoDB server is accessible from that environment.


5. **Review Logs**:
Check the logs for any error messages:
    
    ```
    docker logs llm_mongo

    ```
    
6. **Ensure Correct Port Mapping**:
Verify that the correct ports are mapped. The default port for MongoDB is 27017. Ensure that this port is not blocked by your firewall or other security settings.

## Usage

To run the Financial Agent application, execute the following command:

```
streamlit run scripts/start.py

```

## Project Structure

```
personal_finance_with_LLM/
├── data/
│   │
│   ├── APPL_0000320193.pdf
│   ├── financial_info.csv
│   ├── sample_companies.yml
│   └── output_files/
│
│
├── scripts/
│   │
│   ├── debug_start.py
│   ├── edgar_data_load.py
│   ├── extract_macroeconomic_data.py
│   ├── load_companies.py
│   ├── load_to_vectdb.py
│   ├── start.py
│   └── __init__.py
│
│
├── .env
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory with the following content:

```
MONGO_URI="mongodb://mongoadmin:YourSecretPasswd@localhost/" 
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key

```

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](notion://www.notion.so/LICENSE.md) file for details.
