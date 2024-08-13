# Financial Advisor using LLM
### A Retrieval Augmented Generation Application

Financial Advisor application is a Large Language Model (LLM) based personal financial advising application that leverages the cutting-edge capabilities of LLMs such as Natural Langugage Understanding (NLU), Natural Langugage Processing (NLP) and Generation (GenAI). It digests and understands copious amount of freely available and proprietary financial data to advise on financial status of publicly traded companies and enables users to make informed decisions on personal investing in stocks and other financial instruments. These Retrieval Augmented Generation capabilities combined with search tool enable users to not only get historical data of a given company, but also current financial information such as opening, closing stock prices and market sentiment about company from search results.

As part of this limited prototype and demonstration of Large Language Models technical capabilities of NLU, NLP, GenAI, Agentic and RAG capabilities for financial advising, we used Form 10-Q (Quarterly) and Form 10-K (Annual) filings of sample public traded companies such as Apple (APPL), Microsoft (MSFT) and Tesla (TSLA). All publicly traded companies are mandated by United States  Securities and Exchange Comission (SEC) to file Form 10-Qs and Form-10Ks. These forms include wealth of information on the health of the publicly traded companies to allow investors to make informed decisions. A sample Apple (APPL) Form 10-K with file name `APPL_0000320193.pdf` is included in this repository to provide details on various sections of financial information such as revenue, profit, business units, risks and executive compensation etc. that are included in these forms. This information is used by LLMs to provide answers on relevant questions asked by users of this application. 

While this prototype is limited to Form 10-Q and Form-K data of sample companies from SEC Edgar data source, this application is not limited to such data. In a production deployment, a dedicated data pipeline can be built to load structured (spreadsheets, JSON, SQL data etc.,) and unstructured data (Text, PDF and Word documents etc.,) into data stores and make them available to LLM for natural language related tasks to generate responses. This prototype is designed to demonstrates the capabilities of LLMs to understand large amounts of financial data and current events (in the form search tool) to provide data-driven advices. 

**Responsible AI Disclaimer:** As with any Artificial Intelligence (AI) generated responses, the AI advices can be incorrect. User of the application should verify the generated responses to make informed investment decisions. Also, note that all investments carry financial risks and can result in loss of money. This is a prototype to demonstrate LLM technical capabilities, use at your own risk.


## Table of Contents

- [Financial Advisor using LLM](#financial-advisor-using-llm)
  - [Technology Stack](#technology-stack)
  - [System Design](#system-design)
  - [Installation](#installation)
  - [MongoDB Setup](#mongodb-setup)
  - [Debugging MongoDB Connection](#debugging-mongodb-connection)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Environment Variables](#environment-variables)
  - [Contributing](#contributing)
  - [License](#license)

## Technology Stack

The Financial Advisor application is built using modern AI technology stack from various providers both open source and proprietary technologies. The major components utilized and their functions are outlined below:

1. **Open AI GPT-4o Model:** Core Large Language Model (LLM) for natural language tasks such as NLU, NLP and GenAI
2. **Open AI text-embedding-ada-002 Model:** For generating embeddings for the text data.
3. **LangChain:** A framework for building LLM based AI applications. Serves as overall orchestrator.
4. **Mongo DB:** A NoSQL document database for storing unstructured data such as text, html, json and other files. Used for storing Form 10-Q and Form 10-K documents.
5. **Facebook AI Similarity Search (FAISS):** A vector database store for storing and efficiently searching the vectors generated for documents using Open AI embedding model.
6. **Streamlit:** A user experience framework for building and sharing AI model and data based applications.
7. **Python-edgar and SEC Edgar API:** Python-edgar library is used for getting the 10-K and 10-Q documents from SEC EDGAR Datasource API.

## System Design

Please see system design documents included in /system_design/ folder.

## Installation

To get started with the Financial Advisor using LLM applicaation, follow these steps:

1. Clone the repository:
    
    ```
    git clone https://github.com/peterphan1101/Artificial_Intelligence_Projects.git

    ```
    
2. Install the required Python packages:
    
    ```
    pip install -r requirements.txt

    ```
3. Create a .env file with your API keys/secrets. This project utilizes the following API keys for various functionality requirements. Content of .env should look like the below once API key values are replaced with valid values.

    ```
    OPENAI_API_KEY = "sk-proj-7**************"
    SERPAPI_API_KEY = "0******************3"
    LANGCHAIN_API_KEY = "lsv2_pt_c***************0"
    LANGCHAIN_TRACING_V2 = "true"
    MONGO_URI = "mongodb://mongoadmin:YourSecretPasswd@localhost/"
    HF_API_KEY = "hf_l************************c"
    HF_HUB_DISABLE_SYMLINKS_WARNING = "True"
    
    ```

## MongoDB Setup

To set up MongoDB for this application, follow these instructions:

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
    
5. **Restarting stopped MongoDB Container**:
    
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
Ensure there are no network issues preventing access to MongoDB. You may need to open port *27017* for accessing MangoDB. If you are running the application in a different environment, make sure the MongoDB server is accessible from that environment.


5. **Review Logs**:
Check the logs for any error messages:
    
    ```
    docker logs llm_mongo

    ```
    
6. **Ensure Correct Port Mapping**:
Verify that the correct ports are mapped. The default port for MongoDB is 27017. Ensure that this port is not blocked by your firewall or other security settings.

## Usage

To run the Financial Advisor application, execute the following command Python environment specific terminal:

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
