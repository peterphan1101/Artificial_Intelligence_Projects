# Langchain Extract Financial Info

## Overview

This project streamlines the process of extracting financial information for companies from the SEC EDGAR database. The extraction is carried out using Python scripts, which can be run either locally or within Docker containers. The financial data gathered is then stored in CSV files for subsequent analysis.

## Project Structure
```
Langchain_Extract_Financial_Info/
├── input_files/
│   └── companies.csv
├── output_files/
│   └── companies_with_cik.yml
│   └── financial_data.csv
├── system_design/
├── docker-compose.yml
├── Dockerfile
├── generate_cik_yaml.py
├── extract_financial_info.py
├── requirements.txt
└── README.md
```


## Prerequisites

- Python 3.11
- Docker and Docker Compose

## Setup

### Local Setup

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/Langchain_Extract_Financial_Info.git
   cd Langchain_Extract_Financial_Info
   ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3 **Run the Scripts:**:
    - Generate CIK YAML:

        ```sh
        python generate_cik_yaml.py
        ```
    - Extract Financial Information:

        ```sh
        python extract_financial_info.py --yaml_path=/Users/peter/Documents/Projects/Langchain_Extract_Financial_Info/output_files/companies_with_cik.yml --output_csv_path=/Users/peter/Documents/Projects/Langchain_Extract_Financial_Info/output_files/financial_data.csv
        ```


### Docker Setup
1. **Build the Docker Image**:
        ```sh
        docker-compose build
        ```

2. **Run Docker Compose**:
        ```sh
        docker-compose build
        ``` 

3. **Run the Ingestion Script to Save Data to CSV**:
        ```sh
        docker-compose run app python extract_financial_info.py --yaml_path=/app/output_files/companies_with_cik.yml --output_csv_path=/app/output_files/financial_data.csv
        ```

## Configuration
- **Input File**: `input_files/companies.csv`
    - A CSV file containing the names of companies to fetch CIK numbers.

- **Output Files**:
    - `output_files/companies_with_cik.yml`: YAML file containing company names and their corresponding CIK numbers.
    - `output_files/financial_data.csv`: CSV file containing the extracted financial data.


## Scripts:
   - **generate_cik_yaml.py**:
    Fetches CIK numbers for companies listed in `input_files/companies.csv and saves them to output_files/companies_with_cik.yml`.

   - **extract_financial_info.py**:
    Reads the CIK numbers from `output_files/companies_with_cik.yml`, fetches the financial data from the SEC EDGAR API, and saves it to `output_files/financial_data.csv`.

file_path = "/mnt/data/README.md"

with open("/mnt/data/README.md", "w") as file:
    file.write(readme_content)

file_path
