import requests
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from joblib import load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import time
from bs4 import BeautifulSoup

class AnnualFilingCluster:
    def __init__(self, email):
        self.headers = {'User-Agent': email}
        self.names = []
        self.data = None
        self.results = None
        self.companyData = None

        self._companyData()

    def _companyData(self):
        companyTickers = requests.get('https://www.sec.gov/files/company_tickers.json', headers=self.headers)
        companyData = pd.DataFrame.from_dict(companyTickers.json(), orient='index')
        companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10)
        self.companyData = companyData
  
    def retrieveData(self, tickers, form='10-K', include=None, exclude=None):
        """
        Fetches the 10-K filing documents for a given list of companies by their CIKs.
        
        :param ciks: List of strings representing the company CIKs.
        :return: A dictionary where the key is the CIK and the value is the filing information.
        """

        data = {}

        if not isinstance(tickers, list):
            tickers = [tickers]  # Convert a single CIK string to a list for uniform handling

        for ticker in tickers:
            # Get form data
            time.sleep(1)
            cik = self.companyData[self.companyData['ticker'] == ticker]['cik_str'].values[0]
            filingMetadata = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=self.headers)
            filingMetadata.raise_for_status()  # Raise an error for bad responses
            allForms = pd.DataFrame.from_dict(filingMetadata.json()['filings']['recent'])
            formAccessionNumber = allForms[allForms['form'] == form]['accessionNumber'].iloc[0]
            formDocumentLink = allForms[allForms['form'] == form]['primaryDocument'].iloc[0]
            cik_url, accession_url = cik.lstrip('0'), formAccessionNumber.replace('-', '')
            documentData = requests.get(f'https://www.sec.gov/Archives/edgar/data/{cik_url}/{accession_url}/{formDocumentLink}')
            soup = BeautifulSoup(documentData.content, 'html.parser')            
            print(soup.title.text)

            # Get financial indicators reported
            time.sleep(1)
            companyFacts = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json', headers=self.headers)
            companyFactsData = companyFacts.json()['facts']['us-gaap']

            # Get XBRL disclosures, assets data
            time.sleep(1)
            companyConcept = requests.get((f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}' f'/us-gaap/Assets.json'), headers=self.headers)
            assetsData = pd.DataFrame.from_dict((companyConcept.json()['units']['USD']))
            
        return data
    
    def preprocessData(data):
        return

    def vectorizeData(processed_data):
        return

    def trainModel(vectors, clusters, type="KMeans"):
        return

    def loadModel(model_path):
        return

    def findNearestCluster(model, data):
        return

    def visualizeClusters(cluster_labels, vectors):
        return

# Example usage
model = AnnualFilingCluster('hamzehhamdan@college.harvard.edu')
ciks = ['AAPL', 'DJT']  # Example CIKs for Apple and Amazon
filings_info = model.retrieveData(ciks)
