import requests
import nltk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from ratelimiter import RateLimiter
from edgar import *

from dotenv import load_dotenv
import os

from sec_api import ExtractorApi
import re
from html import unescape
import spacy
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
import hdbscan

from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import seaborn as sns
from scipy.spatial.distance import cosine

# Request handling
rate_limiter = RateLimiter(10, 1)
s = requests.Session()

s = requests.Session()

class AnnualFilingCluster:
    def __init__(self, email):
        self.headers = {'User-Agent': email}
        self.names = []
        self.tickers = None
        self.data = None
        self.results = None
        self.companyIdentifiers = get_companyIdentifiers(self.headers)

    def retrieveFinancialData(self, tickers=None, include=None, exclude=None):
        """
        Fetches the 10-K filing financial numerical data for a given list of companies by their tickers.
        
        :param tickers: List of strings representing the company tickers.
        :return: A dictionary where the key is the CIK and the value is the filing information.
        """

        form='10-K' # TODO: If I want to allow users to get other forms: 
                        # 1) add in forms into the parameters of retrieveData
                        # 2) adapt BeautifulSoup data parsing for other forms
                        # 3) add errors to make sure that the form inputted by user into this function is one of those accepted

        data = {}

        if not isinstance(tickers, list):
            tickers = [tickers]  # Convert a single CIK string to a list for uniform handling

        for ticker in tickers:
            ticker = ticker.upper().replace('.', '-') # handle ticker differences
            # Get form data
            company_data = get_companyData(headers=self.headers, ticker=ticker, form=form, companyIdentifiers=self.companyIdentifiers)
            data[ticker] = company_data
            
        self.tickers = tickers
        return data
    
    def retreiveTextData(self, tickers, sections, years_back, sec_api_key):
        '''
        Fetches 10-K text for a given list of companies by their tickers.

        :param tickers: List of strings representing the company tickers.
        :param sections: List of 10-K sections to include.
        :param years_back: Integer value of how many years back you'd like to consider data from.
        :param sec_api_key: SEC API Key from sec-api.io
        :return: A Pandas DataFrame with the text data.
        '''
        companyTickers = requests.get('https://www.sec.gov/files/company_tickers.json', headers=self.headers)
        companyIdentifiers = pd.DataFrame.from_dict(companyTickers.json(), orient='index')
        companyIdentifiers['cik_str'] = companyIdentifiers['cik_str'].astype(str).str.zfill(10)
        companyIdentifiers = companyIdentifiers.drop_duplicates(subset='ticker', keep='first')

        if tickers == None:
            tickers = companyIdentifiers['ticker'].tolist()

        missing_tickers = set(tickers) - set(companyIdentifiers['ticker'])
        tickers = [ticker for ticker in tickers if ticker not in missing_tickers]
 
        titles = companyIdentifiers[companyIdentifiers['ticker'].isin(tickers)]['title'].tolist()

        ciks_raw = [get_cik(ticker=ticker, companyIdentifiers=companyIdentifiers) for ticker in tickers]

        submissionMetadatas = [get_submissionMetadata(headers=self.headers, cik=cik) for cik in ciks_raw]
        allForms_data = [get_allForms(submissionMetadata) for submissionMetadata in submissionMetadatas]

        formAccessionNumbers_data_raw = [get_formAccessionNumbers(allforms, '10-K') for allforms in allForms_data]
        formAccessionNumbers_data = [formAccessionNumbers.iloc[0]['accessionNumber'] if formAccessionNumbers is not None and not formAccessionNumbers.empty else None for formAccessionNumbers in formAccessionNumbers_data_raw]
        filing_date_data = [formAccessionNumbers.index[0] if formAccessionNumbers is not None and not formAccessionNumbers.empty else None for formAccessionNumbers in formAccessionNumbers_data_raw]

        links_raw = [get_documentLink(allForms, '10-K') for allForms in allForms_data]
        links = [link for link in links_raw if link is not None]

        data = {
            'Title': titles,
            'Ticker': tickers,
            'CIK': ciks_raw,
            'Accession Number': formAccessionNumbers_data,
            'Filing Date': filing_date_data,
            'Raw Links': links_raw
        }

        df = pd.DataFrame(data)
        df = df.dropna()

        df['Filing Date'] = pd.to_datetime(df['Filing Date'])
        df = df[df['Filing Date'] > pd.Timestamp.now() - pd.DateOffset(years=years_back)]

        df['Links'] = df.apply(lambda row: f"https://www.sec.gov/Archives/edgar/data/{row['CIK']}/{row['Accession Number'].replace('-', '')}/{row['Raw Links']}", axis=1)

        df.to_csv('data/market_data.csv', index=False, header=True)
        df['Links'].to_csv('data/market_links.txt', index=False, header=False)

        links = df['Links'].tolist()

        extractorApi = ExtractorApi(sec_api_key)

        all_sections = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9A", "9B", "10", "11", "12", "13", "14"]
        items = [section for section in sections if section in all_sections]

        for link in links:

            url_filter = df['Links'] == link
            if not df[url_filter].empty:
                for item in items:
                    try:
                        section_text = extractorApi.get_section(filing_url=link, section=item, return_type="text")
                        df.loc[url_filter, item] = section_text

                    except Exception as e:
                        print(e)

        self.tickers = tickers
        return df

    def preprocessData(self, df, SECBERT=False):
        '''
        Preprocesses 10-K text for data returned by retreiveTestData.

        :param df: data returned by retreiveTestData.
        :param SECBERT: True if preprocessing is being done for SECBERT.
        :return: A Pandas DataFrame with the processed data.
        '''

        all_sections = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9A", "9B", "10", "11", "12", "13", "14"]
        items = [col for col in df.columns if col in all_sections]

        if SECBERT:
            tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
            spacy_tokenizer = spacy.load("en_core_web_sm")

        def preprocess_text(text):
            if not isinstance(text, str):
                # If it's NaN or some other non-string type, return an empty string
                return ""
            # Decode HTML entities
            text = unescape(text)
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            # Convert to lowercase
            text = text.lower()
            # Replace newline characters with space
            text = text.replace('\n', ' ')
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def sec_bert_shape_preprocess(text):
            # Tokenize the text using spaCy
            tokens = [t.text for t in spacy_tokenizer(text)]
            processed_text = []
            for token in tokens:
                # Match tokens that are entirely numeric (including those with commas and decimals)
                if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", token):
                    # Replace each digit with 'X' to get the shape
                    shape = '[' + re.sub(r'\d', 'X', token) + ']'
                    # Check if this shape is a special token in the tokenizer
                    if shape in tokenizer.additional_special_tokens:
                        processed_text.append(shape)
                    else:
                        # If not recognized, use a generic number token
                        processed_text.append('[NUM]')
                else:
                    processed_text.append(token)
            return ' '.join(processed_text)
        
        df[items] = df[items].map(preprocess_text)
        if SECBERT:
            df[items] = df[items].map(sec_bert_shape_preprocess)

        return df

    def tokenizeData(self, processed_data, SECBERT=False):
        df = processed_data
        if SECBERT:
            tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
            nltk.download('punkt', quiet=True)
        
        def chunk_and_tokenize(text):
            sentences = sent_tokenize(text)
            current_chunk = []
            current_length = 0
            all_tokens = []  # This will hold all tokens for the text

            for i in range(len(sentences)):
                sentence = sentences[i]
                tokens = tokenizer.tokenize(sentence)
                num_tokens = len(tokens)

                # Handle long sentences that exceed the maximum token limit
                if num_tokens > 512:
                    part_size = 512 // 2  # Split long sentence into parts smaller than max_tokens
                    parts = [tokens[j:j + part_size] for j in range(0, len(tokens), part_size)]
                    for part in parts:
                        all_tokens.extend(part)  # Store the tokens directly
                    continue  # Skip the regular processing since we've handled this long sentence

                # Check if adding this sentence would exceed the max token limit
                if current_length + num_tokens > 512:
                    if current_chunk:
                        all_tokens.extend(tokenizer.tokenize(' '.join(current_chunk)))  # Store the tokens directly
                    # Start a new chunk considering the overlap
                    current_chunk = sentences[max(i - 2, 0):i] if i - 2 > 0 else []
                    current_length = sum(len(tokenizer.tokenize(s)) for s in current_chunk)

                # Add the current sentence to the chunk
                current_chunk.append(sentence)
                current_length += num_tokens

            # Tokenize and add the last chunk
            if current_chunk:
                all_tokens.extend(tokenizer.tokenize(' '.join(current_chunk)))  # Store the tokens directly

            return all_tokens

        # Apply the function to each applicable column and create a new column for aggregated tokens
        for column in df.columns[7:]:  # Skip 'Ticker' column
            df[column] = df[column].apply(lambda text: chunk_and_tokenize(text) if isinstance(text, str) else [])

        # Aggregate all tokens into a new column
        df['All Tokens'] = df.apply(lambda row: [token for col in df.columns[7:] if col != 'All Tokens' for token in row[col]], axis=1)
        return df
    
    def embedData(self, tokenized_data, col_embeddings=False):

        df = tokenized_data
        tokenizer = AutoTokenizer.from_pretrained('nlpaueb/sec-bert-shape')
        model = AutoModel.from_pretrained('nlpaueb/sec-bert-shape')

        def get_bert_embeddings(text):
            text = ' '.join(text)
            tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
            # Convert token IDs to a tensor, truncate to the model's max input size
            input_ids = torch.tensor([tokens])  # Ensure it's wrapped in a list to create the right shape
            with torch.no_grad():
                outputs = model(input_ids)
                # Use the mean of the last hidden state as the sentence embedding
                embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy()
           
        df['Embeddings'] = df['All Tokens'].apply(get_bert_embeddings)

        def apply_embeddings(df, columns):
            if isinstance(columns, str):
                columns = [columns]  # Convert to list if only one column is provided
            for column in columns:
                print(df[column])
                df[f'{column}_Embeddings'] = df[column].apply(get_bert_embeddings)
            return df
        
        if col_embeddings:
            df = apply_embeddings(df, df.columns[7:-2])

        return df

    def clusterData(self, embedded_data, model_type="KMeans", KMEANS_n_clusters=11, HBDSCAN_min_cluster_size=11, random_seed=False):
        '''
        model_type == KMeans or HBDSCAN
        '''

        df = embedded_data
        embedding_matrix = np.vstack(df['Embeddings'].values)
        if len(embedding_matrix) >= 100:
            n_components = 100
        else:
            n_components = len(embedding_matrix)
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embedding_matrix)

        if model_type=='KMeans' or 'KMeans' in model_type:
            if random_seed:
                kmeans = KMeans(n_clusters=KMEANS_n_clusters, random_state=random_seed)
            else:
                kmeans = KMeans(n_clusters=KMEANS_n_clusters)
            clusters = kmeans.fit_predict(reduced_embeddings)
            df['KNN Cluster'] = clusters

        if model_type=='HDBSCAN' or 'HDBSCAN' in model_type:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=HBDSCAN_min_cluster_size, prediction_data=True)
            clusters = clusterer.fit_predict(reduced_embeddings)
            df['HBDSCAN Cluster'] = clusters

        print(df.columns)
        return df, reduced_embeddings

    def visualizeClusters(self, cluster_data, reduced_embeddings, perplexity=None, label_df=None, save_file_path=None):
        df = cluster_data
        if label_df:
            df = pd.merge(df, label_df[['Ticker', 'Industry']], on='Ticker', how='left')

        n_samples = reduced_embeddings.shape[0] 
        perplexity = min(40, max(5, n_samples - 1))
        
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
        tsne_results = tsne.fit_transform(reduced_embeddings)
        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]

        reducer = umap.UMAP()
        umap_results = reducer.fit_transform(reduced_embeddings)
        df['umap-one'] = umap_results[:, 0]
        df['umap-two'] = umap_results[:, 1]

        if label_df is not None:
            label_encoder = LabelEncoder()
            df['Industry'] = label_encoder.fit_transform(df['Industry'])

        def plot_with_annotations(ax, x, y, labels, title, cmap):
            scatter = ax.scatter(x, y, c=labels, cmap=cmap, alpha=0.5)
            ax.set_title(title)
            plt.colorbar(scatter, ax=ax)
            # Loop through and annotate each point
            for i, txt in enumerate(df['Ticker']):
                ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, alpha=0.7)

        def plot_without_annotations(ax, x, y, title, cmap):
            scatter = ax.scatter(x, y, c=df['KNN Cluster'], cmap=cmap, alpha=0.5)
            ax.set_title(title)
            plt.colorbar(scatter, ax=ax)
            for i, txt in enumerate(df['Ticker']):
                ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, alpha=0.7)

        if label_df is not None:
            plt.figure(figsize=(32, 20))

            # t-SNE by Clusters
            ax1 = plt.subplot(2, 2, 1)
            plot_with_annotations(ax1, df['tsne-2d-one'], df['tsne-2d-two'], df['KNN Cluster'], 't-SNE Visualization of KNN Clusters', 'viridis')

            # t-SNE by Industry
            ax2 = plt.subplot(2, 2, 2)
            plot_with_annotations(ax2, df['tsne-2d-one'], df['tsne-2d-two'], df['Industry'], 't-SNE Visualization by Industry', 'tab20')

            # UMAP by Clusters
            ax3 = plt.subplot(2, 2, 3)
            plot_with_annotations(ax3, df['umap-one'], df['umap-two'], df['KNN Cluster'], 'UMAP Visualization of KNN Clusters', 'viridis')

            # UMAP by Industry
            ax4 = plt.subplot(2, 2, 4)
            plot_with_annotations(ax4, df['umap-one'], df['umap-two'], df['Industry'], 'UMAP Visualization by Industry', 'tab20')

            plt.suptitle('Embeddings of Business Description using SEC BERT Shape and PCA Embeddings', fontsize=16)
            if save_file_path:
                plt.savefig(save_file_path)
            else:
                plt.show()
            plt.clf()

        else:
            plt.figure(figsize=(16, 20))

            # t-SNE by Clusters
            ax1 = plt.subplot(2, 1, 1)
            plot_without_annotations(ax1, df['tsne-2d-one'], df['tsne-2d-two'], 't-SNE Visualization of KNN Clusters', 'viridis')

            # UMAP by Clusters
            ax3 = plt.subplot(2, 1, 2)
            plot_without_annotations(ax3, df['umap-one'], df['umap-two'], 'UMAP Visualization of KNN Clusters', 'viridis')

            plt.suptitle('Embeddings of Business Description using SEC BERT Shape and PCA Embeddings', fontsize=16)
            if 'save_file_path':
                plt.savefig(save_file_path)
            else:
                plt.show()
            plt.clf()

    def plotCosineSimilaryHeatmap(self, embeddings, labels=None, save_file_path=None, only_return_embeddings=False, threshold=None):
        """
        Plot a heatmap of the cosine similarity matrix.

        :param embeddings: A 2D numpy array where each row represents an embedding.
        :param labels: Optional list of labels for the heatmap axis (default is None).
        :return: A 2D numpy array containing the cosine similarity scores between each pair of embeddings.
        """

        similarity_matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]))

        if labels is None:
            labels = [f'Company {i}' for i in range(len(embeddings))]

        for i in range(embeddings.shape[0]):
            for j in range(i, embeddings.shape[0]):
                if i != j:
                    similarity = 1 - cosine(embeddings[i], embeddings[j])
                    similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
                    if threshold != None:
                        if similarity<threshold:
                            similarity_matrix[i, j] = similarity_matrix[j, i] = 0
                else:
                    similarity_matrix[i, j] = np.nan

        if only_return_embeddings:
            return similarity_matrix
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, annot=True, fmt=".3f", cmap='coolwarm', center=0, xticklabels=labels, yticklabels=labels, annot_kws={"size": 8})
        plt.title("Cosine Similarity Heatmap of 10-K Filings")
        if save_file_path:
            plt.savefig(save_file_path)
        else:
            plt.show()
        plt.clf()

        return similarity_matrix

    def proximityValidation(self, embeddings, labels, threshold=0.5):
        """
        Validate the proximity of embeddings based on a given threshold. This function assumes that embeddings from
        the same label should be more similar to each other than to embeddings from different labels.

        :param embeddings: List of embeddings.
        :param labels: List of labels corresponding to the embeddings.
        :param threshold: Threshold for considering two embeddings as 'close'.
        :return: A dictionary with validation results.
        """
        sim_matrix = self.plotCosineSimilaryHeatmap(embeddings=embeddings, only_return_embeddings=True)
        n = len(labels)
        print(n)
        correct_close = 0
        incorrect_close = 0
        correct_distant = 0
        incorrect_distant = 0

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:  # Expected to be similar
                    if sim_matrix[i][j] >= threshold:
                        correct_close += 1
                    else:
                        incorrect_distant += 1
                else:  # Expected to be dissimilar
                    if sim_matrix[i][j] < threshold:
                        correct_distant += 1
                    else:
                        incorrect_close += 1

        return {
            'correct_close': correct_close,
            'incorrect_close': incorrect_close,
            'correct_distant': correct_distant,
            'incorrect_distant': incorrect_distant,
            'accuracy': (correct_close + correct_distant) / (correct_close + correct_distant + incorrect_close + incorrect_distant)
        }


# Example usage
print('Getting ticker data.')
df = pd.read_csv('data/sp100.csv')
tickers = df['Ticker'].tolist()
labels = df['Industry'].tolist()

tickers = ['AAPL', 'MSFT', 'GOOGL', 'INTC', 'JPM', 'GS', 'C', 'MS', 'XOM', 'CVX', 'COP', 'PG', 'KO', 'PEP', 'NKE'] 
labels = ['Information Technology', 'Information Technology', 'Communication Services', 'Information Technology', 'Financials', 'Financials', 'Financials', 'Financials', 'Energy', 'Energy', 'Energy', 'Consumer Staples', 'Consumer Staples', 'Consumer Staples', 'Consumer Discretionary']

tickers = ['CMCSA', 'DIS', 'CHTR', 'COP', 'XOM', 'CVX', 'GS', 'MS', 'BAC', 'WFC', 'SBUX', 'MCD', 'AAPL', 'MSFT', 'GOOG', 'FB', 'NVDA', 'INTC', 'LLY', 'JNJ', 'PFE', 'NEE', 'DUK', 'SO']
labels = df[df['Ticker'].isin(tickers)]['Industry'].tolist()

sections = ['1', '1A']
n_clusters = 10

print('Initializing model.')
model = AnnualFilingCluster(os.getenv('EDGAR_HEADER'))
sec_api_key = os.getenv('SEC_API_KEY')

#print('Getting quant data.')
#quant_data = model.retrieveFinancialData(tickers)

print('Getting text data.')
text_data = model.retreiveTextData(tickers=tickers, sections=sections, years_back=1, sec_api_key=sec_api_key)
text_data.to_csv('longer_test/text_data.csv', index=False)

tickers = model.tickers
labels = df[df['Ticker'].isin(tickers)]['Industry'].tolist()

print('Processing data.')
processed_data = model.preprocessData(df=text_data, SECBERT=True)
processed_data.to_csv('longer_test/processed_data.csv', index=False)

print('Tokenizing data.')
tokenized_data = model.tokenizeData(processed_data=processed_data, SECBERT=True)
tokenized_data.to_csv('longer_test/tokenized_data.csv', index=False)

print('Embedding data.')
embedded_data = model.embedData(tokenized_data=tokenized_data, col_embeddings=True)
embedded_data.to_csv('longer_test/embeddings.csv', index=False)

print('Clustering data.')
clustered_data, reduced_embeddings = model.clusterData(embedded_data=embedded_data, model_type='KMeans', KMEANS_n_clusters=n_clusters, random_seed=109)
clustered_data.to_csv('longer_test/clustered_data.csv', index=False)
np.save('longer_test/reduced_embeddings.npy', reduced_embeddings)

print('Visualizing clusters.')
model.visualizeClusters(cluster_data=clustered_data, reduced_embeddings=reduced_embeddings, save_file_path='longer_test/visualize_clusters.png')

print('Plotting cosine similarity heatmap.')
cosine_similarity_matrix = model.plotCosineSimilaryHeatmap(embeddings=reduced_embeddings, labels=tickers, save_file_path='longer_test/cosine_similarity.png')
cosine_similarity_matrix = model.plotCosineSimilaryHeatmap(embeddings=reduced_embeddings, labels=tickers, threshold=0.4, save_file_path='longer_test/cosine_similarity_threshold04.png')
cosine_similarity_matrix = model.plotCosineSimilaryHeatmap(embeddings=reduced_embeddings, labels=tickers, threshold=0.5, save_file_path='longer_test/cosine_similarity_threshold05.png')
df = pd.DataFrame(cosine_similarity_matrix)
df.to_csv('longer_test/cosine_similarity_matrix.csv', index=False)

print('Running Proximity Validation.')
threshhold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
proximity_results = [model.proximityValidation(embeddings=reduced_embeddings, labels=labels, threshold=threshold_value)['accuracy'] for threshold_value in threshhold_values]

results_data = {
    "Threshold": threshhold_values,
    "Accuracy": proximity_results
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('longer_test/proximity_results.csv', index=False)

print(results_df)

'''
print(data['AAPL']['Company Facts'].keys())
print(data['AAPL']['Annual Financial Facts']) # return PD dataframes
print(data['AAPL']['Quarterly Financial Facts']) # return PD dataframes
print(data['AAPL']['10K Data'].keys()) # returns dictionary
print(data['AAPL']['Assets Data'].head())


https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/
'''
