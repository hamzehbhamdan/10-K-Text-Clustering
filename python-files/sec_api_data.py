from edgar import *

import pandas as pd
from ratelimiter import RateLimiter

from sec_api import ExtractorApi

import os

extractorApi = ExtractorApi('5d47cc711eb441b212b8c1bcf52e2a234191c9816b2d955b30abca4ab2b2d7e9')

########################################################################################
# Generate list of S&P 500 tickers from txt file (368 available through API)           #
########################################################################################

# Get S&P 500 symbols as a list
with open('data/sp500_symbols.txt', 'r') as file:
    tickers = [line.strip() for line in file]

########################################################################################
# Generate text file of URLs from a list of S&P 500 tickers                            #
########################################################################################
'''
# Get links using EDGAR SEC api
rate_limiter = RateLimiter(10, 1)
s = requests.Session()
headers = {'User-Agent': 'hamzehhamdan@college.harvard.edu'}

## Get company identifiers
companyTickers = requests.get('https://www.sec.gov/files/company_tickers.json', headers=headers)
companyIdentifiers = pd.DataFrame.from_dict(companyTickers.json(), orient='index')
companyIdentifiers['cik_str'] = companyIdentifiers['cik_str'].astype(str).str.zfill(10)

tickers = companyIdentifiers['ticker'].tolist() # to get data on all available stocks
titles = companyIdentifiers['title'].tolist() # to get data on all available stocks

## Get CIKs
ciks_raw = [get_cik(ticker=ticker, companyIdentifiers=companyIdentifiers) for ticker in tickers]
none_count = ciks_raw.count(None) # 129 missing tickers
ciks = [cik for cik in ciks_raw if cik is not None]

## Get submission metadata
submissionMetadatas = [get_submissionMetadata(headers=headers, cik=cik) for cik in ciks_raw]

## Get allforms
allForms_data = [get_allForms(submissionMetadata) for submissionMetadata in submissionMetadatas]

## Get accession numbers
formAccessionNumbers_data_raw = [get_formAccessionNumbers(allforms, '10-K') for allforms in allForms_data]
formAccessionNumbers_data = [formAccessionNumbers.iloc[0]['accessionNumber'] if formAccessionNumbers is not None and not formAccessionNumbers.empty else None for formAccessionNumbers in formAccessionNumbers_data_raw]
filing_date_data = [formAccessionNumbers.index[0] if formAccessionNumbers is not None and not formAccessionNumbers.empty else None for formAccessionNumbers in formAccessionNumbers_data_raw]
none_count = formAccessionNumbers_data.count(None) # 132 total missing tickers

# Get links
links_raw = [get_documentLink(allForms, '10-K') for allForms in allForms_data]
none_count = links_raw.count(None) # 3 companies didn't file 10-K
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

df['Links'] = df.apply(lambda row: f"https://www.sec.gov/Archives/edgar/data/{row['CIK']}/{row['Accession Number'].replace('-', '')}/{row['Raw Links']}", axis=1)

df.to_csv('data/market_data.csv', index=False, header=True)
df['Links'].to_csv('data/market_links.txt', index=False, header=False)

links = df['Links'].tolist()
'''
########################################################################################
# Use API to get structured text data from the links txt file                          #
########################################################################################
'''
with open('data/sp500_10K_links.txt', 'r') as file:
    links = [link.strip() for link in file]

df = pd.read_csv('data/sp500.csv')

def extract_items_10k(filing_url):
    items = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9A", "9B", "10", "11", "12", "13", "14"]

    url_filter = df['Links'] == filing_url
    if not df[url_filter].empty:
        for item in items:
            try:
                section_text = extractorApi.get_section(filing_url=filing_url, section=item, return_type="text")
                df.loc[url_filter, item] = section_text

            except Exception as e:
                print(e)

for link in links:
   extract_items_10k(link)

# saving the data
df.to_pickle('data/market_data_with_10k.pkl')
'''

########################################################################################
# Load data for preprocessing                                                          #
########################################################################################

df = pd.read_pickle('data/market_data_with_10k.pkl')
df['Filing Date'] = pd.to_datetime(df['Filing Date'])

# Filtering for companies that have filed in the past year: 6517 exist
df = df[df['Filing Date'] > pd.Timestamp.now() - pd.DateOffset(years=1)]


lengths = df.map(lambda x: len(str(x)))
average_lengths = lengths.mean()
# average lengths are below:

'''
Title                   21.81
Ticker                   4.04
CIK                        10
Accession Number           20
Filing Date                19
Raw Links               17.30
Links                   87.30
1                   67,286.75 # Business
1A                 103,971.51 # Risk Factors
1B                     103.44
2                    3,367.82
3                    1,145.03
4                      456.61
5                    4,315.21
6                      548.89
7                   64,378.04 # Management’s Discussion and Analysis of Financial Condition and Results of Operations
7A                   3,629.06
8                  101,779.62 # Financial Statements and Supplementary Data
9A                   5,215.97 
9B                     722.31
10                   8,020.93
11                   5,536.84
12                   2,610.16
13                   2,625.61
14                   1,269.60
'''

########################################################################################
# Retreiving data                                                                      #
########################################################################################

# Return the data only for the tickers needed
def filter_tickers(df, tickers, columns=df.columns, keep_ticker=True):
    filtered_df = df[df['Ticker'].isin(tickers)]
    if keep_ticker:
        if 'Ticker' not in columns:
            columns.insert(0, 'Ticker')
        filtered_df = filtered_df[columns]
    else:
        filtered_df = filtered_df[columns]
    return filtered_df

########################################################################################
# Preprocessing Data                                                                   #
########################################################################################

import re
from html import unescape
import spacy
from transformers import AutoTokenizer

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

items = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9A", "9B", "10", "11", "12", "13", "14"]

if not os.path.exists('data/processed_SP100_data_with_10k.csv'):
    sp100_df = pd.read_csv('data/sp100.csv')
    sp100_tickers = sp100_df['Ticker'].tolist()
    existing_tickers = sp100_df['Ticker'].tolist()
    tickers = [ticker for ticker in sp100_tickers if ticker in existing_tickers]
    df = filter_tickers(df, tickers, ["Ticker", "1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9A", "9B", "10", "11", "12", "13", "14"])
    df[items] = df[items].map(preprocess_text)
    df[items] = df[items].map(sec_bert_shape_preprocess)
    df.to_csv('data/processed_SP100_data_with_10k.csv')
    df.to_pickle('data/processed_SP100_data_with_10k.pkl')
else:
    df = pd.read_pickle('data/processed_SP100_data_with_10k.pkl')

########################################################################################
# Tokenize Data                                                                        #
########################################################################################

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import nltk

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
nltk.download('punkt', quiet=True)  # Ensure the punkt tokenizer is downloaded

df = filter_tickers(df, tickers=df['Ticker'].tolist(), columns=['1'], keep_ticker=True)

def prepare_bert_subsections(df, max_tokens=512, overlap=2):
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
            if num_tokens > max_tokens:
                part_size = max_tokens // 2  # Split long sentence into parts smaller than max_tokens
                parts = [tokens[j:j + part_size] for j in range(0, len(tokens), part_size)]
                for part in parts:
                    all_tokens.extend(part)  # Store the tokens directly
                continue  # Skip the regular processing since we've handled this long sentence

            # Check if adding this sentence would exceed the max token limit
            if current_length + num_tokens > max_tokens:
                if current_chunk:
                    all_tokens.extend(tokenizer.tokenize(' '.join(current_chunk)))  # Store the tokens directly
                # Start a new chunk considering the overlap
                current_chunk = sentences[max(i - overlap, 0):i] if i - overlap > 0 else []
                current_length = sum(len(tokenizer.tokenize(s)) for s in current_chunk)

            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_length += num_tokens

        # Tokenize and add the last chunk
        if current_chunk:
            all_tokens.extend(tokenizer.tokenize(' '.join(current_chunk)))  # Store the tokens directly

        return all_tokens

    # Apply the function to each applicable column and create a new column for aggregated tokens
    for column in df.columns[1:]:  # Skip 'Ticker' column
        df[column] = df[column].apply(lambda text: chunk_and_tokenize(text) if isinstance(text, str) else [])

    # Aggregate all tokens into a new column
    df['All Tokens'] = df.apply(lambda row: [token for col in df.columns[1:] if col != 'All Tokens' for token in row[col]], axis=1)
    return df

#if not os.path.exists('data/tokenized_SP100_data_with_10k.csv'):
df = prepare_bert_subsections(df)
df.to_csv('data/tokenized_SP100_data_with_10k.csv')
df.to_pickle('data/tokenized_SP100_data_with_10k.pkl')
#else:
#    df = pd.read_pickle('data/tokenized_SP100_data_with_10k.pkl')

########################################################################################
# Embed Data                                                                           #
########################################################################################

from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
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

df['Embeddings'] = df['1'].apply(get_bert_embeddings)

########################################################################################
# Cluster Embedded Data                                                                #
########################################################################################

embedding_matrix = np.vstack(df['Embeddings'].values)

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
reduced_embeddings = pca.fit_transform(embedding_matrix)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=11, random_state=42)  # Adjust the number of clusters
clusters = kmeans.fit_predict(reduced_embeddings)
df['KNN Cluster'] = clusters

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=11, prediction_data=True)  # min_cluster_size depends on your data
clusters = clusterer.fit_predict(reduced_embeddings)
df['HBDSCAN Cluster'] = clusters

########################################################################################
# Visualize Clusters                                                                   #
########################################################################################

sp100_df = pd.read_csv('data/sp100.csv')
df = pd.merge(df, sp100_df[['Ticker', 'Industry']], on='Ticker', how='left')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming 'reduced_embeddings' is your high-dimensional data reduced via PCA or directly from embeddings
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(reduced_embeddings)
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

# Using UMAP
import umap
reducer = umap.UMAP()
umap_results = reducer.fit_transform(reduced_embeddings)
df['umap-one'] = umap_results[:, 0]
df['umap-two'] = umap_results[:, 1]

# Plotting the data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Industry'] = label_encoder.fit_transform(df['Industry'])

def plot_with_annotations(ax, x, y, labels, title, cmap):
    scatter = ax.scatter(x, y, c=labels, cmap=cmap, alpha=0.5)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
    # Loop through and annotate each point
    for i, txt in enumerate(df['Ticker']):
        ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, alpha=0.7)

# Now plot both t-SNE and UMAP
plt.figure(figsize=(32, 20))  # Increased size for better visibility

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
plt.show()