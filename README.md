# **10-K-Text-Clustering**
A toolkit that allows users to run their own text-clustering models on a select set of companies' 10-K filings.

# **Usage**
* **Pipeline**:
    * Retreive financial data from EDGAR API using retrieveFinancialData
    * Retreive text data from sec-api.io using retreiveTextData
    * Preprocess data using preprocessData
    * Tokenize data using tokenizeData using SEC BERT Shape
    * Embed data using embedData using SEC BERT Shape
    * Cluster data using Kmeans or HBDSCAN (optional, should run for reduced embeddings required for remaining steps)
    * Visualize clusters using t-SNE and UMAP
    * Plot a cosine similarity heatmap for validation
    * If you have labels, find proximity validation accuracy 
* **Requirements**:
    * 1) EDGAR_HEADER, 2) SEC_API_KEY in .env file represent 1) the email address (not necessarily registered) used for the SEC EDGAR databse data requests, and 2) the api request from sec-api.io.
    * List of tickers for main functions.
    * List of sections to include from 10-K filings.
    * List of labels for proximity validation (optional).
* **Returns**:
    *  Dataframes with processed text, tokens, and embeddings
    *  Structured quantitative data usign SEC's EDGAR API
    *  Embedding visualization with t-SNE and/or UMAP
    *  Cosine similarity heatmap
    *  Proximity validation accuracy

**Example Usage**
'''python
tickers = ['AAPL', 'MSFT', 'GOOGL', 'INTC', 'JPM', 'GS', 'C', 'MS', 'XOM', 'CVX', 'COP', 'PG', 'KO', 'PEP', 'NKE'] # Tickers to include

sections = ['1', '1A'] # Sections to include in 10-K filings; 1: Business Description, 1A: Risk Factors.

model = AnnualFilingCluster(os.getenv('EDGAR_HEADER')) # Initializing the model
sec_api_key = os.getenv('SEC_API_KEY') # Your sec-api.io API key

quant_data = model.retrieveFinancialData(tickers) # Optional, retreived quant data for extra analysis

text_data = model.retreiveTextData(tickers=tickers, sections=sections, years_back=1, sec_api_key=sec_api_key) # Retreive text data

tickers = model.tickers # Update tickers list; previous method deletes tickers that were not registered with the SEC (check your inputs) 
labels # Update your labels too, if applicable

processed_data = model.preprocessData(df=text_data, SECBERT=True) # Process data
tokenized_data = model.tokenizeData(processed_data=processed_data, SECBERT=True) # Tokenize data
embedded_data = model.embedData(tokenized_data=tokenized_data, col_embeddings=True) # Embed data
clustered_data, reduced_embeddings = model.clusterData(embedded_data=embedded_data, model_type='KMeans', KMEANS_n_clusters=n_clusters, random_seed=109) # Cluster data

model.visualizeClusters(cluster_data=clustered_data, reduced_embeddings=reduced_embeddings, save_file_path='visualize_clusters.png') # Visualize clusters and save image as 'visualize_clusters.png'

cosine_similarity_matrix = model.plotCosineSimilaryHeatmap(embeddings=reduced_embeddings, labels=tickers, save_file_path='cosine_similarity.png') # Plot cosine similarity heatmap, save image as 'cosine_similarity.png', and save matrix as a variable called 'cosine_similarity_matrix' (optional to save, only used for external analysis)

resutls = model.proximityValidation(embeddings=reduced_embeddings, labels=labels, threshold=threshold_value)['accuracy'] # Returns accuracy from proximity validation for some threshold_value
'''

# **Project Motivation**

While the capabilities of natural language processing are growing in their ability to model the complexities of verbal communication and text, the use of textual analysis in empirical finance is very limited, and typically only occurs in studying the “sentiment” of a text. 

As the main focus lies on a sentiment analysis, there has been little exploration into using text-based clustering based on 10-K filings. A simple search on GitHub would return several projects on 10-K sentiment analysis stability, but none on text-clustering.

The purpose of this project is to provide a tool that allows users to find similar companies to a select company given its 10-K filings. They will be able to run their own clustering model on a select set of companies. I am also considering including a pretrained model with a lot of different companies’ data, but am unsure if that would be inefficient / not very feasible (I’d love your opinion on this front, if possible). 

Additionally, as the data is retrieved, users will be able to specify what portion of the 10-K filings to include / exclude. This can allow for larger control over what the similarities will be based on (i.e. a risk analysis can be made by including only the risk factors section, and any other relevant sections).

The motivation behind this project is rooted in the largely untapped usage of NLPs beyond sentiment analysis. I hope that this approach can uncover nuanced patterns and similarities between companies that are not apparent through traditional numerical data analysis alone.

Please reference the next section on technical steps to understand more about this tool and how it will work.

# **Software Usage**

Example usage:

* Using this tool to identify clusters of similar companies based on their 10-K filings, a user might uncover underlying themes or strategies not immediately apparent through traditional financial analysis. Looking at similar companies within a cluster might reveal some important information about competitors and potential opportunities for collaboration. Recall that as the data is retrieved, users can specify what portion of the 10-K filings to include / exclude, allowing for larger control over what the similarities will be based on (i.e. a risk analysis can be made by including only the risk factors section, and any other relevant sections).
* Researchers can use this tool to study trends and segment industries or markets into subsections. Read more about how this tool could have been used in established studies in the following section.

# **Similar Projects**

Searching through Google, Kaggle and GitHub, I could not find any tools for text-clustering specifically for 10-K filings (I spent quite a bit of time searching, and the closest thing I found for 10-K data was a sentiment analysis stability, which I did not think was relevant enough to include in this section). 

One relevant that could be used for some portions of my project is Extract10K by user jwkuo87 on GitHub, which provides methods for extracting data from 10-K filings using SEC’s EDGAR database. 

Looking at general text-clustering tools, I found Carrot2, a programing library for clustering text that can automatically discover groups of related documents for any textual data.

I also decided to look at published work that could have used this tool. One that I found by Tingyue Gan at UC Berkeley was on “Linking 10-K and the GICS - through Experiments of Text Classification and Clustering”.[^1] There has also been some research on hierarchical clustering being used for industry classification, as well as market orientation using textual analysis of 10-K filings.[^2][^3]


# **Potential Extensions**

While this will likely fall outside of the scope of this class, I would still appreciate your feedback in this area. I would love to add GNN models to better analyze the relational data between companies as they can capture complex patterns in the network-style data between companies within industries. The motivation here is that GNN models can provide a deeper analysis beyond textual similarity by considering the broader context of a company’s position in the market network. My main potential issue is finding the data, but I am wondering what public filings I would need for this data, as I imagine it would extend beyond just 10-K filings.

I would start with a basic GNN architecture focusing on company relationships based on industry classification or direct competition, using PyTorch Geometric for my implementation.

Perhaps a more feasible first step would be to include more data sources for companies that can be used for the text-clustering model.


<!-- Footnotes themselves at the bottom. -->
## Citations

[^1]:
     Gan, T. (2019, April 16). Linking 10-K and the GICS - through Experiments of Text Classification and Clustering. UC Berkeley. https://cdar.berkeley.edu/sites/default/files/tgan190416risk.pdf

[^2]:
     Yang, H., Lee, H. J., Cho, S., & Cho, E. (2016). Automatic classification of securities using hierarchical clustering of the 10-Ks. 2016 IEEE International Conference on Big Data (Big Data), 3936–3943. https://doi.org/10.1109/BigData.2016.7841069

[^3]:
     Andreou, P.C., Harris, T. and Philip, D. (2020), Measuring Firms’ Market Orientation Using Textual Analysis of 10-K Filings. Brit J Manage, 31: 872-895. https://doi.org/10.1111/1467-8551.12391
