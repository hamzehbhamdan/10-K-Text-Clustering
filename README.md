# **10-K-Text-Clustering**
A toolkit that allows users to run their own text-clustering models on a select set of companies' 10-K filings.


# **Project Motivation**

While the capabilities of natural language processing are growing in their ability to model the complexities of verbal communication and text, the use of textual analysis in empirical finance is very limited, and typically only occurs in studying the “sentiment” of a text. 

As the main focus lies on a sentiment analysis, there has been little exploration into using text-based clustering based on 10-K filings. A simple search on GitHub would return several projects on 10-K sentiment analysis stability, but none on text-clustering.

The purpose of this project is to provide a tool that allows users to find similar companies to a select company given its 10-K filings. They will be able to run their own clustering model on a select set of companies. I am also considering including a pretrained model with a lot of different companies’ data, but am unsure if that would be inefficient / not very feasible (I’d love your opinion on this front, if possible). 

Additionally, as the data is retrieved, users will be able to specify what portion of the 10-K filings to include / exclude. This can allow for larger control over what the similarities will be based on (i.e. a risk analysis can be made by including only the risk factors section, and any other relevant sections).

The motivation behind this project is rooted in the largely untapped usage of NLPs beyond sentiment analysis. I hope that this approach can uncover nuanced patterns and similarities between companies that are not apparent through traditional numerical data analysis alone.

Please reference the next section on technical steps to understand more about this tool and how it will work.


# **Technical Steps**

Organizationally, this project will be publicly available on GitHub and will be published on PyPI, the standard location for Python package publication. Additionally, I will clearly document all external dependencies with their versions to avoid conflicts, using either pipenv or conda for environment management. I will also likely add some form of feedback form, as I plan to update this library and potentially build on it with GNNs (outside of the scope of this class, likely to be done over the summer).

As for my developmental steps, I will first create the documentation for the methods, implement a few tests for the repository, and set up a README file that I will update with usage as I work on the project. Once I have the entire repository set up, I will start with my coding of the actual project. The code will be one Python class with several methods, outlined below.

Rough Outline of Methods:

* retrieveData
    * Purpose: Fetches the 10-K filing documents for a given list of companies by their legal names. This method uses the SEC’s EDGAR database API.
    * Input:
        * names: a list of strings representing the legal company names
        * exclude / include: specify what category of data to include or exclude (e.g. risk category)
    * Output:
        * 10-K data: a dictionary or structured data format where the key is the company name and value is the text content of the 10-K filing
* preprocessData
    * Purpose: Cleans and prepares the 10-K filings text for analysis.
    * Input:
        * data: the raw text data of 10-K filings
    * Output:
        * processed_data: preprocessed text data, ready for vectorization and clustering. This will likely include stemming, removal of stopwords, and other normalization steps.
* vectorizeData
    * Purpose: Converts preprocessed text data into a numerical format that can be used for the cluster analysis. This will likely use either TF-IDF vectors or, most likely, a version of BERT.
    * Input:
        * processed_data: preprocessed 10-K filings text data
    * Output:
        * vectors: a numerical representation of the text data, suitable for clustering
* trainModel
    * Purpose: Trains a new model based on a subset of 10-K filing data to cluster companies into groups based on textual similarities.
    * Input:
        * vectors: numerical representations of the companies’ 10-K filings
        * clusters: an integer specifying the number of clusters to form
        * type: type of model to train
    * Output:
        * model: a trained text-clustering model
        * cluster_labels: labels indicating the cluster each company belongs to
* loadModel
    * Purpose: loads a pre-trained clustering model, allowing for quick classification of new companies into existing clusters.
    * Input:
        * model_path: path to the saved model file
    * Output:
        * Model: a loaded text-clustering model ready to be used
* findNearestCluster
    * Purpose: finds the nearest cluster for a new company based on its 10-K filing, using a pre-trained model.
    * Input:
        * model: a trained text-clustering model
        * 10-K data: the 10-K filing text of a new company
    * Output:
        * cluster: the cluster to which the company is most similar
        * similar_companies: a list of companies in the identified cluster
* visualizeClusters
    * Purpose: generates a visual representation of the clustered companies.
    * Input:
        * cluster_labels: labels indicating the cluster each company belongs to
        * vectors: numerical representations of the companies’ 10-K filings
    * Output:
        * a plot showing the clusters of companies, which can help in understanding the distribution and relationships among them.


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
## Notes

[^1]:
     Gan, T. (2019, April 16). Linking 10-K and the GICS - through Experiments of Text Classification and Clustering. UC Berkeley. https://cdar.berkeley.edu/sites/default/files/tgan190416risk.pdf

[^2]:
     Yang, H., Lee, H. J., Cho, S., & Cho, E. (2016). Automatic classification of securities using hierarchical clustering of the 10-Ks. 2016 IEEE International Conference on Big Data (Big Data), 3936–3943. https://doi.org/10.1109/BigData.2016.7841069

[^3]:
     Andreou, P.C., Harris, T. and Philip, D. (2020), Measuring Firms’ Market Orientation Using Textual Analysis of 10-K Filings. Brit J Manage, 31: 872-895. https://doi.org/10.1111/1467-8551.12391
