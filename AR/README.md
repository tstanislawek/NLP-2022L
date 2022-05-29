# gc-bert

## Intro

A considerable number of texts that one encounters on daily basis, are somehow logically connected with each other. For example, Wikipedia articles refer to other articles via hyperlinks, scientific papers relate to others via citations or (co)authors, while tweets relate via users that follow each other, or reshare content.  

The aim of this study is to integrate the graph-based context information into a language model (e.g., BERT model). Research hypothesis is that extraction and integration of context information (context in which text is “located”) might be helpful in facilitating better understanding of the text. 

Exemplary task, in which the proposed approach might be used, is detection of hateful tweets on Twitter, via delivering to the language model data about the author and other users that are connected with her/him. Here, motivation is that friends (and their own content, opinions, activity, ...) of a particular user have significant impact on what he/she publishes. This additional information might be helpful in building a better classifier.  

There are several works describing integration of graph data, via graph neural networks, with language models, however just a few of them accept as an input linked (directly or indirectly) documents. They are mostly focusing on integration of knowledge graph, or creation of a graph representing relations between documents and words in a dictionary.


## Dataset


In order to accomplish the requirements of this work the dataset on which experiments are based on should have following features: (1) there should be a set of $n$ documents $D = \{d_1, d_2, d_3, ..., d_n\}$, (2) documents should be connected with each other via direct or indirect links (there should be a set of edges $E = \{(d_1, d_2), (d_1, d_5), ..., (d_i, d_j)\}$). In summary, the input data must be a graph of texts that are somehow connected with each other in a meaningful manner.

For simplicity in this study we take only the largest connected component of the graph.

As in this study we use GNN that utilize the adjacency matrix $A$ we do not enforce any limitation about the direction of the edge. It means that $A$ does not have to be symmetric.

Edges might be weighted to reflect the strength of the connection. In such a case GNN architecture that is used must be build in such a way to take those weights into consideration during learning process. Weight of the edge can reflect e.g. number of references to another document or number of common coauthors between two documents. 


The main dataset on which this study was done is Pubmed dataset. It consists of 19717 articles regarding various aspects of diabetes. Each document is assigned to one of three classes. As citation network was not contained in the original datasets they were fetched via Pubmed API along with other metadata.

# Models 

Basic building blocks used for this study are Graph Neural Networks (GNNs) and BERT language model. 

## GNNs

Graph Neural Networks are powerful models capable of utilization of graph information in order to create representational node vectors from which prediction can be performed. In this study following GNN architectures were used:

- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE
- Graph Isomorphism Network (GIN)

## Language Model (LM)

Main language model used in this study was BERT model. It was chosed due to its simplicity and extensive usage in other 
studies so that comparison with other works is simpler.

## Proposed LM+GNN architectures

There is a plenty of ways we can try to connect LM and GNN.
The general idea of LM + GNN is to have two representations of each of the document - one refering to text and the other refering to graph information. Text representations are stored in $T$ matrix, while node representations in $N$ matrix.

In this study several general layouts has been proposed:

- late fusion: BERT receives pure text and creates textual representation vectors. Paralelly each text is vectorized (it can be done using TFIDF or BERT) and the vectorized version along with graph data are introduced into GNN model that produces node 
representation of each of the texts. Then node and text representations are merged and processed by classifier. 

- early fusion: In this approach texts are vectorized into vectors that are processed by GNN which creates node representation. 
Node representation is then inserted into input words tokens of the text and processed by BERT and then by classifier. 

- compositonal architecture: Here data is processed at first by BERT and later by GNN.

There are several modifications that can be applied to all of abovementioned architectures:

- skip connections that transfer node or text representation into classifier

- whether node representation is inserted into BERT input data

- the way of how node representation is inserted into BERT input data (it can be inserted as second token; it can be transformed by linear layer, etc..)

- some models can be freezed to obtain static node representation

- way of merging $T[i]$ and $N[i]$ vectors - they can be concatenated or added in an element-wise manner


# Implementational details

## Dataset

Main logic regarding datasets is implemented within `gc_bert/dataset.py` file. It contains one parent class that handles graph text data, its processing, building of the graph, adjacency matrix, etc.. Child classes implements special operations necesary for each of the datasets - currently just Pubmed and DBpedia.

The user is required to have downloaded data in a proper directory. 

In case of Pubmed (main dataset used in this study) text data (abstract) should be stored in json file along with other metadata, like date of publication, authors list, authors affiliation, pmid (Pubmed ID), etc.. Citations are stored in csv file containing two columns: `source` and `target` with pmids of articles that are used.

```
dataset = PubmedDataset('pubmed/data/articles.json', 'pubmed/data/citations.csv')
dataset.load_data()  # loads and cleans the data
```

As in this study  the transductive learning is being performed for GNN the model has acces to entire graph. However, BERT is trained in inductive way. In order to handle this dataset object always returns entire graph and adjacency matrix, however in case of texts it will return only those which are from one of the group: train, valid or test. It will depend on the mode that is currently set.

```
G = dataset.create_graph()
A = dataset.create_adj_matrix()  # it will be n x n; you can specify whether you want sparse or dense version
dataset.change_mode('train')  
```


# How to run?

At first you need to have proper Python3 environment with torch. You should run 

```
pip install -r requirements.txt
pip install -e .
```

All the necessary data for pubmed are stored `pubmed/data/` directory. Before running the traiing you should unpack the tar archive with this command:

```
tar -xvf pubmed/data/articles.json.tar.gz 
```


The main script for running the project is contained in the `train.py` file that can be launched from the CLI. There are several arguments that can be specifed: 

- `dataset`: you can choose the dataset you like, currently only pubmed is available
- `model`: model you want to train (one of `gcn`, `gat`, `gin`, `bert`, `bert-gcn`, `gcbert`)
- `model-load-path`: path to the model weights that you want to load
- `run-name`: name of the run; it will be used when you save the model
- `save-dir`: top-level directory in which results and the model should be saved

For all args just type `python train.py -h`.

If you want to adjust some hyperparameters, like feature space of GNNs just chenage them in `train.py`. In case of some design changes in LM+GNN models you will need to modify the class of the model.



