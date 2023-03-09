GNN based text classification
=====
This repo contains code for 3 papers:

1. **GNoM**: Graph Neural Network Enhanced Language Models for Disaster Related Multilingual Text Classification (https://dl.acm.org/doi/pdf/10.1145/3501247.3531561)
2. **GLEN**: Unsupervised Domain Adaptation With Global and Local Graph Neural Networks Under Limited Supervision and Its Application to Disaster Response (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9744724)
3. **GCPT**: Supervised Graph Contrastive Pretraining for Text Classification (https://dl.acm.org/doi/pdf/10.1145/3477314.3507194)

Among the above mentioned papers, **GNoM** and **GLEN** works in Unsupervised Domain Adaptation (UDA) setting and **GCPT** works in Transfer Learning setting.

-----
Generate a large graph using labelled and unlabelled data from source domain (S) and only unlabelled data from target domain (T). Each node in the graph is a token in the combined (C = S + T) corpus and edges are formed based the co-occurrence of two tokens in a single tweet (or within a sliding window inside the tweet). Edge weights may be calculated using Pair-wise Mutual Information (PMI) on each domain. Each node has two extra attributes, containing information if it occurs in S or T or C. The goal is to utilize global and local context: global from token graph, local from tweet token relation using GNN.

### Parameter priority order 
1. command-line arguments 
2. config.py
3. default arguments

### File name convention 
As this repo was created before the name **GNoM**, we use the prefix `glen_bert` to refer code for **GNoM**

## Directory Structure

- `main_glen_bert.py` [main_glen_bert.py](main_glen_bert.py): **GNoM** starting code
- `Layers`: Contains Neural Network models and layers (e.g. LSTM, MLP, GCN etc)
- `Data_Handlers`: Contains text data and dataset construction codes
    * [instance_handler_dgl.py](Data_Handlers%2Finstance_handler_dgl.py): contains instance graph construction code written in DGL.
    * [token_handler_nx.py](Data_Handlers%2Ftoken_handler_nx.py): constructs global graph in NetworkX and converts to DGL graph. Also, calculates edge weights.
    * [create_datasets.py](Data_Handlers%2Fcreate_datasets.py): Converts dataframe to PyTorch Dataset and DataLoader
- `File_Handlers`: Contains specific file format related codes (e.g. json, csv, pkl, etc)
- `Graph_Construction`: Domain specific text processing (e.g. cleaning, tokenization) and graph construction using `Data_Handlers`
- `Text_Encoder`: Encodes text into vectors (e.g. uses GloVe embeddings for GLEN) and handles OOV tokens
- `Metrics`: Calculates metrics
- `stf_classification`: Transformer based text classification codes taken from [simpletransformer](https://github.com/ThilinaRajapakse/simpletransformers.git)
- `Pretrain`: **GCPT** pre-training related codes
- `Text_Processesor`: Text processing related codes
- `Trainer`: Training logic files
- `Disaster_Models`: Disaster related neural network codes (e.g. CNN, XML-CNN, 
  DenseCNN, etc)
- `config.py`: contains all configuration details (e.g. experiment configs, cuda configs, dataset names, etc.)
- `main_kd.py`: Knowledge Distillation related code for **GCPT**
- `main.py`: **GLEN** starting code
- `main_bert.py`: BERT baseline codes


## GLEN Data Flow:
1. Preprocess the token graph to generate domain invariant token representation
    1. Forward pass via GCN: `H(t+1) = Sigmoid(D-1/2 A D-1/2 H(t))`
    2. `H(t)` represents node features 
    3. `H(t+1)` will contain information from tokens of both domains, i.e. domain invariant 
2. Obtain sample text graph:
    1. Fetch k-hop neighbor induced subgraph from token graph for token in the sample text
    2. If nodes are disconnected in induced subgraph, connect consecutive token present in sample text with edge
     weight `e`
3. Architecture: 
    1. Pass individual sample text graphs through GNNs to generate new token representations
    2. Foreach text graph (`G_i`), use `Xi` and `Ai` to get aggregate sample representation: `Xi’ = GNN(Xi, Ai)`
    3. Concatenate `Xi” = Xi’ + Xi`
    4. Pass Xi” through LSTM: `Xi’” = LSTM(Xi”)`
    5. Forward Xi”' to a classifier `result = C(Xi’”)`

## GLEN Motivations and Ideas:

1. Construct a joint token graph (`G`) using `S` and `T` data with domain information
    1. Calculate edge weights
    2. Generate token embeddings
    3. Fine-tune token embeddings
2. Preprocess G such that node representations are domain-invariant (Using GNN)
    1. Forward pass through a GNN (possibly GCN)
    2. Verify if processed embeddings are domain invariant by classification
        1. Train classifier using `S`
        2. Test trained classifier on `T`
        3. Do previous steps for normal and processed embeddings; processed should work better 
3. Construct subgraph `H_i` by following:
    1. Construct a single sample graph (`H1`) by connecting nodes of adjacent tokens
    2. Fetch n-hop neighbor subgraph of `H`'s nodes from `G` to construct `H2`.
    3. Merge `H1` and `H2` by merging the neighbors from `H2` to `H1`.
    4. `H1` now represents sample graph with domain invariant feature
4. Pass `H1` through GNN followed by LSTM such that linguistic features are captured to get `f(H1)`
5. Pass `f(H1)` through an aggregate network `g()` to 
6. Use Domain Specific vectorizer: If a token occurs with high freq in `S` and not in `T` or vice versa, That token is informative. Treat domain as class and use class specific features.


## Requirements
Requirement files are located in:

**Conda**:
[requirements/conda_requirements.txt](requirements/conda_requirements.txt)

**Pip**:
[requirements/pip_requirements.txt](requirements/pip_requirements.txt)

### Installing Pytorch_Geometric (CPU only):

```bash
conda install pytorch torchvision cpuonly -c pytorch

pip install torch-scatter==latest+cpu torch-sparse==latest+cpu torch-spline-conv==latest+cpu torch-cluster==latest+cpu -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.5.0.html

pip install torch-geometric -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.5.0.html
```

## Data Details

### Data Locations

**FIRE216** [https://crisislex.org/data-collections.html#CrisisLexT6](https://crisislex.org/data-collections.html#CrisisLexT6)

1. Download the data
2. Convert label names from "on-topic" and "off-topic" to 1 and 0 respectively.
3. Remove quotations from tweet_id column.

**SMERP17** [https://crisislex.org/data-collections.html#CrisisLexT6](https://crisislex.org/data-collections.html#CrisisLexT6)

1. Download the data
2. Convert label names from "on-topic" and "off-topic" to 1 and 0 respectively.
3. Remove quotations from tweet_id column.

**NEQ-QFL** [https://crisislex.org/data-collections.html#CrisisLexT6](https://crisislex.org/data-collections.html#CrisisLexT6)

1. Download the data
2. Convert label names from "on-topic" and "off-topic" to 1 and 0 respectively.
3. Remove quotations from tweet_id column.

**Multilingual Disaster Response Messages** [https://www.kaggle.com/landlord/multilingual-disaster-response-messages](https://www.kaggle.com/landlord/multilingual-disaster-response-messages)

1. Download `disaster_response_messages_training.csv`, 
   `disaster_response_messages_validation.csv` 
   and `disaster_response_messages_test.
   csv` files

**Amazon Reviews Sentiment**

1. Download the data
2. Merge *_positive and *_negative files into a single csv file with texts from
   *_positive anootated with 1 and *_negative annotated with 0
3. Add auto-incremented row id


### Data Format

Traning data format: DataFrame with columns `["id", "text", "labels"]`


## Experiments

- All hyper-parameter and path details are set in `config.py` file

- Run FIRE16-SMERP17 (multi-label english) datasets with 
  1. Set `'name': 'smerp17_fire16','num_classes': 4, 'multi_label': True,` 
     under "data" in `config.py`
  2. Run `python main_glen_bert.py -multilabel True -mt 'bert' -m 
     'bert-base-uncased'`

- Run NEQ-QFL (binary english) datasets with
    1. Set `'name': 'NEQ_QFL','num_classes': 1, 'multi_label': False,`
       under "data" in `config.py`
    2. Run `python main_glen_bert.py -mt 'bert' -m 'bert-base-uncased'`

- Run FIRE16-SMERP17 (multi-label english) datasets with
    1. Set `'name': 'smerp17_fire16','num_classes': 4, 'multi_label': True,`
       under "data" in `config.py`
    2. Run `python main_glen_bert.py -multilabel True -mt 'bert' -m
       'bert-base-uncased'`
