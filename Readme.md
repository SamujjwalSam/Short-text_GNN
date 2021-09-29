Multilingual Extension to GLEN
=====

This work is an extension to [GLEN paper](https://arxiv.org/abs/2104.01436)
-----
    
    
* Replace GAT with BERT / XLM-R / mT5: to capture local context
    
* Attention over global component

* Embedding similarity edges


## Directory Structure
- `Layers`: Contains Neural Network model and layer related codes (e.g. 
  LSTM, MLP, GCN etc)
- `Metrics`: Calculates metrics
- `stf_classification`: Transformer model related codes
- `Pretrain`: GCPT pre-training related codes
- `Disaster_Models`: Disaster related neural network codes (e.g. CNN, XML-CNN, 
  DenseCNN, etc)
- `config.py`: contains all configuration details


## Requirements
Requirement files are located in:

**Conda**:

[requirements/conda_requirements.txt](requirements/conda_requirements.txt)

**Pip**:

[requirements/pip_requirements.txt](requirements/pip_requirements.txt)

## Data Details

### Data Locations

**FIRE216** [https://crisislex.org/data-collections.html#CrisisLexT6]
(https://crisislex.org/data-collections.html#CrisisLexT6)

1. Download the data
2. Convert label names from "on-topic" and "off-topic" to 1 and 0 respectively.
3. Remove quotations from tweet_id column.

**SMERP17** [https://crisislex.org/data-collections.html#CrisisLexT6]
(https://crisislex.org/data-collections.html#CrisisLexT6)

1. Download the data
2. Convert label names from "on-topic" and "off-topic" to 1 and 0 respectively.
3. Remove quotations from tweet_id column.

**NEQ-QFL** [https://crisislex.org/data-collections.html#CrisisLexT6]
(https://crisislex.org/data-collections.html#CrisisLexT6)

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
