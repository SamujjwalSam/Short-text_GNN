# Domain Invariant Short-text Classification Framework using Graph Neural Network

Problem with tweets are the availability of limited context. To increase context, generate a large graph using labelled 
 and unlabelled data from source domain (S) and unlabelled data from target domain (T). Each node in the graph is a 
 token in the mixed corpus and edges are formed based the co-occurrence of two tokens in a single tweet (or within a 
 sliding window inside the tweet). Edge weights may be calculated using Pair-wise Mutual Information (PMI). Each note 
 has two attributes containing information if it occurs in S or T or both.
    
    
* Utilize global and local context: global from token graph, local from tweet token relation:
    Capture global context from token graph and local context from GNN.
    
* Parameter priority: 
    argparse > config > default
    
## Approach:
    1 Preprocess the token graph to generate domain invariant token representation
        1.1 Forward pass via GCN: H(t+1) = Sigmoid(D-1/2 A D-1/2 H(t))
        1.2 H(t) represents node features 
        1.3 H(t+1) will contain information from tokens of both domains, i.e. domain invariant 
    2 Obtain sample text graph:
        2.1 Fetch k-hop neighbor induced subgraph from token graph for token in the sample text
        2.2 If nodes are disconnected in induced subgraph, connect consecutive token present in sample text with edge
         weight e
    3. Architecture: 
        3.1 Pass individual sample text graphs through GNNs to generate new token representations
        3.2 Foreach text graph (G_i), use Xi and Ai to get aggregate sample representation: Xi’ = GNN(Xi, Ai)
        3.3 Concatenate Xi” = Xi’ + Xi
        3.4 Pass Xi” through LSTM: Xi’” = LSTM(Xi”)
        3.4 Forward output of to a classifier C(Xi’”) to 

## Ideas:
    1 Construct a joint token graph (G) using S and T data with domain information
        1.1 Calculate edge weights
        1.2 Generate token embeddings
        1.3 Fine-tune token embeddings
    2 Preprocess G such that node representations are domain-invariant (Using GNN)
        2.1 Forward pass through a GNN (possibly GCN)
        2.2 Verify if processed embeddings are domain invariant by classification
            2.2.1 Train classifier using S
            2.2.2 Test trained classifier on T
            2.2.3 Do previous steps for normal and processed embeddings; processed should work better 
    3 Construct subgraph $H_i$ by following:
        3.1 Construct a single sample graph (H1) by connecting nodes of adjacent tokens
        3.2 Fetch n-hop neighbor subgraph of H's nodes from G to construct H2.
        3.3 Merge H1 and H2 by merging the neighbors from H2 to H1.
        3.4 H1 now represents sample graph with domain invariant feature
    4 Pass H1 through GNN followed by LSTM such that linguistic features are captured to get f(H1)
    5 Pass f(H1) through an aggregate network g() to 
    6. Use Domain Specific vectorizer: If a token occurs with high freq in S and not in T or vice versa, That token
     is informative. Treat domain as class and use class specific features.

## Baselines:
    * With and without token graph
    * Word2Vec trained on whole corpus
    * With and without GNN
    * With and without concatenation: without global context
    * DA based approaches
    * Compare with CrisisBERT?

## Experiments:
    1. Comparison various sliding-window sizes
    2. Experiment with k-hops (k=0...n) neighbor in sample graph
    3. Effect of token graph DA preprocessing

### Implementation Tasks: 
[]=pending, [-]=working, [x]=done
    
    TODO:
    # 1. Create separate target domain test data -> TEST
    # 2. Generate result with different portions of target data as train set
    # and a fixed target test set
    # 3. Class-wise toekn distribution of source and target domains
    # 4. Ways to pretrain GCN:
    #   4.1 Domain classification
    #   4.2 Link prediction
    # 5. Restrict GCN propagation using class information
    # 6. Integrate various plotting functions
    # 7 Think about adversarial setting
    # 8. Use BERT for local embedding
    # 9. Concatenate Glove and GCN embedding and evaluate POC
    # 10. Think about preprocessing GCN and GNN
    # 11. Add option to read hyper-params from config
    
    - [] 1. Text preprocessing:
        - [] 1.1. Clean and process OOV tokens
        - [] 1.2. Improve tokenization
    - [] 2. GCN Pre-training:
        - [] 2.1. Link prediction of domain classification
        - [] 2.2. Improve edge weight calculation
    - [] 4. GNN application on tweet graphs
        - [] 4.1. Decide GNN architecture
        - [] 4.2. Write code for GNN
        - [] 4.3. Concatenate GNN representation with GCN representation
    - [] 5. LSTM for classification

### Text Processing Tasks:

    - [] 1. Lemmatize tokens
    - [] 2. Correct Spelling
    - [] 3. Search and replace phone numbers
    - [] 4. Replace numbers with #D
    
    
#### Installing Pytorch_Geometric (CPU only):
```shell script
conda install pytorch torchvision cpuonly -c pytorch

pip install torch-scatter==latest+cpu torch-sparse==latest+cpu torch-spline-conv==latest+cpu torch-cluster==latest+cpu -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.5.0.html

pip install torch-geometric -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.5.0.html
```
