# Domain Adaptation based tweet classification using Graph Neural Networks

Problem with tweets are the availability of limited context. To increase context, generate a large graph using labelled 
 and unlabelled data from source domain (S) and unlabelled data from target domain (T). Each node in the graph is a 
 token in the mixed corpus and edges are formed based the co-occurrence of two tokens in a single tweet (or within a 
 sliding window inside the tweet). Edge weights may be calculated using Pair-wise Mutual Information (PMI). Each note 
 has two attributes containing information if it occurs in S or T or both.
    
    
* Utilize global and local context: global from token graph, local from tweet token relation
* Parameter priority: argparse > config > default
    
## Approach:
    1 Construct a joint token graph (G) using S and T data with domain information
        1.1. Calculate edge weights
        1.2. Generate token embeddings
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
    4 Pass H1 through GNN such that linguistic features are captured to get f(H1)
    5 Pass f(H1) through an aggregate network g() to 

## Baselines:
    * TextGCN
    * TexTing
    * DA based approaches

## Experiments:
    1. Comparison between tweet level and sliding-window co-occurrance
    2. Experiment with multi-hops (e.g. 0-hop, 1-hop, 2-hop, ...)
    3. Effect of token graph DA preprocessing


### Tasks (Minimal):

    - [x] 1. Tokenize tweets (:heavy_check_mark:)
    - [x] 2. Generate graph based on co-occurrance of tokens within a tweet (&#9744;)
    - [x] 3. Mark each token with it's domain count (&#9745;)
    - [x] 4. Aggregate codes in main()
    - [] 5. Generate sample graphs 
    - [] 6. Pass through GNN
    - [] 7. 
    - [] 8. 
    - [] 9. 
    - [] 10. 

### Tasks (Preferred):

    - [] 1. Lemmatize tokens
    - [] 2. Correct Spelling
    - [] 3. Edges based on token occurrance within a Sliding window
    - [] 4. Expand URLs and add words from the URLs
    - [] 5. Calculate edge weights
    - [] 6. Use BERTweet to generate token embeddings
    - [] 7. Replace numbers with #D
    - [] 8. Search and replace phone numbers
