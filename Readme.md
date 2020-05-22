# Domain Adaptation based tweet classification using Graph Neural Networks

Problem with tweets are the availability of limited context. To increase context, generate a large graph using labelled
 and unlabelled data
 from source domain (S) and unlabelled data from target domain (T). Each node in the graph is a token in the mixed
  corpus and edges are formed based the co-occurrence of two tokens in a single tweet (or within a sliding window
   inside the tweet). Edge weights may be calculated using Pair-wise Mutual Information (PMI). Each note has two
    attributes containing information if it occurs in S or T or both.
    
    
* Utilize global and local context: global from token graph, local from tweet token relation
    
## Approach:
    1. Construct a joint token graph (G) using S and T data with domain information
    2. Preprocess G such that node representations are domain-invariant
    3. Construct subgraph $H_i$ by following:
        3.1 Construct a sample graph (H1) by connecting adjacent tokens
        3.2 Fetch n-hop neighbor subgraph of H's nodes from G to construct H2.
        3.3 Merge H1 and H2 by merging the neighbors from H2 to H1.
        3.4 H1 now represents sample graph with domain invariant feature
    4. Pass H1 through GNN such that linguistic features are captured to get f(H1)
    5. Pass f(H1) through an aggregate network g() to 

## Baselines:
    * TextGCN
    * TexTing
    * DA based approaches

## Experiments:
    1. Comparison between tweet level and sliding-window co-occurrance
    2. Experiment with multi-hops (e.g. 0-hop, 1-hop, 2-hop, ...)
    3. Effect of token graph DA preprocessing


### Tasks (Minimal):

    1. Tokenize tweets
    2. Generate graph based on co-occurrance of tokens within a tweet
    3. Mark each token with it's domain 

### Tasks (Preferred):

    1. Lemmatize tokens
    2. Correct Spelling
    3. Edges based on token occurrance within a Sliding window
    4. Expand URLs and add words from the URLs
    5. Calculate edge weights
    6. Use BERTweet to generate token embeddings
    7. Replace numbers with #D
    8. Search and replace phone numbers
