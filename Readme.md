# Domain Adaptation based tweet classification using Graph Neural Networks

Problem with tweets are the availability of limited context. To increase context, generate a large graph using labelled
 and unlabelled data
 from source domain (S) and unlabelled data from target domain (T). Each node in the graph is a token in the mixed
  corpus and edges are formed based the co-occurrence of two tokens in a single tweet (or within a sliding window
   inside the tweet). Edge weights may be calculated using Pair-wise Mutual Information (PMI). Each note has two
    attributes containing information if it occurs in S or T or both.


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
