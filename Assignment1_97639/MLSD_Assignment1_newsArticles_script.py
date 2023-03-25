from pyspark import SparkContext
from pyspark.sql import SparkSession
import argparse

import numpy as np
from tqdm import tqdm # progress bar
from nltk.corpus import stopwords
from string import punctuation
from sympy import nextprime
from itertools import combinations
import json

np.set_printoptions(threshold=np.inf, precision=0, suppress=True) # prevent scientific notation when printing

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--similarity", help = "similarity threshold", type = float, default = 0.85, nargs = 1)
parser.add_argument("-b", "--bands", help = "number of bands", type = int, default = 13, nargs = 1)
parser.add_argument("-r", "--rows", help = "number of rows", type = int, default = 11, nargs = 1)
parser.add_argument("directory", help = "directory of file", type = str, nargs = 1)
parser.add_argument("cores", help = "number of cores", type = int, nargs = 1)
parser.add_argument("--shingle", help = "shingle size", type = int, nargs = 1, default = 10)
parser.add_argument("signature", help = 'load signature matrix, default True', action = 'store_false')
parser.add_argument("suggestions", help = 'load suggestions dictionary, default True', action = 'store_false')
args = parser.parse_args()

if __name__ == "__main__":
    
    # Assign variables from command line
    sim_threshold = args.similarity
    bands = args.bands
    rows = args.rows
    n_cores = args.cores
    shingle_size = args.shingle
    f_dir = args.directory

    # Function to verify parameters
    def tune_params(bands, rows, sim_threshold):
        """Computes if number of bands and rows is appropriate for given similarity threshold."""
        
        sim = (1/bands)**(1/rows)
        print(f"We obtained a s_threshold of {sim}, should be close to {sim_threshold}, with {rows} rows and {bands} bands.")

        sim_probability = 0.85
        p1 = sim_probability ** rows
        print(f"Probability of C1 and C2 identical in a given band = s^r = {p1}")

        p2 = (1 - p1) ** bands
        print(f"Probability C1, C2 are not similar in all of the {bands} bands = (1-s^r)^b = {p2}")
        if p2 < .1:
            print("Condition 1 obtained\n")

        not_sim_probability = 0.6
        p3 = not_sim_probability ** rows
        print(f"Probability C1, C2 are identical in a given band = s^r = {p3}")

        p4 = 1-(1-p3) ** bands
        print(f"Probability C1, C2 identical in at least 1 of 20 bands: 1-(1-s^r)^b = {p4}")
        if p4 < 0.05:
            print("Condition 2 obtained")
    
    tune_params(bands, rows, sim_threshold)
    
    sc = SparkContext(appName="Assignment1 - News Articles")
    spark = SparkSession.builder.appName("Assignment1 - News Articles").getOrCreate()
    
    # Read file
    df = spark.read.json(f_dir)
    print('File structure:')
    df.printSchema()

    import sys
    sys.exit()
    
    # SHINGLING
    punctuation_table = str.maketrans(dict.fromkeys(punctuation, '')) # Table to remove punctuation from text
    stop_words = set(stopwords.words('portuguese')) # Set with stop words

    news_rdd = df.rdd.map(lambda item: (item['tweet_id'], item['text'])) # Removes url

    shingle_rdd = (news_rdd.map(lambda item: (item[0], item[1].translate(punctuation_table).split()))
                 # Remove punctuation and stop words
                .map(lambda item: (item[0], [w for w in item[1] if w.lower() not in stop_words]))
                # Filter out articles with less than 'shingle_size' words
                .filter(lambda item: len(item[1]) >= shingle_size) 
                # Shingle the documents
                .map(lambda item: (item[0], set([tuple(item[1][i:i + shingle_size]) for i in range(len(item[1]) - shingle_size+1)])))
                ) 
    
    # number of total distinct shingles to use in the hash functions
    num_shingles = shingle_rdd.flatMap(lambda item: item[1]).distinct().count()

    p = nextprime(num_shingles) # Next smallest prime number, larger than num_shingles

    num_permutations = 13 * 11 # bands x rows obtained from 2.1. b*r=n; n is the number of permutations
    a = np.random.randint(1, p, size = (num_permutations)) # a can't be 0
    b = np.random.randint(0, p, size = (num_permutations))

    a_bc = sc.broadcast(a)
    b_bc = sc.broadcast(b)
    
    def hash_function(x, i):
        """Universal hashing function according to class slides."""
        return ((a_bc.value[i] * hash(x) + b_bc.value[i]) % p) % num_shingles
    
    # Hash shingles and convert list to set to remove duplicates
    hashed_shingles = shingle_rdd.map(lambda item: (item[0], set([hash(shingle) for shingle in item[1]])))
    num_docs = hashed_shingles.count()

    print(f'Number of unique shingles: {num_shingles}')
    print(f'Number of documents: {num_docs}')
    
    # MINHASHING
    def minhash(document):
        """Applies minhashing to document."""
        minhash_values = []
        
        # For each permutation, get hash for every shingle and keep lowest
        for i in range(num_permutations):
            permuted_hash = [hash_function(shingle, i) for shingle in document]
            minhash_values.append(min(permuted_hash))
        
        return minhash_values
    
    def signature_matrix(shingles, num_docs, block_size):
        """Create signature matrix using minhash."""
        signature_matrix = []
        signature_matrix_blocks = []
        
        # Repartition RDD to lower bound (2 * cores) to reduce overhead and speed up process
        shingles_with_ids = shingles.zipWithIndex().map(lambda x: (x[1], x[0])).repartition(n_cores * 2).cache()

        # Process the data in blocks
        for i in tqdm(range(0, num_docs, block_size), desc = 'Hashing blocks'):
            # Get the next block of data
            block = shingles_with_ids.filter(lambda x: x[0] >= i and x[0] < i + block_size).map(lambda doc: doc[1][1])
            
            # Compute the MinHash values for each document in the block
            minhash_block = block.map(lambda doc: minhash(doc))
            signature_matrix_blocks.append(np.array(minhash_block.collect()).T)

        signature_matrix = np.concatenate(signature_matrix_blocks, axis = 1)
        return signature_matrix
    
    # This part starts taking some time, so you can open the Spark UI
    print(sc.uiWebUrl)
    
    if args.signature:
        sig_mat = np.load('sig_mat.npy')
    else:
        sig_mat = signature_matrix(hashed_shingles, num_docs, 500)
        np.save('sig_mat.npy', sig_mat)
    
    # LSH
    def lsh(signature_matrix, num_docs, num_bands):
        # Split signature matrix into bands
        bands = np.split(signature_matrix, num_bands)
        buckets = []
        
        # Hash buckets by bands
        for band in tqdm(bands, desc = 'Bucket hashing'):
            # Hash each column r rows at a time
            band_hashes = [hash_function(tuple(band[:, j]), 0) for j in range(num_docs)]
            bucket_dict = {}
            
            # Distribute hashes over buckets
            for j, b_hash in enumerate(band_hashes):
                if b_hash not in bucket_dict:
                    bucket_dict[b_hash] = []
                bucket_dict[b_hash].append(j)
            buckets.append(bucket_dict)
        
        # Generate candidate pairs if they are in the same bucket
        candidate_pairs = set()
        for bucket in tqdm(buckets, desc = 'Candidate pairs generation'):
            for b_hash, docs in list(bucket.items()):
                if len(docs) < 2:
                    continue
                for pair in combinations(docs, 2):
                    candidate_pairs.add(pair)

        return list(candidate_pairs)
    
    candidate_pairs = lsh(sig_mat, num_docs, 13)
    print(f'Total candidate pairs found: {len(candidate_pairs)}')
    
    #SIMILARITY SEARCH
    def jaccard(set1: set[int], set2: set[int]):
        """Jaccard similarity between set1 and set2."""
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def get_similar(id, candidate_pairs, threshold, idx_to_id, hashed_rdd):
        """Find similar articles to the input based on the Jaccard similarity."""
        # Unpack pairs 
        candidate_documents = set([x for pair in candidate_pairs for x in pair])
        candidates_rdd = hashed_rdd.filter(lambda item: item[1] in candidate_documents).collect() # Collect 'real' candidates
        
        document_dict = {key: x for x, key in candidates_rdd} # Dict of document data -> {index: (id, shingles)}
        similar_articles = []

        for pair in candidate_pairs:
            if id in pair:
                sim = jaccard(document_dict[pair[0]][1], document_dict[pair[1]][1])
                if sim >= threshold:
                    suggestion_id = set(pair) - set((id,))
                    similar_articles.append(idx_to_id[suggestion_id.pop()])
        return similar_articles
        
    ### PERFORMANCE EVALUATION
    ### Trying to 1000 random pairs
    # Dictionary with articles index and id
    idx_to_id = {key: id[0] for id, key in hashed_shingles.zipWithIndex().collect()}
    cached_hashed_shingles = hashed_shingles.zipWithIndex().repartition(n_cores * 2).cache()
    
    if args.similarity:
        # Unpack random 200 pairs
        random_candidates = set([candidate_pairs[idx][0] for idx in np.random.choice(len(candidate_pairs), 200)])

        # Make sure there are 200 candidates
        while len(random_candidates) < 200:
            random_candidates.add(candidate_pairs[np.random.randint(0, len(candidate_pairs), 1)[0]][0])

        suggestions = {idx_to_id[id]: [] for id in random_candidates}

        # Verify how many candidate pairs are similar, save to dictionary regardless of being similar or not
        for candidate in tqdm(random_candidates):
            similar_articles = get_similar(candidate, candidate_pairs, 0.85, idx_to_id, cached_hashed_shingles)
            candidate_id = idx_to_id[candidate] # Convert the index to the article id
            suggestions[candidate_id].extend(similar_articles)
    
    else:
        random_keys = json.load(open('suggestions.json', 'r'))
        print('Dictionary with suggestions for randomly picked articles loaded.')
        id_to_idx = {key[0]: id for id, key in idx_to_id.items()} # convert back to index
        random_ids = {id_to_idx[id]: vals for id, vals in random_keys.items()}

    # Dictionary with similarities
    sim_dict = {id: {'similar': [], 'not similar': []} for id in random_ids} 

    for id in tqdm(random_ids):
        possible_pairs = set([(id, idx) for idx in range(num_docs) if id != idx])

        for pair in possible_pairs:
            sim = jaccard(idx_to_id[pair[0]][1], idx_to_id[pair[1]][1])
            if sim >= 0.85:
                sim_dict[id]['similar'].append(sim)
            else:
                sim_dict[id]['not similar'].append(sim)
        
    
    # False positives and negatives
    # Choose 200 random ids to test the function
    random_ids = set([idx for idx in np.random.choice(len(candidate_pairs), 100)])

    # Make sure there are 100 candidates
    while len(random_candidates) < 100:
        random_candidates.add(np.random.randint(0, len(candidate_pairs), 1))

    # Dictionary with similarities
    sim_dict = {id: {'similar': [], 'not similar': []} for id in random_ids} 
    document_dict = {key: x for x, key in hashed_shingles.collect()}

    # Get 'real' similar documents
    for id in tqdm(random_ids):
        possible_pairs = set([(id, idx) for idx in range(num_docs) if id != idx])
        
        for pair in possible_pairs:
            sim = jaccard(document_dict[pair[0]][1], document_dict[pair[1]][1])
            if sim >= 0.85:
                sim_dict[id]['similar'].append(sim)
            else:
                sim_dict[id]['not similar'].append(sim)
    
    # Get FPs and FNs
    performance = {id: {'FP': 0, 'FN': 0} for id in random_ids.keys()}

    for id in tqdm(random_ids):
        for pair in candidate_pairs:
            if id in pair:
                if pair[0] in sim_dict[id]['not similar'] or pair[1] in sim_dict[id]['not similar']:
                    performance[id]['FP'] += 1
            performance[id]['FN'] = len(sim_dict[id]['similar']) - performance[id]['FP']
    
    # print metrics
    for idx, metrics in performance.items():
        print(idx_to_id[idx][0], metrics)