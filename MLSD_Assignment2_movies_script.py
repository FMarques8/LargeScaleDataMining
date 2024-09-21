# %% [markdown]
# # MLSD Assignment 2 - User Movie ratings
# 
# ### We want to recommend new movies to an user based on users with similar ratings that watched the movie.

# %%
!pip install pyspark
!pip install pandas
!pip install numpy
!pip install sympy
!pip install tqdm

# %%
import pandas as pd 
import numpy as np
from tqdm import tqdm # progress bar
from sympy import nextprime

# %%
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext(appName="exercise1")
spark = SparkSession.builder.appName("BFR").getOrCreate()

# %% [markdown]
# Read the dataset

# %%
df = spark.read.option("sep", "\t").csv("data/movielens/u.data") 

# %% [markdown]
# Rename columns

# %%
new_columns = ["user id", "item id", "rating","timestamp" ]

df_rdd = df.rdd
df = df_rdd.toDF(new_columns)
df.show()

# %%
from pyspark.sql.functions import asc
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

df = df.select('user id', 'item id', 'rating')

# loop through columns and cast to float
for col_name in df.columns:
    df = df.withColumn(col_name, col(col_name).cast("int")) # Convert string values to int values
# assuming df is your dataframe
sorted_df = df.orderBy(asc("user id"), asc("item id"), asc("rating"))

# %%
sorted_df.show()

# %% [markdown]
# Create a item-user table with the ratings

# %%
# pivot the data on item id and aggregate the rating by user id
df = sorted_df.groupBy('item id').pivot('user id').agg({'rating': 'first'}).na.fill(int(0))

# %%
# order the index
df = df.orderBy(asc("item id"))
new_column_name = "item/ user"
old_column_name = "item id"
df = df.withColumnRenamed(old_column_name,new_column_name)

df.show()

# %% [markdown]
# Function of the Pearson correlation

# %%
def  N(row):
  "Returns the number of values that are different from 'null' in order to compute its mean"
  
  N = 0
  for i in row:
    if i != 0:
      N += 1
  return N 

# %%
def pearson_corr(row, N):
  "Returns a row of the Pearson correlation"
  mean = np.sum(row)/N
  new_row = [] 
  
  for i in row:
    if i != 0:
      new_number = i - mean
      new_row.append(new_number) 
    else:
      new_row.append(i)
  return new_row


# %%
# count the number of non-zero ratings
user_item_matrix = df.rdd.map(lambda x: x[1:])

user_item_matrix = user_item_matrix.map(lambda x: (x, N(x)))

# %%
# apply the pearson correlation to the user-item matrix
matrix_scaled = user_item_matrix.map(lambda x: pearson_corr(x[0],x[1]))

# %% [markdown]
# Add the index again

# %%
matrix_scaled = matrix_scaled.zipWithIndex().map(lambda x: (x[1], x[0])).cache()
new_matrix_scaled = matrix_scaled.collect()

# %%
num_items = matrix_scaled.count()
num_items

# %% [markdown]
# ## Apply LSH to find candidate pairs
# 
# The same way it was done in the first assignment

# %%
num_users = 943
num_items = 1682 

# %%
num_permutations = 13 * 11 # bands x rows obtained from b*r=n; n is the number of permutations

p = nextprime(num_users)
a = np.random.randint(1, p, size = (num_permutations))
b = np.random.randint(0, p, size = (num_permutations))

a_bc = sc.broadcast(a)
b_bc = sc.broadcast(b)

def hash_function(x, i):
  return ((a_bc.value[i] * hash(x) + b_bc.value[i]) % p) % num_users

# %%
def minhash(item, permutations):
    sign_values = []
    for i in range(permutations):
      hash_value = [hash_function(user,i) for user in item]
      sign_values.append(min(hash_value))
    return sign_values


# %%
def signature_matrix(user_item_mat, permutations, number_of_items , block_size):
    # Create a set for each hash_set
    signature_matrix_blocks = []
    # Shingles with an index
    user_item_mat_with_ids = user_item_mat.zipWithIndex().map(lambda x: (x[1], x[0])).cache()

    # Process the data in blocks
    for i in tqdm(range(0, number_of_items , block_size)):
        # Get the next block of data and compute the MinHash values for each document in the block
        block = user_item_mat_with_ids.filter(lambda x: x[0] >= i and x[0] < i + block_size)\
        .map(lambda x: (minhash(x[1][1], permutations)))

        # Construct the signature matrix for the block
        signature_matrix_block = np.array(block.collect()).T
        
        # Add the block's signature matrix to the list of blocks
        signature_matrix_blocks.append(signature_matrix_block)
        
    # Combine the signature matrices for each block
    signature_matrix = np.concatenate(signature_matrix_blocks, axis=1)
    
    return  signature_matrix


# %%
def get_buckets(bands,number_of_items):
  # This function hash band columns into buckets
  buckets = []
  for band in tqdm(bands, desc = 'Bucket hashing'):
    
    band_hashed = [hash_function(tuple(band[:, j]), 0) for j in range(number_of_items)]
    bucket_dict = {}
        
    for j, b_hash in enumerate(band_hashed):
      if b_hash not in bucket_dict:
        bucket_dict[b_hash] = []
      bucket_dict[b_hash].append(j)
    buckets.append(bucket_dict)
  return buckets


# %%
from itertools import combinations

def lsh(sig_mat,number_of_items,num_bands,num_permutations):
    # Hash bukets
    bands = np.split(sig_mat, num_bands)
    buckets = get_buckets(bands,number_of_items)

    # Generate candidate pairs
    candidate_pairs = set()
    for bucket in tqdm(buckets, desc = 'Candidate pairs generation'):
        for b_hash, items in list(bucket.items()):
            if len(items) < 2:
                continue
            for pair in combinations(items, 2):
                candidate_pairs.add(pair)

    candidate_pairs = list(candidate_pairs)
    return candidate_pairs # It returns the ID of each pair

# %%
block_size = 100
signature_mat = signature_matrix(matrix_scaled, num_permutations,num_items, block_size)

# %%
num_bands = 13
candidate_pairs = lsh(signature_mat,num_items,num_bands,num_permutations)

# %% [markdown]
# Functions to obtain cosine similarity between items

# %%
def cosine_sim(r1, r2):
  return np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))

# %% [markdown]
# Testing on the first 2 items

# %%
cosine_sim(new_matrix_scaled[0][1], new_matrix_scaled[1][1])

# %%
print("candidate pairs: ",len(candidate_pairs))

def compute_weights(c_pairs, ratings_mat):
    sim_weights = {}

    # get similarities between candidate pairs
    for item1, item2 in candidate_pairs:
        r1 = new_matrix_scaled[item1 - 1][1]
        r2 = new_matrix_scaled[item2 - 1][1]
        # many are 0 because norm of some items are 0
        sim = cosine_sim(r1, r2)
        try:
            if np.nan:
                sim_weights[item1].append(0)
            else:
                sim_weights[item1].append(sim)
        except KeyError:
            sim_weights[item1] = []
        
    return sim_weights

sim_weights = compute_weights(candidate_pairs, new_matrix_scaled)

# %%
def calculate_new_ratings(item_i, user, ratings_mat, sim_weights):

  '''
  Given the id of item_i, it calculates the new rating using the weighted sum
  '''

  item_i = ratings_mat[item_i][1][user]
  # new_rating = np.sum([w * x for x ])
  # sum_weigth[item_i][user]
  # Initialize our coefficients
  
  new_rating = 0
  
  for item, row in sim_weights.items():
    new_rating += row[user] * ratings_mat[item - 1][1][user]
  
  return new_rating

# %%
new_matrix = matrix_scaled.map(lambda row: [i if i != 0 
                                            else calculate_new_ratings(row, idx, new_matrix_scaled, sim_weights) 
                                            for idx, i in enumerate(row[1])] )

# %%
new_matrix.collect()


