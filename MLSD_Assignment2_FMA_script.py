# %% [markdown]
# # MLSD Assignment 2 - FMA dataset
# 
# We want to cluster a dataset made up of music tracks, where the features are several measures of the audio file.
# 
# First we will apply clustering to a small number of samples (the small dataset), and find the optimal $k$ then apply clustering to the whole dataset.

# %%
%pip install pyspark
%pip install numpy
%pip install pandas==1.5.3
%pip install scikit-learn 
%pip install matplotlib
%pip install seaborn

%matplotlib inline

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm #progress bar

sns.set_style('darkgrid')
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# %% [markdown]
# ### Get every track id belonging to the small dataset (subset == 'small'). Need to use tracks.csv for this
# From tracks.csv, the column that we want is '32'.

# %%
tracks_df = pd.read_csv('data/fma/tracks.csv', header = None)
tracks_df[32].head(10)

# %%
# Get rows that are from the small dataset
small_subset = tracks_df[tracks_df[32] == 'small']
small_subset = small_subset[[0, 32]]
small_subset

# %%
tracks_small_ids = small_subset[0].tolist()
tracks_small_ids

# %% [markdown]
# Verify if it has 8000 samples as it is expected.

# %%
len(tracks_small_ids)

# %% [markdown]
# ### Select the correct tracks from features.csv

# %%
features_df = pd.read_csv('data/fma/features.csv')
features_df.head(10)

# %% [markdown]
# ### Get the rows from tracks.csv that belong to the small dataset

# %%
features_small_df = features_df[features_df.feature.isin(tracks_small_ids)]
features_small_df.drop(columns='feature')

# %%
features_small_df.shape

# %% [markdown]
# Doesn't have all 8000 tracks.
# 
# ### Now apply the in-memory clustering using sklearn AgglomerativeClustering for $k=[8,16]$

# %%
n_clusters = range(8, 17) # number of cluster to evaluate

# euclidean distance matrix of the small dataset
distance_matrix = pairwise_distances(features_small_df.drop(columns = 'feature'))

labels = []

# Perform Agglomerative Clustering on the small dataset
for k in tqdm(n_clusters):
    clustering = AgglomerativeClustering(k).fit(features_small_df.drop(columns = 'feature'))
    labels.append(clustering.labels_)

# %% [markdown]
# ### Now get the centroids using another sklearn function, and compute the several metrics such as radius and density. For evaluation we will use the average diameter of the clusters for each $k$.

# %%
results_dict = {col: [] for col in ['n_clusters', 'cluster', 'radius', 'radius_squared', 'diameter', 'diameter_squared', 'density']}

for i, k in enumerate(n_clusters):
    for n in range(k):
        
        # compute nearest centroids
        clf = NearestCentroid().fit(distance_matrix, labels[i])
        centroids = clf.centroids_[n]
        
        # compute metrics and add to results dictionary
        tmp_points = distance_matrix[np.where(labels[i] == n)]
        diameter = np.max(tmp_points) # largest distance between two points in each cluster
        radius = np.max(abs(tmp_points - centroids))
        density = (len(tmp_points) * features_small_df.shape[0]) / (diameter)
    
        results_dict['n_clusters'].append(k)
        results_dict['cluster'].append(n)
        results_dict['radius'].append(radius)
        results_dict['radius_squared'].append(radius**2)
        results_dict['diameter'].append(diameter)
        results_dict['diameter_squared'].append(diameter**2)
        results_dict['density'].append(density)

# %%
results_df = pd.DataFrame(results_dict)
results_df

# %% [markdown]
# The curse of dimensionality if very clear in these results, very high values for every metric.
# To find the best $k$ for the BFR exercise we use the $k$ that has little improvement of the diameter.

# %%
avg_density = {}
avg_diameter = {}

for k in n_clusters:
    avg_diameter[k] = results_df[results_df.n_clusters == k].describe().loc['mean', 'diameter']

sns.lineplot(avg_diameter)
plt.axvline(14, linestyle = 'dashed', color = 'red')
plt.title('Average Cluster Diameter vs Number of Cluster')
plt.xlabel('Number of Clusters')
plt.ylabel('Diameter')
plt.show()

# %% [markdown]
# The diameter is a good metric to evaluate the clustering since it tells us the maximum distance between any point in the cluster. When increasing the number of clusters from 14 to 15 the diameter decrease is very small, so $k=14$ can be considered the optimal value of $k$. 
# # 1.2. BFR algorithm

# %%
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext(appName="Assignment2 - FMA")
spark = SparkSession.builder.appName("Assignment2 - FMA").getOrCreate()
spark

# %%
# read features.csv
features_df = spark.read.csv('data/fma/features.csv')

# take rows that are the names of the columns
original_columns = features_df.take(3)

# rename columns according to the guide sheet
new_columns = ['track_id'] + [c1 + '_' + c2 + '_' + c3 for c1, c2, c3 in zip(*original_columns)][1:] # skips first column with track_ids

# %%
# Remove first 4 rows and rename the columns by using toDF()
features_df = (features_df.filter(~features_df._c0.isin(['feature', 'statistics', 'number', 'track_id'])) 
               .toDF(*new_columns) 
               )
features_df.show() # final dataframe

# %%
k = 14 # optimal number of clusters

# %% [markdown]
# ### Here we define some function that will be used in the BFR algorithm

# %%
def compute_center(points):
    "Compute centroid of group of points"
    
    n = len(points)
    
    # skip empty groups
    if n == 0:
        return None
    return list(np.sum(points, axis = 0)/n)

# %%
def euclidean_distance(centroid, point):
    "Compute the euclidean distance between a centroid and a point"
    return np.sqrt(np.sum(np.power(np.subtract(centroid, point), 2)))

# %%
def mahalanobis_distance(centroid, stdev, point):
    """Compute Mahalanobis distance of a data point to a given cluster"""

    y = np.subtract(point, centroid) / stdev
    
    # if stdev == 0, this distance would be nan, this way we get some comparable value instead
    if any(x == 0.0 for x in stdev):
        return np.inf
    return np.sqrt(np.sum(y**2))

# %%
def compute_summary(points, n):
    "Compute the initial summary of a cluster"
    
    # metrics according to class slides
    sum_i = np.array(points)
    mean = sum_i / n
    sumsq_i = sum_i ** 2
    variance = (sumsq_i / n) - (sum_i / n)**2
    
    summary = {'N': n, 'SUM_i': sum_i, 'SUMSQ_i': sumsq_i, 'AVERAGE': mean, 'VARIANCE': variance, 'STDEV': np.sqrt(variance), 'ID': {points[0]}}
    return summary

# %%
def update_summary(points, summary):
    "Update an existing summary with new group of points"
    
    ids = list(map(lambda x: x[0], points)) # gets all ids in a list
    points = list(map(lambda x: x[1], points)) # gets all coordinates in a list
    
    summary['N'] = summary['N'] + len(points)
    summary['SUM_i'] = summary['SUM_i'] + np.sum(points, axis = 0)
    summary['SUMSQ_i'] = summary['SUM_i'] ** 2
    summary['VARIANCE'] = (summary['SUMSQ_i'] / summary['N']) - (summary['SUM_i'] / summary['N'])**2
    summary['STDEV'] = np.sqrt(summary['VARIANCE'])
    summary['ID'].update(ids)

# %%
def assign_to_cluster(summaries, point, threshold):
    "Return the cluster index that the point is assigned to"
    
    # if any cluster only has 1 element use the euclidean distance instead
    if any(summary['N'] < 2 for summary in summaries.values()):
        # use the summary 'AVERAGE' since it is also the centroid
        distances = [euclidean_distance(summary['AVERAGE'], point) for summary in summaries.values()]
        cluster = np.argmin(distances) # index of minimum value is the cluster index
        
    else:
        distances = [mahalanobis_distance(summary['AVERAGE'], summary['STDEV'], point) for summary in summaries.values()]
        cluster = np.argmin(distances)
        if distances[cluster] > threshold:
            return None
    
    return cluster

# %%
def cs_assign(summaries, point, threshold):
    "Assign point from the compression set to cluster"
    
    # if any cluster only has 1 element use the euclidean distance instead
    if any(summary['N'] < 2 for summary in summaries.values()):
        distances = [euclidean_distance(summary['AVERAGE'], point) for summary in summaries.values()]
        cluster = np.argmin(distances)
        
    else:
        distances = [mahalanobis_distance(summary['AVERAGE'], summary['STDEV'], point) for summary in summaries.values()]
        cluster = np.argmin(distances)
        
        # if no cluster is within the threshold create a new cluster with the point
        if distances[cluster] > threshold:
            summaries[len(summaries)] = compute_summary(point, 1)
    
    return cluster

# %%
def bfr(rdd, k, threshold_distance):
    "Implementation of the BFR algorithm"
    
    # Get a random sample from the rdd and get its summaries
    centroids = rdd.takeSample(False, k, 12345)
    summaries = {c: compute_summary(centroid[1], 1) for c, centroid in enumerate(centroids)}
    discard_set = []
    retained_set = []

    # Divide the remaining data into compression sets and process them iteratively
    compression_sets = rdd.filter(lambda x: x[1] not in centroids).randomSplit([10]*10, 12345)
    for comp_set in tqdm(compression_sets):

        # Assign points to initial clusters
        assignments = comp_set.map(lambda x: (assign_to_cluster(summaries, x[1], threshold_distance), x))

        # current discard set is made up of every point that was initially assigned to a cluster
        discarded = (assignments.groupByKey().mapValues(list)
                  .filter(lambda x: x[0] != None)
                  .collect()
                  )
        # update summaries with the points from the intial discard set
        for cluster, points in discarded:
            update_summary(points, summaries[cluster])
        
        # retained set made up of remaining points
        retained = assignments.filter(lambda x: x[0] == None)
        
        # from these retained set, select 10 random points and compute their summaries
        cs_centroids = retained.map(lambda x: x[1]).takeSample(False, 10, 12345)
        cs_summaries = {c: compute_summary(centroid[1], 1) for c, centroid in enumerate(cs_centroids)}
        
        # assign points from retained set to the compression set clusters
        cs_assignments = retained.map(lambda x: (assign_to_cluster(cs_summaries, x[1][1], threshold_distance), x[1])).groupByKey().mapValues(list)
        
        # add recently assigned points to discard set
        new_discard = cs_assignments.filter(lambda x: x[0] != None).collect()
        
        # update compression set centroids again
        for cluster, points in new_discard:
            update_summary(points, cs_summaries[cluster])
    
    # remove clusters that only have 1 point and keep them as outliers
    for cluster, summary in cs_summaries.items():
        if summary['N'] < 2:
            retained.append(summary)
            del cs_summaries[cluster]
    
    discard_set = [summaries, cs_summaries]

    # Return final set of clusters and their summaries
    return discard_set, retained_set

# %% [markdown]
# ### Functions defined we can now run the algorithm and cluster the dataset

# %%
features_rdd = features_df.rdd.map(lambda row: (row[0], [float(item) for item in row[1:]]))
features_rdd.count()

# %%
discard, retained = bfr(features_rdd, k, 3)

# %% [markdown]
# ## 1.3. Use the previous cluster and see the most common genres in each one.

# %%
# columns with id and genres
tracks_df[tracks_df.columns[[0, 40]]]

# %% [markdown]
# Create new dataframe with id as index and the genre as the only column, drop nan values which reduced the number of rows to about half

# %%
genre_df = tracks_df[[0, 40]]
genre_df = genre_df.dropna().set_index(0)
print(pd.unique(genre_df[40]))
print(f'shape of tracks dataframe: {tracks_df.shape}')
print(f'shape of genres dataframe: {genre_df.shape} -> dropped nan values')
genre_df.head()

# %% [markdown]
# Join the dictionaries into one

# %%
discard_clusters = {c: {'N': discard[0][c]['N'], 'CENTROID': discard[0][c]['AVERAGE'], 'ID': discard[0][c]['ID']} for c in discard[0]}

for c in discard[1]:
    discard_clusters[len(discard_clusters) + c] = {'N': discard[1][c]['N'], 'CENTROID': discard[1][c]['AVERAGE'], 'ID': discard[1][c]['ID']}

# %% [markdown]
# By counting the number of elements in the clustered points, we see that they have approximately the same number as the BFR clusters.

# %%
count = 0
for c in discard_clusters:
    print(c, discard_clusters[c]['N'])
    count += discard_clusters[c]['N']
count

# %% [markdown]
# Associate each id with a genre and count in each cluster, then return the most counted genre

# %%
for c in discard_clusters:
    discard_clusters[c]['genres'] = []
    
    for point in discard_clusters[c]['ID']:
        # some 'ids' had anomalous values such as floats
        if point in genre_df.index:
                # append the genre association
                discard_clusters[c]['genres'].append(genre_df.loc[point])
                # count each genre in the cluster, return_counts returns an array with the count for each genre
                genres = np.unique(discard_clusters[c]['genres'], return_counts=True)
                # get most counted genre
                top_genre = np.argmax(genres[1])
                discard_clusters[c]['TopGenre'] = genres[0][top_genre]

# %% [markdown]
# Finally we can print the results

# %%
for c in discard_clusters:
    print(f"Cluster {c} top genre: {discard_clusters[c]['TopGenre']}")


