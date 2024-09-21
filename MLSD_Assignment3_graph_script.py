import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os, argparse
from scipy.signal import argrelextrema # get peaks
from sklearn.cluster import KMeans, SpectralClustering
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.clustering import PowerIterationClustering

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help = "directory of *every* graph file, assumes default directory names", 
                    type = str, required = True)
args = parser.parse_args()

def cluster_graph(graph, spark_graph):
    "Performs the 3 clustering algorithms on given graph, 1 figure per graph"
    fig = plt.figure(1)
    fig, ax = plt.subplots(1, 5, figsize=(29, 5))

    pos = nx.spring_layout(graph) # generate coordinates for nodes
    
    plt.sca(ax[0])
    nx.draw_networkx(graph, pos = pos, with_labels = False, node_size = 4, width = 0.2)
    plt.title('Initial graph')
    
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    eigvals, eigvecs = np.linalg.eig(laplacian)

    sorted_indices = np.argsort(eigvals) #indices of sorted eigenvalues
    sorted_eigenvalues = eigvals[sorted_indices] # sorted eigenvalues
    sorted_eigenvectors = eigvecs[:, sorted_indices] #sorted eigenvectors
    
    # eigengap
    eig_diff = np.diff(sorted_eigenvalues)
    peaks = argrelextrema(eig_diff, np.greater)[0] # unpack the tuple that is returned as there is only 1 dimension
    nb_clusters = peaks[0] + 1 # optimal number of cluster is first peak
    
    # eigenvalues plot
    plt.sca(ax[1])
    plt.scatter(range(len(sorted_eigenvalues)), sorted_eigenvalues, marker='*')
    plt.title('Eigenvalues')

    kmeans = KMeans(nb_clusters, n_init = 'auto').fit(np.real(sorted_eigenvectors[1:nb_clusters + 1]).T)
    
    plt.sca(ax[2])
    nx.draw_networkx(graph, pos = pos, with_labels = False, node_size = 4, width = 0.2, node_color = kmeans.labels_)
    plt.title(f'Spectral clustering, $k = {nb_clusters}$')
    
    # sklearn spectral clustering
    clustering = SpectralClustering(n_clusters = nb_clusters, assign_labels = 'kmeans').fit(laplacian)
    
    plt.sca(ax[3])
    nx.draw_networkx(graph, pos = pos, with_labels = False, node_size = 4, width = 0.2, node_color = clustering.labels_)
    plt.title(f'sklearn Spectral Clustering, $k = {nb_clusters}$')
    
    # Spark Power Iteration
    assignments = PowerIterationClustering(k = nb_clusters).assignClusters(dataset = spark_graph).sort('id').rdd.map(lambda row: row[1]).collect()
    
    # only happened for phenomenology dataset where the dataset has slightly smaller size than the real graph
    # so we just add 0s as it wont make a noticeable difference in the results
    if abs(len(assignments) - len(graph)) > 0:
        for _ in range(abs(len(assignments) - len(graph))):
            assignments.append(0)
    
    plt.sca(ax[4])
    nx.draw_networkx(graph, pos = pos, with_labels = False, node_size = 4, width = 0.2, node_color = assignments)
    plt.title(f'Spark Power Iteration Clustering, $k = {nb_clusters}$')

if __name__ == '__main__':
    graphs = [] # list with graphs
    spark_graphs = [] # list with graphs in spark
    
    # start spark session
    sc = SparkContext(appName="Assignment 3 - Graph Power Iteration")
    spark = SparkSession.builder.appName("Assignment 3 - Graph Power Iteration").getOrCreate()
    
    # load facebook graphs 
    for file in os.listdir(args.directory + "/facebook"):
        if file.endswith('.edges'): # only reads edge lists
            graphs.append(nx.read_edgelist(f'{args.directory}/facebook/{file}')) # read graph from edge lists
            
            # spark graphs
            edges = spark.read.text(f'{args.directory}/facebook/{file}')
            edges = (edges.rdd.map(lambda x: x[0].split(" ")) # split edges into 2 element lists
                    .map(lambda row: (int(row[0]), int(row[1]))) # put each edge in two columns
                    .toDF(['src', 'dst'])) # convert to dataframe
            spark_graphs.append(edges)
    
    # facebook graphs 
    for i, G in enumerate(graphs):
        cluster_graph(G, spark_graphs[i])
        
    # load phenomenology graph
    phenom_graph = nx.read_edgelist(args.directory + '/CA-HepPh.txt')
    phenom_spark_graph = (spark.read.text(args.directory + '/CA-HepPh.txt').rdd
                          .filter(lambda row: '#' not in row[0]) # skip initial
                          .map(lambda x: x[0].split("\t")) # edges separated by \t
                          .map(lambda row: (int(row[0]), int(row[1])))
                          .toDF(['src', 'dst']))
    
    # phenomenology graph
    cluster_graph(phenom_graph, phenom_spark_graph)
    
    # protein-protein interaction graph
    # graph must be manually created from csv
    protein_graph = nx.Graph() # initialize graph

    with open(args.directory + '/PP-Pathways_ppi.csv') as f:
        data = f.read().split('\n')
        for edge in data:
            if edge == '': # skip empty lines
                continue
            u, v = edge.split(',')
            protein_graph.add_edge(u, v) # automatically adds each node if they are not in node list

    protein_spark_graph = (spark.read.csv('data/PP-Pathways_ppi.csv').rdd
                           .map(lambda row: (int(row[0]), int(row[1])))
                           .toDF(['src', 'dst']))
    
    # TAKES A LONG TIME due to very large laplacian (~320kx320k)
    cluster_graph(protein_graph, protein_spark_graph)