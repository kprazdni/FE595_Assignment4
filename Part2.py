# Katelin Prazdnik
# FE595 Assignment 4 - SKLearn Review Part 2
# December 18, 2020
# I pledge on my honor that I have not given or received any unauthorized assistance on
# this assignment/examination. I further pledge that I have not copied any material from
# a book, article, the Internet or any other source except where I have expressly cited the
# source.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

# Iris Dataset

# Upload the Boston data set from SKLearn
data_1 = load_iris()

# Create features dataframe
features_1 = pd.DataFrame(data_1.data, columns=data_1['feature_names'])

# Perform KNN for all clusters with k values 1-15
SSE_1 = {}
for k in range(1, 15):
    KNN = KMeans(n_clusters=k, max_iter=1000).fit(features_1)
    features_1["clusters"] = KNN.labels_
    SSE_1[k] = KNN.inertia_

# Plot to see where the elbow is to pick optimal number of populations
plt.figure()
plt.plot(list(SSE_1.keys()), list(SSE_1.values()))
plt.title('Iris Elbow Method Graph')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
# 3 does seem to be the optimal number of populations

# Wine Dataset

# Upload the Boston data set from SKLearn
data_2 = load_wine()

# Create features dataframe
features_2 = pd.DataFrame(data_2.data, columns=data_2['feature_names'])

# Perform KNN for all clusters with k values 1-15
SSE_2 = {}
for k in range(1, 15):
    KNN = KMeans(n_clusters=k, max_iter=1000).fit(features_2)
    features_2["clusters"] = KNN.labels_
    SSE_2[k] = KNN.inertia_

# Plot to see where the elbow is to pick optimal number of populations
plt.figure()
plt.plot(list(SSE_2.keys()), list(SSE_2.values()))
plt.title('Wine Elbow Method Graph')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
# 3 does not seem to be the optimal number of populations for this dataset, it looks like it may be 4
