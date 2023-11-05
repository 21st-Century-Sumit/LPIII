# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load and preprocess the dataset
df = pd.read_csv("sales_data_sample.csv")  # Replace with the path to your dataset
data = df[['Quantity', 'Gross margin percentage', 'Gross income']]

# Standardize the data (important for K-Means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Determine the optimal number of clusters using the elbow method
wcss = []  # Within-Cluster-Sum-of-Squares

# Try different numbers of clusters from 1 to a reasonable maximum
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Step 4: Perform K-Means clustering with the chosen number of clusters
# Based on the elbow method graph, choose the number of clusters (the "elbow" point)
k = 3  # You may adjust this value based on the elbow method plot

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(data_scaled)

# Add cluster labels to the original dataset
df['Cluster'] = kmeans.labels_

# Step 5: Visualize the results (scatterplot)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1,], s=200, c='red')
plt.title('K-Means Clusters')
plt.xlabel('Quantity')
plt.ylabel('Gross Margin Percentage')
plt.show()
