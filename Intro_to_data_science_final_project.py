#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the breast cancer dataset
data = load_breast_cancer().data

# Reduce the dimensions of the dataset to 2 features using PCA
pca = PCA(n_components=2)
df = pca.fit_transform(data)

# Create an object of KMeans with n_clusters = 10
kmeans = KMeans(n_clusters=10)

# Fit and predict the transformed dataset
labels = kmeans.fit_predict(df)

# Plot all the observations on a 2D coordinate system
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.scatter(df[labels == i, 0], df[labels == i, 1], label=f'Cluster {i}')

plt.title('K-means Clustering - Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[14]:


#part2
# Import required libraries
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the digits dataset
data = load_digits().data

# Reduce dimensions to 2 using PCA
pca = PCA(2)
df = pca.fit_transform(data)

# Create a DBSCAN clustering object with min_samples = 10 and eps = 1.5
dbscan = DBSCAN(min_samples=10, eps=1.5)

# Fit and predict the transformed dataset
labels = dbscan.fit_predict(df)

# Get unique labels
unique_labels = set(labels)

# Plot the observations for each label
for label in unique_labels:
    if label == -1:
        plt.scatter(df[labels == label, 0], df[labels == label, 1], label='Noise')
    else:
        plt.scatter(df[labels == label, 0], df[labels == label, 1], label='Cluster ' + str(label))
plt.title('Density-based Clustering (DBSCAN) - Digits Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[9]:


#part3
# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# Load the iris dataset
data = load_iris().data

# Reduce the dimensions of the dataset to 2 features using PCA
pca = PCA(n_components=2)
df = pca.fit_transform(data)

# Create an object of AgglomerativeClustering with n_clusters = 5
agg_clustering = AgglomerativeClustering(n_clusters=5)

# Fit and predict the transformed dataset
labels = agg_clustering.fit_predict(df)

# Plot all the observations on a 2D coordinate system
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.scatter(df[labels == i, 0], df[labels == i, 1], label=f'Cluster {i}')

plt.title('Hierarchical Clustering - Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[10]:


# Part 1 (K-means clustering) with n_clusters = 20 and digits dataset
# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the digits dataset
data = load_digits().data

# Reduce the dimensions of the dataset to 2 features using PCA
pca = PCA(n_components=2)
df = pca.fit_transform(data)

# Create an object of KMeans with n_clusters = 20
kmeans = KMeans(n_clusters=20)

# Fit and predict the transformed dataset
labels = kmeans.fit_predict(df)

# Plot all the observations on a 2D coordinate system
plt.figure(figsize=(8, 6))
for i in range(20):
    plt.scatter(df[labels == i, 0], df[labels == i, 1], label=f'Cluster {i}')

plt.title('K-means Clustering (n_clusters=20) - Digits Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# Part 3 (Hierarchical Clustering) with n_clusters = 20 and digits dataset
# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# Load the digits dataset
data = load_digits().data

# Reduce the dimensions of the dataset to 2 features using PCA
pca = PCA(n_components=2)
df = pca.fit_transform(data)

# Create an object of AgglomerativeClustering with n_clusters = 20
agg_clustering = AgglomerativeClustering(n_clusters=20)

# Fit and predict the transformed dataset
labels = agg_clustering.fit_predict(df)

# Plot all the observations on a 2D coordinate system
plt.figure(figsize=(8, 6))
for i in range(20):
    plt.scatter(df[labels == i, 0], df[labels == i, 1], label=f'Cluster {i}')

plt.title('Hierarchical Clustering (n_clusters=20) - Digits Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[ ]:




