import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from google.colab import drive
import csv
from scipy.io import loadmat
from sklearn.cluster import kmeans_plusplus

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/P4/breast_data.csv')
labels = pd.read_csv('/content/drive/My Drive/P4/breast_labels.csv')

#a.

def kmeanscluster(X, k, mu, tol=1e-4, maxIter=300):
    n_samples, n_features = X.shape
    C = np.zeros(n_samples, dtype=int)

    for iteration in range(maxIter):
        # Step 1: Assign each data point to the closest center
        distances = np.linalg.norm(X[:, np.newaxis, :] - mu, axis=2)  # Modified this line
        new_C = np.argmin(distances, axis=1)

        # Step 2: Compute new centers
        new_mu = np.array([X[new_C == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(new_mu == mu) or np.linalg.norm(new_mu - mu) < tol:
            break

        mu = new_mu
        C = new_C

    return C, mu

# Convert Pandas DataFrame to NumPy array
X = data.to_numpy()
y_true = labels.to_numpy()

# Use kmeans_plusplus to initialize cluster centers
centroids, _ = kmeans_plusplus(X, n_clusters=k)

# Extract the cluster centers from the centroids
mu = centroids

k = mu.shape[0]
print(k)

# Perform K-Means clustering
C, final_mu = kmeanscluster(X, k, mu)

from sklearn.metrics import accuracy_score

# Assuming true labels are in y_true
accuracy1 = accuracy_score(y_true, C)
accuracy2 = accuracy_score(y_true, 1 - C)
accuracy = max(accuracy1, accuracy2)
print(f"Accuracy: {accuracy}")

################################################

accuracies = []

for i in range(5):
    np.random.seed(i)
    initial_mu = X[np.random.choice(X.shape[0], k, replace=False)]
    C, final_mu = kmeanscluster(X, k, initial_mu)
    accuracy = accuracy_score(y_true, C)
    accuracies.append(accuracy)
    print(f"Run {i+1}: Accuracy = {accuracy}")

average_accuracy = np.mean(accuracies)
print(f"Average Accuracy: {average_accuracy}")

#################################################

# Convert Pandas DataFrame to NumPy array
X = data.to_numpy()
y_true = labels.to_numpy()

# Load initial centers from MATLAB file
mu = loadmat('init_mu.mat')['mu_init'].T
k = mu.shape[0]
print(k)
# Perform K-Means clustering
C, final_mu = kmeanscluster(X, k, mu)


#Initializing with the true centers means using the centroids of the clusters as determined by the true labels. This usually results in perfect clustering:


# Convert Pandas DataFrame to NumPy array
X = data.to_numpy()
y_true = labels.to_numpy().ravel()  # Ensure y_true is a 1D array

# Initialize k using the number of unique labels
k = len(np.unique(y_true))

# Use kmeans_plusplus to initialize cluster centers
centroids, _ = kmeans_plusplus(X, n_clusters=k)

# Perform K-Means clustering
C, final_mu = kmeanscluster(X, k, centroids)

# Calculate accuracy with the true centers
true_centers = np.array([X[y_true == j].mean(axis=0) for j in np.unique(y_true)])
C, final_mu = kmeanscluster(X, k, true_centers)
accuracy = accuracy_score(y_true, C)
print(f"Accuracy with true centers: {accuracy}")

"""## e.

 Unsupervised Learning Methods

Hierarchical Clustering:

Hierarchical clustering can be an alternative. It does not require specifying the number of clusters in advance and can be visualized using a dendrogram.
"""

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import accuracy_score

# Perform hierarchical clustering
Z = linkage(X, method='ward')
clusters = fcluster(Z, k, criterion='maxclust')
hierarchical_accuracy = accuracy_score(y_true, clusters)
print(f"Accuracy with Hierarchical Clustering: {hierarchical_accuracy}")

"""Supervised Learning Methods

Support Vector Machine (SVM):

SVMs can often yield better results because they use label information during training.
"""

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with SVM: {svm_accuracy}")

"""Supervised methods, such as SVM, generally achieve higher accuracy compared to unsupervised methods like K-Means because they utilize the label information during training, leading to more accurate predictions."""