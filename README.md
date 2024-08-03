# Breast Cancer Clustering and Classification

This repository contains code for clustering and classification of breast cancer data using both unsupervised and supervised learning methods. The primary goal is to apply various clustering techniques and compare their performance with supervised learning methods.

## Overview

The project uses a dataset of breast cancer samples to perform clustering using the K-Means algorithm and hierarchical clustering. It also compares these unsupervised learning methods with supervised learning methods such as Support Vector Machines (SVM).

## Dataset

The dataset used in this project consists of breast cancer data with features and labels:
- `breast_data.csv`: Contains the feature values for each sample.
- `breast_labels.csv`: Contains the labels indicating the presence of cancer.

## Project Structure

- **a. K-Means Clustering with Custom Initialization**:
  - Implementation of the K-Means clustering algorithm with custom initialization.
  - Initialization using the kmeans++ algorithm to select initial cluster centers.
  - Accuracy measurement of the clustering results.

- **b. K-Means Clustering with Multiple Random Initializations**:
  - Running the K-Means algorithm multiple times with different random initializations.
  - Calculation of accuracy for each run and computation of the average accuracy.

- **c. K-Means Clustering with MATLAB Initialization**:
  - Initialization of cluster centers using pre-defined values from a MATLAB file (`init_mu.mat`).
  - Performance evaluation of the clustering results.

- **d. K-Means Clustering with True Centers**:
  - Initialization of cluster centers using the true centroids of the clusters.
  - Comparison of clustering accuracy with the true centers.

- **e. Comparison with Other Methods**:
  - Hierarchical clustering as an alternative unsupervised learning method.
  - Supervised learning using Support Vector Machines (SVM) to demonstrate the advantage of using label information during training.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: numpy, pandas, matplotlib, scipy, scikit-learn, opencv

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/breast-cancer-clustering.git
   cd breast-cancer-clustering
   ```

2. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Load the dataset and labels:
   ```python
   data = pd.read_csv('path_to_your/breast_data.csv')
   labels = pd.read_csv('path_to_your/breast_labels.csv')
   ```

2. Perform K-Means clustering:
   ```python
   centroids, _ = kmeans_plusplus(data.to_numpy(), n_clusters=k)
   C, final_mu = kmeanscluster(data.to_numpy(), k, centroids)
   ```

3. Evaluate accuracy:
   ```python
   accuracy = accuracy_score(labels.to_numpy(), C)
   print(f"Accuracy: {accuracy}")
   ```

4. Run hierarchical clustering:
   ```python
   Z = linkage(data.to_numpy(), method='ward')
   clusters = fcluster(Z, k, criterion='maxclust')
   hierarchical_accuracy = accuracy_score(labels.to_numpy(), clusters)
   print(f"Accuracy with Hierarchical Clustering: {hierarchical_accuracy}")
   ```

5. Train and evaluate SVM:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(data.to_numpy(), labels.to_numpy(), test_size=0.2, random_state=42)
   svm = SVC(kernel='linear')
   svm.fit(X_train, y_train)
   y_pred = svm.predict(X_test)
   svm_accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy with SVM: {svm_accuracy}")
   ```

## Results

- K-Means clustering shows variable performance depending on the initialization method.
- Hierarchical clustering provides an alternative unsupervised approach with competitive accuracy.
- Supervised learning using SVM achieves higher accuracy by leveraging label information during training.

## Conclusion

This project demonstrates the application of various clustering and classification techniques on breast cancer data. It highlights the importance of initialization methods in K-Means clustering and the advantage of supervised learning methods over unsupervised approaches.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
