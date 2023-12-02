# KMeansClustering.py
# This program implements a version of K-means clustering on machinery data of a mill. It separates the dataset
# into testing data and training data then implements the clustering analysis on two separate groups
# the "Machine Failure" group and "No Machine Failure" group. The training data goes through an elbow curve and
# silhouette scoring to determine optimal 'k' values for each group. Once the centroids are found for each group the
# test data is sent and tested to see which group each point is closer to based on this is where the prediction is made.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# This method is used to find the optimal number of clusters for the give dataset
# It implements both the elbow method and finds the silhouette score for the dataset
# for 'k' values from 2 to 10.
def find_purity(x):
    wcss = {}
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x)
        wcss[i] = kmeans.inertia_

    plt.plot(wcss.keys(), wcss.values(), 'gs-')
    plt.xlabel("Values of 'k'")
    plt.ylabel('WCSS')
    plt.show()

    limit = 10
    # determining number of clusters
    # using silhouette score method
    for k in range(2, limit + 1):
        model = KMeans(n_clusters=k)
        model.fit(x)
        pred = model.predict(x)
        score = silhouette_score(x, pred)
        print('Silhouette Score for k = {}: {:<.3f}'.format(k, score))


# Load data from ai4i2020-edit-1.csv
data = pd.read_csv('ai4i2020-edit-1.csv')

# Drop the machine failure label as this is what is being predicted
X = data.drop('machine_failure', axis=1)
y = data['machine_failure']

# Scale the data so that the different values in columns will not affect prediction
scaler = StandardScaler()
features = scaler.fit_transform(X)

# Convert to scaled dataframe
scaled_df = pd.DataFrame(features, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=42)

# Separate data into two groups one with all machine failure and one with no machine failure
X_failure = X_train[y_train == 1]
X_no_failure = X_train[y_train == 0]

# These find the silhouette score and elbow method for each group comment
find_purity(X_failure)
find_purity(X_no_failure)

# Apply K-Means separately to both sets of data
kmeans_failure = KMeans(n_clusters=8, random_state=42)
kmeans_no_failure = KMeans(n_clusters=3, random_state=42)

# Fit K-Means on the failure and no_failure data
kmeans_failure.fit(X_failure)
kmeans_no_failure.fit(X_no_failure)

# Get centroids for each cluster
centroids_failure = kmeans_failure.cluster_centers_
centroids_no_failure = kmeans_no_failure.cluster_centers_

# Reshape centroids to be 2D arrays
centroids_failure = centroids_failure[:, np.newaxis]
centroids_no_failure = centroids_no_failure[:, np.newaxis]

# Calculate distances from each test point to the centroids
distances_failure = np.linalg.norm(X_test.values - centroids_failure, axis=2)
distances_no_failure = np.linalg.norm(X_test.values - centroids_no_failure, axis=2)


# Assign each test point to the cluster with the nearest centroid
test_labels_failure = np.argmin(distances_failure, axis=0)
test_labels_no_failure = np.argmin(distances_no_failure, axis=0)

# Assign predictions based on the group to which the centroid belongs
final_prediction = np.where(test_labels_failure < test_labels_no_failure, 1, 0)

# Evaluate and print classification report
report = classification_report(y_test, final_prediction)
print('Report:\n', report)
