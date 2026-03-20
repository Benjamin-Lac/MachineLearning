"""
kmeans_clustering.py
Unsupervised Learning: K-Means Clustering
Dataset: Iris flower (via sklearn.datasets)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# load iris dataset
iris   = load_iris()
X      = iris.data
y_true = iris.target 

print("Iris dataset loaded.")
print(f"Features : {iris.feature_names}")
print(f"Shape    : {X.shape}")

# k means with k=4
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X)

labels    = kmeans.labels_
centroids = kmeans.cluster_centers_ 

print(f"\nK-Means fitted with K={K}")
print(f"Cluster assignments (first 20): {labels[:20]}")
print(f"\nCentroids:\n{centroids}")

# RMSE for each cluster
print("\n── RMSE per Cluster ──")
for k in range(K):
    cluster_points = X[labels == k]
    centroid       = centroids[k]
    # Distance of each point from its centroid
    diffs          = cluster_points - centroid
    rmse_k         = np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))
    print(f"  Cluster {k}: {len(cluster_points):3d} samples | RMSE = {rmse_k:.4f}")

# compare clusters to true
print("\n── Cluster vs True Class ──")
class_names = iris.target_names

for k in range(K):
    in_cluster = y_true[labels == k]
    total      = len(in_cluster)
    print(f"\n  Cluster {k} ({total} samples):")
    for cls_idx, cls_name in enumerate(class_names):
        count = np.sum(in_cluster == cls_idx)
        pct   = count / total * 100 if total > 0 else 0
        print(f"    {cls_name:15s}: {count:3d} ({pct:.1f}%)")