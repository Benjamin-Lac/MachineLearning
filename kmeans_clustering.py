import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

print("Loading Iris dataset")
iris = load_iris()
X = iris.data
y_true = iris.target

print("\n(a) K-Means clustering with K=4")
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("\n(b) Computing RMSE for each cluster")
for k in range(K):
    cluster_points = X[labels == k]
    centroid = centroids[k]
    diffs = cluster_points - centroid
    rmse_k = np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))
    print(f"    Cluster {k}: {len(cluster_points)} samples, RMSE = {rmse_k:.4f}")

print("\nComparing clusters to true classes")
class_names = iris.target_names

for k in range(K):
    in_cluster = y_true[labels == k]
    total = len(in_cluster)
    print(f"\n    Cluster {k} ({total} samples):")
    for cls_idx, cls_name in enumerate(class_names):
        count = np.sum(in_cluster == cls_idx)
        pct = count / total * 100 if total > 0 else 0
        print(f"      {cls_name}: {count} ({pct:.1f}%)")