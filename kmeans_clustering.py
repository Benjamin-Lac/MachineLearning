"""
kmeans_clustering.py
Unsupervised Learning: K-Means Clustering
Dataset: Iris flower (via sklearn.datasets)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# ── 1. Load Iris dataset (drop labels — unsupervised!) ────────────────────────
iris   = load_iris()
X      = iris.data    # features only — no labels used for clustering
y_true = iris.target  # kept aside ONLY for post-hoc validation

print("Iris dataset loaded.")
print(f"Features : {iris.feature_names}")
print(f"Shape    : {X.shape}")

# ── 2. Apply K-Means with K=4 ─────────────────────────────────────────────────
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X)

labels    = kmeans.labels_       # cluster assignment for each sample
centroids = kmeans.cluster_centers_  # shape (K, n_features)

print(f"\nK-Means fitted with K={K}")
print(f"Cluster assignments (first 20): {labels[:20]}")
print(f"\nCentroids:\n{centroids}")

# ── 3. RMSE per cluster ───────────────────────────────────────────────────────
print("\n── RMSE per Cluster ──")
for k in range(K):
    cluster_points = X[labels == k]
    centroid       = centroids[k]
    # Distance of each point from its centroid
    diffs          = cluster_points - centroid
    rmse_k         = np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))
    print(f"  Cluster {k}: {len(cluster_points):3d} samples | RMSE = {rmse_k:.4f}")

# ── 4. Compare clusters vs true labels ───────────────────────────────────────
print("\n── Cluster vs True Class Breakdown ──")
class_names = iris.target_names

for k in range(K):
    in_cluster = y_true[labels == k]
    total      = len(in_cluster)
    print(f"\n  Cluster {k} ({total} samples):")
    for cls_idx, cls_name in enumerate(class_names):
        count = np.sum(in_cluster == cls_idx)
        pct   = count / total * 100 if total > 0 else 0
        print(f"    {cls_name:15s}: {count:3d} ({pct:.1f}%)")

# ── 5. Analysis questions ─────────────────────────────────────────────────────
print("""
── Analysis ──

Q1: How well do clusters align with true classes?
  Iris has 3 true classes (setosa, versicolor, virginica) and 3 natural clusters
  in feature space. K=4 forces the algorithm to split one natural class into two
  sub-clusters. Setosa is usually perfectly isolated (linearly separable). The
  versicolor/virginica boundary is less clear, so one of those classes may be
  split across two clusters. Overall alignment is moderately good but not perfect.

Q2: Which clusters mix classes?
  Clusters corresponding to versicolor and virginica regions tend to mix — these
  two classes overlap in feature space. The cluster that captures the boundary
  region will contain samples from both. Setosa almost always forms a pure cluster.

Q3: Why might K=4 be suboptimal?
  The true number of natural groups in Iris is 3. Using K=4 over-segments the data:
  one natural class gets artificially split into two clusters to satisfy the K=4
  constraint. This increases within-cluster distances for that class and reduces
  interpretability. The optimal K for Iris is K=3, as confirmed by methods like
  the Elbow method or Silhouette score.
""")