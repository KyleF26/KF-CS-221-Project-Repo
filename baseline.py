import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json

# Parameters
input_file = "KF_CS221_ProjectNormalizedData.csv"
n_clusters = 5

# Step 1: Load the normalized data
df = pd.read_csv(input_file)

# Step 2: Separate state names and features
state_col = df.columns[0]
states = df[state_col]
features = df.drop(columns=[state_col])

# Step 3: Run K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(features)

# Step 4: Compute clustering metrics
inertia_baseline = kmeans.inertia_
sil_score_baseline = silhouette_score(features, clusters)
cluster_sizes_baseline = {int(i): int((clusters == i).sum()) for i in range(n_clusters)}

# Step 5: Attach cluster labels to the state names
df_clusters = pd.DataFrame({
    state_col: states,
    "Cluster": clusters
})

# Step 6: Print results grouped by cluster
print("States grouped by cluster (baseline):\n")
for cluster_num in sorted(df_clusters["Cluster"].unique()):
    cluster_states = df_clusters[df_clusters["Cluster"] == cluster_num][state_col].tolist()
    print(f"Cluster {cluster_num}:")
    for state in cluster_states:
        print(f"  - {state}")
    print()

# Step 7: Save cluster assignments
df_clusters.to_csv("KF_CS221_Clusters.csv", index=False)

# Step 8: Plot clusters using PCA
try:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters,
                          cmap='Set1', s=100, edgecolors='k')
    plt.title("Baseline K-Means Clusters of Brazilian States (2D PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("KF_CS221_Baseline_Clusters_Plot.png", dpi=300)
    plt.show()

    loadings = pd.DataFrame(pca.components_, columns=features.columns, index=["PCA 1", "PCA 2"])
    print("\nPCA Component Loadings (baseline):\n")
    print(loadings.T.sort_values("PCA 1", ascending=False))

except ImportError:
    print("Install matplotlib and sklearn.decomposition.PCA to enable plotting.")

# Step 9: Bar chart of cluster sizes
plt.figure(figsize=(6, 4))
plt.bar(cluster_sizes_baseline.keys(), cluster_sizes_baseline.values(), color='skyblue', edgecolor='k')
plt.xticks(list(cluster_sizes_baseline.keys()))
plt.xlabel("Cluster ID")
plt.ylabel("Number of States")
plt.title("Baseline Cluster Sizes")
plt.tight_layout()
plt.savefig("KF_CS221_Baseline_Cluster_Sizes.png", dpi=300)
plt.show()

# Step 10: Save clustering metrics
baseline_metrics = {
    "n_clusters": n_clusters,
    "inertia": inertia_baseline,
    "silhouette_score": sil_score_baseline,
    "cluster_sizes": cluster_sizes_baseline
}
with open("KF_CS221_Baseline_Metrics.json", "w") as f:
    json.dump(baseline_metrics, f, indent=2)

print("\n▶ Baseline metrics saved to KF_CS221_Baseline_Metrics.json")
