import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import json

# === Step 1: Load normalized data ===
input_file = "KF_CS221_ProjectNormalizedData.csv"
df_full = pd.read_csv(input_file)

state_col = df_full.columns[0]

# Step 1.1: Exclude specified states (if any)
states_to_exclude = ['São Paulo']  # Modify list to exclude other states
if states_to_exclude:
    df = df_full[~df_full[state_col].isin(states_to_exclude)].copy()
else:
    df = df_full.copy()

excluded_count = len(df_full) - len(df)

states = df[state_col]
features = df.drop(columns=[state_col])

# === Step 2: Find optimal number of clusters using silhouette score ===
print("Finding optimal number of clusters...")
scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels_test = kmeans_test.fit_predict(features)
    score = silhouette_score(features, labels_test)
    scores.append(score)

best_k = k_range[np.argmax(scores)]
print(f"✅ Optimal number of clusters based on silhouette score: {best_k}")

# Plot silhouette score vs. number of clusters
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), scores, marker='o', linestyle='-')
plt.xticks(list(k_range))
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. k (Main Model)")
plt.grid(True)
plt.tight_layout()
plt.savefig("KF_CS221_Main_Silhouette_vs_k.png", dpi=300)
plt.show()

# === Step 3: Fit final KMeans model using optimal k ===
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(features)

# Compute clustering metrics
inertia_main = kmeans.inertia_
sil_score_main = silhouette_score(features, clusters)
cluster_sizes_main = {int(i): int((clusters == i).sum()) for i in range(best_k)}

# === Step 4: Combine clustering results with state names ===
df_clusters = df.copy()
df_clusters["Cluster"] = clusters

# === Step 5: PCA for 2D visualization ===
pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters,
            cmap='tab10', s=100, edgecolors='k')
for i, name in enumerate(states):
    plt.text(reduced[i, 0], reduced[i, 1], name, fontsize=8)
plt.title("Clustered Brazilian States (2D PCA Projection) — Main Model")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("KF_CS221_Main_Clusters_Plot.png", dpi=300)
plt.show()

# === Step 6: Print summary by cluster ===
print("\n=== Cluster Profiles (Main Model) ===")
for i in range(best_k):
    group = df_clusters[df_clusters["Cluster"] == i]
    print(f"\nCluster {i} — {len(group)} states")
    print("States:", ", ".join(group[state_col].tolist()))
    cluster_means = group.drop(columns=["Cluster", state_col]).mean().sort_values(ascending=False)
    print("Top characteristics:")
    print(cluster_means.head(5).round(2))

# === Step 7: Save outputs ===
df_clusters.to_csv("KF_CS221_MainClusterOutput.csv", index=False)

summary_table = df_clusters.drop(columns=[state_col]).groupby("Cluster").mean().round(2)
summary_table.to_csv("KF_CS221_ClusterSummary.csv")

# Plot cluster sizes
plt.figure(figsize=(6, 4))
plt.bar(cluster_sizes_main.keys(), cluster_sizes_main.values(), color='lightgreen', edgecolor='k')
plt.xticks(list(cluster_sizes_main.keys()))
plt.xlabel("Cluster ID")
plt.ylabel("Number of States")
plt.title("Main Model Cluster Sizes (excluding: {})".format(", ".join(states_to_exclude) if states_to_exclude else "None"))
plt.tight_layout()
plt.savefig("KF_CS221_Main_Cluster_Sizes.png", dpi=300)
plt.show()

# Save metrics to JSON
main_metrics = {
    "states_excluded": states_to_exclude,
    "excluded_count": excluded_count,
    "n_clusters": best_k,
    "inertia": inertia_main,
    "silhouette_score": sil_score_main,
    "cluster_sizes": cluster_sizes_main
}
with open("KF_CS221_Main_Metrics.json", "w") as f:
    json.dump(main_metrics, f, indent=2)

print("\n✅ All done. Files saved:")
print("- KF_CS221_MainClusterOutput.csv")
print("- KF_CS221_ClusterSummary.csv")
print("- KF_CS221_Main_Clusters_Plot.png")
print("- KF_CS221_Main_Silhouette_vs_k.png")
print("- KF_CS221_Main_Cluster_Sizes.png")
print("- KF_CS221_Main_Metrics.json")
