import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib   
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

def normalize_data(df):
    # normalize cost column
    cost_columns = [col for col in df.columns if col.endswith('_cost')]
    
    if not cost_columns:
        print("No columns ending with '_cost' found")
        return df
    all_cost_values = df[cost_columns].values.flatten()
    all_cost_values = all_cost_values[~pd.isna(all_cost_values)]  # Remove NaN values
    
    global_mean = all_cost_values.mean()
    global_std = all_cost_values.std()
    
    for col in cost_columns:
        df[col] = (df[col] - global_mean) / global_std
        
    print(f"Applied Standard Scaling to {len(cost_columns)} cost columns")
    print(f"Mean: {global_mean:.6f}, Std: {global_std:.6f}")
    return df

df = pd.read_csv('embedded_routerbench_sample.csv')
# convert openai_embedding to numpy array
df = normalize_data(df)
df['openai_embedding'] = df['openai_embedding'].apply(eval)

matrix = np.vstack(df.openai_embedding.values)
n_clusters = 8

kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)
df['k_means_cluster'] = kmeans.labels_

# perform DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10, metric='cosine')
dbscan.fit(matrix)
df['dbscan_cluster'] = dbscan.labels_

# visualize clusters
# matrix = df.openai_embedding.to_list()

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

colors = ["red", "darkorange", "gold", "cyan", "darkgreen", "blue", "purple", "pink", "gray", "brown"]
x = [x for x,y in vis_dims]
y = [y for x,y in vis_dims]
color_indices = df.k_means_cluster.values - 1

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
plt.title("RouterBench clusters using t-SNE")

# print average stats for each cluster
for i in range(n_clusters):
    print(f"Cluster {i+1}:")
    print(df[df['k_means_cluster'] == i].describe())
    # write to file
    df[df['k_means_cluster'] == i].describe().to_csv(f'cluster_stats/k_means_cluster_{i+1}_describe.csv', index=True)
    print("\n")

plt.savefig('routerbench_clusters.png')