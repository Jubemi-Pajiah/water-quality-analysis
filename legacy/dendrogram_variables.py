import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Step 1: Load your data
df = pd.read_excel("water_quality_data.xlsx", sheet_name="StorageWater")

# Step 2: Drop non-metal and ID columns (keep only heavy metal columns)
X = df.drop(columns=["Sample ID", "pH", "EC", "TDS", "Sal."], errors="ignore")

# Step 3: Transpose the data so we cluster columns (variables) instead of rows (samples)
X_T = X.T

# Step 4: Compute pairwise distances between metals (rows now) and generate linkage matrix
dist_matrix = pdist(X_T, metric='euclidean')
Z = linkage(dist_matrix, method='ward')

# Step 5: Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=X_T.index, orientation='right')
plt.title("Dendrogram of Heavy Metals")
plt.xlabel("Distance")
plt.ylabel("Heavy Metals")
plt.tight_layout()
plt.show()
