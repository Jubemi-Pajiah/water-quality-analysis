import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_excel("water_quality_data.xlsx", sheet_name="StorageWater")

# Drop non-numeric columns
X = df.drop(columns=["Sample ID", "pH", "EC", "TDS", "Sal."])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create linkage matrix
linked = linkage(X_scaled, method='ward')

# Plot horizontal dendrogram
plt.figure(figsize=(10, 8))
dendrogram(linked,
           labels=df["Sample ID"].values,
           orientation='right')  # Horizontal style
plt.title("Dendrogram of Water Samples")
plt.xlabel("Euclidean Distance")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()
