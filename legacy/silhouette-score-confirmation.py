import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

ground_water = {
    'Fe': [0.937, 0.675, 0.736, 0.528, 0.468, 0.663, 0.548, 0.859, 0.513, 0.863, 0.403, 0.413, 0.352, 0.424, 0.467, 0.392, 0.416, 0.386, 0.413],
    'Zn': [0.453, 0.246, 0.311, 0.184, 0.236, 0.275, 0.216, 0.259, 0.205, 0.416, 0.257, 0.203, 0.183, 0.257, 0.244, 0.217, 0.368, 0.207, 0.245],
    'Cu': [0.256, 0.087, 0.143, 0.097, 0.113, 0.102, 0.117, 0.108, 0.094, 0.114, 0.075, 0.072, 0.082, 0.063, 0.065, 0.078, 0.094, 0.077, 0.091],
    'Mn': [0.092, 0.038, 0.056, 0.046, 0.084, 0.057, 0.089, 0.069, 0.037, 0, 0, 0, 0, 0.017, 0.026, 0.022, 0.027, 0.022, 0.032],
    'Ni': [0.035, 0.023, 0.031, 0.022, 0.037, 0.026, 0.036, 0.028, 0.017, 0.048, 0.033, 0.032, 0.043, 0.028, 0.034, 0.029, 0.0311, 0.041, 0.042],
    'Cr': [0.048, 0.034, 0.046, 0.035, 0.043, 0.032, 0.05, 0.025, 0.024, 0.032, 0.022, 0.019, 0.026, 0.017, 0.025, 0.024, 0.039, 0.032, 0.025],
    'Co': [0.04, 0.027, 0.034, 0.028, 0.031, 0.032, 0.04, 0.034, 0.023, 0.057, 0.046, 0.047, 0.059, 0.01, 0.014, 0.025, 0.021, 0.013, 0.019],
}

df = pd.DataFrame(ground_water)

silhouette_scores = []
for i in range(2, 11):  # Test clusters from 2 to 10 (silhouette score needs at least 2)
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(df)
    silhouette_scores.append(silhouette_score(df, labels))

plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()