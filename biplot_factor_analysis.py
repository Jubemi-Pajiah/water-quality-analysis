import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
df = pd.read_excel("water_quality_data.xlsx", sheet_name="GroundWater")

# Step 2: Drop non-numeric or weak columns
X = df.drop(columns=["Sample ID", "pH", "EC", "TDS", "Sal."])

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Perform factor analysis with 2 factors
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(X_scaled)

# Step 5: Get factor scores and loadings
factor_scores = fa.transform(X_scaled)
loadings = fa.loadings_

# Step 6: Plot biplot
plt.figure(figsize=(10, 7))

# Plot sample scores
plt.scatter(factor_scores[:, 0], factor_scores[:, 1], color='purple', alpha=0.6, label='Samples')

# Plot variable vectors (loadings)
for i, var in enumerate(X.columns):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='red', alpha=0.9, head_width=0.05)
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, var, color='red', fontsize=12)

plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Biplot of Factor Analysis (2 Factors)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
