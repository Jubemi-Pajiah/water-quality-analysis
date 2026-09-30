import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer

# Step 1: Load your dataset
df = pd.read_excel("water_quality_data.xlsx", sheet_name="GroundWater")

# Step 2: Drop non-numeric or weak variables
X = df.drop(columns=["Sample ID", "pH", "EC", "TDS", "Sal."])

# Step 3: Fit Factor Analysis (using 2 factors, based on scree plot)
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(X)

# Step 4: Extract loadings and convert to DataFrame
loadings = pd.DataFrame(fa.loadings_, 
                        index=X.columns, 
                        columns=["Factor 1", "Factor 2"])

# Step 5: Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Factor Loadings")
plt.tight_layout()
plt.show()
