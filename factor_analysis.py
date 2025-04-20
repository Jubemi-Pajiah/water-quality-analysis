import pandas as pd
from factor_analyzer import FactorAnalyzer

# Step 1: Load Excel data
df = pd.read_excel("water_quality_data.xlsx", sheet_name="GroundWater")

# Step 2: Drop non-numeric or weak columns
X = df.drop(columns=["Sample ID", "pH", "EC", "TDS", "Sal."]) 

# Step 3: Fit FactorAnalyzer with 2 factors (from scree plot)
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(X)

# Step 4: Get and display the factor loadings
loadings = pd.DataFrame(fa.loadings_, index=X.columns, columns=["Factor 1", "Factor 2"])
print("Rotated Factor Loadings:\n")
print(loadings.round(3))
